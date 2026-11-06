import numpy as np
import time

def design(problem):
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    # Coordinates for the 32x32 grid, centered at 0
    idx = np.arange(size)
    x = (idx + 0.5) / size - 0.5
    y = (idx + 0.5) / size - 0.5
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Precompute C4v-symmetric basis functions
    # We use a set of low-order harmonics that are invariant under C4v
    # f(x,y) = A0 + A1*B1 + A2*B2 + A3*B3 + A4*B4 + A5*B5 + A6*B6 + A7*B7
    
    # B0: Constant
    B0 = np.ones((size, size))
    
    # B1: (1,1) harmonic -> cos(2pi(x+y)) + cos(2pi(x-y))
    B1 = np.cos(2 * np.pi * (X + Y)) + np.cos(2 * np.pi * (X - Y))
    
    # B2: (2,0) harmonic -> cos(4pi x) + cos(4pi y)
    B2 = np.cos(4 * np.pi * X) + np.cos(4 * np.pi * Y)
    
    # B3: (2,2) harmonic -> cos(4pi(x+y)) + cos(4pi(x-y))
    B3 = np.cos(4 * np.pi * (X + Y)) + np.cos(4 * np.pi * (X - Y))
    
    # B4: (3,0) harmonic -> cos(6pi x) + cos(6pi y)
    B4 = np.cos(6 * np.pi * X) + np.cos(6 * np.pi * Y)
    
    # B5: (3,3) harmonic -> cos(6pi(x+y)) + cos(6pi(x-y))
    B5 = np.cos(6 * np.pi * (X + Y)) + np.cos(6 * np.pi * (X - Y))
    
    # B6: (4,0) harmonic -> cos(8pi x) + cos(8pi y)
    B6 = np.cos(8 * np.pi * X) + np.cos(8 * np.pi * Y)
    
    # B7: (4,4) harmonic -> cos(8pi(x+y)) + cos(8pi(x-y))
    B7 = np.cos(8 * np.pi * (X + Y)) + np.cos(8 * np.pi * (X - Y))
    
    bases = [B0, B1, B2, B3, B4, B5, B6, B7]
    N_BASIS = len(bases)
    
    # PWE Surrogate Parameters
    M_PWE = 5  # Indices -5 to 5, so 11x11 = 121 PWs (faster than 289, still accurate enough for screening)
    G_indices = np.arange(-M_PWE, M_PWE + 1)
    Gx, Gy = np.meshgrid(G_indices, G_indices, indexing='ij')
    Gx_flat = Gx.flatten()
    Gy_flat = Gy.flatten()
    N_PW = len(Gx_flat)
    
    # Precompute pairwise differences for Epsilon matrix construction
    dx_matrix = Gx_flat[:, None] - Gx_flat[None, :]
    dy_matrix = Gy_flat[:, None] - Gy_flat[None, :]
    
    # Key k-points in the irreducible BZ
    # We sample a denser set of points to better capture the band edges
    k_points = []
    # Gamma to X (0,0) to (0.5, 0)
    for i in range(6):
        k_points.append((0.5 * i / 5.0, 0.0))
    # X to M (0.5, 0) to (0.5, 0.5)
    for i in range(1, 6):
        k_points.append((0.5, 0.5 * i / 5.0))
    # M to Gamma (0.5, 0.5) to (0, 0) - but we only need the line kx=ky
    for i in range(1, 6):
        k_points.append((0.5 * (5.0 - i) / 5.0, 0.5 * (5.0 - i) / 5.0))
        
    # Remove duplicates
    k_points_unique = []
    for kp in k_points:
        if kp not in k_points_unique:
            k_points_unique.append(kp)
    k_points = k_points_unique

    def calculate_surrogate_gap(cell):
        """
        Calculates an approximate gap using a small PWE expansion.
        Returns the score (gap-midgap ratio).
        """
        # Compute Fourier coefficients of the pixel map
        # cell is 0/1. We need epsilon.
        # epsilon = eps_low + (eps_high - eps_low) * cell
        eps_cell = eps_low + (eps_high - eps_low) * cell
        
        fft_vals = np.fft.fft2(eps_cell)
        fft_vals /= (size * size)
        
        def get_eps(Gx_val, Gy_val):
            ix = int(Gx_val) % size
            iy = int(Gy_val) % size
            return fft_vals[ix, iy]
        
        min_omega_5 = np.inf
        max_omega_4 = 0.0
        
        for kx, ky in k_points:
            kGx = kx + Gx_flat
            kGy = ky + Gy_flat
            
            D = np.outer(kGx, kGx) + np.outer(kGy, kGy)
            
            # Vectorized computation of epsilon matrix
            # E[i,j] = eps(G_i - G_j)
            E = np.zeros((N_PW, N_PW), dtype=complex)
            # We can use the precomputed dx, dy matrices
            # But get_eps is a lookup. 
            # To speed up, we can precompute the E matrix structure? No, E depends on cell.
            # However, the indices (i,j) are fixed.
            # Let's just loop, it's 121x121 = 14641 ops. Fast enough.
            
            # Optimization: Use numpy indexing on fft_vals
            # ix = (Gx_flat[:, None] - Gx_flat[None, :]) % size
            # iy = (Gy_flat[:, None] - Gy_flat[None, :]) % size
            # E = fft_vals[ix, iy]
            
            ix = (Gx_flat[:, None] - Gx_flat[None, :]) % size
            iy = (Gy_flat[:, None] - Gy_flat[None, :]) % size
            E = fft_vals[ix, iy]
            
            A = D * E
            A = (A + A.conj().T) / 2.0
            A = np.real(A)
            
            try:
                eigvals = np.linalg.eigvalsh(A)
            except np.linalg.LinAlgError:
                return -np.inf
                
            eigvals = np.sort(eigvals)
            
            if len(eigvals) >= 5:
                w4 = eigvals[3]
                w5 = eigvals[4]
                
                if w4 > 0 and w5 > 0:
                    max_omega_4 = max(max_omega_4, w4)
                    min_omega_5 = min(min_omega_5, w5)
        
        if min_omega_5 == np.inf or max_omega_4 == 0.0:
            return -np.inf
            
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        
        if midgap == 0:
            return 0.0
            
        score = gap / midgap
        return score

    def generate_cell(params):
        """Generate a binary cell from Fourier parameters."""
        A = params[:N_BASIS]
        field = np.zeros((size, size))
        for i in range(N_BASIS):
            field += A[i] * bases[i]
        
        # Normalize field to have mean 0 and std 1 to make thresholding stable?
        # Or just threshold at 0.5?
        # The offset is part of A (A0).
        # Let's threshold at 0.5
        cell_raw = (field > 0.5).astype(int)
        
        # Enforce C4v symmetry
        sym_cells = [
            cell_raw,
            cell_raw[:, ::-1],
            cell_raw[::-1, :],
            cell_raw.T,
            cell_raw[::-1, ::-1],
            cell_raw.T[::-1, :],
            cell_raw[:, ::-1].T,
            cell_raw[::-1, :].T
        ]
        
        avg_cell = np.mean(sym_cells, axis=0)
        cell_sym = (avg_cell >= 0.5).astype(int)
        
        return cell_sym

    def objective(params):
        cell = generate_cell(params)
        fill = np.sum(cell) / (size * size)
        if fill < 0.10 or fill > 0.90:
            return -10.0 # Penalize extreme fill factors
        score = calculate_surrogate_gap(cell)
        return -score # Minimize negative score

    # Initial guess
    # A0 is offset. A1..A7 are amplitudes.
    # Start with a known good structure: e.g. square rod
    # Square rod corresponds to low order harmonics.
    # Let's try a random start or a heuristic start.
    
    best_score = -np.inf
    best_params = np.zeros(N_BASIS)
    
    # Stage 1: Random Search
    np.random.seed(42)
    n_random = 20
    for _ in range(n_random):
        # A0 in [0.3, 0.7]
        A0 = np.random.uniform(0.3, 0.7)
        # Amplitudes in [-1, 1]
        A_rest = np.random.uniform(-1.0, 1.0, size=N_BASIS - 1)
        params = np.concatenate(([A0], A_rest))
        
        score = -objective(params)
        if score > best_score:
            best_score = score
            best_params = params.copy()
            
    # Stage 2: Local Optimization (Nelder-Mead)
    # We use scipy.optimize if available, otherwise a simple implementation
    try:
        from scipy.optimize import minimize
        # Maximize score -> Minimize -score
        result = minimize(objective, best_params, method='Nelder-Mead', 
                          options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-4})
        if result.fun < -best_score: # i.e. -result.fun > best_score
            best_score = -result.fun
            best_params = result.x
    except ImportError:
        # Fallback: Simple coordinate descent or random perturbation
        # Given the time budget, we can do a few rounds of random perturbation around best
        for _ in range(50):
            perturbation = np.random.normal(0, 0.1, size=N_BASIS)
            new_params = best_params + perturbation
            score = -objective(new_params)
            if score > best_score:
                best_score = score
                best_params = new_params.copy()
            else:
                # Shrink perturbation?
                pass

    # Generate final cell
    best_cell = generate_cell(best_params)
    
    # Final check and return
    return best_cell.tolist()
