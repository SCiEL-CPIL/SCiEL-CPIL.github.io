"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def design(problem):
    """Design a 32x32-pixel unit cell to maximize the TM 4-5 band gap."""
    size = problem["pixels"]
    eps_high = problem["eps_high"]
    
    start_time = time.time()
    time_limit = 195.0  # Safety margin before 230s kill

    # --- 1. Parametric Representation ---
    # We use a sum of cosines to ensure C4v symmetry automatically.
    # eps(x, y) = 0.5 * (1 + tanh(sum c_i cos(k_i . r)))
    # This maps to a smooth dielectric profile which, when thresholded, gives a valid 0/1 cell.
    # The Fourier coefficients of a sum of cosines are sparse and can be computed analytically or via FFT.
    
    # Basis functions (symmetric under C4v):
    # cos(2*pi*x), cos(2*pi*y), cos(2*pi*(x+y)), cos(2*pi*(x-y)), cos(2*pi*(x-y)/2) ...
    # Let's stick to integer frequencies to match the lattice periodicity naturally.
    # Frequencies: (1,0), (0,1), (1,1), (1,-1), (2,0), (0,2), (2,2), (2,-2)
    
    freqs = [
        (1, 0), (0, 1), (1, 1), (1, -1),
        (2, 0), (0, 2), (2, 2), (2, -2),
        (3, 0), (0, 3), (3, 3), (3, -3)
    ]
    
    def get_eps_grid(params):
        """
        Generate a 32x32 grid of epsilon values (1 to 12) from parameters.
        params: array of coefficients for the cosine terms.
        """
        x = (np.arange(size) + 0.5) / size
        y = (np.arange(size) + 0.5) / size
        X, Y = np.meshgrid(x, y, indexing='xy')
        
        val = np.zeros_like(X)
        for i, (fx, fy) in enumerate(freqs):
            val += params[i] * np.cos(2 * np.pi * (fx * X + fy * Y))
        
        # Map to [1, 12] using a sigmoid/tanh to keep values bounded
        # tanh maps R -> (-1, 1). We want 0.5*(1+tanh) -> (0, 1).
        # Then scale to [1, 12].
        norm_val = 0.5 * (1 + np.tanh(val))
        eps_grid = 1 + (eps_high - 1) * norm_val
        return eps_grid

    def get_cell(params):
        """Threshold the continuous profile to a 0/1 integer cell."""
        eps_grid = get_eps_grid(params)
        # Threshold at the midpoint (6.5) to get a balanced fill factor
        # Actually, better to threshold such that the average is ~0.5
        # But for photonic crystals, fill factor matters. 
        # Let's just binarize: 1 if eps > 6.5 else 0.
        cell = (eps_grid > (1 + eps_high) / 2).astype(np.uint8)
        
        # Enforce strict symmetry (just in case of float errors)
        cell = (cell + cell[::-1, ::-1]) // 2
        cell = (cell + cell.T) // 2
        
        return cell.tolist()

    # --- 2. Fast Surrogate Evaluator ---
    # Use a small plane wave basis (N=5, 25 PWs) to estimate the gap.
    # This is much faster than the full 289 PW calculation.
    
    N_small = 5
    idx_small = np.arange(-N_small//2, N_small//2 + 1)
    G_small = [(i, j) for i in idx_small for j in idx_small]
    N_pw = len(G_small)
    
    # Precompute G vectors and |G|^2
    G_arr = np.array(G_small, dtype=float)
    G2_arr = np.sum(G_arr**2, axis=1)
    
    # Critical k-points in the irreducible BZ (in units of 2pi/a)
    # Gamma (0,0), M (0.5, 0.5), X (0.5, 0)
    # And some interior points to avoid missing the gap minimum/maximum
    k_points = [
        (0.0, 0.0),    # Gamma
        (0.5, 0.5),    # M
        (0.5, 0.0),    # X
        (0.25, 0.25),  # Gamma-M mid
        (0.5, 0.25),   # M-X mid
        (0.25, 0.0),   # Gamma-X mid
        (0.1, 0.1),    # Interior
        (0.4, 0.2),    # Interior
    ]

    def fast_score(params):
        """
        Calculate the approximate gap-midgap ratio for the given parameters.
        Returns negative score if no gap or error.
        """
        # Compute Fourier coefficients of epsilon
        # eps_G = (1/A) * sum_r eps(r) exp(-i G.r)
        # For a sum of cosines, this is sparse, but for the surrogate,
        # we can just compute the FFT of the grid for the required Gs.
        # Since N_small=5, we need G components from -2 to 2.
        # We can use the grid directly.
        
        eps_grid = get_eps_grid(params)
        
        # FFT of the grid. Note: numpy fftfreq gives frequencies in cycles/sample.
        # G = 2*pi * (n_x, n_y).
        # We need eps_{G} for G in the set of differences of G_small vectors.
        # Max difference is 4. So we need frequencies up to 4.
        # Let's compute a 9x9 FFT (frequencies -4 to 4) to be safe.
        
        N_fft = 9
        # Pad or resize? The grid is 32x32.
        # We can just compute the DFT for the specific Gs we need.
        # Gs needed for M_ij = eps_{G_i - G_j} * |G_j + k|^2
        # G_i, G_j in [-2, 2]. Diff in [-4, 4].
        
        # Precompute phase factors for all needed Gs?
        # Or just compute the 9x9 FFT of the 32x32 grid?
        # No, the grid is 32x32 pixels. The fundamental period is 1.
        # The Fourier coefficient is approx (1/32^2) * sum_{px} eps(px) exp(-i G px).
        
        # Let's precompute the pixel coordinates
        px = (np.arange(size) + 0.5) / size
        PX, PY = np.meshgrid(px, py, indexing='xy')
        
        # We need eps_G for Gx, Gy in [-4, 4]
        G_range = np.arange(-4, 5)
        eps_f = np.zeros((9, 9), dtype=complex)
        
        # Vectorized computation of all 81 coefficients at once?
        # Phase matrix: 9x9x32x32 might be too big for memory if not careful, but 81*1024 is fine.
        # Let's do it in a loop for the 81 coeffs, it's fast enough in numpy.
        
        for i, gx in enumerate(G_range):
            for j, gy in enumerate(G_range):
                # G = 2*pi*(gx, gy)
                phase = np.exp(-1j * 2 * np.pi * (gx * PX + gy * PY))
                eps_f[i, j] = np.sum(eps_grid * phase) / (size * size)
        
        # Find max omega_4 and min omega_5
        max_omega4 = 0.0
        min_omega5 = np.inf
        
        for kx, ky in k_points:
            # Construct matrix M and D
            # M_ij = eps_{G_i - G_j} * |G_j + k|^2
            # D_ii = |G_i + k|^2
            
            # G_i + k
            Kg = G_arr + np.array([kx, ky])
            Kg2 = np.sum(Kg**2, axis=1)
            Kg2[Kg2 < 1e-10] = 1e-10 # Avoid division by zero at Gamma for G=0
            
            D = Kg2
            D_inv_sqrt = 1.0 / np.sqrt(D)
            
            M = np.zeros((N_pw, N_pw), dtype=complex)
            
            # Fill M
            for i in range(N_pw):
                gi, gj = G_small[i]
                for j in range(N_pw):
                    gij, gjj = G_small[j]
                    dgx = gi - gij
                    dgy = gj - gjj
                    # Map dgx, dgy to eps_f index
                    # eps_f index 0 corresponds to G=-4
                    ix = int(dgx) + 4
                    iy = int(dgy) + 4
                    M[i, j] = eps_f[ix, iy] * D[j]
            
            # Symmetric eigenvalue problem: A y = lambda y
            # A = D^{-1/2} M D^{-1/2}
            A = M * D_inv_sqrt[:, None] * D_inv_sqrt[None, :]
            A = (A + A.conj().T) / 2.0 # Ensure Hermitian
            
            try:
                evals = np.linalg.eigvalsh(A)
                evals = np.real(evals)
                evals = np.sort(evals)
                
                # We need bands 4 and 5 (indices 3 and 4)
                if len(evals) >= 5:
                    omega4 = np.sqrt(evals[3])
                    omega5 = np.sqrt(evals[4])
                    max_omega4 = max(max_omega4, omega4)
                    min_omega5 = min(min_omega5, omega5)
            except:
                pass
        
        if min_omega5 == np.inf or max_omega4 == 0:
            return -1.0
        
        gap = 2 * (min_omega5 - max_omega4)
        midgap = min_omega5 + max_omega4
        if midgap == 0:
            return -1.0
        
        return gap / midgap

    # --- 3. Optimization ---
    
    # Initial guess: Try to create a 4-rod-like structure using the Fourier basis.
    # A 4-rod structure has high (1,1) and (1,-1) components.
    # Let's start with a few different random initializations and pick the best.
    
    np.random.seed(42)
    
    best_params = None
    best_score = -np.inf
    
    # Try a few random starts
    for _ in range(5):
        # Random coefficients
        p0 = np.random.randn(len(freqs)) * 0.5
        # Bias towards (1,1) and (1,-1) which are good for M-point gaps
        p0[2] += 1.0 # (1,1)
        p0[3] += 1.0 # (1,-1)
        
        # Nelder-Mead
        try:
            res = minimize(
                lambda p: -fast_score(p), # Minimize negative score
                p0,
                method='Nelder-Mead',
                options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-4}
            )
            
            if time.time() - start_time > time_limit - 10:
                break
                
            score = -res.fun
            if score > best_score:
                best_score = score
                best_params = res.x
                
        except Exception as e:
            pass

    # If optimization failed or didn't find anything, try a default
    if best_params is None:
        best_params = np.zeros(len(freqs))
        best_params[2] = 1.0
        best_params[3] = 1.0

    # Final check on time
    if time.time() - start_time > time_limit - 2:
        # Return a simple fallback if we are running out of time
        # Just threshold the best params found
        pass

    # Convert best params to cell
    cell = get_cell(best_params)
    
    # Ensure valid output format
    if len(cell) != size or len(cell[0]) != size:
        return [[0]*size for _ in range(size)]
        
    return cell

# EVOLVE-BLOCK-END
