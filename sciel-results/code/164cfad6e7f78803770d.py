"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def design(problem):
    """
    Design a symmetric 32x32-pixel unit cell using spectral-domain optimization.
    
    Strategy:
    1. Define the dielectric permittivity epsilon(x,y) as a sum of low-order 
       Fourier modes that respect the square symmetry (C4v) and inversion symmetry.
    2. The modes are chosen to couple strongly to the 4th and 5th TM bands.
    3. Use an internal plane-wave solver to compute the band structure and 
       the gap-midgap ratio.
    4. Optimize the amplitudes and phases of these modes using Nelder-Mead.
    5. Threshold the optimized continuous permittivity field to 0/1 pixels.
    6. Enforce symmetry by averaging with its transpose and reverses.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # 120s budget, keep 10s margin
    
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    lower_band = problem["lower_band"]
    
    # 1. Set up the internal plane-wave solver
    # We need to solve the TM band structure.
    # The equation is: sum_G' [ (k+G)^2 / epsilon_G-G' ] c_G' = omega^2 c_G
    # Or in the standard form for TM: 
    # det( (k+G)^2 delta_GG' - omega^2 epsilon_G-G' ) = 0 is NOT correct for TM.
    # For TM, the field is perpendicular to the plane. The equation is:
    # sum_G' ( (k+G')^2 / epsilon_{G-G'} ) E_{G'} = omega^2 E_G
    # This requires inverting the permittivity tensor in Fourier space.
    
    # Let's implement a simplified but accurate enough solver for 289 plane waves.
    # We need the Fourier coefficients of 1/epsilon(x,y).
    
    def get_fourier_coeffs(cell_binary):
        """Compute Fourier coefficients of 1/epsilon for a binary cell."""
        N = size
        # Create the permittivity grid
        eps_grid = np.where(np.array(cell_binary) == 1, eps_high, eps_low)
        # We need 1/epsilon
        inv_eps_grid = 1.0 / eps_grid
        
        # Compute Fourier coefficients using FFT
        # The Fourier series is: f(x,y) = sum_G f_G exp(i G . r)
        # f_G = (1/N^2) sum_{x,y} f(x,y) exp(-i G . r)
        # Using numpy FFT: fft2(f) gives sum f(x,y) exp(-i 2pi (kx x + ky y) / N)
        # We need to map this to G vectors.
        
        # The grid is N x N.
        # G = (Gx, Gy) where Gx = 2pi * n_x, Gy = 2pi * n_y.
        # The FFT index k corresponds to n = k if k < N/2, else k - N.
        
        F = np.fft.fft2(inv_eps_grid)
        # Normalize by N^2
        F = F / (N * N)
        
        # We need to extract coefficients for specific G vectors.
        # The evaluator uses 289 plane waves, which corresponds to a cutoff.
        # 289 = 17^2. This suggests G vectors with |G| <= some cutoff.
        # For a square lattice, the number of points in a circle of radius R is ~ pi R^2.
        # If we take G components from -8 to 8, that's 17x17 = 289 points.
        # So the cutoff is |Gx|, |Gy| <= 8.
        
        # We will build the matrix for the eigenvalue problem.
        # We need the set of G vectors.
        G_list = []
        for gx in range(-8, 9):
            for gy in range(-8, 9):
                G_list.append((gx, gy))
        
        # Map G to FFT indices
        def g_to_idx(gx, gy):
            idx_x = gx % N
            idx_y = gy % N
            return idx_y * N + idx_x  # Flatten index for 2D array? No, we need 2D indices.
        
        # Let's just use the FFT array directly.
        # F[gx, gy] in FFT convention.
        # numpy fft2 output F[u, v] corresponds to frequency u, v.
        # u=0..N-1. 
        # The physical frequency is u if u < N/2, else u-N.
        
        # Let's extract the coefficients for our G_list.
        coeffs = {}
        for gx, gy in G_list:
            # Map to FFT indices
            u = gx % N
            v = gy % N
            coeffs[(gx, gy)] = F[v, u]
            
        return coeffs, G_list

    def compute_bands(cell_binary):
        """Compute the first 5 TM bands at high-symmetry points."""
        # This is a simplified solver. We will evaluate at a few key k-points.
        # The gap is usually determined at the M point (pi, 0) or X point (pi, 0) or Gamma (0,0).
        # For TM bands 4 and 5, the gap is often at M or X.
        # Let's evaluate at M=(pi,0), X=(pi,0) (same for square), and Gamma=(0,0).
        # Actually, for square lattice, M is (pi, pi) and X is (pi, 0).
        # Irreducible BZ is triangle (0,0) - (pi,0) - (pi,pi).
        
        # We need to solve the eigenvalue problem:
        # sum_G' ( (k+G')^2 / epsilon_{G-G'} ) c_{G'} = omega^2 c_G
        
        # This is a generalized eigenvalue problem if we don't invert epsilon.
        # But we have 1/epsilon in Fourier space.
        # The equation is:
        # sum_G' ( (k+G')^2 delta_{G,G'} - omega^2 epsilon_{G-G'} ) c_{G'} = 0  <-- This is TE.
        # For TM:
        # sum_G' ( (k+G')^2 / epsilon_{G-G'} ) c_{G'} = omega^2 c_G
        # This is a standard eigenvalue problem: A c = omega^2 c
        # where A_{G,G'} = (k+G')^2 delta_{G,G'} / epsilon_{G-G'}? No.
        # The operator is multiplication by 1/epsilon in real space, which is convolution in Fourier space.
        # So (1/eps * E)_G = sum_G' (1/eps)_{G-G'} E_{G'}
        # Then (k+G)^2 (1/eps * E)_G = omega^2 E_G
        # So sum_G' (k+G)^2 (1/eps)_{G-G'} E_{G'} = omega^2 E_G
        # This is NOT a standard eigenvalue problem because (k+G)^2 depends on G, not G-G'.
        # We can write it as:
        # sum_G' [ (k+G)^2 (1/eps)_{G-G'} ] E_{G'} = omega^2 E_G
        # This is a matrix M where M_{G,G'} = (k+G)^2 (1/eps)_{G-G'}.
        
        coeffs, G_list = get_fourier_coeffs(cell_binary)
        N_pw = len(G_list)
        
        # Precompute G vectors
        G_vecs = np.array(G_list)
        
        # We need (1/eps)_{G-G'} for all pairs.
        # This is a circulant-like matrix structure but in 2D.
        # We can build the matrix M for a given k.
        
        # To save time, we only evaluate at 3 points: Gamma, X, M.
        k_points = [
            (0.0, 0.0),  # Gamma
            (np.pi, 0.0), # X
            (np.pi, np.pi) # M
        ]
        
        min_omega_5 = np.inf
        max_omega_4 = 0.0
        
        for kx, ky in k_points:
            k_vec = np.array([kx, ky])
            
            # Build matrix M
            M = np.zeros((N_pw, N_pw), dtype=complex)
            
            for i, G in enumerate(G_list):
                G_vec_i = np.array(G)
                # (k+G)^2
                k_plus_G = k_vec + 2 * np.pi * G_vec_i
                factor = np.dot(k_plus_G, k_plus_G)
                
                for j, G_prime in enumerate(G_list):
                    G_prime_vec = np.array(G_prime)
                    # Delta = G - G_prime
                    delta_G = G_vec_i - G_prime_vec
                    # Get coefficient (1/eps)_{delta_G}
                    # delta_G components are integers.
                    dgx, dgy = int(delta_G[0]), int(delta_G[1])
                    if (dgx, dgy) in coeffs:
                        M[i, j] = factor * coeffs[(dgx, dgy)]
                    else:
                        # If the difference is outside the cutoff, it's 0?
                        # No, the Fourier series is infinite. But we truncated 1/eps.
                        # This approximation is standard in PWE.
                        M[i, j] = 0.0
            
            # Solve eigenvalue problem
            # M c = omega^2 c
            # Since M is Hermitian (if epsilon is real and symmetric), we can use eigh.
            # Check Hermiticity: M_{ij} = (k+G_i)^2 eps_{G_i-G_j}
            # M_{ji}^* = ((k+G_j)^2 eps_{G_j-G_i})^* = (k+G_j)^2 eps_{-(G_i-G_j)}^*
            # Since eps is real, eps_{-d} = eps_d^*.
            # So M is not necessarily Hermitian unless (k+G_i)^2 = (k+G_j)^2, which is false.
            # Wait, the operator is (k+G)^2 * (1/eps).
            # The operator is not Hermitian in the standard sense because of the (k+G)^2 term.
            # However, the eigenvalues are real.
            # We should use eig, not eigh.
            
            eigvals, eigvecs = np.linalg.eig(M)
            
            # Filter real positive eigenvalues
            real_eigvals = np.real(eigvals)
            positive_eigvals = real_eigvals[real_eigvals > 0]
            positive_eigvals.sort()
            
            if len(positive_eigvals) >= 5:
                omega_4 = np.sqrt(positive_eigvals[3])
                omega_5 = np.sqrt(positive_eigvals[4])
                max_omega_4 = max(max_omega_4, omega_4)
                min_omega_5 = min(min_omega_5, omega_5)
        
        if min_omega_5 == np.inf:
            return -1.0
            
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        if midgap == 0:
            return 0.0
        return gap / midgap

    # 2. Parameterize the design
    # We will optimize a small number of Fourier coefficients for epsilon.
    # Symmetry constraints:
    # 1. Epsilon is real -> coeffs are Hermitian: c_{-G} = c_G^*
    # 2. Square symmetry -> c_{Gx,Gy} = c_{Gy,Gx} (transpose) and c_{Gx,Gy} = c_{-Gx,-Gy} (inversion)
    # 3. We focus on low-order modes that affect bands 4 and 5.
    # Modes: (2,0), (0,2), (2,2), (1,1), (3,0), etc.
    
    # Let's use a set of free parameters: amplitudes and phases of a few modes.
    # To ensure symmetry, we define the independent modes.
    # (2,0) and (0,2) are related by rotation.
    # (2,2) is its own rotation.
    # (1,1) is its own rotation.
    
    # Let's define a function to generate the epsilon grid from parameters.
    # params: [amp_20, phase_20, amp_22, phase_22, amp_11, phase_11, amp_30, phase_30]
    
    def params_to_grid(params):
        N = size
        # Initialize with average epsilon (fill factor)
        # We need to determine the DC component (average epsilon) to control fill factor.
        # Let's fix the DC component to a reasonable value, e.g., corresponding to 50% fill.
        # eps_avg = (eps_low + eps_high) / 2 = 6.5
        
        # We will sum the modes.
        x = (np.arange(N) + 0.5) / N
        y = (np.arange(N) + 0.5) / N
        xx, yy = np.meshgrid(x, y)
        
        eps_grid = np.full((N, N), 6.5) # Start with average
        
        # Add modes
        # Mode (2,0): cos(4pi x)
        # Mode (2,2): cos(4pi x) cos(4pi y)
        # Mode (1,1): cos(2pi x) cos(2pi y)
        
        # Let's allow a general set of modes.
        # To keep it simple and fast, we use:
        # 1. (2,0) and (0,2) -> combined as cos(4pi x) + cos(4pi y)
        # 2. (2,2) -> cos(4pi x) cos(4pi y)
        # 3. (1,1) -> cos(2pi x) cos(2pi y)
        
        # params: [a20, a22, a11, a30]
        # We will optimize amplitudes only, keeping phases fixed to 0 or pi for symmetry.
        
        a20 = params[0]
        a22 = params[1]
        a11 = params[2]
        a30 = params[3] if len(params) > 3 else 0.0
        
        # Construct the field
        # f(x,y) = a20 * (cos(4pi x) + cos(4pi y)) + a22 * cos(4pi x) cos(4pi y) + ...
        
        eps_grid += a20 * (np.cos(4 * np.pi * xx) + np.cos(4 * np.pi * yy))
        eps_grid += a22 * np.cos(4 * np.pi * xx) * np.cos(4 * np.pi * yy)
        eps_grid += a11 * np.cos(2 * np.pi * xx) * np.cos(2 * np.pi * yy)
        eps_grid += a30 * (np.cos(6 * np.pi * xx) + np.cos(6 * np.pi * yy))
        
        # Threshold to 0/1
        # We need to choose a threshold that gives the right fill factor.
        # Let's assume a target fill factor of 0.5.
        # Sort the grid and pick the median as threshold.
        threshold = np.median(eps_grid)
        cell = (eps_grid > threshold).astype(int)
        
        # Enforce symmetry
        # 1. Transpose
        cell = (cell + cell.T) / 2
        # 2. Reverse rows
        cell = (cell + cell[::-1, :]) / 2
        # 3. Reverse cols
        cell = (cell + cell[:, ::-1]) / 2
        
        # Round to binary
        cell = (cell > 0.5).astype(int)
        
        return cell.tolist()

    # 3. Optimization
    # Objective function: -score (minimize negative score)
    
    def objective(params):
        if time.time() > deadline:
            return 1000.0 # Large penalty
            
        try:
            cell = params_to_grid(params)
            score = compute_bands(cell)
            return -score
        except Exception:
            return 1000.0

    # Initial guess
    # Based on the four-rod structure, the (2,0) and (2,2) modes are significant.
    # Let's start with small amplitudes.
    x0 = [0.5, 0.5, 0.5, 0.0]
    
    best_score = -1.0
    best_cell = None
    
    # Run Nelder-Mead
    # We have a tight time budget. We will run a limited number of iterations.
    
    # First, evaluate the initial guess
    cell0 = params_to_grid(x0)
    score0 = compute_bands(cell0)
    best_score = score0
    best_cell = cell0
    
    # Check time
    if time.time() > deadline - 10:
        return best_cell
    
    # Run optimization
    # Use a small number of iterations to stay within time.
    # Nelder-Mead is robust but slow.
    
    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 50, 'xatol': 1e-3, 'fatol': 1e-3, 'adaptive': True})
        
        if res.success or res.fun < -best_score:
            best_params = res.x
            best_cell = params_to_grid(best_params)
            best_score = compute_bands(best_cell)
            
    except Exception:
        pass
    
    # Final check
    if time.time() > deadline - 2:
        pass # Just return best found
    
    # Ensure the returned cell is valid
    if best_cell is None:
        # Fallback to the original four-rod design
        cell = [[0 for _ in range(size)] for _ in range(size)]
        centers = [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)]
        radius = 0.15
        for i in range(size):
            y = (i + 0.5) / size
            for j in range(size):
                x = (j + 0.5) / size
                is_dielectric = False
                for cx, cy in centers:
                    dx = x - cx
                    dy = y - cy
                    if dx * dx + dy * dy < radius * radius:
                        is_dielectric = True
                        break
                cell[i][j] = 1 if is_dielectric else 0
        best_cell = cell
        best_score = compute_bands(best_cell)

    return best_cell

# EVOLVE-BLOCK-END
