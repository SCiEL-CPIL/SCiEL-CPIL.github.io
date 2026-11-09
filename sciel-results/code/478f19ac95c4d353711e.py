"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
import random

def _enforce_c4v(cell):
    """Enforce C4v symmetry: cell == cell[::-1] rows, cell == cell[::-1] cols, cell == cell.T"""
    size = len(cell)
    cell = np.array(cell, dtype=np.int8)
    # Enforce 180-degree rotation symmetry (row reverse + col reverse)
    cell = (cell + np.rot90(cell, 2)) // 2
    # Enforce transpose symmetry
    cell = (cell + cell.T) // 2
    # Enforce row-reverse symmetry
    cell = (cell + cell[::-1, :]) // 2
    # Enforce col-reverse symmetry
    cell = (cell + cell[:, ::-1]) // 2
    # Final cleanup to ensure binary
    cell = (cell > 0).astype(np.int8)
    return cell.tolist()

def _compute_gap(cell, problem):
    """
    Compute the TM band gap ratio between bands 4 and 5 using plane-wave method.
    This is a simplified evaluator that approximates what the real evaluator does.
    """
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    cell = np.array(cell, dtype=np.float64)
    
    # Compute Fourier coefficients of epsilon
    # For a 32x32 cell, we need Fourier coefficients up to some order
    # Use a subset of plane waves for speed during optimization
    
    N = 7  # Number of plane waves in each direction (smaller for speed)
    kx_vals = []
    ky_vals = []
    for ix in range(-N, N+1):
        for iy in range(-N, N+1):
            if ix == 0 and iy == 0:
                continue
            kx_vals.append(ix)
            ky_vals.append(iy)
    
    kx = np.array(kx_vals, dtype=np.float64)
    ky = np.array(ky_vals, dtype=np.float64)
    Npw = len(kx)
    
    # Compute epsilon Fourier coefficients
    # epsilon(k) = sum of epsilon_r * exp(-i*k*r)
    # For a 2D grid, we can compute via 2D FFT
    eps_grid = np.zeros((size, size), dtype=np.float64)
    for i in range(size):
        for j in range(size):
            eps_grid[i, j] = eps_high if cell[i, j] == 1 else eps_low
    
    # 2D FFT to get Fourier coefficients
    eps_fft = np.fft.fft2(eps_grid)
    eps_fft = eps_fft / (size * size)
    
    # Map FFT indices to k-space indices
    # FFT index (i, j) corresponds to frequency (i - size//2, j - size//2) in units of 2*pi/size
    # We need epsilon(kx, ky) where kx, ky are in units of 2*pi
    
    def get_eps_coeff(ix, iy):
        """Get epsilon Fourier coefficient for wavevector (ix, iy) in units of 2*pi"""
        # FFT gives us coefficients for frequencies in [-size//2, size//2)
        # Map ix, iy to FFT indices
        iix = ix % size
        iiy = iy % size
        return eps_fft[iix, iiy].real  # epsilon is real and symmetric
    
    # Build the TM eigenvalue problem
    # For TM polarization, the eigenvalue problem is:
    # sum_k' epsilon^{-1}(k-k') * |k'|^2 * c_{k'} = omega^2 * c_k
    # where omega is in units of 2*pi*c/a
    
    # Build epsilon^{-1} matrix
    # We need epsilon^{-1}(k) for k = k_i - k_j
    # This is more complex, so let's use a simplified approach
    
    # Actually, let's use a simpler surrogate: compute the dielectric contrast
    # and use a heuristic that correlates with band gaps
    
    # Better approach: use the actual plane-wave method but with fewer wave vectors
    # for the optimization, and rely on the final evaluation for accuracy
    
    # For speed during optimization, use a rough proxy:
    # 1. Compute the average dielectric constant
    # 2. Compute the dielectric contrast
    # 3. Use a heuristic based on known good designs
    
    # Actually, let's implement a proper but fast plane-wave calculation
    # with a reduced set of wave vectors
    
    # Select wave vectors in the irreducible BZ
    # Irreducible BZ for square lattice: 0 <= ky <= kx, 0 <= kx <= pi/a
    # In units of 2*pi/a: 0 <= ky/2pi <= kx/2pi <= 0.5
    # Let's use a small set of high-symmetry points
    
    k_points = [
        (0.0, 0.0),      # Gamma
        (0.5, 0.0),      # X
        (0.5, 0.5),      # M
        (0.25, 0.0),     # Interior
        (0.5, 0.25),     # Interior
        (0.25, 0.25),    # Interior
    ]
    
    # For each k-point, solve the eigenvalue problem
    # Use a small number of plane waves for speed
    
    Npw_opt = 5  # Number of plane waves in each direction for optimization
    
    def solve_tm_bands(kx_unit, ky_unit, n_bands=5):
        """Solve for the first n_bands TM eigenvalues at given k-point"""
        # Generate plane wave vectors
        pw_kx = []
        pw_ky = []
        for ix in range(-Npw_opt, Npw_opt+1):
            for iy in range(-Npw_opt, Npw_opt+1):
                pw_kx.append(kx_unit + ix)
                pw_ky.append(ky_unit + iy)
        
        pw_kx = np.array(pw_kx)
        pw_ky = np.array(pw_ky)
        Npw_total = len(pw_kx)
        
        # Compute |k|^2 for each plane wave
        k_sq = pw_kx**2 + pw_ky**2
        
        # Build the epsilon^{-1} coupling matrix
        # H[k, k'] = epsilon^{-1}(k - k') * |k'|^2
        # We need to compute epsilon^{-1}(G) for various G vectors
        
        # Precompute epsilon^{-1} coefficients
        # epsilon^{-1}(G) = (1/N) sum_r exp(i*G*r) / epsilon(r)
        
        inv_eps_grid = np.zeros((size, size), dtype=np.float64)
        for i in range(size):
            for j in range(size):
                inv_eps_grid[i, j] = 1.0 / (eps_high if cell[i, j] == 1 else eps_low)
        
        inv_eps_fft = np.fft.fft2(inv_eps_grid) / (size * size)
        
        def get_inv_eps_coeff(ix, iy):
            iix = ix % size
            iiy = iy % size
            return inv_eps_fft[iix, iiy].real
        
        # Build H matrix
        H = np.zeros((Npw_total, Npw_total), dtype=np.float64)
        for i in range(Npw_total):
            for j in range(Npw_total):
                Gx = int(round(pw_kx[i] - pw_kx[j]))
                Gy = int(round(pw_ky[i] - pw_ky[j]))
                H[i, j] = get_inv_eps_coeff(Gx, Gy) * k_sq[j]
        
        # Solve eigenvalue problem
        eigvals, eigvecs = np.linalg.eigh(H)
        
        # Sort eigenvalues
        idx = np.argsort(eigvals)
        eigvals = eigvals[idx]
        
        # The eigenvalue lambda = omega^2 * (a/2*pi)^2 / c^2
        # So omega = sqrt(lambda) * 2*pi*c/a
        # We want the ratio, so we can work with sqrt(lambda)
        omegas = np.sqrt(np.maximum(eigvals, 0))
        
        return omegas[:n_bands]
    
    # Compute bands at each k-point
    max_omega4 = 0.0
    min_omega5 = float('inf')
    
    for kx_u, ky_u in k_points:
        omegas = solve_tm_bands(kx_u, ky_u, n_bands=5)
        if len(omegas) >= 5:
            max_omega4 = max(max_omega4, omegas[3])
            min_omega5 = min(min_omega5, omegas[4])
    
    if min_omega5 == float('inf') or max_omega4 == 0:
        return -1.0
    
    gap = min_omega5 - max_omega4
    midgap = (min_omega5 + max_omega4) / 2.0
    
    if midgap == 0:
        return -1.0
    
    return 2.0 * gap / midgap

def design(problem):
    """
    Design a C4v-symmetric 32x32 unit cell that maximizes the TM band gap
    between bands 4 and 5 using a multi-start local search with a
    Chebyshev level-set representation.
    """
    start_time = time.time()
    deadline = start_time + 200  # Safety margin before 230s kill
    
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    # Start with a good initial guess: four-rod configuration
    def make_initial_cell():
        cell = np.zeros((size, size), dtype=np.int8)
        # Place four small rods at (8,8), (8,24), (24,8), (24,24)
        rod_centers = [(8, 8), (8, 24), (24, 8), (24, 24)]
        rod_radius = 3
        for cy, cx in rod_centers:
            for i in range(size):
                for j in range(size):
                    if (i - cy)**2 + (j - cx)**2 < rod_radius**2:
                        cell[i, j] = 1
        return _enforce_c4v(cell.tolist())
    
    # Initial cell
    cell = make_initial_cell()
    best_cell = cell
    best_score = _compute_gap(best_cell, problem)
    
    # Multi-start local search
    num_starts = 3
    max_iter = 500
    step_size = 0.5
    
    for start_idx in range(num_starts):
        if time.time() > deadline - 30:
            break
        
        # Initialize with perturbation from best or random
        if start_idx == 0:
            current_cell = best_cell[:]
        else:
            # Random C4v-symmetric initialization
            # Only need to define the fundamental domain: upper-right 1/4
            # with symmetry constraints
            current_cell = np.zeros((size, size), dtype=np.int8)
            # Fill the fundamental domain randomly
            for i in range(size // 2):
                for j in range(size // 2):
                    if random.random() < 0.3:
                        current_cell[i, j] = 1
            # Enforce C4v symmetry
            current_cell = _enforce_c4v(current_cell.tolist())
            current_cell = np.array(current_cell, dtype=np.int8)
        
        current_score = _compute_gap(current_cell.tolist(), problem)
        
        # Local search: flip random pixels while maintaining C4v symmetry
        for it in range(max_iter):
            if time.time() > deadline - 30:
                break
            
            # Pick a random pixel in the fundamental domain and flip it
            # Fundamental domain for C4v: i in [0, size//2), j in [0, size//2)
            # But we need to be careful about the symmetry axes
            i = random.randint(0, size // 2 - 1)
            j = random.randint(0, size // 2 - 1)
            
            # Create a new cell by flipping this pixel and all its symmetric partners
            new_cell = current_cell.copy()
            
            # C4v symmetry partners for (i, j):
            # 1. (i, j)
            # 2. (i, size-1-j)  [col reverse]
            # 3. (size-1-i, j)  [row reverse]
            # 4. (size-1-i, size-1-j)  [180 rot]
            # 5. (j, i)  [transpose]
            # 6. (j, size-1-i)  [transpose + row reverse]
            # 7. (size-1-j, i)  [transpose + col reverse]
            # 8. (size-1-j, size-1-i)  [transpose + 180 rot]
            
            partners = [
                (i, j),
                (i, size - 1 - j),
                (size - 1 - i, j),
                (size - 1 - i, size - 1 - j),
                (j, i),
                (j, size - 1 - i),
                (size - 1 - j, i),
                (size - 1 - j, size - 1 - i),
            ]
            
            # Remove duplicates
            partners = list(set(partners))
            
            # Flip all partners
            for pi, pj in partners:
                new_cell[pi, pj] = 1 - new_cell[pi, pj]
            
            # Evaluate
            new_score = _compute_gap(new_cell.tolist(), problem)
            
            # Accept if better or with some probability (simulated annealing-like)
            if new_score > current_score:
                current_cell = new_cell
                current_score = new_score
                
                if current_score > best_score:
                    best_score = current_score
                    best_cell = new_cell.tolist()
            else:
                # Small probability of accepting worse moves to escape local minima
                if random.random() < 0.1:
                    current_cell = new_cell
                    current_score = new_score
        
        # Update best if this start found something better
        if current_score > best_score:
            best_score = current_score
            best_cell = current_cell.tolist()
    
    # Final check: ensure the best cell is valid
    best_cell = _enforce_c4v(best_cell)
    
    return best_cell
# EVOLVE-BLOCK-END
