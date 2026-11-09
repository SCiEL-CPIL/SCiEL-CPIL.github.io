"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def _compute_mu_fft(cell, size=32):
    """Compute the Fourier coefficients of 1/epsilon for the given cell."""
    eps = np.array([[12.0 if c == 1 else 1.0 for c in row] for row in cell], dtype=float)
    mu = 1.0 / eps
    mu_fft = np.fft.fft2(mu)
    return mu_fft / (size * size)

def _solve_tm_bands(mu_fft, k_points, n_pw=8, size=32):
    """
    Solve for the lowest 5 TM bands using the plane-wave method.
    n_pw=8 means indices -8..8, total 17x17=289 plane waves.
    Returns min_omega_5 and max_omega_4.
    """
    G_indices = np.arange(-n_pw, n_pw + 1)
    Gx, Gy = np.meshgrid(G_indices, G_indices, indexing='ij')
    Gx_flat = Gx.ravel()
    Gy_flat = Gy.ravel()
    M = len(Gx_flat)
    
    # Precompute difference indices for mu(G_i - G_j)
    # diff_x[i, j] = Gx_flat[i] - Gx_flat[j]
    diff_x = Gx_flat[:, None] - Gx_flat[None, :]
    diff_y = Gy_flat[:, None] - Gy_flat[None, :]
    
    # Map differences to FFT indices (0..size-1)
    idx_x = diff_x % size
    idx_y = diff_y % size
    
    # Extract mu values: mu_vals[i, j] = mu(G_i - G_j)
    mu_vals = mu_fft[idx_x, idx_y]
    
    min_omega_5 = float('inf')
    max_omega_4 = float('-inf')
    
    for k in k_points:
        kx, ky = k
        # (k + G_j)^2
        kg_x = kx + 2.0 * np.pi * Gx_flat
        kg_y = ky + 2.0 * np.pi * Gy_flat
        kg_sq = kg_x**2 + kg_y**2
        
        # A = mu_vals * kg_sq[None, :]
        A = mu_vals * kg_sq[None, :]
        
        # Solve for the smallest 5 eigenvalues
        # eigvalsh with subset_by_index is faster
        evals = np.linalg.eigvalsh(A, subset_by_index=[0, 4])
        
        w4 = np.sqrt(evals[3])
        w5 = np.sqrt(evals[4])
        
        max_omega_4 = max(max_omega_4, w4)
        min_omega_5 = min(min_omega_5, w5)
        
    gap = min_omega_5 - max_omega_4
    midgap = (min_omega_5 + max_omega_4) / 2.0
    if midgap == 0:
        return 0.0
    return gap / midgap

def _generate_k_points(num_points=30):
    """Generate 30 k-points in the irreducible BZ of a square lattice."""
    points = []
    # Gamma (0,0) to X (pi, 0)
    for i in range(10):
        kx = (i + 0.5) * (np.pi / 10)
        points.append((kx, 0))
        
    # X (pi, 0) to M (pi, pi)
    for i in range(10):
        ky = (i + 0.5) * (np.pi / 10)
        points.append((np.pi, ky))
        
    # M (pi, pi) to Gamma (0, 0) along diagonal
    for i in range(10):
        t = (i + 0.5) * (np.pi / 10)
        points.append((t, t))
        
    return np.array(points)

def _params_to_cell(params, size=32):
    """
    Convert parameters to a 32x32 cell.
    params: [r1, cx1, r2, cx2]
    - r1, cx1: radius and x-offset for the first set of circles (near corners/edges)
    - r2, cx2: radius and x-offset for the second set
    Due to C4v symmetry, we define circles in the 1st quadrant and replicate.
    Positions are (cx, cy) where cy is fixed or derived?
    Let's use 4 circles at (cx, cy), (cx, -cy), (-cx, cy), (-cx, -cy) relative to center.
    Actually, simpler: define circles centered at (cx, cy) with cx, cy > 0.
    Replicate to 4 quadrants.
    params: [r, cx, cy]
    """
    r, cx, cy = params
    cell = np.zeros((size, size), dtype=int)
    
    # Centers in the 1st quadrant (relative to cell center 16,16)
    # Cell coordinates: 0..31. Center is 15.5, 15.5.
    # Let's place centers at (15.5 + cx, 15.5 + cy) and symmetric equivalents.
    
    centers = [
        (15.5 + cx, 15.5 + cy),
        (15.5 + cx, 15.5 - cy),
        (15.5 - cx, 15.5 + cy),
        (15.5 - cx, 15.5 - cy)
    ]
    
    for i in range(size):
        for j in range(size):
            x = i + 0.5
            y = j + 0.5
            for cx_c, cy_c in centers:
                if (x - cx_c)**2 + (y - cy_c)**2 < r*r:
                    cell[i][j] = 1
                    break
                    
    return cell.tolist()

def _evaluate_cell(cell, k_points, size=32):
    """Evaluate the gap-midgap ratio for a given cell."""
    mu_fft = _compute_mu_fft(cell, size)
    return _solve_tm_bands(mu_fft, k_points, n_pw=8, size=size)

def design(problem):
    size = problem["pixels"]
    start_time = time.time()
    deadline = start_time + 220.0 # 10s margin
    
    k_points = _generate_k_points(30)
    
    # Initial parameters: [radius, cx, cy]
    # Try a few initial guesses
    initial_guesses = [
        [0.2, 0.2, 0.2],
        [0.3, 0.3, 0.3],
        [0.4, 0.4, 0.4],
        [0.25, 0.1, 0.1],
        [0.15, 0.25, 0.25]
    ]
    
    best_cell = None
    best_score = -float('inf')
    
    for guess in initial_guesses:
        if time.time() > deadline - 10.0:
            break
            
        cell = _params_to_cell(guess, size)
        score = _evaluate_cell(cell, k_points, size)
        
        if score > best_score:
            best_score = score
            best_cell = cell
            
        # Local optimization around this guess using random perturbation
        current_params = guess[:]
        current_score = score
        
        # Simple hill climbing / random walk
        step_size = 0.05
        for step in range(50):
            if time.time() > deadline - 5.0:
                break
                
            # Perturb one parameter
            p_idx = np.random.randint(0, 3)
            delta = np.random.uniform(-step_size, step_size)
            new_params = current_params[:]
            new_params[p_idx] += delta
            
            # Clamp parameters
            new_params[0] = max(0.05, min(0.5, new_params[0])) # Radius
            new_params[1] = max(0.0, min(0.8, new_params[1]))  # cx
            new_params[2] = max(0.0, min(0.8, new_params[2]))  # cy
            
            new_cell = _params_to_cell(new_params, size)
            new_score = _evaluate_cell(new_cell, k_points, size)
            
            if new_score > current_score:
                current_params = new_params
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_cell = new_cell
            else:
                # Decay step size if no improvement
                step_size *= 0.95
                if step_size < 0.01:
                    break
                    
    if best_cell is None:
        # Fallback: empty cell or full cell
        best_cell = [[0]*size for _ in range(size)]
        
    return best_cell
# EVOLVE-BLOCK-END
