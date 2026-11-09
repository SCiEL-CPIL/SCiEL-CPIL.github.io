import numpy as np
import time

def design(problem):
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    # Time management
    start_time = time.time()
    time_limit = 110.0 # Safety margin
    
    # PWM Setup
    N_G = 17
    g_range = np.arange(-N_G//2, N_G//2 + 1)
    Gx, Gy = np.meshgrid(g_range, g_range, indexing='ij')
    G_vectors = np.column_stack((Gx.ravel(), Gy.ravel()))
    N_pw = len(G_vectors)
    
    # K points in irreducible BZ (30 points)
    k_points = []
    for i in range(10):
        t = i / 9.0 * 0.5
        k_points.append([t, 0.0])
        k_points.append([0.5, t])
        k_points.append([t, t])
    k_points.append([0.25, 0.1])
    k_points.append([0.35, 0.2])
    k_points.append([0.45, 0.3])
    k_points = np.array(k_points)
    
    # Precompute diff indices for H matrix
    dx = G_vectors[:, 0][:, None] - G_vectors[:, 0][None, :]
    dy = G_vectors[:, 1][:, None] - G_vectors[:, 1][None, :]
    valid_diffs = (np.abs(dx) <= N_G//2) & (np.abs(dy) <= N_G//2)
    diff_indices = np.where(valid_diffs, (dx + N_G//2) * N_G + (dy + N_G//2), 0)
    
    def get_fourier_eps_inv(cell_arr):
        eps = np.where(cell_arr == 1, eps_high, eps_low)
        eps_inv = 1.0 / eps
        coeffs_2d = np.fft.fft2(eps_inv) / (size * size)
        Gx_grid = G_vectors[:, 0] % size
        Gy_grid = G_vectors[:, 1] % size
        return coeffs_2d[Gx_grid, Gy_grid]
    
    def evaluate(cell_arr):
        eps_inv_coeffs = get_fourier_eps_inv(cell_arr)
        
        min_omega_5 = float('inf')
        max_omega_4 = float('-inf')
        
        for k in k_points:
            k_vec = np.array(k)
            k_plus_G = k_vec[None, :] + G_vectors
            k_plus_G_sq = np.sum(k_plus_G**2, axis=1)
            
            H = eps_inv_coeffs[diff_indices] * k_plus_G_sq[None, :]
            H[~valid_diffs] = 0.0
            
            evals = np.linalg.eigvals(H)
            evals = np.sort(np.real(evals))
            
            w4 = evals[3]
            w5 = evals[4]
            
            max_omega_4 = max(max_omega_4, w4)
            min_omega_5 = min(min_omega_5, w5)
            
        if min_omega_5 == float('inf'):
            return 0.0
            
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        if midgap == 0:
            return 0.0
        return gap / midgap

    # --- New Structural Change: Chebyshev Level Set Representation ---
    
    # Precompute pixel coordinates in [-1, 1]
    # Center is at (15.5, 15.5) for 32x32? Or (16,16)?
    # The symmetry is about the cell centre.
    # For 32x32, center is between pixels.
    # Let's map pixel (i,j) to x,y in [-1, 1].
    # i, j in 0..31.
    # x = (i - 15.5) / 16.0 -> range [-0.968, 0.968]
    # This preserves C4v symmetry if we define the level set in terms of x^2, y^2, x^2 y^2 etc.
    
    x_coords, y_coords = np.meshgrid(
        (np.arange(size) - (size - 1) / 2.0) / (size / 2.0),
        (np.arange(size) - (size - 1) / 2.0) / (size / 2.0),
        indexing='ij'
    )
    
    # Basis functions for C4v symmetry:
    # Invariant under x->-x, y->-y, x<->y
    # Terms: 1, x^2, y^2, x^4, y^4, x^2 y^2, x^6, y^6, x^4 y^2, x^2 y^4, x^3 y^3 ...
    # Let's use up to order 4 (degree 8 in original variables) to keep it low order.
    # Basis:
    # 1
    # x^2, y^2
    # x^4, y^4, x^2 y^2
    # x^6, y^6, x^4 y^2, x^2 y^4, x^3 y^3
    
    basis_funcs = [
        np.ones_like(x_coords),
        x_coords**2,
        y_coords**2,
        x_coords**4,
        y_coords**4,
        x_coords**2 * y_coords**2,
        x_coords**6,
        y_coords**6,
        x_coords**4 * y_coords**2,
        x_coords**2 * y_coords**4,
        x_coords**3 * y_coords**3
    ]
    
    # Stack into matrix B of shape (32*32, num_coeffs)
    B = np.column_stack([bf.ravel() for bf in basis_funcs])
    num_coeffs = B.shape[1]
    
    # Initial coefficients: A centered rod
    # f(x,y) = x^2 + y^2 - R^2 < 0 is dielectric?
    # Let's say f(x,y) > 0 is dielectric.
    # Rod: x^2 + y^2 - 0.2^2 < 0 -> Air? No, usually rod is dielectric.
    # Let's define: cell[i,j] = 1 if f(x,y) < threshold.
    # Let's just optimize coefficients c such that f(x,y) = B @ c.
    # cell = (f(x,y) < 0).astype(int)
    
    # Initial c for a rod of radius ~0.2 (in normalized coords)
    # x^2 + y^2 - 0.04 < 0
    # c = [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, -0.04]
    # Wait, basis indices:
    # 0: 1
    # 1: x^2
    # 2: y^2
    # ...
    c_init = np.zeros(num_coeffs)
    c_init[1] = 1.0
    c_init[2] = 1.0
    c_init[0] = -0.04 # constant term to shift radius
    
    best_cell = None
    best_score = -np.inf
    
    # Stochastic Hill Climbing with Adaptive Step Size
    c = c_init.copy()
    step_size = 0.05
    
    # Precompute B for speed
    # B is (1024, 11)
    
    for _ in range(5000): # Max iterations, break by time
        if time.time() - start_time > time_limit:
            break
            
        f_vals = B @ c
        cell = (f_vals < 0).reshape(size, size).astype(int)
        
        # Ensure symmetry is perfect (it should be by construction, but just in case of numerical issues)
        # The basis is strictly symmetric, so the pixel map will be symmetric.
        
        score = evaluate(cell)
        
        if score > best_score:
            best_score = score
            best_cell = cell.copy()
            
        # Generate a new candidate
        # Perturb coefficients
        mask = np.random.rand(num_coeffs) < 0.3 # Only perturb some coefficients
        delta = np.random.normal(0, step_size, num_coeffs)
        delta[~mask] = 0
        
        c_new = c + delta
        
        # Simple adaptation: if we accept, keep step size or reduce slightly if stuck
        # For now, simple random walk with occasional restart
        if np.random.rand() < 0.05:
             # Restart towards a random point or center
             c_new = c_init + np.random.normal(0, 0.5, num_coeffs)
        else:
             c_new = c + delta
             
        # Evaluate candidate? To save time, we can just accept if it looks similar or use a simpler metric first?
        # No, we must evaluate to know if it's better.
        # But evaluating is expensive.
        # Let's do a "look-ahead" or just update c and evaluate in next loop?
        # Standard hill climbing: evaluate new point.
        
        f_vals_new = B @ c_new
        cell_new = (f_vals_new < 0).reshape(size, size).astype(int)
        score_new = evaluate(cell_new)
        
        if score_new > score:
            c = c_new
            # Shrink step size slightly to refine
            step_size = max(0.001, step_size * 0.95)
        else:
            # Expand step size to escape
            step_size = min(0.5, step_size * 1.1)
            
    # If best_cell is not found (should not happen), fallback to seed
    if best_cell is None:
        cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                if x*x + y*y < 0.2**2:
                    cell[i, j] = 1
        best_cell = cell

    return best_cell.tolist()
