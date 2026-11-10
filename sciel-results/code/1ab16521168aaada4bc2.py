import numpy as np
import time

def design(problem):
    """
    Designs a 32x32 unit cell using a parameterized Superellipse shape
    and a fast plane-wave band solver.
    """
    start_time = time.time()
    deadline = start_time + 190.0  # Safety margin
    
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    # Precompute constants for the solver
    n_pw = 6
    g_range = np.arange(-n_pw, n_pw + 1)
    g_m, g_n = np.meshgrid(g_range, g_range, indexing='ij')
    g_m = g_m.flatten()
    g_n = g_n.flatten()
    g_sq = g_m**2 + g_n**2
    
    # K-points in irreducible BZ: 0 <= kx <= ky <= 0.5
    # High symmetry points and some interior
    k_pts = [
        (0.0, 0.0),   # Gamma
        (0.5, 0.5),   # M
        (0.5, 0.0),   # X
        (0.25, 0.25), # Interior
        (0.25, 0.5),  # Mid M-X
        (0.0, 0.25),  # Mid Gamma-X
        (0.1, 0.3),
        (0.3, 0.3),
        (0.2, 0.4),
        (0.4, 0.4),
    ]
    
    def generate_superellipse(n_exp, scale, offset=0.0):
        """
        Generates a 32x32 grid for a superellipse |x|^n + |y|^n <= scale^n.
        Coordinates are centered at (15.5, 15.5).
        """
        cell = np.zeros((size, size), dtype=np.int8)
        # Center is at 15.5, 15.5
        y_coords = np.arange(size) - (size - 1) / 2.0
        x_coords = np.arange(size) - (size - 1) / 2.0
        
        # Normalize coords to [-0.5, 0.5] range roughly?
        # Let's use pixel centers relative to cell center.
        # Max distance is ~ 15.5.
        # Let scale be in units of pixels.
        
        yy, xx = np.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Superellipse equation: |x|^(2n) + |y|^(2n) <= scale^(2n)
        # We use 2n to allow n=1 (diamond), n=2 (circle), n->inf (square)
        # Actually, standard superellipse is |x/a|^n + |y/b|^n = 1.
        # Let's use |x|^(2n) + |y|^(2n) <= scale^(2n).
        
        val = np.abs(xx) ** (2 * n_exp) + np.abs(yy) ** (2 * n_exp)
        cell[val <= scale ** (2 * n_exp)] = 1
        
        return cell

    def compute_fourier_inv_eps(cell):
        """Compute 2D FFT of 1/epsilon, normalized."""
        eps = np.where(cell == 1, eps_high, eps_low)
        inv_eps = 1.0 / eps
        fft_2d = np.fft.fft2(inv_eps) / (size * size)
        return fft_2d

    def solve_bands(cell, fft_2d):
        """
        Fast TM band solver.
        Returns gap-midgap ratio for bands 4 and 5.
        """
        max_omega4_sq = -1e10
        min_omega5_sq = 1e10
        
        # Precompute diff indices for G_i - G_j
        diff_m = g_m[:, None] - g_m[None, :]
        diff_n = g_n[:, None] - g_n[None, :]
        idx_m = diff_m % size
        idx_n = diff_n % size
        eps_inv_diff = fft_2d[idx_m, idx_n]
        
        for kx, ky in k_pts:
            kx_g = kx + g_m
            ky_g = ky + g_n
            k_g_sq = kx_g**2 + ky_g**2
            
            M = eps_inv_diff * k_g_sq[None, :]
            
            try:
                eigvals = np.linalg.eigvals(M)
                eigvals = np.real(eigvals)
                # Filter positive real eigenvalues
                eigvals = eigvals[eigvals > 1e-6]
                if len(eigvals) >= 5:
                    eigvals = np.sort(eigvals)
                    omega4_sq = eigvals[3]
                    omega5_sq = eigvals[4]
                    
                    if omega4_sq > max_omega4_sq:
                        max_omega4_sq = omega4_sq
                    if omega5_sq < min_omega5_sq:
                        min_omega5_sq = omega5_sq
            except:
                pass
        
        if max_omega4_sq <= 0 or min_omega5_sq <= 0 or min_omega5_sq <= max_omega4_sq:
            return -1.0
        
        gap = min_omega5_sq - max_omega4_sq
        midgap = (min_omega5_sq + max_omega4_sq) / 2.0
        
        if midgap <= 0:
            return -1.0
        
        return gap / midgap

    # Search over superellipse parameters
    # n_exp: 1.0 (diamond) to 4.0 (square-like). 2.0 is circle.
    # scale: 5.0 to 14.0 pixels (radius)
    
    best_score = -1.0
    best_cell = np.zeros((size, size), dtype=np.int8)
    
    # Define a grid of parameters to search
    n_exps = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 8.0]
    scales = [5.0, 7.0, 9.0, 11.0, 13.0]
    
    candidates = []
    for n_exp in n_exps:
        for scale in scales:
            candidates.append((n_exp, scale))
            
    # Also add some random perturbations or specific known good shapes?
    # Let's just stick to the grid for now, it's fast.
    
    for n_exp, scale in candidates:
        if time.time() > deadline:
            break
            
        cell = generate_superellipse(n_exp, scale)
        fft_2d = compute_fourier_inv_eps(cell)
        score = solve_bands(cell, fft_2d)
        
        if score > best_score:
            best_score = score
            best_cell = cell.copy()
            
    # If we have time, do a finer search around the best
    if time.time() < deadline - 5.0 and best_score > -1.0:
        # Find the best params from the grid
        # We didn't store them, so let's just do a random local search
        # Or re-evaluate a few random points
        pass # Skipping for simplicity, grid should be sufficient
        
    # Fallback: if best_score is bad, try a standard square hole or something
    if best_score < 0.0:
        # Try a simple square dielectric block
        cell = np.zeros((size, size), dtype=np.int8)
        cell[8:24, 8:24] = 1
        fft_2d = compute_fourier_inv_eps(cell)
        score = solve_bands(cell, fft_2d)
        if score > best_score:
            best_score = score
            best_cell = cell

    # Convert to nested list
    result = []
    for i in range(size):
        row = []
        for j in range(size):
            row.append(int(best_cell[i, j]))
        result.append(row)
    
    return result
