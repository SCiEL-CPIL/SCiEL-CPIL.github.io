"""Photonic crystal unit cell design for large TM band gap between bands 4 and 5."""
import numpy as np
import time
from scipy.optimize import minimize

def design(problem):
    """Design a 32x32 symmetric unit cell maximizing TM band gap between bands 4 and 5."""
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)

    t_start = time.time()
    DEADLINE = 220.0  # safety margin before 230s kill

    def time_ok():
        return time.time() - t_start < DEADLINE

    # Plane wave setup: 17x17 = 289 plane waves
    n_pw = 17
    half = n_pw // 2
    G_vals = np.arange(-half, half + 1)
    Gx_grid, Gy_grid = np.meshgrid(G_vals, G_vals, indexing='ij')
    Gx = Gx_grid.ravel()
    Gy = Gy_grid.ravel()
    n_pw_total = n_pw * n_pw  # 289

    # K-points in irreducible BZ (30 points)
    pi = np.pi
    k_points = []
    for i in range(10):
        t = i / 9.0
        k_points.append((t * pi, 0.0))
    for i in range(10):
        t = i / 9.0
        k_points.append((pi, t * pi))
    for i in range(10):
        t = i / 9.0
        k_points.append((pi * (1 - t), pi * (1 - t)))
    k_points = np.array(k_points)

    # Precompute (k+G)^2
    k2_all = np.zeros((len(k_points), n_pw_total))
    for i, (kx, ky) in enumerate(k_points):
        k2_all[i] = (kx + Gx) ** 2 + (ky + Gy) ** 2

    # Precompute difference indices for Fourier coupling
    diff_idx = np.zeros((n_pw_total, n_pw_total), dtype=int)
    for i in range(n_pw_total):
        for j in range(n_pw_total):
            dGx = Gx[i] - Gx[j]
            dGy = Gy[i] - Gy[j]
            idx_x = int(dGx) + half
            idx_y = int(dGy) + half
            if 0 <= idx_x < n_pw and 0 <= idx_y < n_pw:
                diff_idx[i, j] = idx_x * n_pw + idx_y
            else:
                diff_idx[i, j] = -1
    valid_mask = diff_idx >= 0
    valid_rows, valid_cols = np.where(valid_mask)

    def get_fourier_coeffs(cell):
        """Compute Fourier coefficients of 1/eps using FFT."""
        eps_map = np.where(cell == 1, eps_high, eps_low)
        inv_eps = 1.0 / eps_map
        fft_2d = np.fft.fft2(inv_eps)
        coeffs = np.zeros(n_pw_total, dtype=complex)
        for idx in range(n_pw_total):
            gx = int(Gx[idx])
            gy = int(Gy[idx])
            fft_ix = gx % size
            fft_iy = gy % size
            coeffs[idx] = fft_2d[fft_ix, fft_iy] / (size * size)
        return coeffs

    def compute_band_ratio(coeffs):
        """Compute gap-midgap ratio."""
        max_band4 = -np.inf
        min_band5 = np.inf

        for k_idx in range(len(k_points)):
            k2 = k2_all[k_idx]
            # Build H matrix efficiently
            H_vals = k2[valid_rows] * coeffs[diff_idx[valid_rows, valid_cols]]
            H = np.zeros((n_pw_total, n_pw_total), dtype=complex)
            H[valid_rows, valid_cols] = H_vals
            H = (H + H.conj().T) / 2.0
            eigvals = np.linalg.eigvalsh(H)
            eigvals = np.sort(eigvals)
            if len(eigvals) >= 5:
                if eigvals[3] > max_band4:
                    max_band4 = eigvals[3]
                if eigvals[4] < min_band5:
                    min_band5 = eigvals[4]

        gap = min_band5 - max_band4
        midgap = (min_band5 + max_band4) / 2.0
        if midgap > 1e-10:
            return gap / midgap
        return 0.0

    def enforce_symmetry(cell):
        """Enforce 4-fold rotational + reflection symmetry."""
        cell = np.array(cell, dtype=int)
        rot90 = np.rot90(cell, 1)
        rot180 = np.rot90(cell, 2)
        rot270 = np.rot90(cell, 3)
        hflip = cell[:, ::-1].copy()
        vflip = cell[::-1, :].copy()
        trans = cell.T.copy()
        diag_flip = np.rot90(trans, 1)
        avg = (cell + rot90 + rot180 + rot270 + hflip + vflip + trans + diag_flip)
        result = (avg >= 4).astype(int)
        return result

    def generate_superellipses(radius, offset, exponent):
        """Generate four superellipse rods at (±offset, ±offset).
        Superellipse: |x|^(2/n) + |y|^(2/n) <= 1, where n = exponent.
        n=2 is circle, n>2 is squarer, n<1 is star-like.
        """
        cell = np.zeros((size, size), dtype=int)
        # Precompute pixel centers
        xs = (np.arange(size) + 0.5) / size - 0.5
        ys = (np.arange(size) + 0.5) / size - 0.5
        X, Y = np.meshgrid(xs, ys, indexing='ij')
        
        # Superellipse equation: (|dx|/r)^(2/n) + (|dy|/r)^(2/n) <= 1
        inv_n = 2.0 / exponent
        centers = [(offset, offset), (offset, -offset), (-offset, offset), (-offset, -offset)]
        for cx, cy in centers:
            dx = np.abs(X - cx) / radius
            dy = np.abs(Y - cy) / radius
            # Avoid 0^0 issues
            dx_safe = np.where(dx == 0, 1e-10, dx)
            dy_safe = np.where(dy == 0, 1e-10, dy)
            mask = (dx_safe ** inv_n + dy_safe ** inv_n) <= 1.0
            cell[mask] = 1
        
        return enforce_symmetry(cell)

    def objective(params):
        """Objective: negative gap ratio (for minimization)."""
        radius, offset, exponent = params
        # Clamp to valid ranges
        radius = np.clip(radius, 0.03, 0.20)
        offset = np.clip(offset, 0.10, 0.35)
        exponent = np.clip(exponent, 0.5, 4.0)
        
        cell = generate_superellipses(radius, offset, exponent)
        coeffs = get_fourier_coeffs(cell)
        ratio = compute_band_ratio(coeffs)
        return -ratio

    # Phase 1: Evaluate a diverse set of initial candidates to find a good starting point
    candidates_params = []
    # Superellipses with varying shapes
    for r in [0.06, 0.08, 0.10, 0.12, 0.14]:
        for off in [0.18, 0.22, 0.26, 0.30]:
            for n in [1.0, 1.5, 2.0, 3.0]:
                candidates_params.append((r, off, n))
    
    best_ratio = -np.inf
    best_cell = None
    best_params = None
    
    for params in candidates_params:
        if not time_ok():
            break
        r, off, n = params
        cell = generate_superellipses(r, off, n)
        coeffs = get_fourier_coeffs(cell)
        ratio = compute_band_ratio(coeffs)
        if ratio > best_ratio:
            best_ratio = ratio
            best_cell = cell.copy()
            best_params = params

    if best_cell is None:
        best_cell = generate_superellipses(0.10, 0.25, 2.0)
        best_params = (0.10, 0.25, 2.0)
        best_ratio = compute_band_ratio(get_fourier_coeffs(best_cell))

    # Phase 2: Nelder-Mead optimization over the 3D parameter space
    current_cell = best_cell.copy()
    current_ratio = best_ratio
    current_params = np.array(best_params, dtype=float)
    
    # Set up Nelder-Mead
    nm_options = {
        'xatol': 1e-4,
        'fatol': 1e-6,
        'maxiter': 200,
        'adaptive': True,
    }
    
    search_start = time.time()
    search_time_limit = 170.0  # leave time for final eval
    
    try:
        # Track best during optimization
        best_found = {'ratio': current_ratio, 'cell': current_cell.copy(), 'params': current_params.copy()}
        
        def obj_wrapper(params):
            if not time_ok() or time.time() - search_start > search_time_limit:
                return best_found['ratio'] * -1  # return current best (negative for minimization)
            val = objective(params)
            if -val > best_found['ratio']:
                best_found['ratio'] = -val
                best_found['params'] = params.copy()
                # Reconstruct cell for tracking
                r, off, n = params
                r = np.clip(r, 0.03, 0.20)
                off = np.clip(off, 0.10, 0.35)
                n = np.clip(n, 0.5, 4.0)
                best_found['cell'] = generate_superellipses(r, off, n)
            return val
        
        result = minimize(obj_wrapper, current_params, method='Nelder-Mead', options=nm_options)
        
        if time_ok() and time.time() - search_start < search_time_limit:
            if -result.fun > current_ratio:
                current_ratio = -result.fun
                current_params = result.x
                current_cell = best_found['cell'].copy()
        
        # Use the best found during optimization
        if best_found['ratio'] > current_ratio:
            current_ratio = best_found['ratio']
            current_cell = best_found['cell'].copy()
            current_params = best_found['params'].copy()
            
    except Exception as e:
        # If optimization fails, keep the best from Phase 1
        pass

    # Phase 3: Fine-tune with local pixel flips around the optimized shape
    # Only a few iterations to stay within time budget
    if time_ok() and time.time() - search_start < search_time_limit - 20:
        indep_pixels = [(i, j) for i in range(16) for j in range(16)]
        np.random.shuffle(indep_pixels)
        
        improved_count = 0
        for (i, j) in indep_pixels[:40]:  # Limit to 40 flips
            if not time_ok() or time.time() - search_start > search_time_limit:
                break
            
            old_val = current_cell[i, j]
            new_cell = current_cell.copy()
            new_cell[i, j] = 1 - old_val
            new_cell = enforce_symmetry(new_cell)
            
            coeffs = get_fourier_coeffs(new_cell)
            new_ratio = compute_band_ratio(coeffs)
            
            if new_ratio > current_ratio + 1e-8:
                current_cell = new_cell
                current_ratio = new_ratio
                improved_count += 1
    
    # Final verification
    final_coeffs = get_fourier_coeffs(current_cell)
    final_ratio = compute_band_ratio(final_coeffs)
    
    if final_ratio < current_ratio - 0.01:
        # Fallback
        current_cell = best_cell
        current_ratio = best_ratio
    else:
        current_ratio = final_ratio

    return current_cell.tolist()
