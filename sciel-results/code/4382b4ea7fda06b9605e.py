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
    DEADLINE = 210.0  # safety margin before 230s kill
    
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
        avg = (cell + rot90 + rot180 + rot270 + hflip + vflip + trans + np.rot90(trans, 1))
        result = (avg >= 4).astype(int)
        return result

    def generate_superellipse_single(n_exp, a, b):
        """
        Generate a cell with a single centrally-located superellipse.
        Equation: |x/a|^n + |y/b|^n <= 1
        """
        cell = np.zeros((size, size), dtype=int)
        coords = (np.arange(size) + 0.5) / size - 0.5
        X, Y = np.meshgrid(coords, coords, indexing='xy')
        
        val = np.abs(X / a) ** n_exp + np.abs(Y / b) ** n_exp
        cell[val <= 1.0] = 1
        
        return enforce_symmetry(cell)

    def generate_four_superellipses(n_exp, a, b, offset):
        """
        Generate a cell with 4 superellipses at (±offset, ±offset).
        Equation: |x/a|^n + |y/b|^n <= 1
        """
        cell = np.zeros((size, size), dtype=int)
        coords = (np.arange(size) + 0.5) / size - 0.5
        X, Y = np.meshgrid(coords, coords, indexing='xy')
        
        centers = [(offset, offset), (-offset, offset), (offset, -offset), (-offset, -offset)]
        
        for cx, cy in centers:
            dx = X - cx
            dy = Y - cy
            val = np.abs(dx / a) ** n_exp + np.abs(dy / b) ** n_exp
            cell[val <= 1.0] = 1
            
        return enforce_symmetry(cell)

    def generate_cross(radius):
        """Generate a cross-shaped dielectric pattern."""
        cell = np.zeros((size, size), dtype=int)
        center = size // 2
        for i in range(size):
            for j in range(size):
                if abs(i - center) <= radius or abs(j - center) <= radius:
                    cell[i, j] = 1
        return enforce_symmetry(cell)

    def generate_ring(inner_r, outer_r):
        """Generate a ring-shaped dielectric pattern."""
        cell = np.zeros((size, size), dtype=int)
        coords = (np.arange(size) + 0.5) / size - 0.5
        X, Y = np.meshgrid(coords, coords, indexing='xy')
        R = np.sqrt(X**2 + Y**2)
        cell[(R >= inner_r) & (R <= outer_r)] = 1
        return enforce_symmetry(cell)

    def generate_stars(n_points, inner_r, outer_r):
        """Generate a star-shaped dielectric pattern."""
        cell = np.zeros((size, size), dtype=int)
        coords = (np.arange(size) + 0.5) / size - 0.5
        X, Y = np.meshgrid(coords, coords, indexing='xy')
        theta = np.arctan2(Y, X)
        r = np.sqrt(X**2 + Y**2)
        
        # Star shape: r(theta) = inner_r + (outer_r - inner_r) * |cos(n_points * theta / 2)|
        r_star = inner_r + (outer_r - inner_r) * np.abs(np.cos(n_points * theta / 2.0))
        cell[r <= r_star] = 1
        return enforce_symmetry(cell)

    # Phase 1: Evaluate diverse initial candidates
    best_ratio = -np.inf
    best_cell = None
    
    # Evaluate single superellipses with various shape exponents
    for n_exp in [1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0]:
        for a in [0.15, 0.18, 0.20, 0.22, 0.25, 0.28]:
            for b in [0.15, 0.18, 0.20, 0.22, 0.25, 0.28]:
                if not time_ok():
                    break
                cell = generate_superellipse_single(n_exp, a, b)
                coeffs = get_fourier_coeffs(cell)
                ratio = compute_band_ratio(coeffs)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_cell = cell.copy()
        if not time_ok():
            break
    if not time_ok():
        return best_cell.tolist() if best_cell is not None else np.zeros((size, size), dtype=int).tolist()

    # Evaluate four superellipses
    for n_exp in [2.0, 3.0, 4.0]:
        for a in [0.08, 0.10, 0.12]:
            for b in [0.08, 0.10, 0.12]:
                for offset in [0.20, 0.22, 0.25, 0.28]:
                    if not time_ok():
                        break
                    cell = generate_four_superellipses(n_exp, a, b, offset)
                    coeffs = get_fourier_coeffs(cell)
                    ratio = compute_band_ratio(coeffs)
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_cell = cell.copy()
                    if not time_ok():
                        break
                if not time_ok():
                    break
            if not time_ok():
                break
        if not time_ok():
            break
    if not time_ok():
        return best_cell.tolist() if best_cell is not None else np.zeros((size, size), dtype=int).tolist()

    # Evaluate cross patterns
    for radius in [1, 2, 3, 4, 5]:
        if not time_ok():
            break
        cell = generate_cross(radius)
        coeffs = get_fourier_coeffs(cell)
        ratio = compute_band_ratio(coeffs)
        if ratio > best_ratio:
            best_ratio = ratio
            best_cell = cell.copy()
    if not time_ok():
        return best_cell.tolist() if best_cell is not None else np.zeros((size, size), dtype=int).tolist()

    # Evaluate ring patterns
    for inner_r in [0.15, 0.18, 0.20]:
        for outer_r in [0.25, 0.28, 0.30]:
            if not time_ok():
                break
            cell = generate_ring(inner_r, outer_r)
            coeffs = get_fourier_coeffs(cell)
            ratio = compute_band_ratio(coeffs)
            if ratio > best_ratio:
                best_ratio = ratio
                best_cell = cell.copy()
    if not time_ok():
        return best_cell.tolist() if best_cell is not None else np.zeros((size, size), dtype=int).tolist()

    # Evaluate star patterns
    for n_points in [4, 6, 8]:
        for inner_r in [0.10, 0.12, 0.15]:
            for outer_r in [0.25, 0.28, 0.30]:
                if not time_ok():
                    break
                cell = generate_stars(n_points, inner_r, outer_r)
                coeffs = get_fourier_coeffs(cell)
                ratio = compute_band_ratio(coeffs)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_cell = cell.copy()
                if not time_ok():
                    break
            if not time_ok():
                break
        if not time_ok():
            break
    if not time_ok():
        return best_cell.tolist() if best_cell is not None else np.zeros((size, size), dtype=int).tolist()

    if best_cell is None:
        best_cell = generate_superellipse_single(2.0, 0.20, 0.20)
        best_ratio = compute_band_ratio(get_fourier_coeffs(best_cell))

    # Phase 2: Local optimization using scipy
    current_cell = best_cell.copy()
    current_ratio = best_ratio
    
    search_start = time.time()
    search_time_limit = 140.0
    
    # Try to optimize using pixel-level local search
    indep_pixels = [(i, j) for i in range(16) for j in range(16)]
    np.random.shuffle(indep_pixels)
    
    no_improve_count = 0
    max_no_improve = 80
    
    while time.time() - search_start < search_time_limit and no_improve_count < max_no_improve and time_ok():
        improved = False
        
        # Pick a random independent pixel and try flipping it
        (i, j) = indep_pixels[np.random.randint(0, len(indep_pixels))]
        
        old_val = current_cell[i, j]
        new_cell = current_cell.copy()
        new_cell[i, j] = 1 - old_val
        new_cell = enforce_symmetry(new_cell)
        
        coeffs = get_fourier_coeffs(new_cell)
        new_ratio = compute_band_ratio(coeffs)
        
        if new_ratio > current_ratio + 1e-8:
            current_cell = new_cell
            current_ratio = new_ratio
            improved = True
            no_improve_count = 0
        else:
            no_improve_count += 1
    
    # Phase 3: More aggressive local search with multiple flips
    if time_ok() and (DEADLINE - (time.time() - t_start)) > 40:
        refine_start = time.time()
        refine_limit = 30.0
        
        no_improve_refine = 0
        while time.time() - refine_start < refine_limit and no_improve_refine < 50 and time_ok():
            improved = False
            
            # Try flipping 2-3 random independent pixels simultaneously
            n_flips = np.random.randint(1, 4)
            pixels_to_flip = np.random.choice(len(indep_pixels), n_flips, replace=False)
            
            new_cell = current_cell.copy()
            for idx in pixels_to_flip:
                i, j = indep_pixels[idx]
                new_cell[i, j] = 1 - new_cell[i, j]
            new_cell = enforce_symmetry(new_cell)
            
            coeffs = get_fourier_coeffs(new_cell)
            new_ratio = compute_band_ratio(coeffs)
            
            if new_ratio > current_ratio + 1e-8:
                current_cell = new_cell
                current_ratio = new_ratio
                improved = True
                no_improve_refine = 0
            else:
                no_improve_refine += 1
    
    # Final verification
    final_coeffs = get_fourier_coeffs(current_cell)
    final_ratio = compute_band_ratio(final_coeffs)
    
    if final_ratio < best_ratio - 0.01:
        current_cell = best_cell
    else:
        best_ratio = final_ratio
        
    return current_cell.tolist()
