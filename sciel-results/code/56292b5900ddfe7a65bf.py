"""Photonic crystal unit cell design for large TM band gap between bands 4 and 5."""
import numpy as np
import time
import random

# EVOLVE-BLOCK-START
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
    
    # K-points in irreducible BZ (30 points as specified)
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
    
    def compute_band_ratio(coeffs, k_indices=None):
        """Compute gap-midgap ratio."""
        if k_indices is None:
            k_indices = range(len(k_points))
        
        max_band4 = -np.inf
        min_band5 = np.inf
        
        for k_idx in k_indices:
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
        """Enforce 4-fold rotational + reflection symmetry (d4 group)."""
        cell = np.array(cell, dtype=int)
        # All 8 symmetry operations of the square group
        s0 = cell
        s1 = np.rot90(cell, 1)
        s2 = np.rot90(cell, 2)
        s3 = np.rot90(cell, 3)
        s4 = cell[:, ::-1].copy()
        s5 = cell[::-1, :].copy()
        s6 = cell.T.copy()
        s7 = np.rot90(cell.T, 1)
        avg = (s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7)
        result = (avg >= 4).astype(int)
        return result
    
    def evaluate_cell(cell):
        """Evaluate a cell and return the ratio."""
        coeffs = get_fourier_coeffs(cell)
        return compute_band_ratio(coeffs)
    
    # Generate initial candidates using various geometric primitives
    def generate_cross(arm_width, arm_length):
        """Generate a cross pattern."""
        cell = np.zeros((size, size), dtype=int)
        center = size // 2
        half_w = arm_width // 2
        half_l = arm_length // 2
        
        # Horizontal arm
        for i in range(center - half_l, center + half_l + 1):
            for j in range(center - half_w, center + half_w + 1):
                if 0 <= i < size and 0 <= j < size:
                    cell[i, j] = 1
        
        # Vertical arm
        for i in range(center - half_w, center + half_w + 1):
            for j in range(center - half_l, center + half_l + 1):
                if 0 <= i < size and 0 <= j < size:
                    cell[i, j] = 1
        
        return enforce_symmetry(cell)
    
    def generate_four_squares(sq_size, offset):
        """Generate four squares at corners."""
        cell = np.zeros((size, size), dtype=int)
        half_sq = sq_size // 2
        centers = [(offset, offset), (size - offset - 1, offset), 
                   (offset, size - offset - 1), (size - offset - 1, size - offset - 1)]
        for cx, cy in centers:
            for i in range(max(0, cy - half_sq), min(size, cy + half_sq + 1)):
                for j in range(max(0, cx - half_sq), min(size, cx + half_sq + 1)):
                    cell[i, j] = 1
        return enforce_symmetry(cell)
    
    def generate_ring(inner_r, outer_r):
        """Generate a ring (annulus) centered in the cell."""
        cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                r2 = x * x + y * y
                if inner_r * inner_r <= r2 <= outer_r * outer_r:
                    cell[i, j] = 1
        return enforce_symmetry(cell)
    
    def generate_four_rods(radius, offset):
        """Four dielectric rods at (±offset, ±offset)."""
        cell = np.zeros((size, size), dtype=int)
        centers = [(offset, offset), (size - offset - 1, offset), 
                   (offset, size - offset - 1), (size - offset - 1, size - offset - 1)]
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                for cx, cy in centers:
                    dx = x - (cx + 0.5) / size + 0.5
                    dy = y - (cy + 0.5) / size + 0.5
                    if dx * dx + dy * dy <= radius * radius:
                        cell[i, j] = 1
                        break
        return enforce_symmetry(cell)
    
    def generate_diamond(radius, offset):
        """Four diamond shapes at (±offset, ±offset)."""
        cell = np.zeros((size, size), dtype=int)
        centers = [(offset, offset), (size - offset - 1, offset), 
                   (offset, size - offset - 1), (size - offset - 1, size - offset - 1)]
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                for cx, cy in centers:
                    dx = x - (cx + 0.5) / size + 0.5
                    dy = y - (cy + 0.5) / size + 0.5
                    if abs(dx) + abs(dy) <= radius:
                        cell[i, j] = 1
                        break
        return enforce_symmetry(cell)
    
    # Phase 1: Evaluate diverse initial candidates
    candidates = []
    
    # Crosses
    for w in [2, 3, 4, 5, 6]:
        for l in [6, 8, 10, 12, 14, 16]:
            candidates.append(generate_cross(w, l))
    
    # Four squares
    for sq in [4, 6, 8, 10, 12]:
        for off in [4, 6, 8, 10, 12]:
            candidates.append(generate_four_squares(sq, off))
    
    # Rings
    for ir in [0.1, 0.15, 0.2, 0.25]:
        for orr in [0.3, 0.35, 0.4, 0.45, 0.5]:
            if ir < orr:
                candidates.append(generate_ring(ir, orr))
    
    # Four rods
    for r in [0.05, 0.08, 0.10, 0.12, 0.15]:
        for off in [6, 8, 10, 12, 14]:
            candidates.append(generate_four_rods(r, off))
    
    # Four diamonds
    for r in [0.08, 0.12, 0.15, 0.18, 0.22]:
        for off in [6, 8, 10, 12, 14]:
            candidates.append(generate_diamond(r, off))
    
    # Single centered shapes
    def generate_single_diamond(radius):
        cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                if abs(x) + abs(y) <= radius:
                    cell[i, j] = 1
        return enforce_symmetry(cell)
    
    for r in [0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
        candidates.append(generate_single_diamond(r))
    
    def generate_single_square(sq_size):
        cell = np.zeros((size, size), dtype=int)
        half_sq = sq_size // 2
        center = size // 2
        for i in range(center - half_sq, center + half_sq + 1):
            for j in range(center - half_sq, center + half_sq + 1):
                if 0 <= i < size and 0 <= j < size:
                    cell[i, j] = 1
        return enforce_symmetry(cell)
    
    for sq in [8, 10, 12, 14, 16, 18]:
        candidates.append(generate_single_square(sq))
    
    # Evaluate all candidates
    best_ratio = -np.inf
    best_cell = None
    scored_cells = []
    
    for idx, cell in enumerate(candidates):
        if not time_ok():
            break
        ratio = evaluate_cell(cell)
        scored_cells.append((ratio, cell))
        if ratio > best_ratio:
            best_ratio = ratio
            best_cell = cell.copy()
    
    if best_cell is None:
        best_cell = generate_four_rods(0.10, 10)
        best_ratio = evaluate_cell(best_cell)
    
    # Sort by ratio descending to use top candidates for local optimization
    scored_cells.sort(key=lambda x: -x[0])
    
    # Phase 2: Local optimization with pixel flips
    # Use multiple starting points from top candidates
    independent_pixels = [(i, j) for i in range(16) for j in range(16)]
    
    def local_optimize(start_cell, start_ratio, max_iter=200):
        """Local optimization via pixel flips."""
        current_cell = start_cell.copy()
        current_ratio = start_ratio
        no_improve = 0
        max_no_improve = 40
        
        for iteration in range(max_iter):
            if not time_ok():
                break
            if no_improve >= max_no_improve:
                break
            
            # Try flipping each independent pixel
            random.shuffle(independent_pixels)
            improved = False
            
            for (i, j) in independent_pixels:
                if not time_ok():
                    break
                
                old_val = current_cell[i, j]
                new_cell = current_cell.copy()
                new_cell[i, j] = 1 - old_val
                new_cell = enforce_symmetry(new_cell)
                
                new_ratio = evaluate_cell(new_cell)
                
                if new_ratio > current_ratio + 1e-9:
                    current_cell = new_cell
                    current_ratio = new_ratio
                    improved = True
                    no_improve = 0
                    break
                else:
                    no_improve += 1
            
            if not improved:
                no_improve += 1
        
        return current_cell, current_ratio
    
    # Optimize from top 3 starting points
    search_time_limit = 180.0
    search_start = time.time()
    
    for idx in range(min(3, len(scored_cells))):
        if not time_ok() or time.time() - search_start > search_time_limit:
            break
        
        start_ratio, start_cell = scored_cells[idx]
        opt_cell, opt_ratio = local_optimize(start_cell, start_ratio, max_iter=150)
        
        if opt_ratio > best_ratio:
            best_ratio = opt_ratio
            best_cell = opt_cell.copy()
    
    # Phase 3: Additional local search from best result
    if time_ok():
        opt_cell, opt_ratio = local_optimize(best_cell, best_ratio, max_iter=100)
        if opt_ratio > best_ratio:
            best_ratio = opt_ratio
            best_cell = opt_cell.copy()
    
    # Final verification
    final_ratio = evaluate_cell(best_cell)
    if final_ratio > best_ratio:
        best_ratio = final_ratio
    
    return best_cell.tolist()
# EVOLVE-BLOCK-END
