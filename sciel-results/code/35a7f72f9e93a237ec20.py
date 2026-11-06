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
    
    # K-points in irreducible BZ
    pi = np.pi
    k_points_full = []
    for i in range(10):
        t = i / 9.0
        k_points_full.append((t * pi, 0.0))
    for i in range(10):
        t = i / 9.0
        k_points_full.append((pi, t * pi))
    for i in range(10):
        t = i / 9.0
        k_points_full.append((pi * (1 - t), pi * (1 - t)))
    k_points_full = np.array(k_points_full)
    
    # Reduced set for fast search (Gamma, M, K, and midpoints)
    # Gamma: (0,0)
    # M: (pi, 0) and (0, pi) -> (pi, 0) is in list, (0, pi) is not explicitly but (pi, 0) covers edge.
    # K: (pi, pi)
    # Let's pick indices: 0 (Gamma), 9 (M1), 19 (M2), 29 (K)
    # And a few interior ones: 4, 14, 24
    k_search_indices = [0, 4, 9, 14, 19, 24, 29]
    
    k_points_search = k_points_full[k_search_indices]
    
    # Precompute (k+G)^2 for full and search sets
    k2_full = np.zeros((len(k_points_full), n_pw_total))
    for i, (kx, ky) in enumerate(k_points_full):
        k2_full[i] = (kx + Gx) ** 2 + (ky + Gy) ** 2
        
    k2_search = k2_full[k_search_indices]

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
    
    # Precompute symmetric pixel mappings
    # Independent pixels: i in [0, 16), j in [0, 16)
    # Symmetry group C4v
    # (i, j) -> 
    # (i, j)
    # (31-i, j)
    # (i, 31-j)
    # (31-i, 31-j)
    # (j, i)
    # (31-j, i)
    # (j, 31-i)
    # (31-j, 31-i)
    
    sym_mappings = {}
    for i in range(16):
        for j in range(16):
            indices = set()
            indices.add((i, j))
            indices.add((31-i, j))
            indices.add((i, 31-j))
            indices.add((31-i, 31-j))
            indices.add((j, i))
            indices.add((31-j, i))
            indices.add((j, 31-i))
            indices.add((31-j, 31-i))
            sym_mappings[(i, j)] = list(indices)
            
    indep_pixels = list(sym_mappings.keys())

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

    def compute_band_ratio_fast(coeffs):
        """Compute gap-midgap ratio using reduced k-points."""
        max_band4 = -np.inf
        min_band5 = np.inf
        
        for k_idx in range(len(k_points_search)):
            k2 = k2_search[k_idx]
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

    def compute_band_ratio_full(coeffs):
        """Compute gap-midgap ratio using all k-points."""
        max_band4 = -np.inf
        min_band5 = np.inf
        
        for k_idx in range(len(k_points_full)):
            k2 = k2_full[k_idx]
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
        # 4-fold rotation: 90, 180, 270 degrees
        rot90 = np.rot90(cell, 1)
        rot180 = np.rot90(cell, 2)
        rot270 = np.rot90(cell, 3)
        # Reflections
        hflip = cell[:, ::-1].copy()
        vflip = cell[::-1, :].copy()
        # Transpose
        trans = cell.T.copy()
        # All 8 symmetries
        avg = (cell + rot90 + rot180 + rot270 + hflip + vflip + trans + np.rot90(trans, 1))
        result = (avg >= 4).astype(int)
        return result

    def apply_flip(cell, pixel):
        """Flip the specified pixel and its symmetric counterparts."""
        new_cell = cell.copy()
        for (i, j) in sym_mappings[pixel]:
            new_cell[i, j] = 1 - new_cell[i, j]
        return new_cell

    # Phase 1: Generate diverse initial candidates
    candidates = []
    
    # 1. Four rods
    for r in [0.08, 0.12, 0.16]:
        for off in [0.20, 0.25, 0.30]:
            cell = np.zeros((size, size), dtype=int)
            centers = [(off, off), (-off, off), (off, -off), (-off, -off)]
            for i in range(size):
                for j in range(size):
                    x = (i + 0.5) / size - 0.5
                    y = (j + 0.5) / size - 0.5
                    for cx, cy in centers:
                        dx = x - cx
                        dy = y - cy
                        if dx * dx + dy * dy <= r * r:
                            cell[i, j] = 1
                            break
            candidates.append(enforce_symmetry(cell))
            
    # 2. Single rod
    for r in [0.25, 0.30]:
        cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                if x * x + y * y <= r * r:
                    cell[i, j] = 1
        candidates.append(enforce_symmetry(cell))
        
    # 3. Cross pattern
    cell = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            if i in [15, 16] or j in [15, 16]:
                cell[i, j] = 1
    candidates.append(enforce_symmetry(cell))
    
    # 4. Random noise
    for _ in range(5):
        cell = np.random.randint(0, 2, (size, size))
        candidates.append(enforce_symmetry(cell))

    # Evaluate initial candidates
    best_ratio = -np.inf
    best_cell = None
    
    for cell in candidates:
        if not time_ok():
            break
        coeffs = get_fourier_coeffs(cell)
        ratio = compute_band_ratio_fast(coeffs)
        if ratio > best_ratio:
            best_ratio = ratio
            best_cell = cell.copy()

    if best_cell is None:
        best_cell = np.zeros((size, size), dtype=int)
        best_ratio = 0.0

    # Phase 2: Simulated Annealing
    current_cell = best_cell.copy()
    current_ratio = best_ratio
    global_best_cell = best_cell.copy()
    global_best_ratio = best_ratio
    
    T_start = 0.5
    T_end = 0.01
    cooling_rate = 0.995
    
    T = T_start
    no_improve = 0
    
    while time_ok() and T > T_end:
        # Pick a random independent pixel
        pixel = random.choice(indep_pixels)
        
        # Flip it
        new_cell = apply_flip(current_cell, pixel)
        
        coeffs = get_fourier_coeffs(new_cell)
        new_ratio = compute_band_ratio_fast(coeffs)
        
        delta = new_ratio - current_ratio
        
        if delta > 0 or random.random() < np.exp(delta / T):
            current_cell = new_cell
            current_ratio = new_ratio
            no_improve = 0
            
            if current_ratio > global_best_ratio:
                global_best_ratio = current_ratio
                global_best_cell = current_cell.copy()
        else:
            no_improve += 1
            
        T *= cooling_rate
        
        # If stuck for too long, restart temperature or jump
        if no_improve > 100:
            T = max(T, 0.1)
            no_improve = 0

    # Phase 3: Local Search with Full K-points
    # Use the best cell found in SA
    current_cell = global_best_cell.copy()
    
    # Verify with full k-points
    coeffs = get_fourier_coeffs(current_cell)
    current_ratio = compute_band_ratio_full(coeffs)
    global_best_ratio = current_ratio
    global_best_cell = current_cell.copy()
    
    # Local search: flip single pixels, accept only if full ratio improves
    # Limit steps to ensure we finish
    steps = 0
    max_steps = 200
    
    while steps < max_steps and time_ok():
        improved = False
        random.shuffle(indep_pixels)
        
        for pixel in indep_pixels:
            if steps >= max_steps or not time_ok():
                break
                
            new_cell = apply_flip(current_cell, pixel)
            coeffs = get_fourier_coeffs(new_cell)
            new_ratio = compute_band_ratio_full(coeffs)
            
            if new_ratio > current_ratio + 1e-8:
                current_cell = new_cell
                current_ratio = new_ratio
                improved = True
                global_best_cell = current_cell.copy()
                global_best_ratio = current_ratio
            steps += 1
            
        if not improved:
            break

    return global_best_cell.tolist()
# EVOLVE-BLOCK-END
