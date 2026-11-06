"""Photonic crystal unit cell design for large TM band gap between bands 4 and 5."""
import numpy as np
import time
import math

# EVOLVE-BLOCK-START
def design(problem):
    """Design a 32x32 symmetric unit cell maximizing TM band gap between bands 4 and 5."""
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1)
    eps_high = problem.get("eps_high", 12)
    
    start_time = time.time()
    deadline = 215.0  # safety margin before 230s kill
    
    def time_ok():
        return time.time() - start_time < deadline
    
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
    
    def compute_band_ratio_full(coeffs):
        """Compute gap-midgap ratio using all 30 k-points and 289 PWs."""
        max_band4 = -np.inf
        min_band5 = np.inf
        
        for k_idx in range(len(k_points)):
            k2 = k2_all[k_idx]
            H = np.zeros((n_pw_total, n_pw_total), dtype=complex)
            H[valid_rows, valid_cols] = k2[valid_rows] * coeffs[diff_idx[valid_rows, valid_cols]]
            H = (H + H.conj().T) / 2.0
            eigvals = np.linalg.eigvalsh(H)
            eigvals = np.sort(eigvals)
            if len(eigvals) >= 5:
                max_band4 = max(max_band4, eigvals[3])
                min_band5 = min(min_band5, eigvals[4])
        
        gap = min_band5 - max_band4
        midgap = (min_band5 + max_band4) / 2.0
        if midgap > 1e-10:
            return gap / midgap
        return 0.0
    
    def compute_band_ratio_fast(coeffs):
        """Fast approximate evaluation using subset of k-points for search guidance."""
        k_subset = [0, 4, 9, 14, 19, 24, 29]
        max_band4 = -np.inf
        min_band5 = np.inf
        
        for k_idx in k_subset:
            k2 = k2_all[k_idx]
            H = np.zeros((n_pw_total, n_pw_total), dtype=complex)
            H[valid_rows, valid_cols] = k2[valid_rows] * coeffs[diff_idx[valid_rows, valid_cols]]
            H = (H + H.conj().T) / 2.0
            eigvals = np.linalg.eigvalsh(H)
            eigvals = np.sort(eigvals)
            if len(eigvals) >= 5:
                max_band4 = max(max_band4, eigvals[3])
                min_band5 = min(min_band5, eigvals[4])
        
        gap = min_band5 - max_band4
        midgap = (min_band5 + max_band4) / 2.0
        if midgap > 1e-10:
            return gap / midgap
        return 0.0
    
    def enforce_symmetry(cell):
        """Enforce square symmetry by averaging over the group."""
        cell = np.array(cell, dtype=int)
        cell_rot = cell[::-1, ::-1].copy()
        cell_hflip = cell[:, ::-1].copy()
        cell_vflip = cell[::-1, :].copy()
        result = (cell + cell_rot + cell_hflip + cell_vflip)
        result = (result >= 2).astype(int)
        return result
    
    def generate_superellipse(a, b, n_exp):
        """Generate a superellipse |x/a|^n + |y/b|^n <= 1."""
        cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                if a > 0 and b > 0:
                    val = abs(x / a) ** n_exp + abs(y / b) ** n_exp
                    if val <= 1.0:
                        cell[i, j] = 1
        return enforce_symmetry(cell)
    
    def generate_four_rods(radius, offset):
        """Four rods at (±offset, ±offset)."""
        cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                for sx in [1, -1]:
                    for sy in [1, -1]:
                        dx = x - sx * offset
                        dy = y - sy * offset
                        if dx * dx + dy * dy < radius * radius:
                            cell[i, j] = 1
        return enforce_symmetry(cell)
    
    def generate_rod(radius):
        """Single centered rod."""
        cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                if x * x + y * y < radius * radius:
                    cell[i, j] = 1
        return enforce_symmetry(cell)
    
    # Generate diverse candidates
    candidates = []
    
    # Superellipses - structural change: optimize shape exponent and semi-axes
    # Focus on non-circular shapes that generate strong higher-order harmonics
    for n_exp in [1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0]:
        for a in [0.15, 0.20, 0.25, 0.30, 0.35]:
            for b in [0.15, 0.20, 0.25, 0.30, 0.35]:
                if a > 0.05 and b > 0.05:
                    cell = generate_superellipse(a, b, n_exp)
                    fill = np.sum(cell) / (size * size)
                    if 0.1 < fill < 0.7:
                        candidates.append(cell)
    
    # Four-rod patterns (known good baseline)
    for r in [0.06, 0.08, 0.10, 0.12, 0.14]:
        for off in [0.20, 0.22, 0.25, 0.28, 0.30]:
            cell = generate_four_rods(r, off)
            fill = np.sum(cell) / (size * size)
            if 0.1 < fill < 0.7:
                candidates.append(cell)
    
    # Single rods
    for r in [0.15, 0.18, 0.20, 0.22, 0.25, 0.28]:
        cell = generate_rod(r)
        fill = np.sum(cell) / (size * size)
        if 0.1 < fill < 0.7:
            candidates.append(cell)
    
    # Evaluate candidates with fast evaluator first, then refine top ones
    scored_candidates = []
    for cell in candidates:
        if not time_ok():
            break
        
        coeffs = get_fourier_coeffs(cell)
        score = compute_band_ratio_fast(coeffs)
        scored_candidates.append((score, cell))
    
    # Sort by fast score
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Refine top candidates with full evaluator
    best_cell = None
    best_score = -np.inf
    
    for score, cell in scored_candidates[:10]:  # Top 10
        if not time_ok():
            break
        
        coeffs = get_fourier_coeffs(cell)
        full_score = compute_band_ratio_full(coeffs)
        
        if full_score > best_score:
            best_score = full_score
            best_cell = cell
    
    # If no good candidate, fall back to a known good design
    if best_cell is None:
        best_cell = generate_four_rods(0.10, 0.25)
    
    # Convert to nested list
    result = best_cell.tolist()
    return result
# EVOLVE-BLOCK-END
