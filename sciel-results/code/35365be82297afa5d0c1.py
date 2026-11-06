"""Photonic crystal unit cell design for large TM band gap between bands 4 and 5."""
import numpy as np
import time

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
        
    def compute_band_ratio(coeffs, use_full_bz=True):
        """Compute gap-midgap ratio."""
        if not use_full_bz:
            # Use a subset of k-points for speed during search
            # Gamma, X, M and a few intermediates
            subset_indices = [0, 9, 18, 27, 4, 14, 23, 28] 
            k_indices = subset_indices
        else:
            k_indices = range(len(k_points))
            
        max_band4 = -np.inf
        min_band5 = np.inf
        
        for k_idx in k_indices:
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
        """Enforce 4-fold rotational + reflection symmetry (D4)."""
        cell = np.array(cell, dtype=int)
        # Generate all 8 D4 transforms
        transforms = [
            cell,
            np.rot90(cell, 1),
            np.rot90(cell, 2),
            np.rot90(cell, 3),
            cell[:, ::-1].copy(),
            cell[::-1, :].copy(),
            cell.T.copy(),
            np.rot90(cell.T.copy(), 1)
        ]
        # Average and threshold
        avg = np.sum(transforms, axis=0)
        result = (avg >= 4).astype(int)
        return result

    def generate_superellipses(n_exp, a_size, offset):
        """Generate a cell with 4 superellipses at (+/-offset, +/-offset)."""
        cell = np.zeros((size, size), dtype=int)
        coords = (np.arange(size) + 0.5) / size - 0.5
        X, Y = np.meshgrid(coords, coords, indexing='xy')
        
        centers = [(offset, offset), (-offset, offset), (offset, -offset), (-offset, -offset)]
        
        for cx, cy in centers:
            dx = X - cx
            dy = Y - cy
            val = np.abs(dx / a_size) ** n_exp + np.abs(dy / a_size) ** n_exp
            cell[val <= 1.0] = 1
            
        return enforce_symmetry(cell)

    def generate_hexagonal_like(radius):
        """Generate a pattern approximating hexagonal close packing of holes."""
        cell = np.ones((size, size), dtype=int) # Start with dielectric
        coords = (np.arange(size) + 0.5) / size - 0.5
        X, Y = np.meshgrid(coords, coords, indexing='xy')
        
        # Hexagonal lattice points in the unit cell
        # Center (0,0)
        # Neighbors at (1,0), (0,1), etc scaled by radius
        # For a 32x32 cell, we can place holes at:
        # (0,0), (0.5, 0), (0.25, 0.433), etc?
        # Simpler: Just use circles at specific symmetric positions.
        # Holes at (0,0), (0.5, 0), (0, 0.5), (0.5, 0.5) - this is square lattice of holes.
        # Let's try a "quasi-hexagonal" arrangement:
        # Holes at (0,0) and (0.5, 0.5) and (0.5, 0) and (0, 0.5)
        # This is effectively a square lattice of holes with 4 holes per cell.
        
        centers = [
            (0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.5, 0.5),
            (-0.5, 0.0), (0.5, -0.0), (-0.0, -0.5), (-0.5, -0.5)
        ]
        
        for cx, cy in centers:
            dx = X - cx
            dy = Y - cy
            dist = np.sqrt(dx**2 + dy**2)
            cell[dist <= radius] = 0 # Air holes
            
        return enforce_symmetry(cell)

    # Phase 1: Fast Global Search
    best_ratio = -np.inf
    best_cell = None
    
    # Candidate 1: Square holes (Superellipse n=4)
    # Candidate 2: Circular holes (Superellipse n=100)
    # Candidate 3: Hexagonal-like (Multiple circles)
    
    candidates = []
    
    # Superellipse variations
    for n in [4.0, 10.0]:
        for a in [0.15, 0.18, 0.20]:
            for off in [0.25]:
                cell = generate_superellipses(n, a, off)
                candidates.append(cell)
                
    # Hexagonal-like variations
    for r in [0.10, 0.12, 0.14]:
        cell = generate_hexagonal_like(r)
        candidates.append(cell)
        
    # Evaluate candidates with coarse BZ
    for cell in candidates:
        if not time_ok():
            break
        coeffs = get_fourier_coeffs(cell)
        ratio = compute_band_ratio(coeffs, use_full_bz=False)
        if ratio > best_ratio:
            best_ratio = ratio
            best_cell = cell.copy()
            
    if best_cell is None:
        best_cell = generate_superellipses(4.0, 0.18, 0.25)
        
    # Phase 2: Refine best candidate with full BZ and local pixel flipping
    current_cell = best_cell.copy()
    coeffs = get_fourier_coeffs(current_cell)
    current_ratio = compute_band_ratio(coeffs, use_full_bz=True)
    
    # Simple local optimization: try flipping a few pixels in the fundamental domain
    # We limit this to a very small number of steps to ensure we finish in time.
    max_steps = 20
    step = 0
    while step < max_steps and time_ok():
        step += 1
        improved = False
        # Try flipping a random pixel in the fundamental domain
        # Fundamental domain is 16x16 top-left quadrant? No, D4 symmetry implies 8-fold.
        # We can just pick a random pixel and enforce symmetry.
        i = np.random.randint(0, size)
        j = np.random.randint(0, size)
        
        new_cell = current_cell.copy()
        new_cell[i, j] = 1 - new_cell[i, j]
        new_cell = enforce_symmetry(new_cell)
        
        new_coeffs = get_fourier_coeffs(new_cell)
        new_ratio = compute_band_ratio(new_coeffs, use_full_bz=True)
        
        if new_ratio > current_ratio:
            current_cell = new_cell
            current_ratio = new_ratio
            improved = True
            
        if not improved and step > 5:
            # If no improvement in a few steps, stop early to save time
            # We are already using full BZ which is expensive.
            pass

    return current_cell.tolist()
# EVOLVE-BLOCK-END
