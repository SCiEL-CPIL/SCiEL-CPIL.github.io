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
    
    start_time = time.time()
    deadline = 218.0  # safety margin before 230s kill
    
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
    n_kpts = len(k_points)
    
    # Precompute (k+G)^2
    k2_all = np.zeros((n_kpts, n_pw_total))
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
    valid_diff = diff_idx[valid_rows, valid_cols]
    
    # Fast evaluation with configurable k-point subset
    def compute_band_ratio(coeffs, k_indices=None):
        """Compute gap-midgap ratio. k_indices: list of k-point indices to use."""
        if k_indices is None:
            k_indices = range(n_kpts)
        max_band4 = -np.inf
        min_band5 = np.inf
        
        for k_idx in k_indices:
            k2 = k2_all[k_idx]
            H = np.zeros((n_pw_total, n_pw_total), dtype=complex)
            H[valid_rows, valid_cols] = k2[valid_rows] * coeffs[valid_diff]
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
    
    # Coarse k-point subsets for fast search
    k_coarse = [0, 4, 9, 14, 19, 24, 29]  # 7 points
    k_medium = [0, 2, 4, 6, 8, 9, 12, 14, 16, 18, 19, 22, 24, 26, 28, 29]  # 16 points
    
    def cell_to_coeffs(cell):
        """Compute Fourier coefficients of 1/eps from a 0/1 cell."""
        cell = np.array(cell, dtype=float)
        inv_eps = np.where(cell == 1, 1.0/eps_high, 1.0/eps_low)
        fft_2d = np.fft.fft2(inv_eps)
        coeffs = np.zeros(n_pw_total, dtype=complex)
        for idx in range(n_pw_total):
            gx = int(Gx[idx])
            gy = int(Gy[idx])
            fft_ix = gx % size
            fft_iy = gy % size
            coeffs[idx] = fft_2d[fft_ix, fft_iy] / (size * size)
        return coeffs
    
    def enforce_symmetry(cell):
        """Enforce square symmetry by averaging over the group."""
        cell = np.array(cell, dtype=int)
        cell_rot = cell[::-1, ::-1].copy()
        cell_hflip = cell[:, ::-1].copy()
        cell_vflip = cell[::-1, :].copy()
        result = (cell + cell_rot + cell_hflip + cell_vflip)
        result = (result >= 2).astype(int)
        return result
    
    def get_symmetry_orbit(i, j):
        """Get all symmetry-equivalent positions of (i,j)."""
        orbit = [(i, j)]
        # 90-degree rotations about center
        r = (size - 1 - i, j)
        if r not in orbit: orbit.append(r)
        r = (size - 1 - i, size - 1 - j)
        if r not in orbit: orbit.append(r)
        r = (i, size - 1 - j)
        if r not in orbit: orbit.append(r)
        # Also mirror through center
        r = (size - 1 - i, size - 1 - j)
        if r not in orbit: orbit.append(r)
        return orbit
    
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
    
    def generate_ring(inner_r, outer_r):
        """Dielectric ring."""
        cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                r2 = x * x + y * y
                if inner_r * inner_r < r2 < outer_r * outer_r:
                    cell[i, j] = 1
        return enforce_symmetry(cell)
    
    def generate_cross(arm_width, arm_length):
        """Cross shape."""
        cell = np.zeros((size, size), dtype=int)
        c = size // 2
        hw = arm_width // 2
        for i in range(size):
            for j in range(size):
                if abs(i - c) <= hw and abs(j - c) <= arm_length:
                    cell[i, j] = 1
                elif abs(j - c) <= hw and abs(i - c) <= arm_length:
                    cell[i, j] = 1
        return enforce_symmetry(cell)
    
    def generate_voronoi_cell(n_seeds, seed_positions=None):
        """Generate a cell from Voronoi tessellation with seeds on symmetry axes."""
        if seed_positions is None:
            # Generate seeds constrained to high-symmetry positions
            seeds = []
            # Center
            seeds.append((size//2, size//2))
            # On axes
            for _ in range(n_seeds - 1):
                axis = random.choice(['x', 'y', 'diag'])
                if axis == 'x':
                    seeds.append((random.randint(2, size-3), size//2))
                elif axis == 'y':
                    seeds.append((size//2, random.randint(2, size-3)))
                else:
                    d = random.randint(3, size//2 - 2)
                    seeds.append((size//2 + d, size//2 + d))
            # Enforce symmetry: for each seed, add its orbit
            all_seeds = []
            for s in seeds:
                for pos in get_symmetry_orbit(s[0], s[1]):
                    if pos not in all_seeds:
                        all_seeds.append(pos)
            seed_positions = all_seeds
        
        # Assign each pixel to nearest seed
        cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                min_dist = float('inf')
                min_seed = None
                for s in seed_positions:
                    d = (i - s[0])**2 + (j - s[1])**2
                    if d < min_dist:
                        min_dist = d
                        min_seed = s
                # Dielectric if assigned to even-indexed seed
                if min_seed is not None:
                    idx = seed_positions.index(min_seed)
                    if idx % 2 == 0:
                        cell[i, j] = 1
        return enforce_symmetry(cell)
    
    # Phase 1: Generate diverse initial candidates
    candidates = []
    for r in [0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30, 0.32, 0.35]:
        candidates.append(generate_rod(r))
    for r in [0.06, 0.08, 0.10, 0.12, 0.14, 0.16]:
        for off in [0.18, 0.20, 0.22, 0.25, 0.28, 0.30, 0.32]:
            candidates.append(generate_four_rods(r, off))
    for ir, orr in [(0.08, 0.18), (0.10, 0.20), (0.12, 0.22), (0.14, 0.25), (0.15, 0.28), (0.18, 0.30)]:
        candidates.append(generate_ring(ir, orr))
    for aw in [3, 5, 7, 9]:
        for al in [8, 10, 12, 14]:
            candidates.append(generate_cross(aw, al))
    # Voronoi-based candidates
    for _ in range(20):
        n_seeds = random.choice([3, 4, 5, 6, 7, 8])
        candidates.append(generate_voronoi_cell(n_seeds))
    
    # Evaluate all candidates with coarse evaluator
    best_cell = None
    best_score = -1
    
    for cell in candidates:
        if not time_ok():
            break
        fill = np.sum(cell) / (size * size)
        if fill < 0.1 or fill > 0.7:
            continue
        coeffs = cell_to_coeffs(cell)
        score = compute_band_ratio(coeffs, k_coarse)
        if score > best_score:
            best_score = score
            best_cell = cell
    
    # Phase 2: Pixel-level mutation with direct search
    if best_cell is not None and time_ok():
        best_cell = np.array(best_cell, dtype=int)
        
        # Precompute symmetry orbits for all pixels
        pixel_orbits = {}
        for i in range(size):
            for j in range(size):
                if (i, j) not in pixel_orbits:
                    pixel_orbits[(i, j)] = get_symmetry_orbit(i, j)
        
        # Get all unique orbits
        all_orbits = list(pixel_orbits.values())
        
        n_opt_steps = 0
        max_opt_steps = 500
        mutation_rate = 0.15  # fraction of orbits to flip per mutation
        
        while n_opt_steps < max_opt_steps and time_ok():
            n_opt_steps += 1
            
            # Create mutated cell
            new_cell = best_cell.copy()
            n_mutations = max(1, int(len(all_orbits) * mutation_rate))
            
            # Randomly select orbits to flip
            mutated_orbits = random.sample(all_orbits, n_mutations)
            for orbit in mutated_orbits:
                for (oi, oj) in orbit:
                    new_cell[oi, oj] = 1 - new_cell[oi, oj]
            
            # Enforce symmetry (should already be symmetric, but just in case)
            new_cell = enforce_symmetry(new_cell)
            
            # Check filling fraction
            fill = np.sum(new_cell) / (size * size)
            if fill < 0.05 or fill > 0.8:
                continue
            
            # Evaluate
            new_coeffs = cell_to_coeffs(new_cell)
            score = compute_band_ratio(new_coeffs, k_coarse)
            
            if score > best_score:
                best_score = score
                best_cell = new_cell
                # Adaptive: reduce mutation rate when improving
                mutation_rate = max(0.03, mutation_rate * 0.9)
            else:
                # Occasionally increase mutation rate to escape local minima
                if random.random() < 0.15:
                    mutation_rate = min(0.3, mutation_rate * 1.4)
        
        # Phase 2b: Refine with medium k-point set and smaller mutations
        if time_ok():
            refine_steps = 0
            max_refine_steps = 200
            refine_rate = 0.05
            
            while refine_steps < max_refine_steps and time_ok():
                refine_steps += 1
                n_mutations = max(1, int(len(all_orbits) * refine_rate))
                mutated_orbits = random.sample(all_orbits, n_mutations)
                
                new_cell = best_cell.copy()
                for orbit in mutated_orbits:
                    for (oi, oj) in orbit:
                        new_cell[oi, oj] = 1 - new_cell[oi, oj]
                
                new_cell = enforce_symmetry(new_cell)
                
                fill = np.sum(new_cell) / (size * size)
                if fill < 0.05 or fill > 0.8:
                    continue
                
                new_coeffs = cell_to_coeffs(new_cell)
                score = compute_band_ratio(new_coeffs, k_medium)
                
                if score > best_score:
                    best_score = score
                    best_cell = new_cell
                    refine_rate = max(0.01, refine_rate * 0.85)
                else:
                    if random.random() < 0.2:
                        refine_rate = min(0.15, refine_rate * 1.3)
    
    # Phase 3: Final evaluation with full 30 k-points
    if best_cell is not None and time_ok():
        final_coeffs = cell_to_coeffs(best_cell)
        final_score = compute_band_ratio(final_coeffs, None)
    
    # Convert to nested list
    result = best_cell.tolist()
    return result
# EVOLVE-BLOCK-END
