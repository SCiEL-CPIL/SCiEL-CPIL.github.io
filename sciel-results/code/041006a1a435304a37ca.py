import numpy as np
import time
import math

def design(problem):
    t_start = time.time()
    deadline = t_start + 115.0  # Safety margin before 120s limit
    
    size = problem['pixels']  # 32
    eps_low = problem['eps_low']  # 1
    eps_high = problem['eps_high']  # 12
    lower_band = problem['lower_band']  # 4
    
    # Define independent pixels under D4 symmetry
    # For a 32x32 cell (indices 0..31), center is at (15.5, 15.5)
    # Symmetries: 180° rotation, reflection about x-axis, y-axis, and diagonals
    # Fundamental domain: i in 0..15, j in i..31-i
    indep_indices = []
    for i in range(16):
        for j in range(i, 32 - i):
            indep_indices.append((i, j))
    
    n_indep = len(indep_indices)
    
    def expand_cell(indep_vals):
        """Expand independent pixels to full 32x32 cell using D4 symmetry."""
        cell = np.zeros((size, size), dtype=np.uint8)
        for idx, val in enumerate(indep_vals):
            i, j = indep_indices[idx]
            # Apply all 8 symmetry operations
            sym_positions = [
                (i, j),
                (31-i, 31-j),
                (31-i, j),
                (i, 31-j),
                (j, i),
                (31-j, 31-i),
                (j, 31-i),
                (31-j, i)
            ]
            for (ii, jj) in sym_positions:
                cell[ii, jj] = val
        return cell

    def compute_score(cell, n_pw=19, n_kpts=16):
        """
        Compute band gap score using plane wave method.
        Higher n_pw and n_kpts = more accurate but slower.
        """
        inv_eps = np.where(cell == 1, 1.0/eps_high, 1.0/eps_low)
        # Use FFT to get Fourier coefficients of 1/epsilon
        # Note: The evaluator uses exact coefficients, but FFT is a good proxy for search.
        # We need to be careful with the grid size. The cell is 32x32.
        # The plane wave expansion is over the unit cell.
        fft_inv = np.fft.fft2(inv_eps)
        
        # Plane wave indices
        half = n_pw // 2
        g_vals = np.arange(-half, half + 1)
        gx, gy = np.meshgrid(g_vals, g_vals, indexing='ij')
        g_vecs = np.stack([gx.flatten(), gy.flatten()], axis=1)
        n_pw_actual = len(g_vecs)
        
        # K-points along the irreducible Brillouin zone boundary
        # Gamma (0,0) -> X (0.5, 0) -> M (0.5, 0.5) -> Gamma
        k_points = []
        # Gamma to X: (t, 0) for t in [0, 0.5]
        n_seg = max(2, n_kpts // 3)
        for t in np.linspace(0, 0.5, n_seg):
            k_points.append((t, 0))
        # X to M: (0.5, t) for t in [0, 0.5]
        for t in np.linspace(0.5, 0.5, 1): # Just X
            pass
        for t in np.linspace(0, 0.5, n_seg - 1):
            k_points.append((0.5, t))
        # M to Gamma: (0.5 - t, 0.5 - t) for t in [0, 0.5]
        for t in np.linspace(0, 0.5, n_seg - 1):
            k_points.append((0.5 - t, 0.5 - t))
        
        max_omega_4 = 0.0
        min_omega_5 = np.inf
        
        # Pre-compute matrix structure parts
        # The matrix element H_ab = (k+G_b)^2 * eps^{-1}_{G_a - G_b}
        # We can vectorize the k-dependence partially, but let's keep it simple for now.
        
        for k in k_points:
            k_vec = np.array(k)
            matrix = np.zeros((n_pw_actual, n_pw_actual), dtype=np.complex128)
            
            # Vectorized calculation of (k+G)^2 for all G
            k_plus_g = k_vec[None, :] + g_vecs
            k_sq = np.sum(k_plus_g**2, axis=1)
            
            for alpha in range(n_pw_actual):
                ga = g_vecs[alpha]
                for beta in range(n_pw_actual):
                    gb = g_vecs[beta]
                    g_diff = ga - gb
                    idx_x = int(g_diff[0]) % size
                    idx_y = int(g_diff[1]) % size
                    coeff = fft_inv[idx_x, idx_y]
                    
                    val = coeff * k_sq[beta]
                    matrix[alpha, beta] = val
            
            eigs = np.linalg.eigvalsh(matrix)
            if len(eigs) > 4:
                w4 = np.sqrt(max(eigs[3], 0))
                w5 = np.sqrt(max(eigs[4], 0))
                if w4 > max_omega_4:
                    max_omega_4 = w4
                if w5 < min_omega_5:
                    min_omega_5 = w5
        
        if min_omega_5 == np.inf or max_omega_4 == 0:
            return -1.0
        gap = min_omega_5 - max_omega_4
        if gap < 0:
            return -1.0
        ratio = 2 * gap / (min_omega_5 + max_omega_4)
        return ratio

    def generate_candidate(shape_type, params):
        """
        Generate a cell based on a parametric shape definition.
        shape_type: 'circle', 'square', 'diamond', 'ring'
        params: tuple of parameters depending on shape
        """
        current_indep = [0] * n_indep
        center = 15.5
        
        if shape_type == 'circle':
            radius = params[0]
            for idx, (i, j) in enumerate(indep_indices):
                dx = i + 0.5 - center
                dy = j + 0.5 - center
                r = math.sqrt(dx*dx + dy*dy)
                if r < radius:
                    current_indep[idx] = 1
        elif shape_type == 'square':
            half_side = params[0]
            for idx, (i, j) in enumerate(indep_indices):
                dx = abs(i + 0.5 - center)
                dy = abs(j + 0.5 - center)
                if dx < half_side and dy < half_side:
                    current_indep[idx] = 1
        elif shape_type == 'diamond':
            radius_l1 = params[0]
            for idx, (i, j) in enumerate(indep_indices):
                dx = abs(i + 0.5 - center)
                dy = abs(j + 0.5 - center)
                if dx + dy < radius_l1:
                    current_indep[idx] = 1
        elif shape_type == 'ring':
            r_inner = params[0]
            r_outer = params[1]
            for idx, (i, j) in enumerate(indep_indices):
                dx = i + 0.5 - center
                dy = j + 0.5 - center
                r = math.sqrt(dx*dx + dy*dy)
                if r_inner < r < r_outer:
                    current_indep[idx] = 1
        elif shape_type == 'cross':
            width = params[0]
            for idx, (i, j) in enumerate(indep_indices):
                dx = abs(i + 0.5 - center)
                dy = abs(j + 0.5 - center)
                # Cross shape: either x or y is within width
                if dx < width or dy < width:
                    current_indep[idx] = 1
        elif shape_type == 'plus':
            # A plus sign with variable arm length and width
            arm_len = params[0]
            width = params[1]
            for idx, (i, j) in enumerate(indep_indices):
                dx = abs(i + 0.5 - center)
                dy = abs(j + 0.5 - center)
                if (dx < width and dy < arm_len) or (dy < width and dx < arm_len):
                    current_indep[idx] = 1
        elif shape_type == 'air_hole_circle':
            # Dielectric background with circular air hole
            radius = params[0]
            for idx, (i, j) in enumerate(indep_indices):
                dx = i + 0.5 - center
                dy = j + 0.5 - center
                r = math.sqrt(dx*dx + dy*dy)
                if r >= radius:
                    current_indep[idx] = 1
        elif shape_type == 'air_hole_square':
            # Dielectric background with square air hole
            half_side = params[0]
            for idx, (i, j) in enumerate(indep_indices):
                dx = abs(i + 0.5 - center)
                dy = abs(j + 0.5 - center)
                if dx >= half_side or dy >= half_side:
                    current_indep[idx] = 1
        elif shape_type == 'air_hole_diamond':
            # Dielectric background with diamond air hole
            radius_l1 = params[0]
            for idx, (i, j) in enumerate(indep_indices):
                dx = abs(i + 0.5 - center)
                dy = abs(j + 0.5 - center)
                if dx + dy >= radius_l1:
                    current_indep[idx] = 1
        elif shape_type == 'checkerboard':
            # Fine scale pattern
            period = params[0]
            for idx, (i, j) in enumerate(indep_indices):
                if (i // period + j // period) % 2 == 0:
                    current_indep[idx] = 1
        elif shape_type == 'radial_rings':
            # Alternating rings
            n_rings = params[0]
            r_max = params[1]
            for idx, (i, j) in enumerate(indep_indices):
                dx = i + 0.5 - center
                dy = j + 0.5 - center
                r = math.sqrt(dx*dx + dy*dy)
                if r < r_max:
                    ring_idx = int(r / (r_max / n_rings))
                    if ring_idx % 2 == 0:
                        current_indep[idx] = 1
        elif shape_type == 'random_symmetric':
            # Random fill with symmetry
            p_fill = params[0]
            np.random.seed(params[1])
            for idx in range(n_indep):
                if np.random.rand() < p_fill:
                    current_indep[idx] = 1
        else:
            # Default: circle
            radius = params[0] if params else 5.0
            for idx, (i, j) in enumerate(indep_indices):
                dx = i + 0.5 - center
                dy = j + 0.5 - center
                r = math.sqrt(dx*dx + dy*dy)
                if r < radius:
                    current_indep[idx] = 1

        return expand_cell(current_indep)

    # Stage 1: Broad search over parametric shapes
    # We will test a variety of shapes and parameters
    candidates = []
    
    # Circle rods
    for r in [3.0, 4.0, 5.0, 6.0, 7.0]:
        candidates.append(('circle', (r,)))
        
    # Square rods
    for h in [2.5, 3.0, 3.5, 4.0, 4.5]:
        candidates.append(('square', (h,)))
        
    # Diamond rods
    for r in [4.0, 5.0, 6.0, 7.0]:
        candidates.append(('diamond', (r,)))
        
    # Air holes in dielectric
    for r in [3.0, 4.0, 5.0, 6.0]:
        candidates.append(('air_hole_circle', (r,)))
        
    for h in [2.5, 3.0, 3.5, 4.0]:
        candidates.append(('air_hole_square', (h,)))
        
    for r in [4.0, 5.0, 6.0]:
        candidates.append(('air_hole_diamond', (r,)))
        
    # Rings
    for r_in, r_out in [(2.0, 4.0), (3.0, 5.0), (4.0, 6.0), (2.0, 5.0)]:
        candidates.append(('ring', (r_in, r_out)))
        
    # Crosses
    for w in [1.5, 2.0, 2.5, 3.0]:
        candidates.append(('cross', (w,)))
        
    # Plus signs
    for arm, w in [(8.0, 1.5), (10.0, 2.0), (12.0, 2.5), (6.0, 2.0)]:
        candidates.append(('plus', (arm, w)))
        
    # Checkerboards
    for p in [2, 3, 4]:
        candidates.append(('checkerboard', (p,)))
        
    # Radial rings
    for n_r, r_max in [(3, 8.0), (4, 10.0), (2, 6.0)]:
        candidates.append(('radial_rings', (n_r, r_max)))
        
    # Random symmetric (a few samples)
    np.random.seed(0)
    for p_fill in [0.3, 0.4, 0.5, 0.6]:
        for seed in [1, 2, 3]:
            candidates.append(('random_symmetric', (p_fill, seed)))

    # Evaluate all candidates with moderate fidelity
    best_cells = []
    best_scores = []
    
    for shape_type, params in candidates:
        if time.time() > deadline - 30:
            break
        cell = generate_candidate(shape_type, params)
        score = compute_score(cell, n_pw=17, n_kpts=14)
        best_cells.append(cell)
        best_scores.append(score)
        
    # Sort by score
    sorted_indices = np.argsort(best_scores)[::-1]
    
    # Stage 2: Refine the top 5 candidates
    top_cells = []
    for idx in sorted_indices[:5]:
        top_cells.append(best_cells[idx])
        
    final_best_cell = top_cells[0] if top_cells else best_cells[0]
    final_best_score = best_scores[sorted_indices[0]] if top_cells else best_scores[0]
    
    # Local refinement on top cells
    for cell in top_cells:
        if time.time() > deadline - 10:
            break
            
        current_indep = []
        for (i, j) in indep_indices:
            current_indep.append(int(cell[i, j]))
            
        current_score = compute_score(cell, n_pw=17, n_kpts=14)
        
        # Perform a few local flips
        n_flips = 20
        for _ in range(n_flips):
            if time.time() > deadline - 5:
                break
                
            idx = np.random.randint(n_indep)
            old_val = current_indep[idx]
            current_indep[idx] = 1 - old_val
            
            new_cell = expand_cell(current_indep)
            # Use slightly higher fidelity for the refinement step?
            # Let's stick to moderate to ensure we finish in time.
            new_score = compute_score(new_cell, n_pw=17, n_kpts=14)
            
            if new_score > current_score:
                current_score = new_score
                cell = new_cell
            else:
                current_indep[idx] = old_val
                
        if current_score > final_best_score:
            final_best_score = current_score
            final_best_cell = cell
            
    return final_best_cell.tolist()
