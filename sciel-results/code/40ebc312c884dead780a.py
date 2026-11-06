import numpy as np
import time
import math

def design(problem):
    t_start = time.time()
    deadline = t_start + 105.0  # Safety margin before 120s limit
    
    size = problem['pixels']  # 32
    eps_low = problem['eps_low']  # 1
    eps_high = problem['eps_high']  # 12
    lower_band = problem['lower_band']  # 4
    
    # Define independent pixels under D4 symmetry
    # Symmetries: 180° rotation, reflection about x-axis, y-axis, and diagonals
    # For a 32x32 cell (indices 0..31), center is at (15.5, 15.5)
    # Symmetry operations on indices (i, j):
    # 1. Identity: (i, j)
    # 2. 180° rot: (31-i, 31-j)
    # 3. Refl x-axis (horizontal mirror): (31-i, j)
    # 4. Refl y-axis (vertical mirror): (i, 31-j)
    # 5. Refl main diagonal: (j, i)
    # 6. Refl anti-diagonal: (31-j, 31-i)
    # 7. 90° CW rot: (j, 31-i)
    # 8. 90° CCW rot: (31-j, i)
    
    # The fundamental domain for D4 symmetry on a grid is the triangular region
    # in the upper-left corner where i <= j and i <= 31-j, i.e., i <= 15 and i <= j <= 31-i
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
    
    def compute_score_fast(cell, n_pw=15, n_kpts=12):
        """Compute approximate band gap score using plane wave method."""
        inv_eps = np.where(cell == 1, 1.0/eps_high, 1.0/eps_low)
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
        for t in np.linspace(0, 0.5, n_seg - 1):
            k_points.append((0.5, t))
        # M to Gamma: (0.5 - t, 0.5 - t) for t in [0, 0.5]
        for t in np.linspace(0, 0.5, n_seg - 1):
            k_points.append((0.5 - t, 0.5 - t))
        
        max_omega_4 = 0.0
        min_omega_5 = np.inf
        
        for k in k_points:
            k_vec = np.array(k)
            matrix = np.zeros((n_pw_actual, n_pw_actual), dtype=np.complex128)
            
            for alpha in range(n_pw_actual):
                ga = g_vecs[alpha]
                for beta in range(n_pw_actual):
                    gb = g_vecs[beta]
                    g_diff = ga - gb
                    idx_x = int(g_diff[0]) % size
                    idx_y = int(g_diff[1]) % size
                    coeff = fft_inv[idx_x, idx_y]
                    
                    k_plus_g = k_vec + gb
                    val = coeff * (k_plus_g[0]**2 + k_plus_g[1]**2)
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
    
    def generate_initial_cell(strategy):
        """Generate initial cell based on strategy."""
        current_indep = [0] * n_indep
        center = 15.5
        
        if strategy == 0:
            # Dielectric rods (high eps) in air background - circle
            for idx, (i, j) in enumerate(indep_indices):
                dx = i + 0.5 - center
                dy = j + 0.5 - center
                r = math.sqrt(dx*dx + dy*dy)
                if r < 5.0:
                    current_indep[idx] = 1
        elif strategy == 1:
            # Air holes in dielectric background
            for idx, (i, j) in enumerate(indep_indices):
                dx = i + 0.5 - center
                dy = j + 0.5 - center
                r = math.sqrt(dx*dx + dy*dy)
                if r >= 4.0:
                    current_indep[idx] = 1
        elif strategy == 2:
            # Square dielectric rods
            for idx, (i, j) in enumerate(indep_indices):
                dx = abs(i + 0.5 - center)
                dy = abs(j + 0.5 - center)
                if dx < 4.0 and dy < 4.0:
                    current_indep[idx] = 1
        elif strategy == 3:
            # Air square holes in dielectric
            for idx, (i, j) in enumerate(indep_indices):
                dx = abs(i + 0.5 - center)
                dy = abs(j + 0.5 - center)
                if dx >= 3.5 or dy >= 3.5:
                    current_indep[idx] = 1
        elif strategy == 4:
            # Diamond shape (L1 norm)
            for idx, (i, j) in enumerate(indep_indices):
                dx = abs(i + 0.5 - center)
                dy = abs(j + 0.5 - center)
                if dx + dy < 6.0:
                    current_indep[idx] = 1
        elif strategy == 5:
            # Air diamond holes
            for idx, (i, j) in enumerate(indep_indices):
                dx = abs(i + 0.5 - center)
                dy = abs(j + 0.5 - center)
                if dx + dy >= 4.5:
                    current_indep[idx] = 1
        elif strategy == 6:
            # Checkerboard-like with periodicity
            for idx, (i, j) in enumerate(indep_indices):
                if (i // 4 + j // 4) % 2 == 0:
                    current_indep[idx] = 1
        elif strategy == 7:
            # Radial pattern with alternating rings
            for idx, (i, j) in enumerate(indep_indices):
                dx = i + 0.5 - center
                dy = j + 0.5 - center
                r = math.sqrt(dx*dx + dy*dy)
                ring = int(r // 2.0)
                if ring % 2 == 0:
                    current_indep[idx] = 1
        
        return expand_cell(current_indep)
    
    # Evaluate multiple initial strategies and pick the best
    best_cell = None
    best_score = -np.inf
    
    for strat in range(8):
        cell = generate_initial_cell(strat)
        score = compute_score_fast(cell)
        if score > best_score:
            best_score = score
            best_cell = cell
    
    # Hill climbing optimization
    current_cell = best_cell.copy()
    current_score = best_score
    
    np.random.seed(42)
    
    # Get current independent values from best_cell
    current_indep = []
    for (i, j) in indep_indices:
        current_indep.append(int(current_cell[i, j]))
    
    step_size = 1  # Start with single pixel flips
    
    while time.time() < deadline:
        # Random pixel flip
        idx = np.random.randint(n_indep)
        old_val = current_indep[idx]
        current_indep[idx] = 1 - old_val
        
        new_cell = expand_cell(current_indep)
        new_score = compute_score_fast(new_cell)
        
        if new_score > current_score:
            current_score = new_score
            current_cell = new_cell
            best_cell = new_cell
            best_score = new_score
        else:
            # Revert
            current_indep[idx] = old_val
        
        # Occasionally try multi-pixel moves
        if np.random.rand() < 0.1 and time.time() < deadline - 30:
            # Try flipping 2-3 random pixels together
            n_flips = np.random.randint(2, 4)
            flip_indices = np.random.choice(n_indep, size=n_flips, replace=False)
            for fi in flip_indices:
                current_indep[fi] = 1 - current_indep[fi]
            
            new_cell = expand_cell(current_indep)
            new_score = compute_score_fast(new_cell)
            
            if new_score > current_score:
                current_score = new_score
                current_cell = new_cell
                best_cell = new_cell
                best_score = new_score
            else:
                # Revert all
                for fi in flip_indices:
                    current_indep[fi] = 1 - current_indep[fi]
    
    # Fine-tune: try local perturbations around the best solution
    if time.time() < deadline - 20:
        # Get current independent values from best_cell
        current_indep = []
        for (i, j) in indep_indices:
            current_indep.append(int(best_cell[i, j]))
        
        fine_best_cell = best_cell.copy()
        fine_best_score = best_score
        
        for _ in range(50):  # Limited fine-tuning iterations
            if time.time() >= deadline - 10:
                break
            idx = np.random.randint(n_indep)
            old_val = current_indep[idx]
            current_indep[idx] = 1 - old_val
            
            new_cell = expand_cell(current_indep)
            new_score = compute_score_fast(new_cell, n_pw=17, n_kpts=15)
            
            if new_score > fine_best_score:
                fine_best_score = new_score
                fine_best_cell = new_cell
            else:
                current_indep[idx] = old_val
        
        if fine_best_score > best_score:
            best_cell = fine_best_cell
            best_score = fine_best_score
    
    return best_cell.tolist()
