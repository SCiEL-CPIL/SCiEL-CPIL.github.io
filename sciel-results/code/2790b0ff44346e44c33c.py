import numpy as np
import time
import math
import random

def design(problem):
    start_time = time.time()
    deadline = start_time + 210  # Safety margin before 230s wall-clock limit
    
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    # Precompute Fourier coefficient matrix for the coarse solver
    # Using n_g = 4 (9x9 = 81 G-vectors) for speed during search
    n_g_coarse = 4
    n_g_exact = 8
    
    def compute_fourier_matrix_coarse(n_g):
        """Precompute the matrix that maps pixel values to Fourier coefficients."""
        # term[i, n] = integral contribution of pixel i to Fourier mode n
        term = np.zeros((size, 2*n_g+1), dtype=complex)
        for n in range(-n_g, n_g+1):
            idx = n + n_g
            G = 2 * np.pi * n
            if G == 0:
                term[:, idx] = 1.0 / size
            else:
                sinc_factor = (2.0 / G) * np.sin(G / (2.0 * size))
                base_exp = np.exp(-1j * G * 0.5 / size)
                i_vals = np.arange(size)
                exp_terms = np.exp(i_vals * (-1j * G / size)) * base_exp
                term[:, idx] = sinc_factor * exp_terms
        return term
    
    def compute_fourier_matrix_exact(n_g):
        """Precompute the matrix that maps pixel values to Fourier coefficients."""
        term = np.zeros((size, 2*n_g+1), dtype=complex)
        for n in range(-n_g, n_g+1):
            idx = n + n_g
            G = 2 * np.pi * n
            if G == 0:
                term[:, idx] = 1.0 / size
            else:
                sinc_factor = (2.0 / G) * np.sin(G / (2.0 * size))
                base_exp = np.exp(-1j * G * 0.5 / size)
                i_vals = np.arange(size)
                exp_terms = np.exp(i_vals * (-1j * G / size)) * base_exp
                term[:, idx] = sinc_factor * exp_terms
        return term
    
    term_coarse = compute_fourier_matrix_coarse(n_g_coarse)
    term_exact = compute_fourier_matrix_exact(n_g_exact)
    
    def get_eps_coeffs_coarse(mask_flat):
        """Get epsilon Fourier coefficients from flat pixel mask using precomputed term."""
        # mask_flat is 1D array of length size*size
        M = mask_flat.reshape(size, size)
        # f_coeffs[nx, ny] = sum_{i,j} M[i,j] * term[i, nx] * term[j, ny]
        f_coeffs = term_coarse.T @ (M @ term_coarse)
        delta_eps = eps_high - eps_low
        eps_coeffs = f_coeffs * delta_eps
        eps_coeffs[n_g_coarse, n_g_coarse] += eps_low
        return eps_coeffs
    
    def get_eps_coeffs_exact(mask_flat):
        """Get epsilon Fourier coefficients from flat pixel mask using precomputed term."""
        M = mask_flat.reshape(size, size)
        f_coeffs = term_exact.T @ (M @ term_exact)
        delta_eps = eps_high - eps_low
        eps_coeffs = f_coeffs * delta_eps
        eps_coeffs[n_g_exact, n_g_exact] += eps_low
        return eps_coeffs
    
    # Precompute G-vectors and E-index lookup for coarse solver
    G_vecs_coarse = []
    for nx in range(-n_g_coarse, n_g_coarse+1):
        for ny in range(-n_g_coarse, n_g_coarse+1):
            G_vecs_coarse.append((2 * np.pi * nx, 2 * np.pi * ny, nx, ny))
    N_G_coarse = len(G_vecs_coarse)
    
    E_indices_coarse = np.zeros((N_G_coarse, N_G_coarse), dtype=int)
    for idx_g, (Gx, Gy, nx, ny) in enumerate(G_vecs_coarse):
        for idx_gp, (Gxp, Gyp, nxp, nyp) in enumerate(G_vecs_coarse):
            dx = nx - nxp
            dy = ny - nyp
            if -n_g_coarse <= dx <= n_g_coarse and -n_g_coarse <= dy <= n_g_coarse:
                E_indices_coarse[idx_g, idx_gp] = (dx + n_g_coarse) * (2*n_g_coarse+1) + (dy + n_g_coarse)
            else:
                E_indices_coarse[idx_g, idx_gp] = -1
    
    valid_mask_coarse = E_indices_coarse != -1
    
    # Precompute G-vectors and E-index lookup for exact solver
    G_vecs_exact = []
    for nx in range(-n_g_exact, n_g_exact+1):
        for ny in range(-n_g_exact, n_g_exact+1):
            G_vecs_exact.append((2 * np.pi * nx, 2 * np.pi * ny, nx, ny))
    N_G_exact = len(G_vecs_exact)
    
    E_indices_exact = np.zeros((N_G_exact, N_G_exact), dtype=int)
    for idx_g, (Gx, Gy, nx, ny) in enumerate(G_vecs_exact):
        for idx_gp, (Gxp, Gyp, nxp, nyp) in enumerate(G_vecs_exact):
            dx = nx - nxp
            dy = ny - nyp
            if -n_g_exact <= dx <= n_g_exact and -n_g_exact <= dy <= n_g_exact:
                E_indices_exact[idx_g, idx_gp] = (dx + n_g_exact) * (2*n_g_exact+1) + (dy + n_g_exact)
            else:
                E_indices_exact[idx_g, idx_gp] = -1
    
    valid_mask_exact = E_indices_exact != -1
    
    # K-points for coarse solver (fewer for speed)
    k_points_coarse = []
    for i in range(6):
        for j in range(6):
            kx = i / 5.0 * np.pi
            ky = j / 5.0 * np.pi
            if kx <= ky:
                k_points_coarse.append((kx, ky))
    
    # K-points for exact solver (30 points)
    k_points_exact = []
    for i in range(10):
        k_points_exact.append((0, i / 9.0 * np.pi))
    for i in range(10):
        k_points_exact.append((i / 9.0 * np.pi, np.pi))
    for i in range(1, 5):
        k_points_exact.append((i / 4.0 * np.pi, i / 4.0 * np.pi))
    k_points_exact.append((np.pi/4, np.pi/2))
    k_points_exact.append((np.pi/3, 2*np.pi/3))
    k_points_exact.append((np.pi/6, np.pi/3))
    k_points_exact.append((np.pi/5, 3*np.pi/5))
    k_points_exact.append((np.pi/2.5, 7*np.pi/10))
    
    def evaluate_gap_coarse(mask_flat):
        """Fast gap evaluation using coarse solver."""
        eps_coeffs = get_eps_coeffs_coarse(mask_flat)
        eps_flat = eps_coeffs.flatten()
        
        all_omegas = []
        
        for kx, ky in k_points_coarse:
            D = np.zeros(N_G_coarse, dtype=float)
            for idx_g, (Gx, Gy, nx, ny) in enumerate(G_vecs_coarse):
                kx_g = kx + Gx
                ky_g = ky + Gy
                D[idx_g] = kx_g * kx_g + ky_g * ky_g
            
            E = np.zeros((N_G_coarse, N_G_coarse), dtype=complex)
            E[valid_mask_coarse] = eps_flat[E_indices_coarse[valid_mask_coarse]]
            
            try:
                sqrt_D = np.sqrt(D)
                inv_sqrt_D = 1.0 / sqrt_D
                H = (sqrt_D[:, None] * E) * inv_sqrt_D[None, :]
                H = (H + H.conj().T) / 2.0
                
                mu_vals = np.linalg.eigvalsh(H)
                mu_real = np.real(mu_vals)
                
                mu_valid = mu_real[mu_real > 1e-10]
                if len(mu_valid) < 5:
                    continue
                
                mu_sorted = np.sort(mu_valid)
                omegas = 1.0 / np.sqrt(mu_sorted)
                omegas = np.sort(omegas)
                
                if len(omegas) >= 5:
                    all_omegas.append(omegas)
            except Exception:
                continue
        
        if not all_omegas:
            return 0.0
            
        max_band4 = max(o[3] for o in all_omegas)
        min_band5 = min(o[4] for o in all_omegas)
        
        gap = min_band5 - max_band4
        midgap = (min_band5 + max_band4) / 2.0
        
        if midgap == 0:
            return 0.0
            
        ratio = gap / midgap
        return ratio
    
    def evaluate_gap_exact(cell):
        """Exact gap evaluation with 289 G-vectors and 30 k-points."""
        mask_flat = np.array(cell, dtype=float).flatten()
        eps_coeffs = get_eps_coeffs_exact(mask_flat)
        eps_flat = eps_coeffs.flatten()
        
        all_omegas = []
        
        for kx, ky in k_points_exact:
            D = np.zeros(N_G_exact, dtype=float)
            for idx_g, (Gx, Gy, nx, ny) in enumerate(G_vecs_exact):
                kx_g = kx + Gx
                ky_g = ky + Gy
                D[idx_g] = kx_g * kx_g + ky_g * ky_g
            
            E = np.zeros((N_G_exact, N_G_exact), dtype=complex)
            E[valid_mask_exact] = eps_flat[E_indices_exact[valid_mask_exact]]
            
            try:
                sqrt_D = np.sqrt(D)
                inv_sqrt_D = 1.0 / sqrt_D
                H = (sqrt_D[:, None] * E) * inv_sqrt_D[None, :]
                H = (H + H.conj().T) / 2.0
                
                mu_vals = np.linalg.eigvalsh(H)
                mu_real = np.real(mu_vals)
                
                mu_valid = mu_real[mu_real > 1e-10]
                if len(mu_valid) < 5:
                    continue
                
                mu_sorted = np.sort(mu_valid)
                omegas = 1.0 / np.sqrt(mu_sorted)
                omegas = np.sort(omegas)
                
                if len(omegas) >= 5:
                    all_omegas.append(omegas)
            except Exception:
                continue
        
        if not all_omegas:
            return 0.0
            
        max_band4 = max(o[3] for o in all_omegas)
        min_band5 = min(o[4] for o in all_omegas)
        
        gap = min_band5 - max_band4
        midgap = (min_band5 + max_band4) / 2.0
        
        if midgap == 0:
            return 0.0
            
        ratio = gap / midgap
        return ratio
    
    def make_cell_from_mask(mask):
        """Convert numpy mask to nested list."""
        return [[int(mask[i][j]) for j in range(size)] for i in range(size)]
    
    def symmetrize_mask(mask):
        """Ensure 4-fold symmetry about cell center."""
        sym_mask = np.zeros_like(mask)
        for i in range(size):
            for j in range(size):
                # The four symmetric positions
                i1, j1 = i, j
                i2, j2 = size - 1 - i, j
                i3, j3 = i, size - 1 - j
                i4, j4 = size - 1 - i, size - 1 - j
                
                # Take the value from the first position and set all four
                val = mask[i1, j1]
                sym_mask[i1, j1] = val
                sym_mask[i2, j2] = val
                sym_mask[i3, j3] = val
                sym_mask[i4, j4] = val
        return sym_mask
    
    # Phase 1: Generate diverse initial candidates
    candidates = []
    
    # 1. Ring + cross + corner (parametric)
    def make_parametric_cell(ring_w, cross_w, corner_w):
        cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                ax, ay = abs(x), abs(y)
                
                is_dielectric = False
                
                if ring_w > 0.01:
                    r_max = max(ax, ay)
                    if ring_w / 2 < r_max < ring_w:
                        is_dielectric = True
                
                if cross_w > 0.01:
                    if ax < cross_w / 2 or ay < cross_w / 2:
                        is_dielectric = True
                
                if corner_w > 0.01:
                    if ax > 0.5 - corner_w and ay > 0.5 - corner_w:
                        is_dielectric = True
                
                cell[i, j] = 1 if is_dielectric else 0
        return symmetrize_mask(cell)
    
    # 2. Quadrant rods
    def make_quadrant_rods(rod_r, spacing):
        cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                is_dielectric = False
                if rod_r > 0.01:
                    for sx in [1, -1]:
                        for sy in [1, -1]:
                            dx = x - sx * spacing
                            dy = y - sy * spacing
                            if dx * dx + dy * dy < rod_r * rod_r:
                                is_dielectric = True
                cell[i, j] = 1 if is_dielectric else 0
        return symmetrize_mask(cell)
    
    # 3. Circle
    def make_circle(radius):
        cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                if x * x + y * y < radius * radius:
                    cell[i, j] = 1
        return symmetrize_mask(cell)
    
    # 4. Square
    def make_square(half_w):
        cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                if abs(x) < half_w and abs(y) < half_w:
                    cell[i, j] = 1
        return symmetrize_mask(cell)
    
    # 5. Cross
    def make_cross(width):
        cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                ax, ay = abs(x), abs(y)
                if ax < width / 2 or ay < width / 2:
                    cell[i, j] = 1
        return symmetrize_mask(cell)
    
    # 6. Ring
    def make_ring(inner_r, outer_r):
        cell = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                x = (i + 0.5) / size - 0.5
                y = (j + 0.5) / size - 0.5
                r = math.sqrt(x * x + y * y)
                if inner_r < r < outer_r:
                    cell[i, j] = 1
        return symmetrize_mask(cell)
    
    # Generate diverse candidates
    param_configs = [
        (0.1, 0.0, 0.0), (0.15, 0.0, 0.0), (0.2, 0.0, 0.0), (0.25, 0.0, 0.0), (0.3, 0.0, 0.0),
        (0.0, 0.1, 0.0), (0.0, 0.15, 0.0), (0.0, 0.2, 0.0),
        (0.0, 0.0, 0.1), (0.0, 0.0, 0.15), (0.0, 0.0, 0.2),
        (0.15, 0.1, 0.0), (0.2, 0.1, 0.0), (0.25, 0.1, 0.0),
        (0.15, 0.0, 0.1), (0.2, 0.0, 0.1), (0.25, 0.0, 0.1),
        (0.15, 0.1, 0.1), (0.2, 0.1, 0.1), (0.25, 0.1, 0.1),
        (0.1, 0.15, 0.05), (0.15, 0.15, 0.1), (0.2, 0.15, 0.15),
    ]
    for rw, cw, ccw in param_configs:
        if time.time() > deadline - 30:
            break
        mask = make_parametric_cell(rw, cw, ccw)
        candidates.append(mask)
    
    rod_configs = [
        (0.08, 0.15), (0.1, 0.15), (0.12, 0.15), (0.15, 0.15),
        (0.08, 0.2), (0.1, 0.2), (0.12, 0.2), (0.15, 0.2),
        (0.08, 0.25), (0.1, 0.25), (0.12, 0.25), (0.15, 0.25),
    ]
    for rr, sp in rod_configs:
        if time.time() > deadline - 30:
            break
        mask = make_quadrant_rods(rr, sp)
        candidates.append(mask)
    
    for r in [0.1, 0.15, 0.2, 0.25, 0.3]:
        if time.time() > deadline - 30:
            break
        mask = make_circle(r)
        candidates.append(mask)
    
    for hw in [0.1, 0.15, 0.2, 0.25]:
        if time.time() > deadline - 30:
            break
        mask = make_square(hw)
        candidates.append(mask)
    
    for w in [0.1, 0.15, 0.2, 0.25]:
        if time.time() > deadline - 30:
            break
        mask = make_cross(w)
        candidates.append(mask)
    
    ring_configs = [
        (0.1, 0.2), (0.1, 0.25), (0.1, 0.3),
        (0.15, 0.25), (0.15, 0.3), (0.15, 0.35),
        (0.2, 0.3), (0.2, 0.35), (0.2, 0.4),
    ]
    for ir, or_ in ring_configs:
        if time.time() > deadline - 30:
            break
        mask = make_ring(ir, or_)
        candidates.append(mask)
    
    # Add some random masks
    for _ in range(10):
        if time.time() > deadline - 30:
            break
        mask = np.random.randint(0, 2, (size, size))
        mask = symmetrize_mask(mask)
        candidates.append(mask)
    
    # Phase 2: Evaluate all candidates with coarse solver
    scored_candidates = []
    for mask in candidates:
        if time.time() > deadline - 20:
            break
        mask_flat = mask.flatten()
        score = evaluate_gap_coarse(mask_flat)
        scored_candidates.append((score, mask))
    
    # Sort by coarse score
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Phase 3: Local search (hill climbing) on top candidates
    best_score = -1.0
    best_mask = None
    
    if scored_candidates:
        # Start with the best coarse candidate
        best_mask = scored_candidates[0][1].copy()
        best_score = scored_candidates[0][0]
    
    # Hill climbing: try flipping individual pixels (with symmetry)
    if best_mask is not None and time.time() < deadline - 10:
        # Precompute symmetric pixel groups
        # Each group is a set of 4 symmetric positions (or fewer if on symmetry axes)
        pixel_groups = []
        visited = set()
        for i in range(size):
            for j in range(size):
                if (i, j) in visited:
                    continue
                # Get all symmetric positions
                group = set()
                group.add((i, j))
                group.add((size - 1 - i, j))
                group.add((i, size - 1 - j))
                group.add((size - 1 - i, size - 1 - j))
                pixel_groups.append(sorted(group))
                for pos in group:
                    visited.add(pos)
        
        # Local search
        improved = True
        max_iterations = 500
        iteration = 0
        while improved and iteration < max_iterations and time.time() < deadline - 10:
            improved = False
            iteration += 1
            
            # Try flipping each pixel group
            for group in pixel_groups:
                if time.time() > deadline - 10:
                    break
                
                # Get current value
                i0, j0 = group[0]
                current_val = best_mask[i0, j0]
                new_val = 1 - current_val
                
                # Apply flip
                for (i, j) in group:
                    best_mask[i, j] = new_val
                
                # Evaluate
                mask_flat = best_mask.flatten()
                new_score = evaluate_gap_coarse(mask_flat)
                
                if new_score > best_score:
                    best_score = new_score
                    improved = True
                else:
                    # Revert
                    for (i, j) in group:
                        best_mask[i, j] = current_val
        
        # Try some random restarts
        for _ in range(3):
            if time.time() > deadline - 10:
                break
            
            # Random perturbation of best mask
            restart_mask = best_mask.copy()
            for _ in range(5):
                group = random.choice(pixel_groups)
                i0, j0 = group[0]
                val = 1 - restart_mask[i0, j0]
                for (i, j) in group:
                    restart_mask[i, j] = val
            
            # Local search from restart
            restart_score = evaluate_gap_coarse(restart_mask.flatten())
            if restart_score > best_score:
                best_mask = restart_mask
                best_score = restart_score
                # Continue local search
                improved = True
                for _ in range(100):
                    if time.time() > deadline - 10:
                        break
                    group = random.choice(pixel_groups)
                    i0, j0 = group[0]
                    current_val = best_mask[i0, j0]
                    new_val = 1 - current_val
                    for (i, j) in group:
                        best_mask[i, j] = new_val
                    new_score = evaluate_gap_coarse(best_mask.flatten())
                    if new_score > best_score:
                        best_score = new_score
                    else:
                        for (i, j) in group:
                            best_mask[i, j] = current_val
    
    # Phase 4: Evaluate top candidates with exact solver
    if time.time() < deadline - 5:
        # Collect top candidates
        top_candidates = []
        if best_mask is not None:
            top_candidates.append(best_mask)
        
        # Add top 5 from coarse search
        for score, mask in scored_candidates[:5]:
            if mask not in top_candidates:
                top_candidates.append(mask)
        
        best_exact_score = -1.0
        best_exact_cell = None
        
        for mask in top_candidates:
            if time.time() > deadline - 3:
                break
            cell = make_cell_from_mask(mask)
            exact_score = evaluate_gap_exact(cell)
            if exact_score > best_exact_score:
                best_exact_score = exact_score
                best_exact_cell = cell
        
        if best_exact_cell is not None:
            return best_exact_cell
    
    # Fallback
    if best_mask is not None:
        return make_cell_from_mask(best_mask)
    
    # Ultimate fallback: simple circle
    cell = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if x * x + y * y < 0.04:  # r=0.2
                cell[i, j] = 1
    cell = symmetrize_mask(cell)
    return make_cell_from_mask(cell)

