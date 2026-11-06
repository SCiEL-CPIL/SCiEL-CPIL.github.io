"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

# EVOLVE-BLOCK-START
def design(problem):
    """
    Design a 32x32 unit cell to maximize the TM band gap between bands 4 and 5.
    
    Strategy:
    1. Fast internal PWE solver with precomputed G-vectors.
    2. Seed with diverse parametric shapes (rods, rings, crosses).
    3. D4-symmetric pixel-level genetic algorithm over the fundamental domain.
    4. Maintain D4 symmetry and stay within time budget.
    """
    size = problem["pixels"]
    eps_high = problem.get("eps_high", 12)
    lower_band = problem.get("lower_band", 4)
    
    start_time = time.time()
    deadline = start_time + 218.0
    
    def check_time():
        return time.time() > deadline - 5.0
    
    # Precompute G-vectors and k-points once
    n_pw = 13
    half = n_pw // 2
    m_vals = np.arange(-half, half + 1)
    g_indices = [(m, n) for m in m_vals for n in m_vals]
    n_total = len(g_indices)
    g_m = np.array([g[0] for g in g_indices])
    g_n = np.array([g[1] for g in g_indices])
    
    # Precompute k-points in irreducible BZ (15 points)
    k_points = []
    n_per_seg = 5
    for t in np.linspace(0, 0.5, n_per_seg):
        k_points.append((t, 0.0))
    for t in np.linspace(0, 0.5, n_per_seg):
        k_points.append((0.5, t))
    for t in np.linspace(0.5, 0, n_per_seg):
        k_points.append((t, t))
    # Remove duplicates
    k_points_unique = []
    for kp in k_points:
        if not any(np.allclose(kp, kp2, atol=1e-10) for kp2 in k_points_unique):
            k_points_unique.append(kp)
    
    # Precompute |G+k|^2 for each k-point
    gk_sq_cache = []
    for kx, ky in k_points_unique:
        gk_m = g_m + kx
        gk_n = g_n + ky
        gk_sq_cache.append(gk_m**2 + gk_n**2)
    
    # Fast PWE solver using precomputed data
    def compute_tm_bands(cell, n_kpts=None):
        """Compute gap-midgap ratio. cell is 2D array of 0/1."""
        cell = np.asarray(cell, dtype=float)
        eps = 1.0 + (eps_high - 1.0) * cell
        inv_eps = 1.0 / eps
        
        # FFT of inv_eps to get Fourier coefficients
        fft_coeffs = np.fft.fft2(inv_eps) / (size * size)
        
        # Build S matrix: S[i,j] = inv_eps(G_i - G_j)
        dm = g_m[:, None] - g_m[None, :]
        dn = g_n[:, None] - g_n[None, :]
        idx_m = dm % size
        idx_n = dn % size
        S = fft_coeffs[idx_m, idx_n]
        
        # Track max omega_4 and min omega_5
        max_omega4 = -np.inf
        min_omega5 = np.inf
        
        kpts_to_use = k_points_unique if n_kpts is None else k_points_unique[:n_kpts]
        
        for k_idx, (kx, ky) in enumerate(kpts_to_use):
            if check_time():
                break
            
            gk_sq = gk_sq_cache[k_idx]
            A = S * gk_sq[None, :]
            
            try:
                from scipy.linalg import eigh
                # S is symmetric positive definite (it's a Fourier coefficient matrix)
                # Use eigh for symmetric generalized eigenvalue problem
                # A c = lambda B c where B = S
                # This is equivalent to S^{-1} A c = lambda c
                # But S may not be easily invertible; use scipy's eigh for generalized
                from scipy.linalg import eigh as eigh_gen
                vals, _ = eigh_gen(A, S)
                vals = np.real(vals)
                vals = vals[np.isfinite(vals)]
                vals_sorted = np.sort(vals)
                
                if len(vals_sorted) >= 5:
                    omega4 = vals_sorted[3]
                    omega5 = vals_sorted[4]
                    if omega5 > 0 and omega4 > 0:
                        max_omega4 = max(max_omega4, omega4)
                        min_omega5 = min(min_omega5, omega5)
            except Exception:
                pass
        
        if max_omega4 > -np.inf and min_omega5 > 0:
            gap = min_omega5 - max_omega4
            midgap = (min_omega5 + max_omega4) / 2.0
            if midgap > 0:
                return 2.0 * gap / midgap
            else:
                return -np.inf
        else:
            return -np.inf
    
    # D4-symmetric cell construction from fundamental domain
    # The D4 fundamental domain for a 32x32 grid:
    # By symmetry about center and axes, we only need to define
    # the upper-right 16x16 quadrant, but with additional constraints
    # from the diagonal symmetry.
    # Actually, for full D4: cell[i,j] = cell[j,i] = cell[31-i,31-j] = cell[31-j,31-i]
    # The fundamental domain is the set of (i,j) where i <= j and i + j <= 31
    # i.e., the upper triangle of the upper-right quadrant.
    
    def get_fundamental_domain_indices():
        """Get indices (i,j) in the D4 fundamental domain."""
        indices = []
        for i in range(size):
            for j in range(i, size):
                if i + j <= size - 1:
                    indices.append((i, j))
        return indices
    
    def expand_fundamental(domain_vals, fd_indices):
        """Expand a fundamental domain assignment to full 32x32 cell."""
        cell = np.zeros((size, size), dtype=int)
        for (i, j), val in zip(fd_indices, domain_vals):
            # Apply all D4 symmetries
            cell[i, j] = val
            cell[j, i] = val
            cell[size-1-i, size-1-j] = val
            cell[size-1-j, size-1-i] = val
        # Verify and enforce exact D4 symmetry
        cell = ((cell + cell.T) // 2).astype(int)
        cell = ((cell + cell[::-1, ::-1]) // 2).astype(int)
        cell = ((cell + cell[:, ::-1]) // 2).astype(int)  # vertical mirror
        cell = ((cell + cell[::-1, :]) // 2).astype(int)  # horizontal mirror
        return cell
    
    def cell_to_fundamental(cell, fd_indices):
        """Extract fundamental domain from a full cell."""
        cell = np.asarray(cell)
        return np.array([cell[i, j] for i, j in fd_indices], dtype=int)
    
    fd_indices = get_fundamental_domain_indices()
    n_fd = len(fd_indices)
    
    # Parametric seed generators
    def make_four_quadrant_cell(rod_radius, exponent=2.0):
        cell = np.zeros((size, size), dtype=int)
        coords = (np.arange(size) + 0.5) / size - 0.5
        centers = [(-0.25, -0.25), (0.25, -0.25), (-0.25, 0.25), (0.25, 0.25)]
        x_grid, y_grid = np.meshgrid(coords, coords, indexing='ij')
        for (cx, cy) in centers:
            dx = x_grid - cx
            dy = y_grid - cy
            if rod_radius > 0:
                val = np.abs(dx/rod_radius)**exponent + np.abs(dy/rod_radius)**exponent
                cell[val <= 1.0 + 1e-9] = 1
        cell = ((cell + cell.T) // 2).astype(int)
        cell = ((cell + cell[::-1, ::-1]) // 2).astype(int)
        return cell
    
    def make_ring_cell(inner_r, outer_r, exponent=2.0):
        cell = np.zeros((size, size), dtype=int)
        coords = (np.arange(size) + 0.5) / size - 0.5
        x_grid, y_grid = np.meshgrid(coords, coords, indexing='ij')
        r_inner = np.abs(x_grid/inner_r)**exponent + np.abs(y_grid/inner_r)**exponent
        r_outer = np.abs(x_grid/outer_r)**exponent + np.abs(y_grid/outer_r)**exponent
        cell[(r_outer <= 1.0 + 1e-9) & (r_inner > 1.0 + 1e-9)] = 1
        cell = ((cell + cell.T) // 2).astype(int)
        cell = ((cell + cell[::-1, ::-1]) // 2).astype(int)
        return cell
    
    def make_cross_cell(arm_width, exponent=2.0):
        cell = np.zeros((size, size), dtype=int)
        coords = (np.arange(size) + 0.5) / size - 0.5
        x_grid, y_grid = np.meshgrid(coords, coords, indexing='ij')
        h_val = np.abs(x_grid / 0.45)**exponent + np.abs(y_grid / arm_width)**exponent
        cell[h_val <= 1.0 + 1e-9] = 1
        v_val = np.abs(x_grid / arm_width)**exponent + np.abs(y_grid / 0.45)**exponent
        cell[v_val <= 1.0 + 1e-9] = 1
        cell = ((cell + cell.T) // 2).astype(int)
        cell = ((cell + cell[::-1, ::-1]) // 2).astype(int)
        return cell
    
    def make_corner_rods_cell(rod_radius, exponent=2.0):
        cell = np.zeros((size, size), dtype=int)
        coords = (np.arange(size) + 0.5) / size - 0.5
        centers = [(-0.4, -0.4), (0.4, -0.4), (-0.4, 0.4), (0.4, 0.4)]
        x_grid, y_grid = np.meshgrid(coords, coords, indexing='ij')
        for (cx, cy) in centers:
            dx = x_grid - cx
            dy = y_grid - cy
            if rod_radius > 0:
                val = np.abs(dx/rod_radius)**exponent + np.abs(dy/rod_radius)**exponent
                cell[val <= 1.0 + 1e-9] = 1
        cell = ((cell + cell.T) // 2).astype(int)
        cell = ((cell + cell[::-1, ::-1]) // 2).astype(int)
        return cell
    
    def make_central_cell(r, n=2.0):
        cell = np.zeros((size, size), dtype=int)
        coords = (np.arange(size) + 0.5) / size - 0.5
        x_grid, y_grid = np.meshgrid(coords, coords, indexing='ij')
        val = np.abs(x_grid/r)**n + np.abs(y_grid/r)**n
        cell[val <= 1.0 + 1e-9] = 1
        cell = ((cell + cell.T) // 2).astype(int)
        cell = ((cell + cell[::-1, ::-1]) // 2).astype(int)
        return cell
    
    # Generate diverse parametric seeds
    seeds = []
    for r in [0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18]:
        seeds.append(make_four_quadrant_cell(r, 2.0))
    for r in [0.08, 0.10, 0.12]:
        for n in [3.0, 4.0, 6.0]:
            seeds.append(make_four_quadrant_cell(r, n))
    for inner_r in [0.08, 0.10, 0.12]:
        for outer_r in [0.20, 0.25, 0.30]:
            if outer_r > inner_r:
                seeds.append(make_ring_cell(inner_r, outer_r, 2.0))
    for w in [0.04, 0.06, 0.08]:
        seeds.append(make_cross_cell(w, 2.0))
    for r in [0.06, 0.08, 0.10]:
        seeds.append(make_corner_rods_cell(r, 2.0))
    seeds.append(make_central_cell(0.20, 2.0))
    
    # Score seeds
    best_score = -np.inf
    best_cell = seeds[0]
    
    for seed in seeds:
        if check_time():
            break
        score = compute_tm_bands(seed)
        if score > best_score:
            best_score = score
            best_cell = seed
    
    # Convert best to fundamental domain for GA
    best_fd = cell_to_fundamental(best_cell, fd_indices)
    
    # Genetic algorithm on the fundamental domain
    if not check_time():
        pop_size = 24
        mutation_rate = 0.05
        elite_count = 4
        
        # Initialize population: best cell + random variants
        population = [best_fd.copy()]
        scores = [best_score]
        
        # Add parametric seeds to population
        for seed in seeds[:8]:
            if check_time():
                break
            fd = cell_to_fundamental(seed, fd_indices)
            population.append(fd)
            scores.append(compute_tm_bands(expand_fundamental(fd, fd_indices)))
        
        # Add random initial individuals
        while len(population) < pop_size:
            if check_time():
                break
            # Random with bias toward moderate fill fraction
            p_fill = random.uniform(0.2, 0.5)
            fd = np.random.binomial(1, p_fill, n_fd)
            population.append(fd)
            scores.append(compute_tm_bands(expand_fundamental(fd, fd_indices)))
        
        # Sort by score
        order = np.argsort(scores)[::-1]
        population = [population[i] for i in order]
        scores = [scores[i] for i in order]
        best_score = scores[0]
        best_fd = population[0].copy()
        
        # GA loop
        generation = 0
        while not check_time() and generation < 200:
            generation += 1
            
            # Selection: tournament
            def tournament_select(pop, scores, k=3):
                idxs = random.sample(range(len(pop)), k)
                best_idx = max(idxs, key=lambda i: scores[i])
                return pop[best_idx].copy()
            
            # Crossover + mutation
            new_population = []
            new_scores = []
            
            # Elitism: keep top individuals
            for i in range(elite_count):
                new_population.append(population[i].copy())
                new_scores.append(scores[i])
            
            # Generate offspring
            while len(new_population) < pop_size:
                if check_time():
                    break
                
                parent1 = tournament_select(population, scores)
                parent2 = tournament_select(population, scores)
                
                # Uniform crossover
                mask = np.random.random(n_fd) < 0.5
                child = np.where(mask, parent1, parent2).astype(int)
                
                # Mutation: flip random bits in fundamental domain
                n_mut = max(1, int(n_fd * mutation_rate))
                mut_idx = random.sample(range(n_fd), n_mut)
                for idx in mut_idx:
                    child[idx] = 1 - child[idx]
                
                # Ensure cell is valid (D4 symmetry is automatic from fundamental domain)
                cell = expand_fundamental(child, fd_indices)
                sc = compute_tm_bands(cell)
                
                new_population.append(child)
                new_scores.append(sc)
            
            # Sort new population
            order = np.argsort(new_scores)[::-1]
            population = [new_population[i] for i in order]
            scores = [new_scores[i] for i in order]
            
            if scores[0] > best_score:
                best_score = scores[0]
                best_fd = population[0].copy()
            
            # Adaptive mutation rate: increase if stagnant
            if generation % 20 == 0:
                recent_best = max(scores[:10])
                if recent_best <= best_score * 1.001:
                    mutation_rate = min(0.15, mutation_rate * 1.3)
            
            # Periodically add fresh random individuals to maintain diversity
            if generation % 15 == 0 and len(population) > elite_count:
                p_fill = random.uniform(0.15, 0.55)
                fd = np.random.binomial(1, p_fill, n_fd)
                cell = expand_fundamental(fd, fd_indices)
                sc = compute_tm_bands(cell)
                # Replace worst individual
                worst_idx = len(population) - 1
                population[worst_idx] = fd
                scores[worst_idx] = sc
                order = np.argsort(scores)[::-1]
                population = [population[i] for i in order]
                scores = [scores[i] for i in order]
    
    # Final: expand best fundamental domain to full cell
    final_cell = expand_fundamental(best_fd, fd_indices)
    
    # Verify D4 symmetry - enforce all constraints
    final_cell = ((final_cell + final_cell.T) // 2).astype(int)
    final_cell = ((final_cell + final_cell[::-1, ::-1]) // 2).astype(int)
    final_cell = ((final_cell + final_cell[:, ::-1]) // 2).astype(int)
    final_cell = ((final_cell + final_cell[::-1, :]) // 2).astype(int)
    
    return final_cell.tolist()

# EVOLVE-BLOCK-END
