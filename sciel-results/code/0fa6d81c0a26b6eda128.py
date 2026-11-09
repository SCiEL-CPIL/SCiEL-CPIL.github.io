"""Photonic crystal design for maximum TM band gap (bands 4-5)."""
import numpy as np
import time

def design(problem):
    """Design a symmetric 32x32 unit cell maximizing TM band 4-5 gap."""
    start_time = time.time()
    deadline = start_time + 210.0  # Safety margin before 230s kill
    
    size = problem["pixels"]  # 32
    eps_low = problem["eps_low"]  # 1
    eps_high = problem["eps_high"]  # 12
    
    # 289 plane waves => (2*N+1)^2 = 289 => N=8
    N_pw = 8
    G_range = np.arange(-N_pw, N_pw + 1)
    Gx, Gy = np.meshgrid(G_range, G_range, indexing='ij')
    G_vectors = np.column_stack((Gx.ravel(), Gy.ravel()))  # shape (289, 2)
    num_pw = len(G_vectors)
    
    # Precompute G_diff[i,j] = G_vectors[i] - G_vectors[j]
    G_diff = G_vectors[:, np.newaxis, :] - G_vectors[np.newaxis, :, :]  # (num_pw, num_pw, 2)
    
    # Precompute lookup: map (gx, gy) -> index in G_vectors
    G_lookup = {}
    for idx in range(num_pw):
        key = (int(G_vectors[idx, 0]), int(G_vectors[idx, 1]))
        G_lookup[key] = idx
    
    # Precompute which pairs have valid G_diff (within -8..8 range)
    # For each pair (i,j), check if G_diff[i,j] is in G_lookup
    valid_mask = np.zeros((num_pw, num_pw), dtype=bool)
    for i in range(num_pw):
        for j in range(num_pw):
            dx = int(G_diff[i, j, 0])
            dy = int(G_diff[i, j, 1])
            if (dx, dy) in G_lookup:
                valid_mask[i, j] = True
    
    # Precompute G_diff indices for valid pairs
    G_diff_idx = np.full((num_pw, num_pw), -1, dtype=int)
    for i in range(num_pw):
        for j in range(num_pw):
            if valid_mask[i, j]:
                dx = int(G_diff[i, j, 0])
                dy = int(G_diff[i, j, 1])
                G_diff_idx[i, j] = G_lookup[(dx, dy)]
    
    # Symmetry orbits for the 32x32 cell under square group
    def get_orbits(size):
        """Get symmetry orbits for square symmetry about cell center."""
        visited = set()
        orbits = []
        for i in range(size):
            for j in range(size):
                if (i, j) not in visited:
                    orbit = set()
                    orbit.add((i, j))
                    orbit.add((size-1-i, size-1-j))
                    orbit.add((i, size-1-j))
                    orbit.add((size-1-i, j))
                    orbit.add((j, i))
                    orbit.add((size-1-j, size-1-i))
                    visited.update(orbit)
                    orbits.append(list(orbit))
        return orbits
    
    orbits = get_orbits(size)
    num_orbits = len(orbits)
    
    # Map from orbit index to pixel positions (for fast cell construction)
    orbit_pixels = [np.array(orbit, dtype=int) for orbit in orbits]  # list of (n, 2) arrays
    
    def orbit_to_cell(orbit_vals):
        """Convert orbit values to full cell."""
        cell = np.zeros((size, size), dtype=np.uint8)
        for o_idx in range(num_orbits):
            if orbit_vals[o_idx]:
                pixels = orbit_pixels[o_idx]
                cell[pixels[:, 0], pixels[:, 1]] = 1
        return cell
    
    # Precompute FFT-based Fourier coefficient computation
    # For a binary cell, 1/eps = where(cell==0, 1.0, 1.0/eps_high)
    # We need the DFT at G values -8..8
    
    # Precompute phase factors for the DFT
    # inv_eps_G[idx] = (1/size^2) * sum_{i,j} inv_eps[i,j] * exp(-2pi*i*(Gx*i + Gy*j)/size)
    # This is equivalent to: FFT2(inv_eps)[Gx mod size, Gy mod size] / size^2
    
    # K-points for evaluation (irreducible BZ)
    k_points = [
        (0.0, 0.0),       # Gamma
        (0.5, 0.0),       # X
        (0.5, 0.5),       # M
        (0.25, 0.0),      # Midpoint Gamma-X
        (0.25, 0.25),     # Midpoint Gamma-M
        (0.5, 0.25),      # Midpoint X-M
        (0.1, 0.0),
        (0.4, 0.1),
        (0.3, 0.3),
        (0.15, 0.05),
        (0.2, 0.2),
        (0.35, 0.05),
        (0.15, 0.15),
        (0.45, 0.1),
        (0.2, 0.05),
    ]
    
    # Precompute |k+G|^2 for each k-point
    kG2_cache = {}
    for kx, ky in k_points:
        k_plus_G = G_vectors + np.array([kx, ky])
        kG2 = np.sum(k_plus_G ** 2, axis=1)  # shape (num_pw,)
        kG2_cache[(round(kx, 10), round(ky, 10))] = kG2
    
    def compute_coeff_matrix(orbit_vals):
        """Compute the coefficient matrix for the Hamiltonian from orbit values."""
        cell = orbit_to_cell(orbit_vals)
        inv_eps = np.where(cell == 0, 1.0, 1.0/eps_high)
        
        # FFT to get Fourier coefficients
        fft_inv_eps = np.fft.fft2(inv_eps) / (size * size)
        
        # Build coefficient matrix: coeff[i,j] = fft_inv_eps[G_diff[i,j]]
        coeff_matrix = np.zeros((num_pw, num_pw), dtype=complex)
        for i in range(num_pw):
            for j in range(num_pw):
                if G_diff_idx[i, j] >= 0:
                    # G_diff_idx[i,j] gives the index into G_vectors
                    # which corresponds to the (gx, gy) position
                    gx = int(G_diff[i, j, 0])
                    gy = int(G_diff[i, j, 1])
                    # Map to FFT index (mod size)
                    ix = gx % size
                    iy = gy % size
                    coeff_matrix[i, j] = fft_inv_eps[ix, iy]
        
        return coeff_matrix
    
    def evaluate_cell_fast(orbit_vals):
        """Evaluate gap-midgap ratio using precomputed structures."""
        coeff_matrix = compute_coeff_matrix(orbit_vals)
        
        min_w5_sq = float('inf')
        max_w4_sq = 0.0
        
        for kx, ky in k_points:
            key = (round(kx, 10), round(ky, 10))
            kG2 = kG2_cache[key]
            
            # M[i,j] = coeff_matrix[i,j] * kG2[i]
            M = coeff_matrix * kG2[:, np.newaxis]
            
            # Solve eigenvalue problem
            eigenvalues = np.linalg.eigvalsh(M)
            eigenvalues = np.real(eigenvalues)
            eigenvalues.sort()
            
            w4_sq = eigenvalues[3]
            w5_sq = eigenvalues[4]
            
            if w5_sq < min_w5_sq:
                min_w5_sq = w5_sq
            if w4_sq > max_w4_sq:
                max_w4_sq = w4_sq
        
        if max_w4_sq <= 0:
            return 0.0
        
        min_w5 = np.sqrt(min_w5_sq)
        max_w4 = np.sqrt(max_w4_sq)
        
        midgap = (min_w5 + max_w4) / 2.0
        if midgap == 0:
            return 0.0
        
        gap = min_w5 - max_w4
        ratio = 2.0 * gap / midgap
        return ratio
    
    # Genetic algorithm over orbit bitmasks
    np.random.seed(42)
    
    # Population size
    pop_size = 20
    elite_size = 4
    
    # Initialize population
    population = []
    scores = []
    
    # Seed with some structured initial guesses
    # 1. Centered circle
    vals = np.zeros(num_orbits, dtype=int)
    for o_idx in range(num_orbits):
        pixels = orbit_pixels[o_idx]
        # Check if any pixel in orbit is within radius 0.25
        for i, j in pixels:
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if x*x + y*y < 0.25**2:
                vals[o_idx] = 1
                break
    population.append(vals.copy())
    
    # 2. Ring pattern
    vals = np.zeros(num_orbits, dtype=int)
    for o_idx in range(num_orbits):
        pixels = orbit_pixels[o_idx]
        for i, j in pixels:
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            r = np.sqrt(x*x + y*y)
            if 0.15 < r < 0.3:
                vals[o_idx] = 1
                break
    population.append(vals.copy())
    
    # 3. Cross pattern
    vals = np.zeros(num_orbits, dtype=int)
    for o_idx in range(num_orbits):
        pixels = orbit_pixels[o_idx]
        for i, j in pixels:
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if abs(x) < 0.1 or abs(y) < 0.1:
                vals[o_idx] = 1
                break
    population.append(vals.copy())
    
    # Fill rest with random
    while len(population) < pop_size:
        vals = np.random.randint(0, 2, num_orbits)
        population.append(vals)
    
    # Evaluate initial population
    for ind in population:
        scores.append(evaluate_cell_fast(ind))
    
    best_score = max(scores)
    best_vals = population[scores.index(best_score)].copy()
    
    # Evolution loop
    generation = 0
    while time.time() < deadline:
        generation += 1
        
        # Sort by score
        order = np.argsort(scores)[::-1]
        population = [population[i] for i in order]
        scores = [scores[i] for i in order]
        
        # Elitism
        new_population = [population[i].copy() for i in range(min(elite_size, pop_size))]
        new_scores = [scores[i] for i in range(min(elite_size, pop_size))]
        
        # Crossover and mutation to fill population
        while len(new_population) < pop_size:
            if time.time() > deadline - 5:
                break
            
            # Select parents (tournament selection)
            def tournament_select(pop, scores, k=3):
                candidates = np.random.choice(len(pop), k, replace=False)
                best_cand = candidates[np.argmax([scores[c] for c in candidates])]
                return pop[best_cand].copy()
            
            parent1 = tournament_select(population, scores)
            parent2 = tournament_select(population, scores)
            
            # Uniform crossover
            child = np.random.randint(0, 2, num_orbits)
            mask = np.random.random(num_orbits) < 0.5
            child[mask] = parent1[mask]
            child[~mask] = parent2[~mask]
            
            # Mutation
            n_mut = np.random.randint(1, 6)
            mut_idx = np.random.choice(num_orbits, n_mut, replace=False)
            child[mut_idx] = 1 - child[mut_idx]
            
            new_population.append(child)
            new_scores.append(0.0)  # Placeholder
        
        # Evaluate new individuals
        for i in range(elite_size, len(new_population)):
            if time.time() > deadline - 5:
                break
            new_scores[i] = evaluate_cell_fast(new_population[i])
        
        population = new_population
        scores = new_scores
        
        # Track best
        current_best = max(scores)
        if current_best > best_score:
            best_score = current_best
            best_idx = scores.index(current_best)
            best_vals = population[best_idx].copy()
    
    # Final evaluation of best
    final_score = evaluate_cell_fast(best_vals)
    if final_score > best_score:
        best_score = final_score
    
    best_cell = orbit_to_cell(best_vals)
    return best_cell.tolist()
