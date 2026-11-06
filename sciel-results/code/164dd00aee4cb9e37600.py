import numpy as np
import time

def design(problem):
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]

    # --- Build C4v symmetry orbits ---
    orbits = []
    seen = set()
    for i in range(size):
        for j in range(size):
            if (i, j) in seen:
                continue
            orbit = set()
            x, y = i, j
            for _ in range(4):
                orbit.add((x, y))
                orbit.add((x, size - 1 - y))
                x, y = y, size - 1 - x
            orbits.append(orbit)
            seen.update(orbit)

    n_orbits = len(orbits)

    # Pre-compute orbit index lookup for fast grid construction
    orbit_index = {}
    for k, orbit in enumerate(orbits):
        for (i, j) in orbit:
            orbit_index[(i, j)] = k

    def orbit_vals_to_grid(ov):
        """Convert orbit binary vector to 32x32 grid (vectorized)."""
        grid = np.zeros((size, size), dtype=np.int8)
        for k in range(n_orbits):
            if ov[k]:
                for (i, j) in orbits[k]:
                    grid[i, j] = 1
        return grid

    # --- Fast vectorized PWE surrogate (N=7, 49 modes) ---
    N_pw = 7
    G = np.arange(-N_pw // 2, N_pw // 2 + 1)
    Gx, Gy = np.meshgrid(G, G, indexing='ij')
    G_flat = np.stack([Gx.ravel(), Gy.ravel()], axis=1).astype(np.float64)
    N_modes = len(G_flat)

    # Precompute G-differences for Fourier coefficient lookup
    dGx = G_flat[:, 0:1] - G_flat[0:1, 0]
    dGy = G_flat[:, 1:2] - G_flat[0:1, 1]
    dGx = dGx.ravel()
    dGy = dGy.ravel()
    # Map to FFT indices (mod size)
    fft_idx_i = (dGx % size).astype(int)
    fft_idx_j = (dGy % size).astype(int)

    # K-points in irreducible BZ (square lattice, using fractional coords * 2pi)
    # Use a denser set for better gap estimation
    k_points = [
        (0.0, 0.0),
        (0.5, 0.0),
        (0.5, 0.5),
        (0.25, 0.0),
        (0.25, 0.25),
        (0.5, 0.25),
        (0.125, 0.0),
        (0.125, 0.125),
        (0.375, 0.0),
        (0.375, 0.125),
        (0.5, 0.125),
        (0.375, 0.375),
    ]

    # Precompute k+G vectors for each k-point
    kG_list = []
    for kx, ky in k_points:
        kx_pi = kx * 2.0 * np.pi
        ky_pi = ky * 2.0 * np.pi
        kxG = kx_pi + G_flat[:, 0]
        kyG = kx_pi + G_flat[:, 1]  # BUG: should be ky_pi
        # Fix:
        kxG = kx_pi + G_flat[:, 0]
        kyG = ky_pi + G_flat[:, 1]
        kG_list.append((kxG, kyG))

    def fast_evaluate(orbit_vals):
        """
        Vectorized PWE evaluation using 49 plane waves.
        Returns the gap-midgap ratio between bands 4 and 5.
        """
        grid = orbit_vals_to_grid(orbit_vals)
        eps_grid = eps_low + (eps_high - eps_low) * grid

        # Fourier coefficients of epsilon
        eps_fft = np.fft.fft2(eps_grid) / (size * size)

        global_max_4 = 0.0
        global_min_5 = float('inf')

        for idx, (kxG, kyG) in enumerate(kG_list):
            k_sq = kxG**2 + kyG**2  # (N_modes,)

            # Build inverse-epsilon matrix: 1/eps(k+G) * delta_ij + coupling
            # For TM: A_ij = (k_i + G_i) · (k_j + G_j) / eps(G_i - G_j)
            # We build the matrix as: diag(k^2) - off-diagonal coupling / eps
            # Actually the standard PWE TM matrix is:
            # sum_j [ (k+G_i)·(k+G_j) / eps(G_i - G_j) ] c_j = omega^2 c_i
            # So M_ij = (k+G_i)·(k+G_j) / eps(G_i - G_j)

            # Vectorized dot products: (kx_i + Gx_i)(kx_j + Gx_j) + (ky_i + Gy_i)(ky_j + Gy_j)
            # = kx_i*kx_j + kx_i*Gx_j + Gx_i*kx_j + Gx_i*Gx_j + same for y
            # Precompute outer products
            kxG_outer = np.outer(kxG, kxG)
            kyG_outer = np.outer(kyG, kyG)
            dot_matrix = kxG_outer + kyG_outer  # (N_modes, N_modes)

            # Get 1/eps for each pair (i,j)
            # eps(G_i - G_j) -> eps_fft[fft_idx_i[i,j], fft_idx_j[i,j]]
            # We need this for all i,j pairs
            # Reshape dG arrays to (N_modes, N_modes)
            dGx_2d = (G_flat[:, 0:1] - G_flat[0:1, 0]).ravel().reshape(N_modes, N_modes)
            dGy_2d = (G_flat[:, 1:2] - G_flat[0:1, 1]).ravel().reshape(N_modes, N_modes)

            fft_i = (dGx_2d % size).astype(int)
            fft_j = (dGy_2d % size).astype(int)

            eps_vals = eps_fft[fft_i, fft_j].real
            # Avoid division by zero (shouldn't happen with eps >= 1)
            inv_eps = 1.0 / np.maximum(eps_vals, 0.1)

            M = dot_matrix * inv_eps

            try:
                eigvals = np.linalg.eigvalsh(M)
            except:
                return -1.0

            if len(eigvals) < 5:
                return -1.0

            # Take 4th and 5th smallest eigenvalues (0-indexed: 3 and 4)
            w4_sq = max(eigvals[3], 0.0)
            w5_sq = max(eigvals[4], 0.0)

            w4 = np.sqrt(w4_sq)
            w5 = np.sqrt(w5_sq)

            if w4 > global_max_4:
                global_max_4 = w4
            if w5 < global_min_5:
                global_min_5 = w5

        if global_min_5 == float('inf') or global_max_4 == 0.0:
            return -1.0

        gap = global_min_5 - global_max_4
        midgap = (global_min_5 + global_max_4) / 2.0
        if midgap <= 0:
            return 0.0
        return gap / midgap

    # --- Genetic Algorithm over orbit binary vectors ---
    start_time = time.time()
    deadline = start_time + 195.0  # Safety margin

    pop_size = 60
    elite_size = 8
    mutation_rate = 0.08

    # Initialize population
    population = []
    scores = []

    # Seed with some structured initial designs
    # 1. Central dielectric disk
    grid1 = np.zeros((size, size), dtype=int)
    cx, cy = size // 2 - 0.5, size // 2 - 0.5
    for i in range(size):
        for j in range(size):
            if (i - cx)**2 + (j - cy)**2 < (size * 0.25)**2:
                grid1[i, j] = 1
    ov1 = np.zeros(n_orbits, dtype=bool)
    for k, orbit in enumerate(orbits):
        val = sum(grid1[i, j] for (i, j) in orbit)
        ov1[k] = val > len(orbit) / 2
    population.append(ov1)

    # 2. Dielectric frame (border)
    grid2 = np.ones((size, size), dtype=int)
    inner = size // 4
    grid2[inner:size-inner, inner:size-inner] = 0
    ov2 = np.zeros(n_orbits, dtype=bool)
    for k, orbit in enumerate(orbits):
        val = sum(grid2[i, j] for (i, j) in orbit)
        ov2[k] = val > len(orbit) / 2
    population.append(ov2)

    # 3. Cross shape
    grid3 = np.zeros((size, size), dtype=int)
    mid = size // 2
    grid3[mid-2:mid+2, :] = 1
    grid3[:, mid-2:mid+2] = 1
    ov3 = np.zeros(n_orbits, dtype=bool)
    for k, orbit in enumerate(orbits):
        val = sum(grid3[i, j] for (i, j) in orbit)
        ov3[k] = val > len(orbit) / 2
    population.append(ov3)

    # Fill rest with random
    while len(population) < pop_size:
        ov = np.random.randint(0, 2, n_orbits).astype(bool)
        population.append(ov)

    # Evaluate initial population
    for ov in population:
        s = fast_evaluate(ov)
        scores.append(s)

    best_score = max(scores)
    best_ov = population[scores.index(best_score)].copy()

    # GA loop
    gen = 0
    while time.time() < deadline - 10.0:
        gen += 1
        # Sort by score
        order = np.argsort(scores)[::-1]
        population = [population[i] for i in order]
        scores = [scores[i] for i in order]

        # Keep elites
        new_population = [population[i].copy() for i in range(min(elite_size, len(population)))]
        new_scores = scores[:min(elite_size, len(scores))]

        # Update best
        if scores[0] > best_score:
            best_score = scores[0]
            best_ov = population[0].copy()

        # Crossover and mutation to fill population
        while len(new_population) < pop_size:
            # Tournament selection
            idx1 = np.random.randint(0, min(10, len(population)))
            idx2 = np.random.randint(0, min(10, len(population)))
            if scores[idx1] >= scores[idx2]:
                parent1 = population[idx1]
            else:
                parent1 = population[idx2]

            idx3 = np.random.randint(0, min(10, len(population)))
            idx4 = np.random.randint(0, min(10, len(population)))
            if scores[idx3] >= scores[idx4]:
                parent2 = population[idx3]
            else:
                parent2 = population[idx4]

            # Uniform crossover
            mask = np.random.random(n_orbits) < 0.5
            child = np.where(mask, parent1, parent2).astype(bool)

            # Mutation
            mut_mask = np.random.random(n_orbits) < mutation_rate
            child[mut_mask] = ~child[mut_mask]

            new_population.append(child)
            s = fast_evaluate(child)
            new_scores.append(s)

        population = new_population
        scores = new_scores

        # Periodic check
        if gen % 10 == 0:
            current_best = max(scores)
            if current_best > best_score:
                best_score = current_best
                best_idx = scores.index(current_best)
                best_ov = population[best_idx].copy()

    # Final: return best grid
    best_grid = orbit_vals_to_grid(best_ov)

    # Enforce exact symmetry (safety)
    final_grid = np.zeros((size, size), dtype=int)
    for k, orbit in enumerate(orbits):
        val = int(best_ov[k])
        for (i, j) in orbit:
            final_grid[i, j] = val

    # Verify symmetry
    assert np.array_equal(final_grid, final_grid[::-1, :]), "Row symmetry failed"
    assert np.array_equal(final_grid, final_grid[:, ::-1]), "Col symmetry failed"
    assert np.array_equal(final_grid, final_grid.T), "Transpose symmetry failed"

    return final_grid.tolist()
