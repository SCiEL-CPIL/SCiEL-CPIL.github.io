"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s limit

    def _compute_radius(centers):
        """Compute the maximum radius for the given centers."""
        if centers.shape != (N, 3) or not np.all(np.isfinite(centers)):
            return 0.0
        # Wall distances
        wall_dists = 0.5 - np.abs(centers)
        min_wall = np.min(wall_dists)
        if min_wall < 0:
            return 0.0
        # Pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        return min(min_pair, min_wall) / 2.0

    def _softmin_radius_and_grad(centers, beta=50.0):
        """
        Compute a smooth approximation of min(pairwise distances, wall distances) / 2
        using soft-min, and return (value, gradient).
        """
        c = centers
        # Wall distances: for each center i, min over 3 coords of (0.5 - |c_i,d|)
        # Actually wall distance for center i is min(0.5-|x|, 0.5-|y|, 0.5-|z|)
        # But the constraint is 0.5 - |c_ik| >= r for all i,k
        # So the "wall constraint" value is min_{i,k} (0.5 - |c_ik|)
        # And the "pair constraint" value is min_{i<j} |c_i - c_j|
        # The radius is min(wall_val, pair_val) / 2

        # Wall values: (0.5 - |c_ik|) for all i,k -> N*3 values
        wall_vals = 0.5 - np.abs(c)  # shape (N, 3)
        wall_flat = wall_vals.flatten()  # 150 values

        # Pair distances: N*(N-1)/2 values
        # For efficiency, compute all pairwise
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        # Get upper triangle indices
        iu = np.triu_indices(N, k=1)
        pair_dists = dists[iu]  # N*(N-1)/2 values

        # Combine all constraint values into one array
        # We want soft-min of all these values, then divide by 2
        all_vals = np.concatenate([wall_flat, pair_dists])
        n_constraints = len(all_vals)

        # Soft-min: -beta * log(sum(exp(-beta * val)))
        # For numerical stability, subtract max
        max_val = np.max(all_vals)
        exp_terms = np.exp(-beta * (all_vals - max_val))
        soft_min = max_val - np.log(np.sum(exp_terms)) / beta

        # Gradient with respect to centers
        # d(soft_min)/d(all_vals_j) = -exp_j / sum(exp_terms)
        weights = exp_terms / np.sum(exp_terms)  # shape (n_constraints,)

        # Now we need to map back to centers
        # First 150 are wall values: wall_vals[i, k] = 0.5 - |c[i,k]|
        # d(wall_vals[i,k])/dc[i,k] = -sign(c[i,k])
        grad = np.zeros_like(c)

        # Wall contributions
        wall_weights = weights[:N * 3].reshape(N, 3)
        # d(0.5 - |x|)/dx = -sign(x)
        sign_c = np.sign(c)
        # Handle zero: sign(0) = 0, which is fine (subgradient)
        grad_wall = -wall_weights * sign_c
        grad += grad_wall

        # Pair contributions
        pair_weights = weights[N * 3:]  # N*(N-1)/2 values
        # For pair (i,j) with i<j, dist = |c_i - c_j|
        # d(dist)/dc_i = (c_i - c_j)/dist
        # d(dist)/dc_j = (c_j - c_i)/dist
        # We need to accumulate contributions
        # pair_dists[iu] corresponds to (iu[0][m], iu[1][m])
        ii = iu[0]
        jj = iu[1]
        # dists for these pairs
        pair_dist_vals = dists[ii, jj]
        # Avoid division by zero
        safe_dist = np.where(pair_dist_vals > 1e-12, pair_dist_vals, 1.0)
        # Unit vectors
        unit_vec = (c[ii] - c[jj]) / safe_dist[:, np.newaxis]
        # Contribution to c[i]: +w * unit_vec
        # Contribution to c[j]: -w * unit_vec
        for m in range(len(pair_weights)):
            w = pair_weights[m]
            grad[ii[m]] += w * unit_vec[m]
            grad[jj[m]] -= w * unit_vec[m]

        # The objective is -soft_min/2, so gradient of objective is -grad/2
        # But we're maximizing soft_min/2, so in scipy minimize we negate
        value = soft_min / 2.0
        # Return value (to maximize) and gradient (of the value, so minimize -value)
        return value, grad

    def _repulsion_anneal(centers, n_steps=20000, beta_start=5.0, beta_end=200.0,
                          lr_start=0.05, lr_end=0.001, rng=None):
        """
        Repulsion-based annealing: push spheres apart using soft-min gradient.
        """
        if rng is None:
            rng = np.random.default_rng(42)

        c = centers.copy()
        best_c = c.copy()
        best_val = _compute_radius(c)

        for step in range(n_steps):
            if time.time() > deadline - 20:
                break
            # Annealing schedule
            t = step / max(n_steps - 1, 1)
            beta = beta_start + (beta_end - beta_start) * t
            lr = lr_start + (lr_end - lr_start) * t

            # Add small noise for exploration
            noise = rng.normal(0, 0.001 * (1 - t), c.shape)
            c_noisy = c + noise

            val, grad = _softmin_radius_and_grad(c_noisy, beta)

            # Gradient ascent (we want to maximize val)
            c = c + lr * grad

            # Clip to valid range
            c = np.clip(c, -0.5, 0.5)

            # Occasionally check actual radius
            if step % 100 == 0:
                actual_val = _compute_radius(c)
                if actual_val > best_val:
                    best_val = actual_val
                    best_c = c.copy()

        # Final check
        final_val = _compute_radius(c)
        if final_val > best_val:
            best_val = final_val
            best_c = c.copy()

        return best_c, best_val

    def _lloyd_optimize(centers, n_iters=500, rng=None):
        """
        Lloyd-like algorithm: iteratively move each sphere to maximize its
        local minimum distance to neighbors and walls.
        """
        if rng is None:
            rng = np.random.default_rng(42)

        c = centers.copy()
        best_c = c.copy()
        best_val = _compute_radius(c)

        for it in range(n_iters):
            if time.time() > deadline - 15:
                break

            improved = False
            for i in range(N):
                # Try to improve sphere i
                best_pos = c[i].copy()
                best_local_val = -np.inf

                # Current local min for sphere i
                # Distance to all other spheres
                dists_to_others = np.sqrt(np.sum((c[i] - c[np.arange(N) != i]) ** 2, axis=1))
                # Wall distance
                wall_d = 0.5 - np.abs(c[i])
                wall_min = np.min(wall_d)
                current_local = min(np.min(dists_to_others), wall_min)

                # Try several candidate positions
                # 1. Current position
                candidates = [c[i].copy()]

                # 2. Move away from nearest neighbor
                if len(dists_to_others) > 0:
                    min_idx = np.argmin(dists_to_others)
                    other_idx = np.where(np.arange(N) != i)[0][min_idx]
                    direction = c[i] - c[other_idx]
                    norm = np.linalg.norm(direction)
                    if norm > 1e-10:
                        direction /= norm
                        for step in [0.01, 0.02, 0.05]:
                            candidates.append(c[i] + step * direction)

                # 3. Move away from nearest wall
                min_wall_coord = np.argmin(wall_d)
                wall_dir = np.zeros(3)
                wall_dir[min_wall_coord] = -np.sign(c[i, min_wall_coord])
                if np.linalg.norm(wall_dir) > 0:
                    wall_dir /= np.linalg.norm(wall_dir)
                    for step in [0.01, 0.02, 0.05]:
                        candidates.append(c[i] + step * wall_dir)

                # 4. Random perturbations
                for _ in range(5):
                    candidates.append(c[i] + rng.normal(0, 0.03, 3))

                for cand in candidates:
                    cand = np.clip(cand, -0.5, 0.5)
                    # Compute local min for this candidate
                    dists_new = np.sqrt(np.sum((cand - c[np.arange(N) != i]) ** 2, axis=1))
                    wall_new = 0.5 - np.abs(cand)
                    wall_min_new = np.min(wall_new)
                    local_val = min(np.min(dists_new), wall_min_new)

                    if local_val > best_local_val:
                        best_local_val = local_val
                        best_pos = cand

                if np.any(best_pos != c[i]):
                    c[i] = best_pos
                    improved = True

            if not improved:
                break

            # Check global improvement
            val = _compute_radius(c)
            if val > best_val:
                best_val = val
                best_c = c.copy()

        return best_c, best_val

    def _local_search(centers, max_iter=5000, step_start=0.01, step_end=0.0001, rng=None):
        """
        Coordinate-wise local search with adaptive step size.
        """
        if rng is None:
            rng = np.random.default_rng(42)

        c = centers.copy()
        best_c = c.copy()
        best_val = _compute_radius(c)
        step = step_start

        for it in range(max_iter):
            if time.time() > deadline - 10:
                break

            improved = False
            # Randomly select a sphere and coordinate
            i = rng.integers(0, N)
            d = rng.integers(0, 3)

            for delta in [step, -step]:
                trial = c.copy()
                trial[i, d] += delta
                if trial[i, d] < -0.5 or trial[i, d] > 0.5:
                    continue
                val = _compute_radius(trial)
                if val > best_val + 1e-12:
                    c = trial
                    best_val = val
                    best_c = c.copy()
                    improved = True
                    break

            if not improved:
                step *= 0.9
                if step < step_end:
                    break
            else:
                # Occasionally increase step if we're stuck
                if it % 100 == 0:
                    step = min(step * 1.1, step_start)

        return best_c, best_val

    def _seed_fcc_like():
        """
        Create an FCC-like initial configuration for 50 spheres.
        FCC has 4 atoms per unit cell. For 50 spheres, use a distorted grid.
        We'll use a 5x5x2 arrangement with some FCC-like offsets.
        """
        nx, ny, nz = 5, 5, 2
        # Base grid
        xs = np.linspace(-0.45, 0.45, nx)
        ys = np.linspace(-0.45, 0.45, ny)
        zs = np.linspace(-0.45, 0.45, nz)

        pts = []
        for zi, z in enumerate(zs):
            for yi, y in enumerate(ys):
                for xi, x in enumerate(xs):
                    # Add small FCC-like offset for even/odd layers
                    if zi % 2 == 0:
                        offset = 0.02 * ((xi + yi) % 2)
                    else:
                        offset = -0.02 * ((xi + yi) % 2)
                    pts.append([x + offset, y + offset, z])

        pts = np.array(pts[:N])
        return pts

    def _seed_grid():
        """Simple 5x5x2 grid."""
        nx, ny, nz = 5, 5, 2
        xs = np.linspace(-0.45, 0.45, nx)
        ys = np.linspace(-0.45, 0.45, ny)
        zs = np.linspace(-0.45, 0.45, nz)
        pts = np.array([(x, y, z) for z in zs for y in ys for x in xs])
        return pts[:N]

    def _seed_random(rng):
        """Random initial configuration."""
        return rng.uniform(-0.45, 0.45, (N, 3))

    def _seed_shell_core(rng):
        """Shell-core configuration: some spheres near center, others near walls."""
        # Core: 20 spheres in a smaller grid
        n_core = 20
        n_shell = N - n_core

        # Core in 4x5x1 grid
        cx = np.linspace(-0.2, 0.2, 4)
        cy = np.linspace(-0.2, 0.2, 5)
        cz = [0.0]
        core_pts = []
        for z in cz:
            for y in cy:
                for x in cx:
                    core_pts.append([x, y, z])
        core_pts = np.array(core_pts[:n_core])

        # Shell: remaining spheres in a larger grid
        sx = np.linspace(-0.45, 0.45, 7)
        sy = np.linspace(-0.45, 0.45, 7)
        sz = [-0.45, 0.45]
        shell_pts = []
        for z in sz:
            for y in sy:
                for x in sx:
                    shell_pts.append([x, y, z])
        # Shuffle and take n_shell
        rng.shuffle(shell_pts)
        shell_pts = np.array(shell_pts[:n_shell])

        return np.vstack([core_pts, shell_pts])

    # --- Main optimization pipeline ---
    rng = np.random.default_rng(42)

    # Generate multiple initial configurations
    seeds = [
        _seed_fcc_like(),
        _seed_grid(),
        _seed_random(rng),
        _seed_shell_core(rng),
    ]

    # Add some perturbed versions
    for _ in range(3):
        base = seeds[rng.integers(0, len(seeds))]
        perturbed = base + rng.normal(0, 0.05, (N, 3))
        perturbed = np.clip(perturbed, -0.45, 0.45)
        seeds.append(perturbed)

    best_centers = None
    best_val = -np.inf

    for seed_idx, seed in enumerate(seeds):
        if time.time() > deadline - 30:
            break

        # Phase 1: Repulsion annealing
        c1, v1 = _repulsion_anneal(seed, n_steps=15000, rng=rng)
        if v1 > best_val:
            best_val = v1
            best_centers = c1.copy()

        # Phase 2: Lloyd optimization
        c2, v2 = _lloyd_optimize(c1, n_iters=300, rng=rng)
        if v2 > best_val:
            best_val = v2
            best_centers = c2.copy()

        # Phase 3: Local search
        c3, v3 = _local_search(c2, max_iter=3000, rng=rng)
        if v3 > best_val:
            best_val = v3
            best_centers = c3.copy()

        # Phase 4: Another round of repulsion annealing with higher beta
        if time.time() < deadline - 25:
            c4, v4 = _repulsion_anneal(c3, n_steps=10000, beta_start=20.0, beta_end=100.0,
                                       lr_start=0.01, lr_end=0.0005, rng=rng)
            if v4 > best_val:
                best_val = v4
                best_centers = c4.copy()

        # Phase 5: Final local search
        if time.time() < deadline - 15:
            c5, v5 = _local_search(c4, max_iter=2000, step_start=0.005, step_end=0.00005, rng=rng)
            if v5 > best_val:
                best_val = v5
                best_centers = c5.copy()

    # Final high-precision polish
    if time.time() < deadline - 5 and best_centers is not None:
        final_c, final_v = _local_search(best_centers, max_iter=2000, step_start=0.001,
                                         step_end=0.00001, rng=rng)
        if final_v > best_val:
            best_val = final_v
            best_centers = final_c.copy()

    # Ensure validity
    if best_centers is None:
        best_centers = _seed_grid()
    best_centers = np.clip(best_centers, -0.5, 0.5)

    return best_centers

# EVOLVE-BLOCK-END
