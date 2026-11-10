import numpy as np
import time

N = 100
DEADLINE = 145.0  # Leave margin before 150s limit

def construct():
    start_time = time.time()

    def remaining():
        return DEADLINE - (time.time() - start_time)

    def compute_radius(centers):
        """Compute the maximum common radius (min of pair-dist/2 and wall-dist)."""
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        min_pair = np.sqrt(np.min(dist_sq))
        wall = 0.5 - np.abs(centers)
        min_wall = np.min(wall)
        return min(min_pair / 2.0, min_wall)

    def compute_energy(centers, beta):
        """Compute a smooth energy that penalizes close pairs and wall violations.
        Lower energy = better packing. Returns (energy, radius_estimate)."""
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dist_sq, np.inf)

        # Pair penalty: sum of 1/dist^beta over all pairs (only close pairs matter)
        # Use a cutoff to avoid overflow
        d = np.sqrt(dist_sq)
        # Only count pairs closer than some threshold
        mask = d < 0.5  # Only close pairs contribute significantly
        pair_energy = np.sum(np.where(mask, 1.0 / np.maximum(d, 1e-6) ** beta, 0.0))

        # Wall penalty: spheres too close to walls
        wall = 0.5 - np.abs(centers)
        wall_violation = np.maximum(0.0, -wall)  # Only count actual violations
        wall_energy = np.sum(wall_violation ** 2) * 100.0

        # Also penalize being very close to walls (soft)
        wall_soft = np.where(wall > 0, 1.0 / np.maximum(wall, 1e-6) ** beta, 0.0)
        wall_soft_energy = np.sum(wall_soft) * 0.1

        total_energy = pair_energy + wall_energy + wall_soft_energy

        # Estimate radius from energy (for tracking)
        min_pair = np.sqrt(np.min(dist_sq))
        min_wall = np.min(wall)
        radius_est = min(min_pair / 2.0, min_wall)

        return total_energy, radius_est

    def simulated_annealing(centers, t_start, t_end, n_steps, beta=2.0, step_scale=0.02):
        """Simulated annealing to optimize packing."""
        centers = centers.copy().astype(float)
        current_energy, current_r = compute_energy(centers, beta)
        best_energy = current_energy
        best_centers = centers.copy()
        best_r = current_r

        for i in range(n_steps):
            if remaining() < 3.0:
                break

            # Temperature schedule (geometric cooling)
            progress = i / max(n_steps - 1, 1)
            T = t_start * (t_end / t_start) ** progress

            # Random perturbation
            move = np.random.randn(N, 3) * step_scale
            trial = centers + move
            trial = np.clip(trial, -0.4999, 0.4999)

            trial_energy, trial_r = compute_energy(trial, beta)
            delta_E = trial_energy - current_energy

            # Accept if better, or with probability exp(-delta_E / T)
            if delta_E < 0 or (T > 1e-10 and np.random.random() < np.exp(-delta_E / T)):
                centers = trial
                current_energy = trial_energy
                if trial_energy < best_energy - 1e-10:
                    best_energy = trial_energy
                    best_centers = centers.copy()
                    best_r = trial_r

        return best_centers, best_r

    def local_refine(centers, max_iters=2000):
        """Local refinement using gradient-free optimization (Nelder-Mead style on subsets).
        We do coordinate-wise local search."""
        centers = centers.copy().astype(float)
        current_r = compute_radius(centers)
        best_r = current_r
        best_centers = centers.copy()

        for i in range(max_iters):
            if remaining() < 3.0:
                break

            improved = False
            # Try small random moves on random subsets
            n_move = np.random.randint(1, 6)
            idx = np.random.choice(N, size=n_move, replace=False)
            move = np.random.randn(n_move, 3) * 0.005
            trial = centers.copy()
            trial[idx] += move
            trial = np.clip(trial, -0.4999, 0.4999)

            new_r = compute_radius(trial)
            if new_r > best_r + 1e-12:
                centers = trial
                best_r = new_r
                best_centers = centers.copy()
                improved = True

            if not improved and i % 50 == 0:
                # Occasionally try larger moves
                n_move = np.random.randint(3, 10)
                idx = np.random.choice(N, size=n_move, replace=False)
                move = np.random.randn(n_move, 3) * 0.02
                trial = centers.copy()
                trial[idx] += move
                trial = np.clip(trial, -0.4999, 0.4999)
                new_r = compute_radius(trial)
                if new_r > best_r + 1e-12:
                    centers = trial
                    best_r = new_r
                    best_centers = centers.copy()

        return best_centers, best_r

    # Generate diverse initializations
    np.random.seed(42)
    inits = []

    # Init 1: 5x5x4 grid
    nx, ny, nz = 5, 5, 4
    xs = np.linspace(-0.44, 0.44, nx)
    ys = np.linspace(-0.44, 0.44, ny)
    zs = np.linspace(-0.44, 0.44, nz)
    grid_pts = []
    for x in xs:
        for y in ys:
            for z in zs:
                grid_pts.append([x, y, z])
    inits.append(np.array(grid_pts[:N]))

    # Init 2: Random uniform
    inits.append(np.random.uniform(-0.44, 0.44, (N, 3)))

    # Init 3: Jittered grid
    base = inits[0].copy()
    base += np.random.randn(N, 3) * 0.03
    base = np.clip(base, -0.4999, 0.4999)
    inits.append(base)

    # Init 4: FCC-inspired (face-centered cubic)
    # FCC unit cell has 4 atoms; scale to fit
    # Generate FCC points in a larger region and select
    a = 0.2  # lattice constant
    fcc_pts = []
    for ix in range(-2, 3):
        for iy in range(-2, 3):
            for iz in range(-2, 3):
                # Corner
                fcc_pts.append([ix * a, iy * a, iz * a])
                # Face centers
                fcc_pts.append([(ix + 0.5) * a, (iy + 0.5) * a, iz * a])
                fcc_pts.append([(ix + 0.5) * a, iy * a, (iz + 0.5) * a])
                fcc_pts.append([ix * a, (iy + 0.5) * a, (iz + 0.5) * a])
    fcc_arr = np.array(fcc_pts)
    # Center and select N points closest to origin
    fcc_arr -= np.mean(fcc_arr, axis=0)
    dists = np.sum(fcc_arr ** 2, axis=1)
    idx_sorted = np.argsort(dists)
    inits.append(fcc_arr[idx_sorted[:N]])

    # Init 5: BCC-inspired
    a = 0.22
    bcc_pts = []
    for ix in range(-2, 3):
        for iy in range(-2, 3):
            for iz in range(-2, 3):
                bcc_pts.append([ix * a, iy * a, iz * a])
                bcc_pts.append([(ix + 0.5) * a, (iy + 0.5) * a, (iz + 0.5) * a])
    bcc_arr = np.array(bcc_pts)
    bcc_arr -= np.mean(bcc_arr, axis=0)
    dists = np.sum(bcc_arr ** 2, axis=1)
    idx_sorted = np.argsort(dists)
    inits.append(bcc_arr[idx_sorted[:N]])

    # Init 6: Layered structure (5 layers of 20)
    layer_pts = []
    for layer in range(5):
        z = -0.36 + layer * 0.18
        for i in range(4):
            for j in range(5):
                x = -0.36 + i * 0.24 + (0.06 if layer % 2 else 0)
                y = -0.36 + j * 0.24 + (0.06 if (layer + i) % 2 else 0)
                layer_pts.append([x, y, z])
    inits.append(np.array(layer_pts[:N]))

    # Init 7: Random with clustering
    cluster_centers = np.random.uniform(-0.35, 0.35, (10, 3))
    cluster_assign = np.random.randint(0, 10, N)
    init7 = cluster_centers[cluster_assign] + np.random.randn(N, 3) * 0.08
    init7 = np.clip(init7, -0.4999, 0.4999)
    inits.append(init7)

    best_r = 0.0
    best_centers = None

    # Run simulated annealing on each initialization
    for init_idx, init in enumerate(inits):
        if remaining() < 10.0:
            break

        # Multiple SA runs per init with different temperatures
        for sa_run in range(2):
            if remaining() < 8.0:
                break
            np.random.seed(42 + init_idx * 100 + sa_run)
            c, r = simulated_annealing(
                init,
                t_start=0.5,
                t_end=0.001,
                n_steps=5000,
                beta=2.0,
                step_scale=0.03
            )
            if r > best_r:
                best_r = r
                best_centers = c

    # Local refinement on best result
    if best_centers is not None and remaining() > 5.0:
        for refine_pass in range(3):
            if remaining() < 4.0:
                break
            c, r = local_refine(best_centers, max_iters=3000)
            if r > best_r:
                best_r = r
                best_centers = c
            else:
                break

    # Final SA pass on best with lower temperature
    if best_centers is not None and remaining() > 5.0:
        np.random.seed(12345)
        c, r = simulated_annealing(
            best_centers,
            t_start=0.1,
            t_end=0.0001,
            n_steps=5000,
            beta=2.0,
            step_scale=0.01
        )
        if r > best_r:
            best_r = r
            best_centers = c

    if best_centers is None:
        return inits[0]

    return best_centers
