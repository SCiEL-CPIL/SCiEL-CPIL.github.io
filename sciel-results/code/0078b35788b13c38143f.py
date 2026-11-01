import numpy as np
from scipy.spatial import Delaunay, Voronoi, HalfspaceIntersection
from scipy.optimize import minimize
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill

    # --- Helper functions ---

    def compute_radius(c):
        """Compute the common radius for a set of centers."""
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists) / 2.0
        wall_dists = np.min(0.5 - np.abs(c), axis=1)
        min_wall_dist = np.min(wall_dists)
        return min(min_pair_dist, min_wall_dist)

    def pair_dist_matrix(c):
        """Return pairwise distance matrix."""
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        return dists

    def lloyd_step(c):
        """One Lloyd iteration: move each center to the centroid of its Voronoi cell clipped to cube."""
        new_c = np.zeros_like(c)
        # Cube half-spaces: a^T x <= b
        cube_hs = np.array([
            [-1, 0, 0, 0.5],
            [ 1, 0, 0, 0.5],
            [ 0, -1, 0, 0.5],
            [ 0, 1, 0, 0.5],
            [ 0, 0, -1, 0.5],
            [ 0, 0, 1, 0.5]
        ])
        for i in range(N):
            hs_list = [cube_hs]
            for j in range(N):
                if i == j:
                    continue
                a = 2.0 * (c[j] - c[i])
                b = np.dot(c[j], c[j]) - np.dot(c[i], c[i])
                hs_list.append(np.array([[a[0], a[1], a[2], b]]))
            hs = np.vstack(hs_list)
            try:
                inter = HalfspaceIntersection(hs, c[i])
                verts = inter.intersections
                if len(verts) > 0:
                    # Approximate centroid as mean of vertices
                    new_c[i] = np.mean(verts, axis=0)
                else:
                    new_c[i] = c[i]
            except Exception:
                new_c[i] = c[i]
        return np.clip(new_c, -0.5, 0.5)

    def lloyd_optimize(c, time_budget=15.0):
        """Run Lloyd's algorithm for a time budget."""
        t0 = time.time()
        centers = c.copy()
        best_r = compute_radius(centers)
        best_c = centers.copy()
        while time.time() - t0 < time_budget:
            old_c = centers.copy()
            centers = lloyd_step(centers)
            r = compute_radius(centers)
            if r > best_r:
                best_r = r
                best_c = centers.copy()
            if np.linalg.norm(centers - old_c) < 1e-6:
                break
        return best_c, best_r

    def softmin_radius(c, beta):
        """Compute soft-min of all distances (pair and wall)."""
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        pair_vals = dists / 2.0
        # Only upper triangle
        iu = np.triu_indices(N, 1)
        pair_vals_flat = pair_vals[iu]
        wall_dists = np.min(0.5 - np.abs(c), axis=1)
        vals = np.concatenate([pair_vals_flat, wall_dists])
        # Soft min
        v_min = np.min(vals)
        exp_vals = np.exp(-beta * (vals - v_min))
        sum_exp = np.sum(exp_vals)
        soft_min = v_min - np.log(sum_exp) / beta
        return soft_min

    def smooth_obj(x, beta):
        """Objective: negative soft-min radius."""
        c = x.reshape(N, 3)
        return -softmin_radius(c, beta)

    def smooth_grad(x, beta):
        """Gradient of smooth_obj."""
        c = x.reshape(N, 3)
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        pair_vals = dists / 2.0
        iu = np.triu_indices(N, 1)
        pair_vals_flat = pair_vals[iu]
        wall_dists = np.min(0.5 - np.abs(c), axis=1)
        vals = np.concatenate([pair_vals_flat, wall_dists])

        v_min = np.min(vals)
        exp_vals = np.exp(-beta * (vals - v_min))
        sum_exp = np.sum(exp_vals)
        weights = exp_vals / sum_exp

        grads = np.zeros((N, 3))
        # Pair gradients
        idx = 0
        for i in range(N):
            for j in range(i + 1, N):
                d = dists[i, j]
                if d > 1e-10:
                    vec = (c[i] - c[j]) / d
                    grads[i] += weights[idx] * 0.5 * vec
                    grads[j] += weights[idx] * (-0.5) * vec
                idx += 1
        # Wall gradients
        min_wall_idx = np.argmin(np.abs(c), axis=1)
        for i in range(N):
            k = min_wall_idx[i]
            grads[i, k] += weights[idx] * (-np.sign(c[i, k]))
            idx += 1

        # Gradient of -softmin = -sum(w_k * grad(val_k))
        # But we want grad of objective = -softmin, so:
        # d(-softmin)/dx = -d(softmin)/dx = -sum(w_k * d(val_k)/dx)
        # Wait, softmin derivative: d(softmin)/dx_k = w_k (as derived)
        # So d(-softmin)/dx = -sum(w_k * d(val_k)/dx)
        # But grads already contains d(val_k)/dx contributions weighted by w_k
        # So the total gradient is -sum(w_k * d(val_k)/dx) = -grads
        return (-grads).flatten()

    def smooth_polish(c, beta, time_budget=10.0):
        """Polish with L-BFGS-B on the soft-min objective."""
        t0 = time.time()
        x0 = c.flatten()
        try:
            res = minimize(
                smooth_obj, x0, args=(beta,),
                method='L-BFGS-B', jac=smooth_grad,
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res.fun < smooth_obj(x0, beta):
                new_c = np.clip(res.x.reshape(N, 3), -0.5, 0.5)
                return new_c
        except Exception:
            pass
        return c

    def repulsion_anneal(c, time_budget=20.0):
        """Anneal with a repulsive force to spread spheres apart."""
        t0 = time.time()
        centers = c.copy()
        best_r = compute_radius(centers)
        best_c = centers.copy()
        T = 0.05  # initial temperature
        cooling = 0.97
        n_iter = 0
        while time.time() - t0 < time_budget and T > 1e-4:
            # Compute repulsive forces
            forces = np.zeros_like(centers)
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dists, np.inf)
            # Force proportional to 1/d^2 for close pairs
            for i in range(N):
                for j in range(N):
                    if i == j:
                        continue
                    d = dists[i, j]
                    if d < 0.3:
                        f = (0.3 / d)**2 * (centers[i] - centers[j]) / d
                        forces[i] += f
            # Wall repulsion
            for i in range(N):
                for k in range(3):
                    if centers[i, k] > 0.4:
                        forces[i, k] -= (centers[i, k] - 0.4) * 10
                    elif centers[i, k] < -0.4:
                        forces[i, k] -= (centers[i, k] + 0.4) * 10
            # Add random perturbation
            noise = np.random.normal(0, T, centers.shape)
            centers += 0.01 * forces + noise
            centers = np.clip(centers, -0.5, 0.5)
            r = compute_radius(centers)
            if r > best_r:
                best_r = r
                best_c = centers.copy()
            T *= cooling
            n_iter += 1
        return best_c, best_r

    def metropolis_mc(c, time_budget=15.0):
        """Metropolis Monte Carlo to explore configurations."""
        t0 = time.time()
        centers = c.copy()
        best_r = compute_radius(centers)
        best_c = centers.copy()
        T = 0.02
        cooling = 0.98
        step_size = 0.03
        while time.time() - t0 < time_budget and T > 1e-4:
            # Pick a random sphere and move it
            i = np.random.randint(N)
            old_pos = centers[i].copy()
            centers[i] += np.random.normal(0, step_size, 3)
            centers[i] = np.clip(centers[i], -0.5, 0.5)
            new_r = compute_radius(centers)
            delta = new_r - best_r  # simplified acceptance
            if new_r >= best_r or np.random.rand() < np.exp((new_r - best_r) / T):
                best_r = max(best_r, new_r)
                best_c = centers.copy()
            else:
                centers[i] = old_pos
            T *= cooling
        return best_c, best_r

    def local_search(c, time_budget=10.0):
        """Local search: try small moves on each sphere."""
        t0 = time.time()
        centers = c.copy()
        best_r = compute_radius(centers)
        best_c = centers.copy()
        step_sizes = [0.05, 0.02, 0.01, 0.005]
        for ss in step_sizes:
            while time.time() - t0 < time_budget:
                improved = False
                for i in range(N):
                    for axis in range(3):
                        for sign in [1, -1]:
                            old_val = centers[i, axis]
                            centers[i, axis] = np.clip(old_val + sign * ss, -0.5, 0.5)
                            r = compute_radius(centers)
                            if r > best_r:
                                best_r = r
                                best_c = centers.copy()
                                improved = True
                            else:
                                centers[i, axis] = old_val
                if not improved:
                    break
        return best_c, best_r

    def seed_fcc():
        """FCC-like initialization."""
        # FCC unit cell has atoms at corners and face centers
        # Scale to fit in [-0.5, 0.5]^3
        # For 50 spheres, use a 3x3x3 FCC supercell subset
        pts = []
        for ix in range(3):
            for iy in range(3):
                for iz in range(3):
                    # Corner
                    pts.append([ix * 0.333 - 0.5, iy * 0.333 - 0.5, iz * 0.333 - 0.5])
                    # Face centers
                    pts.append([ix * 0.333 - 0.5 + 0.167, iy * 0.333 - 0.5 + 0.167, iz * 0.333 - 0.5])
                    pts.append([ix * 0.333 - 0.5 + 0.167, iy * 0.333 - 0.5, iz * 0.333 - 0.5 + 0.167])
                    pts.append([ix * 0.333 - 0.5, iy * 0.333 - 0.5 + 0.167, iz * 0.333 - 0.5 + 0.167])
        pts = np.array(pts)
        # Filter to cube and take N
        mask = np.all(np.abs(pts) <= 0.5, axis=1)
        pts = pts[mask]
        if len(pts) >= N:
            np.random.seed(42)
            idx = np.random.choice(len(pts), N, replace=False)
            return pts[idx].copy()
        else:
            # Pad with random
            remaining = N - len(pts)
            rand_pts = np.random.uniform(-0.5, 0.5, (remaining, 3))
            return np.vstack([pts, rand_pts])

    def seed_grid():
        """Jittered grid initialization."""
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        np.random.seed(42)
        indices = np.random.choice(len(grid), N, replace=False)
        centers = grid[indices].copy()
        centers += np.random.uniform(-0.01, 0.01, centers.shape)
        return np.clip(centers, -0.5, 0.5)

    def seed_random():
        """Random initialization."""
        np.random.seed(42)
        return np.random.uniform(-0.45, 0.45, (N, 3))

    # --- Main optimization pipeline ---

    best_radius = -1.0
    best_centers = None

    # Phase 1: Try multiple seeds
    seeds = [seed_fcc(), seed_grid(), seed_random()]
    for seed in seeds:
        r = compute_radius(seed)
        if r > best_radius:
            best_radius = r
            best_centers = seed.copy()

    # Phase 2: Lloyd's algorithm on best seed
    if time.time() < deadline - 60.0:
        best_centers, best_radius = lloyd_optimize(best_centers, time_budget=20.0)

    # Phase 3: Repulsion annealing
    if time.time() < deadline - 40.0:
        c, r = repulsion_anneal(best_centers, time_budget=15.0)
        if r > best_radius:
            best_radius = r
            best_centers = c

    # Phase 4: Lloyd's again
    if time.time() < deadline - 30.0:
        best_centers, best_radius = lloyd_optimize(best_centers, time_budget=10.0)

    # Phase 5: Smooth polish (soft)
    if time.time() < deadline - 20.0:
        best_centers = smooth_polish(best_centers, beta=10.0, time_budget=8.0)
        r = compute_radius(best_centers)
        if r > best_radius:
            best_radius = r

    # Phase 6: Smooth polish (sharp)
    if time.time() < deadline - 10.0:
        best_centers = smooth_polish(best_centers, beta=50.0, time_budget=5.0)
        r = compute_radius(best_centers)
        if r > best_radius:
            best_radius = r

    # Phase 7: Local search
    if time.time() < deadline - 5.0:
        best_centers, best_radius = local_search(best_centers, time_budget=5.0)

    # Phase 8: Final sharp polish
    if time.time() < deadline - 2.0:
        best_centers = smooth_polish(best_centers, beta=100.0, time_budget=3.0)
        r = compute_radius(best_centers)
        if r > best_radius:
            best_radius = r
            best_centers = np.clip(best_centers, -0.5, 0.5)

    return best_centers
