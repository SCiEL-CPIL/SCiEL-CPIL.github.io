"""Circle packing in a unit square - 200 equal circles.

Strategy: multi-start hex-grid initialization -> fast gradient
ascent on the minimum-distance objective -> simulated annealing
with coordinate-descent repair -> final gradient polish.
"""
import numpy as np
import time
import math

N = 200


def construct():
    t0 = time.time()
    deadline = t0 + 145.0  # safety margin before 150s kill

    # ---------- helpers ----------
    def compute_radius(c):
        """Return the largest feasible radius for centers c (N,2)."""
        wall = 0.5 - np.abs(c)
        min_wall = float(np.min(wall))
        diff = c[:, None, :] - c[None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dist, np.inf)
        min_pair = float(np.min(dist))
        return max(0.0, min(min_wall, min_pair * 0.5))

    def min_info(c):
        """Return (r, kind, idx) where kind is 'wall' or 'pair'."""
        wall = 0.5 - np.abs(c)
        min_wall = float(np.min(wall))
        wall_idx = int(np.argmin(wall))
        i, j = divmod(wall_idx, 2)
        diff = c[:, None, :] - c[None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dist, np.inf)
        min_pair = float(np.min(dist))
        pair_idx = int(np.argmin(dist))
        pi, pj = divmod(pair_idx, N)
        if min_wall <= min_pair * 0.5:
            return min_wall, 'wall', (i, j)
        else:
            return min_pair * 0.5, 'pair', (pi, pj)

    def gradient_ascent(c, steps, lr0, lr_end, time_limit, start_t):
        """Simple gradient ascent on min distance with adaptive step size."""
        c = c.copy()
        r = compute_radius(c)
        for s in range(steps):
            if time.time() > start_t + time_limit:
                break
            # compute gradient of min distance
            wall = 0.5 - np.abs(c)
            min_wall = np.min(wall)
            diff = c[:, None, :] - c[None, :, :]
            dist = np.sqrt(np.sum(diff * diff, axis=2))
            np.fill_diagonal(dist, np.inf)
            min_pair = np.min(dist)
            r_cur = min(min_wall, min_pair * 0.5)

            if min_wall <= min_pair * 0.5:
                # wall is bottleneck
                grad = np.zeros_like(c)
                idx = int(np.argmin(wall))
                i, j = divmod(idx, 2)
                # push away from wall: gradient of wall_dist wrt c
                # wall_dist = 0.5 - |c_ik|
                # d/dc_ik = -sign(c_ik)
                # we want to maximize, so move in direction of +grad
                grad[i, j] = -np.sign(c[i, j])
                # also push slightly from other close points
            else:
                # pair is bottleneck
                grad = np.zeros_like(c)
                idx = int(np.argmin(dist))
                pi, pj = divmod(idx, N)
                d = dist[pi, pj]
                if d < 1e-12:
                    break
                direction = (c[pi] - c[pj]) / d
                grad[pi] += direction
                grad[pj] -= direction

            # normalize gradient
            gnorm = np.linalg.norm(grad)
            if gnorm < 1e-15:
                break
            grad = grad / gnorm

            # adaptive learning rate
            alpha = 0.5 * (lr0 + lr_end) + 0.5 * (lr0 - lr_end) * math.cos(
                math.pi * s / max(1, steps)
            )
            step = alpha / (1.0 + s * 0.01)
            new_c = c + step * grad
            new_c = np.clip(new_c, -0.4999, 0.4999)

            r_new = compute_radius(new_c)
            if r_new > r:
                c = new_c
                r = r_new
            # if not improved, reduce step
            else:
                step *= 0.5
                new_c = c + step * grad
                new_c = np.clip(new_c, -0.4999, 0.4999)
                r_new = compute_radius(new_c)
                if r_new > r:
                    c = new_c
                    r = r_new
                else:
                    break  # stuck
        return c, r

    def hex_grid(n_rows, n_cols, d_scale=1.0):
        d = d_scale / (n_cols + 0.5)
        row_spacing = d * math.sqrt(3) / 2
        total_height = (n_rows - 1) * row_spacing + d
        if total_height > 1.0 + 1e-6:
            return None
        pts = []
        for row in range(n_rows):
            y = -0.5 + d / 2 + row * row_spacing
            if row % 2 == 0:
                n_in_row = n_cols
                x_start = -0.5 + d / 2
            else:
                n_in_row = n_cols - 1
                x_start = -0.5 + d
            for col in range(n_in_row):
                x = x_start + col * d
                pts.append([x, y])
        pts = np.array(pts)
        if len(pts) >= N:
            return pts[:N]
        return None

    def square_grid(m):
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]

    # ---------- main search ----------
    best_centers = None
    best_radius = 0.0

    # 1) Generate diverse initial configurations
    inits = []
    for n_rows in range(13, 18):
        n_cols = math.ceil(N / n_rows)
        for d_scale in [0.92, 0.97, 1.0, 1.03, 1.06]:
            pts = hex_grid(n_rows, n_cols, d_scale)
            if pts is not None:
                inits.append(pts)
    m = int(math.ceil(math.sqrt(N)))
    inits.append(square_grid(m))

    # Also add a few randomly perturbed hex grids
    rng = np.random.default_rng(42)
    for base in inits[:3]:
        perturbed = base + rng.normal(0, 0.005, (N, 2))
        perturbed = np.clip(perturbed, -0.499, 0.499)
        inits.append(perturbed)

    # 2) Evaluate all initializations, keep top few
    scored = []
    for pts in inits:
        r = compute_radius(pts)
        scored.append((r, pts))
    scored.sort(key=lambda x: -x[0])
    top_inits = [pts for _, pts in scored[:min(6, len(scored))]]

    if not top_inits:
        top_inits = [square_grid(m)]

    # 3) Gradient ascent on each top initialization
    for i, init in enumerate(top_inits):
        if time.time() > deadline - 20:
            break
        r_init = compute_radius(init)
        if r_init > best_radius:
            best_radius = r_init
            best_centers = init.copy()
        # gradient ascent
        c, r = gradient_ascent(
            init, steps=500, lr0=0.03, lr_end=0.002,
            time_limit=min(20.0, deadline - time.time() - 5),
            start_t=time.time()
        )
        if r > best_radius:
            best_radius = r
            best_centers = c.copy()

    # 4) Simulated annealing perturbation on best
    if time.time() < deadline - 10 and best_centers is not None:
        c = best_centers.copy()
        r = best_radius
        T0 = 0.01
        T_end = 0.0001
        n_sa = 8000
        for s in range(n_sa):
            if time.time() > deadline - 3:
                break
            T = T0 * (T_end / T0) ** (s / n_sa)
            # random perturbation: pick 3-5 random points, nudge them
            n_pert = rng.integers(3, 6)
            idxs = rng.choice(N, size=n_pert, replace=False)
            old_positions = c[idxs].copy()
            c[idxs] += rng.normal(0, T, (n_pert, 2))
            c[idxs] = np.clip(c[idxs], -0.4999, 0.4999)

            r_new = compute_radius(c)
            if r_new > r:
                r = r_new
                if r > best_radius:
                    best_radius = r
                    best_centers = c.copy()
            else:
                # accept with probability exp(-(r - r_new)/T)
                delta_r = r - r_new
                if delta_r < 1e-12:
                    pass  # accept (tie)
                elif rng.random() < math.exp(-delta_r / max(T, 1e-10)):
                    pass  # accept
                else:
                    c[idxs] = old_positions  # reject

    # 5) Final gradient polish
    if best_centers is not None and time.time() < deadline - 2:
        c, r = gradient_ascent(
            best_centers, steps=300, lr0=0.01, lr_end=0.001,
            time_limit=max(1.0, deadline - time.time() - 1),
            start_t=time.time()
        )
        if r > best_radius:
            best_radius = r
            best_centers = c.copy()

    if best_centers is None:
        best_centers = square_grid(m)

    return best_centers


if __name__ == "__main__":
    result = construct()
    print(f"Shape: {result.shape}")
    print(f"Radius: {compute_radius(result)}")
