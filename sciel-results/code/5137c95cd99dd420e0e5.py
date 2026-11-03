import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import time

N = 50

# EVOLVE-BLOCK-START
def construct():
    # Best known packing for N=50 has radius ~0.1258254895304
    # Structure: 1 center, then rings of 6, 12, 12, 19
    # We optimize radial distances AND angular offsets for each ring.
    # The ring model is structurally sound; the improvement here comes from
    # a *differentiable* soft-min surrogate + L-BFGS-B, which converges much
    # faster than the previous Nelder-Mead/Powell search, freeing budget for
    # a multi-restart ring search over radii and angular offsets.

    ring_sizes = [6, 12, 12, 19]  # 4 rings plus center

    def get_centers(params):
        """
        params: [r1, r2, r3, r4, a1, a2, a3, a4]
        where r_i are radial distances and a_i are angular offsets
        """
        centers = [[0.0, 0.0]]
        for i in range(4):
            r = params[i]
            offset = params[i + 4]
            n = ring_sizes[i]
            angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + offset
            xs = r * np.cos(angles)
            ys = r * np.sin(angles)
            centers.extend(zip(xs, ys))
        return np.array(centers)

    def compute_radius(centers):
        """Compute the maximum radius for given centers (exact, double precision)."""
        n = len(centers)
        # Boundary constraint
        dist_to_center = np.linalg.norm(centers, axis=1)
        max_dist = np.max(dist_to_center)
        if max_dist >= 1.0:
            return 0.0

        # Minimum distance between any two centers
        dists = cdist(centers, centers)
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)

        return min(min_dist / 2.0, 1.0 - max_dist)

    def soft_min_radius(centers, beta):
        """
        Smooth (differentiable) surrogate for the exact radius, via soft-min
        over pairwise distances/2 and boundary distances (1 - norm).
        As beta grows the surrogate approaches the exact min.
        """
        n = len(centers)
        dists = cdist(centers, centers)
        np.fill_diagonal(dists, np.inf)
        pair_vals = dists / 2.0

        norms = np.linalg.norm(centers, axis=1)
        boundary_vals = 1.0 - norms

        all_vals = np.concatenate([pair_vals.ravel(), boundary_vals])
        all_vals = all_vals[np.isfinite(all_vals)]
        if all_vals.size == 0:
            return 0.0

        m = np.min(all_vals)
        exp_vals = np.exp(-beta * (all_vals - m))
        return m - np.log(np.sum(exp_vals)) / beta

    def objective(params, beta):
        """Negative soft-min radius + hard boundary penalty (differentiable)."""
        radii = np.clip(np.abs(params[:4]), 0.05, 0.95)
        params_clipped = np.concatenate([radii, params[4:]])
        centers = get_centers(params_clipped)

        # Hard penalty for points outside the unit circle (drives them back in)
        norms = np.linalg.norm(centers, axis=1)
        violation = np.maximum(0.0, norms - 1.0)
        penalty = 100.0 * np.sum(violation)

        r = soft_min_radius(centers, beta)
        return -r + penalty

    # Initial guess based on known good values (r_circle ~ 0.1258)
    initial_params = np.array([0.252, 0.45, 0.65, 0.85, 0.0, np.pi / 12, np.pi / 6, 0.0])

    start_time = time.time()
    deadline = start_time + 145.0  # explicit wall-clock guard (150 s limit)

    def time_left():
        return deadline - time.time()

    best_params = initial_params.copy()
    best_centers = get_centers(best_params)
    best_score = compute_radius(best_centers)

    def polish_from(params):
        """
        Refine a candidate with the smooth surrogate (increasing beta) then a
        final exact-objective pass. Returns (centers, score) if improved.
        """
        nonlocal best_params, best_centers, best_score
        current = params.copy()
        cur_score = compute_radius(get_centers(current))

        # 1) Smooth surrogate polish with increasing sharpness
        for beta in [20.0, 60.0, 150.0, 400.0]:
            if time_left() < 3.0:
                break
            try:
                res = minimize(
                    lambda p: objective(p, beta),
                    current,
                    method='L-BFGS-B',
                    options={'maxiter': 400, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 50},
                )
                cand = res.x.copy()
                cand[:4] = np.clip(np.abs(cand[:4]), 0.05, 0.95)
                c = get_centers(cand)
                if np.all(np.isfinite(c)) and np.max(np.linalg.norm(c, axis=1)) < 1.0:
                    s = compute_radius(c)
                    if s > cur_score:
                        current, cur_score = cand, s
            except Exception:
                pass

        # 2) Exact-objective polish (penalty method) for a final squeeze
        if time_left() > 2.0:
            for pw in [50.0, 500.0]:
                if time_left() < 2.0:
                    break
                try:
                    def exact_obj(p):
                        c = get_centers(p)
                        norms = np.linalg.norm(c, axis=1)
                        maxd = np.max(norms)
                        if maxd >= 1.0:
                            return 1.0 + 10.0 * (maxd - 1.0)
                        dists = cdist(c, c)
                        np.fill_diagonal(dists, np.inf)
                        r = min(np.min(dists) / 2.0, 1.0 - maxd)
                        return -r + pw * np.sum(np.maximum(0.0, norms - 1.0)) ** 2

                    res = minimize(exact_obj, current, method='L-BFGS-B',
                                   options={'maxiter': 300, 'ftol': 1e-16, 'gtol': 1e-14})
                    cand = res.x.copy()
                    cand[:4] = np.clip(np.abs(cand[:4]), 0.05, 0.95)
                    c = get_centers(cand)
                    if np.all(np.isfinite(c)) and np.max(np.linalg.norm(c, axis=1)) < 1.0:
                        s = compute_radius(c)
                        if s > cur_score:
                            current, cur_score = cand, s
                except Exception:
                    pass

        if cur_score > best_score:
            best_params, best_centers, best_score = current, get_centers(current), cur_score
        return best_centers, best_score

    # Multi-restart ring search: perturb radii and angular offsets.
    np.random.seed(1234)
    inits = [initial_params.copy()]
    for _ in range(20):
        p = initial_params.copy()
        p[:4] += np.random.randn(4) * 0.05
        p[4:] += np.random.randn(4) * (np.pi / 6.0)
        p[:4] = np.clip(np.abs(p[:4]), 0.05, 0.95)
        inits.append(p)

    for init in inits:
        if time_left() < 6.0:
            break
        polish_from(init)

    # Final high-precision polish of the best found so far
    if time_left() > 4.0:
        polish_from(best_params)

    # Final validation
    if best_centers.shape != (N, 2) or not np.all(np.isfinite(best_centers)) \
            or np.max(np.linalg.norm(best_centers, axis=1)) >= 1.0:
        return _fallback_hexagonal()

    return best_centers

def _fallback_hexagonal():
    """Simple hexagonal packing as a safe fallback."""
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        mask = radius <= 1.0 - spacing / 2.0
        points = points[mask]
        if len(points) >= N:
            return points[np.argsort(radius, kind="stable")[:N]]
        spacing *= 0.98

# EVOLVE-BLOCK-END
