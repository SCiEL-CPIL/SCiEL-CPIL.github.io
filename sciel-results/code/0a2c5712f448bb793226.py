"""Ring-parametric packing: DE over ring radii/angles, then L-BFGS polish in full 2D."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50
DEADLINE = 115.0  # safety margin before 120s kill


def _score(pts):
    """Return the common radius (min pair-dist/2, min (1-|c|))."""
    if pts.shape != (N, 2) or not np.all(np.isfinite(pts)):
        return -1.0
    # pairwise distances
    d = np.sqrt(np.sum((pts[:, None, :] - pts[None, :, :]) ** 2, axis=2))
    np.fill_diagonal(d, np.inf)
    min_pair = d.min() / 2.0
    min_boundary = 1.0 - np.max(np.hypot(pts[:, 0], pts[:, 1]))
    return float(min(min_pair, min_boundary))


def _ring_model(radii, angles_per_ring, n_center=1):
    """Build N points from ring radii and per-ring angle offsets.
    radii: array of length = len(angles_per_ring)
    angles_per_ring: list of ints, circles per ring
    Returns (N,2) array.
    """
    pts = []
    idx = 0
    if n_center > 0:
        pts.append([0.0, 0.0])
        idx += 1
    for r, n in zip(radii, angles_per_ring):
        for k in range(n):
            theta = 2.0 * np.pi * k / n
            pts.append([r * np.cos(theta), r * np.sin(theta)])
            idx += 1
    return np.array(pts[:N])


def _score_from_params(radii, angles_per_ring, n_center=1):
    pts = _ring_model(radii, angles_per_ring, n_center)
    return _score(pts)


def construct():
    t0 = time.time()
    best_pts = None
    best_score = -1.0

    # ---- Phase 1: DE over ring topology ----
    # Try several ring topologies: (n_center, [ring sizes summing to N - n_center])
    # Best-known 50-circle packing: 1 center + rings of 6, 12, 18, 13 (approx)
    # We'll let DE optimize radii and small angular offsets.

    topologies = []
    # 1 center + rings
    ring_splits = [
        [6, 12, 18, 13],
        [6, 12, 19, 12],
        [6, 13, 18, 12],
        [6, 12, 17, 14],
        [6, 14, 18, 11],
        [6, 12, 18, 13],
        [5, 12, 19, 13],
        [6, 11, 19, 13],
        [6, 12, 16, 15],
        [6, 12, 18, 13],
    ]
    for split in ring_splits:
        if sum(split) == N - 1:
            topologies.append((1, split))

    # Also try 0-center with more rings (less likely optimal but cheap to try)
    ring_splits0 = [
        [12, 18, 20],
        [13, 18, 19],
        [12, 19, 19],
        [14, 18, 18],
        [12, 18, 20],
    ]
    for split in ring_splits0:
        if sum(split) == N:
            topologies.append((0, split))

    # DE parameters
    de_pop = 40
    de_iter_max = 300
    F = 0.8
    CR = 0.9

    for n_center, split in topologies:
        if time.time() - t0 > DEADLINE - 30:
            break
        n_rings = len(split)
        # Parameters: radii (n_rings) + angular offset per ring (n_rings) = 2*n_rings
        # Plus maybe a global rotation per ring? Keep simple: radii + phase per ring.
        n_params = 2 * n_rings
        lo = np.zeros(n_params)
        hi = np.array([0.9] * n_rings + [1.0] * n_rings)  # radii in [0,0.9], phase in [0,1)

        # Initialize population
        pop = np.random.uniform(lo, hi, size=(de_pop, n_params))
        # Evaluate
        def eval_pop(x):
            radii = x[:n_rings]
            phases = x[n_rings:]
            # Build points with phase offset per ring
            pts = []
            if n_center > 0:
                pts.append([0.0, 0.0])
            for ri, n in enumerate(split):
                r = radii[ri]
                phase = phases[ri] * 2.0 * np.pi
                for k in range(n):
                    theta = 2.0 * np.pi * k / n + phase
                    pts.append([r * np.cos(theta), r * np.sin(theta)])
            pts = np.array(pts[:N])
            return _score(pts)

        fitness = np.array([eval_pop(p) for p in pop])
        best_idx = np.argmax(fitness)
        best_fit = fitness[best_idx]
        best_x = pop[best_idx].copy()

        # DE loop
        for it in range(de_iter_max):
            if time.time() - t0 > DEADLINE - 25:
                break
            for i in range(de_pop):
                # Pick 3 distinct random indices
                a, b, c = np.random.choice(de_pop, 3, replace=False)
                mutant = pop[a] + F * (pop[b] - pop[c])
                mutant = np.clip(mutant, lo, hi)
                trial = np.empty(n_params)
                for j in range(n_params):
                    if np.random.random() < CR or j == np.random.randint(n_params):
                        trial[j] = mutant[j]
                    else:
                        trial[j] = pop[i][j]
                trial_fit = eval_pop(trial)
                if trial_fit >= fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fit
                    if trial_fit > best_fit:
                        best_fit = trial_fit
                        best_x = trial.copy()
                        best_idx = i

        # Record best for this topology
        if best_fit > best_score:
            best_score = best_fit
            radii = best_x[:n_rings]
            phases = best_x[n_rings:]
            pts = []
            if n_center > 0:
                pts.append([0.0, 0.0])
            for ri, n in enumerate(split):
                r = radii[ri]
                phase = phases[ri] * 2.0 * np.pi
                for k in range(n):
                    theta = 2.0 * np.pi * k / n + phase
                    pts.append([r * np.cos(theta), r * np.sin(theta)])
            best_pts = np.array(pts[:N])

    # ---- Phase 2: L-BFGS polish in full 2D ----
    if best_pts is None:
        # Fallback: hexagonal lattice
        spacing = 2.0 / (np.sqrt(N) + 1.0)
        while True:
            k = int(1.0 / spacing) + 2
            rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                    for i in range(-k, k + 1) for j in range(-k, k + 1)]
            points = np.array(rows)
            radius = np.hypot(points[:, 0], points[:, 1])
            points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
            if len(points) >= N:
                best_pts = points[np.argsort(radius, kind="stable")[:N]]
                break
            spacing *= 0.98

    # L-BFGS polish: maximize r = min(min_pair/2, min(1-|c|))
    # Use smooth surrogate: soft-min via log-sum-exp
    def _neg_soft_score(pts_flat, beta=50.0):
        pts = pts_flat.reshape(N, 2)
        # Pairwise distances
        d = np.sqrt(np.sum((pts[:, None, :] - pts[None, :, :]) ** 2, axis=2))
        np.fill_diagonal(d, np.inf)
        # Boundary distances
        bd = 1.0 - np.hypot(pts[:, 0], pts[:, 1])
        # All "constraint values": pair/2 and boundary
        vals = np.concatenate([d[d < np.inf] / 2.0, bd])
        if vals.min() < 0:
            return -vals.min() * 10.0 + 1.0  # penalty for infeasible
        # soft-min: -log(sum(exp(-beta*v)))/beta
        m = vals.min()
        soft_min = m - np.log(np.sum(np.exp(-beta * (vals - m)))) / beta
        return -soft_min

    # Gradient of soft-min (approximate via finite differences is too slow; use analytic)
    def _neg_soft_score_grad(pts_flat, beta=50.0):
        pts = pts_flat.reshape(N, 2)
        d = np.sqrt(np.sum((pts[:, None, :] - pts[None, :, :]) ** 2, axis=2))
        np.fill_diagonal(d, np.inf)
        bd = 1.0 - np.hypot(pts[:, 0], pts[:, 1])
        vals = np.concatenate([d[d < np.inf] / 2.0, bd])
        m = vals.min()
        if m < 0:
            # Infeasible: push out
            grad = np.zeros_like(pts_flat)
            for i in range(N):
                for j in range(i + 1, N):
                    diff = pts[i] - pts[j]
                    dist = np.linalg.norm(diff)
                    if dist > 1e-12:
                        grad[i] += diff / dist
                        grad[j] -= diff / dist
                r = np.linalg.norm(pts[i])
                if r > 1e-12:
                    grad[i] -= pts[i] / r
            return 10.0 * grad
        exps = np.exp(-beta * (vals - m))
        weights = exps / exps.sum()
        # Soft-min gradient w.r.t. each val: -weights[k]
        # d(soft_min)/d(pts) = -sum_k weights[k] * d(vals[k])/d(pts)
        grad = np.zeros_like(pts_flat)
        # Pair contributions
        pair_idx = 0
        for i in range(N):
            for j in range(i + 1, N):
                w = weights[pair_idx]
                diff = pts[i] - pts[j]
                dist = np.linalg.norm(diff)
                if dist > 1e-12:
                    # d(val)/d(pts[i]) = diff/dist, d(val)/d(pts[j]) = -diff/dist
                    grad[i] += w * diff / dist
                    grad[j] -= w * diff / dist
                pair_idx += 1
        # Boundary contributions
        for i in range(N):
            w = weights[pair_idx]
            r = np.linalg.norm(pts[i])
            if r > 1e-12:
                # d(val)/d(pts[i]) = -pts[i]/r
                grad[i] += w * (-pts[i] / r)
            pair_idx += 1
        return grad

    # Run L-BFGS
    if time.time() - t0 < DEADLINE - 10:
        x0 = best_pts.flatten()
        result = minimize(_neg_soft_score, x0, method='L-BFGS-B',
                          jac=_neg_soft_score_grad,
                          options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50})
        if result.success or result.fun < _neg_soft_score(x0):
            new_pts = result.x.reshape(N, 2)
            new_score = _score(new_pts)
            if new_score > best_score:
                best_pts = new_pts
                best_score = new_score

    # Second polish with higher beta for sharper min
    if time.time() - t0 < DEADLINE - 5:
        def _neg_hard_score(pts_flat):
            pts = pts_flat.reshape(N, 2)
            d = np.sqrt(np.sum((pts[:, None, :] - pts[None, :, :]) ** 2, axis=2))
            np.fill_diagonal(d, np.inf)
            bd = 1.0 - np.hypot(pts[:, 0], pts[:, 1])
            vals = np.concatenate([d[d < np.inf] / 2.0, bd])
            return -vals.min()

        x0 = best_pts.flatten()
        result2 = minimize(_neg_hard_score, x0, method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14})
        if result2.fun < _neg_hard_score(x0):
            new_pts = result2.x.reshape(N, 2)
            new_score = _score(new_pts)
            if new_score > best_score:
                best_pts = new_pts
                best_score = new_score

    return best_pts
