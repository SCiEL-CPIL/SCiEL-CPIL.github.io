"""Packing 50 equal circles in a unit circle.

Structural approach:
- Ring-parameterized representation (radii + angles) for global search via differential evolution.
- Multi-start with hexagonal and ring initializations.
- Progressive beta annealing on full coordinates (soft-min objective).
- Constraint-push projection polish and binary-search radius refinement.
"""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 50
DEADLINE = 145.0  # safety margin before 150s kill


def _pairwise_distances(pts):
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=-1))


def _score(pts):
    if pts is None or len(pts) != N:
        return -1.0
    pts = np.asarray(pts, dtype=float)
    if not np.all(np.isfinite(pts)):
        return -1.0
    d = _pairwise_distances(pts)
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d) / 2.0
    min_boundary = np.min(1.0 - np.sqrt(np.sum(pts ** 2, axis=1)))
    val = min(min_pair, min_boundary)
    if val <= 0:
        return -1.0
    return val


def _softmin_logsumexp(values, beta):
    if len(values) == 0:
        return 0.0
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return 0.0
    neg = -values * beta
    neg_max = np.max(neg)
    if not np.isfinite(neg_max):
        return 0.0
    exp_sum = np.sum(np.exp(neg - neg_max))
    if exp_sum <= 0.0:
        return 0.0
    return -(neg_max + np.log(exp_sum)) / beta


def _smooth_objective(x, beta):
    pts = x.reshape(N, 2)
    if not np.all(np.isfinite(pts)):
        return 1e10
    d = _pairwise_distances(pts)
    np.fill_diagonal(d, np.inf)
    boundary = 1.0 - np.sqrt(np.sum(pts ** 2, axis=1))
    pair_half = d / 2.0
    all_vals = np.concatenate([pair_half.ravel(), boundary])
    all_vals = all_vals[np.isfinite(all_vals)]
    if len(all_vals) == 0:
        return 1e10
    return -_softmin_logsumexp(all_vals, beta)


def _ring_to_points(radii, angles_list):
    """Convert ring parameters (radii, list of angle arrays) to point coordinates."""
    pts = [np.array([0.0, 0.0])]  # Center
    for i in range(1, len(radii)):
        r = radii[i]
        a = angles_list[i]
        pts.append(np.column_stack([r * np.cos(a), r * np.sin(a)]))
    return np.vstack(pts)


def _hexagonal_init(scale=0.12):
    """Generate hexagonal lattice points and filter to fit in unit disk."""
    points = []
    for i in range(-10, 11):
        for j in range(-10, 11):
            x = scale * (i + 0.5 * j)
            y = scale * (np.sqrt(3) / 2 * j)
            if x ** 2 + y ** 2 <= 0.81:
                points.append([x, y])
    points = np.array(points)
    if len(points) == 0:
        return None
    dists = np.sqrt(np.sum(points ** 2, axis=1))
    idx = np.argsort(dists)
    points = points[idx[:N]]
    return points


def _anneal_points(pts, deadline, max_betas=12):
    """Multi-stage annealing optimization on full point coordinates."""
    current = pts.copy()
    current_score = _score(current)
    betas = [3.0, 8.0, 20.0, 50.0, 120.0, 300.0, 700.0, 1500.0, 3000.0, 6000.0, 12000.0, 25000.0]
    betas = betas[:max_betas]
    for beta in betas:
        if time.time() > deadline - 5:
            break
        try:
            res = minimize(
                _smooth_objective,
                current.ravel(),
                args=(beta,),
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxfun': 30000}
            )
            new_pts = res.x.reshape(N, 2)
            if not np.all(np.isfinite(new_pts)):
                continue
            s = _score(new_pts)
            if s > current_score:
                current = new_pts
                current_score = s
        except Exception:
            pass
    return current, current_score


def _nelder_mead_polish(pts, deadline, beta=200.0, max_iter=5000):
    """Derivative-free local polish using Nelder-Mead on the smooth objective.
    This can escape local minima that L-BFGS-B gets stuck in due to
    switching active constraint sets."""
    current = pts.copy()
    current_score = _score(current)
    if time.time() > deadline - 3:
        return current, current_score
    try:
        res = minimize(
            _smooth_objective,
            current.ravel(),
            args=(beta,),
            method='Nelder-Mead',
            options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        new_pts = res.x.reshape(N, 2)
        if not np.all(np.isfinite(new_pts)):
            return current, current_score
        s = _score(new_pts)
        if s > current_score:
            return new_pts, s
    except Exception:
        pass
    return current, current_score


def _constraint_push(pts, r_target, deadline, max_iters=500):
    """Polish solution by iteratively pushing violating constraints."""
    pts = pts.copy()
    for _ in range(max_iters):
        if time.time() > deadline - 3:
            break
        d = _pairwise_distances(pts)
        np.fill_diagonal(d, np.inf)
        pair_margins = d / 2.0
        boundary_margins = 1.0 - np.sqrt(np.sum(pts ** 2, axis=1))
        min_pair = np.min(pair_margins)
        min_bound = np.min(boundary_margins)
        if min_pair < min_bound - 1e-14:
            i, j = np.unravel_index(np.argmin(pair_margins), pair_margins.shape)
            if i == j:
                break
            di = pts[i] - pts[j]
            dist = np.linalg.norm(di)
            if dist < 1e-15:
                break
            di /= dist
            push = (2.0 * r_target - dist) / 2.0
            if push > 1e-15:
                pts[i] += push * di
                pts[j] -= push * di
        else:
            i = int(np.argmin(boundary_margins))
            pi = pts[i]
            dist = np.linalg.norm(pi)
            if dist < 1e-15:
                pts[i] += 1e-4 * np.array([1.0, 0.0])
            else:
                di = pi / dist
                push = (1.0 - r_target - dist)
                if push > 1e-15:
                    pts[i] += push * di
        d = _pairwise_distances(pts)
        np.fill_diagonal(d, np.inf)
        r = min(np.min(d) / 2.0, np.min(1.0 - np.sqrt(np.sum(pts ** 2, axis=1))))
        if r >= r_target - 1e-12:
            break
    return pts


def _binary_search_polish(pts, deadline):
    """Binary search for maximum feasible radius with projection polish."""
    current = pts.copy()
    low = _score(current) - 1e-8
    high = _score(current) + 1e-4
    for _ in range(50):
        if time.time() > deadline - 2:
            break
        mid = (low + high) / 2.0
        polished = _constraint_push(current, mid, deadline, max_iters=300)
        s = _score(polished)
        if s >= mid - 1e-10:
            low = mid
            current = polished
        else:
            high = mid
    return current, _score(current)


def _de_ring_objective(params, ring_sizes):
    """
    Objective for differential evolution on ring parameters.
    params: [r1, r2, r3, r4, a1_1, ..., a1_6, a2_1, ..., a2_12, ...]
    Returns negative score (to minimize).
    """
    n_rings = len(ring_sizes) - 1  # excluding center
    radii = params[:n_rings]
    angles = params[n_rings:]
    angles_list = [np.array([0.0])]
    idx = 0
    for i in range(1, n_rings + 1):
        count = ring_sizes[i]
        a = angles[idx:idx + count]
        angles_list.append(a)
        idx += count
    pts = _ring_to_points(radii, angles_list)
    if len(pts) != N:
        return 1e10
    s = _score(pts)
    if s < 0:
        return 1e10
    return -s


def _de_ring_search(ring_sizes, deadline):
    """Global search over ring parameters using differential evolution."""
    n_rings = len(ring_sizes) - 1
    n_angles = sum(ring_sizes[1:])
    bounds = [(0.0, 1.0)] * n_rings + [(0, 2 * np.pi)] * n_angles
    # Use a reasonable subset of DE for time
    best = None
    try:
        res = differential_evolution(
            _de_ring_objective,
            bounds,
            args=(ring_sizes,),
            maxiter=300,
            popsize=15,
            tol=1e-10,
            seed=42,
            polish=True,
            init='sobol',
            workers=1
        )
        if res.success or res.fun < 1e9:
            best = res.x
    except Exception:
        pass
    return best


def construct():
    t_start = time.time()
    deadline = t_start + DEADLINE

    best_points = None
    best_score = -1.0

    ring_sizes = [1, 6, 12, 18, 13]

    # ---- Stage 1: Multiple initializations ----
    candidates = []

    # Ring-based initialization
    init_radii = np.array([0.0, 0.25, 0.45, 0.65, 0.85])
    angles_list = []
    for i, n in enumerate(ring_sizes):
        if i == 0:
            angles_list.append(np.array([0.0]))
        else:
            angles_list.append(np.linspace(0, 2 * np.pi, n, endpoint=False))
    init_pts = _ring_to_points(init_radii, angles_list)
    init_score = _score(init_pts)
    if init_score > 0:
        candidates.append((init_pts.copy(), init_score))

    # Hexagonal initialization
    hex_pts = _hexagonal_init(scale=0.12)
    if hex_pts is not None and len(hex_pts) == N:
        hex_score = _score(hex_pts)
        if hex_score > 0:
            candidates.append((hex_pts.copy(), hex_score))

    # Random initialization
    np.random.seed(42)
    rand_pts = np.random.randn(N, 2) * 0.8
    rand_score = _score(rand_pts)
    if rand_score > 0:
        candidates.append((rand_pts.copy(), rand_score))

    # ---- Stage 2: DE ring search ----
    if time.time() < deadline - 60:
        de_params = _de_ring_search(ring_sizes, deadline - 30)
        if de_params is not None:
            n_rings = len(ring_sizes) - 1
            radii = de_params[:n_rings]
            angles = de_params[n_rings:]
            angles_list = [np.array([0.0])]
            idx = 0
            for i in range(1, n_rings + 1):
                count = ring_sizes[i]
                a = angles[idx:idx + count]
                angles_list.append(a)
                idx += count
            de_pts = _ring_to_points(radii, angles_list)
            if len(de_pts) == N:
                de_score = _score(de_pts)
                if de_score > 0:
                    candidates.append((de_pts.copy(), de_score))

    # ---- Stage 3: Anneal each candidate ----
    for init_pts, init_score in candidates:
        if time.time() > deadline - 20:
            break
        remaining = deadline - time.time() - 10
        if remaining < 10:
            break
        c_opt, r_opt = _anneal_points(init_pts, deadline - 10, max_betas=10)
        if r_opt > best_score:
            best_score = r_opt
            best_points = c_opt.copy()

    # ---- Stage 4: Nelder-Mead local polish ----
    if best_points is not None and time.time() < deadline - 20:
        c_nm, r_nm = _nelder_mead_polish(best_points, deadline - 10, beta=200.0, max_iter=5000)
        if r_nm > best_score:
            best_score = r_nm
            best_points = c_nm.copy()

    # ---- Stage 5: Binary search polish ----
    if best_points is not None and time.time() < deadline - 15:
        c_polished, r_polished = _binary_search_polish(best_points, deadline - 5)
        if r_polished > best_score:
            best_score = r_polished
            best_points = c_polished.copy()

    # ---- Stage 6: Final high-beta polish + Nelder-Mead ----
    if best_points is not None and time.time() < deadline - 5:
        c_final, r_final = _anneal_points(best_points, deadline - 3, max_betas=4)
        if r_final > best_score:
            best_score = r_final
            best_points = c_final.copy()
        if time.time() < deadline - 3:
            c_final2, r_final2 = _nelder_mead_polish(best_points, deadline - 1, beta=500.0, max_iter=3000)
            if r_final2 > best_score:
                best_score = r_final2
                best_points = c_final2.copy()

    # Ensure validity: clamp points that might be slightly outside
    if best_points is not None:
        c_final = best_points.reshape(N, 2)
        r2 = np.hypot(c_final[:, 0], c_final[:, 1])
        for i in range(N):
            if r2[i] > 1 - best_score + 1e-10:
                scale = (1 - best_score) / r2[i]
                c_final[i] *= scale
        return c_final

    # Fallback: return a valid configuration
    return _ring_to_points(np.array([0.0, 0.25, 0.45, 0.65, 0.85]), angles_list)
