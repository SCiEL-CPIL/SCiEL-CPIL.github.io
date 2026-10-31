import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import warnings
warnings.filterwarnings('ignore')

N = 13

_TRIPLE_IDX = np.array([
    (i, j, k)
    for i in range(N)
    for j in range(i + 1, N)
    for k in range(j + 1, N)
], dtype=np.int32)


def _triangle_areas_vectorized(points):
    pi = points[_TRIPLE_IDX[:, 0]]
    pj = points[_TRIPLE_IDX[:, 1]]
    pk = points[_TRIPLE_IDX[:, 2]]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)


def _hull_area(points):
    try:
        if len(points) < 3:
            return 0.0
        if np.allclose(points, points[0], atol=1e-12):
            return 0.0
        hull = ConvexHull(points)
        return float(hull.volume)
    except Exception:
        return 0.0


def _score(points):
    if len(points) < 3:
        return 0.0
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    areas = _triangle_areas_vectorized(points)
    min_area = float(np.min(areas))
    if min_area < 1e-15:
        return 0.0
    return min_area / hull_area


def _soft_min_value(areas, beta):
    """Smooth approximation of min(areas) via log-sum-exp."""
    a = np.clip(areas, 1e-15, None)
    x = -beta * a
    m = np.max(x)
    return -(m + np.log(np.sum(np.exp(x - m)))) / beta


def _soft_score(points, beta):
    if len(points) < 3:
        return 0.0
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    areas = _triangle_areas_vectorized(points)
    sm = _soft_min_value(areas, beta)
    if sm < 1e-15:
        return 0.0
    return sm / hull_area


def _neg_soft_score(points, beta):
    s = _soft_score(points, beta)
    if s <= 0:
        return 1e10
    return -s


def _annealed_soft_minimize(x0, deadline, beta_start=8.0, beta_end=120.0,
                            max_total_iters=1200, stages=4):
    """
    Run L-BFGS-B on the soft-min surrogate with a beta schedule.
    Returns the best point found (using true score).
    """
    best = x0.copy()
    best_s = _score(best)
    betas = np.geomspace(beta_start, beta_end, stages)
    iters_per_stage = max(50, max_total_iters // stages)

    cur = x0.copy()
    for beta in betas:
        if time.time() > deadline - 3:
            break
        try:
            res = minimize(
                lambda x, b=beta: _neg_soft_score(x.reshape(N, 2), b),
                cur,
                method='L-BFGS-B',
                options={'maxiter': iters_per_stage, 'ftol': 1e-12,
                         'gtol': 1e-10, 'maxfun': iters_per_stage * 4},
            )
            cand = res.x.reshape(N, 2)
            s = _score(cand)
            if s > best_s:
                best_s, best = s, cand.copy()
            cur = cand
        except Exception:
            pass

    # Final Nelder-Mead polish on true objective
    if time.time() > deadline - 4:
        return best, best_s
    try:
        res = minimize(
            lambda x: -_score(x.reshape(N, 2)),
            best.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 600, 'xatol': 1e-12, 'fatol': 1e-12,
                     'adaptive': True},
        )
        cand = res.x.reshape(N, 2)
        s = _score(cand)
        if s > best_s:
            best_s, best = s, cand
    except Exception:
        pass
    return best, best_s


def _seed_regular(radius=1.0):
    ang = np.linspace(0, 2 * np.pi, N, endpoint=False)
    return np.column_stack([radius * np.cos(ang), radius * np.sin(ang)])


def _seed_two_ring(n_out, r_in, angle_offset):
    n_in = N - n_out
    a_out = np.linspace(0, 2 * np.pi, n_out, endpoint=False)
    a_in = np.linspace(angle_offset, 2 * np.pi + angle_offset, n_in, endpoint=False)
    outer = np.column_stack([np.cos(a_out), np.sin(a_out)])
    inner = np.column_stack([r_in * np.cos(a_in), r_in * np.sin(a_in)])
    return np.vstack([outer, inner])


def _seed_three_ring(n1, n2, n3, r2, r3, off2, off3):
    a1 = np.linspace(0, 2 * np.pi, n1, endpoint=False)
    a2 = np.linspace(off2, 2 * np.pi + off2, n2, endpoint=False)
    a3 = np.linspace(off3, 2 * np.pi + off3, n3, endpoint=False)
    return np.vstack([
        np.column_stack([np.cos(a1), np.sin(a1)]),
        np.column_stack([r2 * np.cos(a2), r2 * np.sin(a2)]),
        np.column_stack([r3 * np.cos(a3), r3 * np.sin(a3)]),
    ])


def _seed_quasi_regular(radial_pert, angular_pert, radius=1.0):
    """Regular 13-gon with per-point radial/angular perturbations."""
    ang = np.linspace(0, 2 * np.pi, N, endpoint=False) + angular_pert
    rad = radius * (1.0 + radial_pert)
    return np.column_stack([rad * np.cos(ang), rad * np.sin(ang)])


def _random_perturbation(rng, scale_r, scale_a):
    rp = rng.uniform(-scale_r, scale_r, N)
    ap = rng.uniform(-scale_a, scale_a, N)
    return rp, ap


def _pairwise_repulsion_step(points, deadline, n_steps=200, lr=0.02):
    """
    Simple gradient-free diversification: push points apart in a repulsion
    field, then re-score. Returns best point found during the walk.
    """
    best = points.copy()
    best_s = _score(points)
    cur = points.copy()
    rng = np.random.default_rng(12345)

    for step in range(n_steps):
        if time.time() > deadline - 5:
            break
        # Compute repulsion force (push each point away from all others)
        force = np.zeros_like(cur)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                d = cur[i] - cur[j]
                dist2 = np.dot(d, d) + 1e-12
                force[i] += d / (dist2 ** 1.5)
        # Add small random noise for diversity
        force += rng.normal(0, 0.05, size=cur.shape)
        cur = cur + lr * force
        s = _score(cur)
        if s > best_s:
            best_s, best = s, cur.copy()
        # If score collapsed, revert and shrink step
        if s < best_s * 0.5:
            cur = best.copy()
            lr = max(0.002, lr * 0.5)
    return best, best_s


def construct():
    start_time = time.time()
    deadline = start_time + 143.0  # safety margin under 150s

    best_points = _seed_regular()
    best_score = _score(best_points)

    def _update(pts):
        nonlocal best_points, best_score
        if pts is None or len(pts) != N:
            return
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_points = pts.copy()

    # ---------- Build a diverse pool of seeds ----------
    seeds = []
    seeds.append(_seed_regular())
    seeds.append(_seed_regular(0.95))
    seeds.append(_seed_regular(1.05))

    # Two-ring seeds
    for r_in in [0.4, 0.5, 0.6, 0.7, 0.8]:
        for n_out in [6, 7, 8]:
            for off in [0.0, np.pi / 7, np.pi / 6, np.pi / 5]:
                seeds.append(_seed_two_ring(n_out, r_in, off))

    # Three-ring seeds
    for (n1, n2, n3) in [(5, 4, 4), (6, 4, 3), (7, 3, 3), (6, 5, 2)]:
        for r2 in [0.5, 0.65]:
            for r3 in [0.25, 0.4]:
                for o2 in [0.0, np.pi / 6]:
                    for o3 in [0.0, np.pi / 4]:
                        seeds.append(_seed_three_ring(n1, n2, n3, r2, r3, o2, o3))

    # Quasi-regular with random perturbations
    rng = np.random.default_rng(7)
    for _ in range(30):
        rp, ap = _random_perturbation(rng, 0.15, 0.2)
        seeds.append(_seed_quasi_regular(rp, ap))

    # Random points in a disk (for diversity)
    for _ in range(20):
        r = rng.uniform(0.3, 1.0, N)
        a = rng.uniform(0, 2 * np.pi, N)
        seeds.append(np.column_stack([r * np.cos(a), r * np.sin(a)]))

    # Score all seeds, keep top-K for optimization
    scored = []
    for s in seeds:
        sc = _score(s)
        scored.append((sc, s))
    scored.sort(key=lambda x: -x[0])
    top_seeds = [s for sc, s in scored[:15]]
    if _score(best_points) > scored[0][0]:
        top_seeds.insert(0, best_points.copy())

    # ---------- Phase 1: Annealed soft-min L-BFGS-B from each top seed ----------
    for idx, seed in enumerate(top_seeds):
        if time.time() > deadline - 25:
            break
        per_seed_budget = deadline - 25 - time.time()
        max_iters = min(1500, max(200, int(per_seed_budget * 25)))
        try:
            pts, s = _annealed_soft_minimize(seed, deadline,
                                             beta_start=8.0, beta_end=150.0,
                                             max_total_iters=max_iters, stages=4)
            _update(pts)
        except Exception:
            pass

    # ---------- Phase 2: Pairwise repulsion diversification + re-optimize ----------
    if time.time() < deadline - 20:
        try:
            pts, s = _pairwise_repulsion_step(best_points, deadline,
                                              n_steps=150, lr=0.015)
            _update(pts)
            # Re-optimize from the diversified point
            if time.time() < deadline - 10:
                pts2, s2 = _annealed_soft_minimize(pts, deadline,
                                                   beta_start=15.0, beta_end=120.0,
                                                   max_total_iters=800, stages=3)
                _update(pts2)
        except Exception:
            pass

    # ---------- Phase 3: Second round of restarts from perturbed best ----------
    if time.time() < deadline - 15:
        rng2 = np.random.default_rng(99)
        for k in range(5):
            if time.time() > deadline - 8:
                break
            rp, ap = _random_perturbation(rng2, 0.1, 0.15)
            # Apply perturbation in polar-like fashion to best_points
            base = best_points.copy()
            # rotate slightly and scale
            theta = rng2.uniform(-0.1, 0.1)
            c, s_ = np.cos(theta), np.sin(theta)
            rot = np.array([[c, -s_], [s_, c]])
            perturbed = (base @ rot.T) * (1.0 + rng2.uniform(-0.05, 0.05, (N, 1)))
            try:
                pts, sc = _annealed_soft_minimize(perturbed, deadline,
                                                  beta_start=12.0, beta_end=100.0,
                                                  max_total_iters=600, stages=3)
                _update(pts)
            except Exception:
                pass

    # ---------- Phase 4: Final ultra-fine Nelder-Mead polish ----------
    if time.time() < deadline - 3:
        try:
            res = minimize(
                lambda x: -_score(x.reshape(N, 2)),
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2500, 'xatol': 1e-14, 'fatol': 1e-14,
                         'adaptive': True},
            )
            _update(res.x.reshape(N, 2))
        except Exception:
            pass

    # ---------- Phase 5: One more high-beta soft pass ----------
    if time.time() < deadline - 1:
        try:
            res = minimize(
                lambda x: _neg_soft_score(x.reshape(N, 2), 250.0),
                best_points.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12,
                         'maxfun': 800},
            )
            cand = res.x.reshape(N, 2)
            s = _score(cand)
            if s > best_score:
                best_score = s
                best_points = cand.copy()
        except Exception:
            pass

    if best_points is None:
        best_points = _seed_regular()

    return best_points
