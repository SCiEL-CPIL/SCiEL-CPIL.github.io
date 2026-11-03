"""Packing 50 equal circles in a unit circle.

Combined design:
- Full-coordinate soft-min (log-sum-exp) optimization (L-BFGS-B multi-stage annealing),
  ring + hexagonal initializations, simulated annealing, multi-start.
- Epigraph smooth formulation (auxiliary t) as an additional polish surrogate.
- Vectorized Adam gradient polish (fast local optimizer).
- Constraint-push + binary-search radius polish (high-precision final stage).
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
DEADLINE = 145.0  # safety margin before 150 s kill


# ---------------------------------------------------------------------------
# Core score / distance helpers
# ---------------------------------------------------------------------------
def _pairwise_indices(n):
    iu = np.triu_indices(n, k=1)
    return np.column_stack((iu[0], iu[1]))


def _pairwise_distances(pts):
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=-1))


def _score_vec(points):
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    n = len(points)
    if n != N:
        return -1.0
    if not np.all(np.isfinite(points)):
        return -1.0
    radii = 1.0 - np.linalg.norm(points, axis=1)
    r = float(np.min(radii))
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(n, 1)
    pair_dists = dists[iu] / 2.0
    r = min(r, float(np.min(pair_dists)))
    return r


# ---------------------------------------------------------------------------
# Smooth surrogates
# ---------------------------------------------------------------------------
def _softmin_logsumexp(values, beta):
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


def _epigraph_objective(x, beta):
    """Epigraph formulation: x = [positions (2N), t]. Minimize -t subject to
    smooth constraints t <= each margin. Returns large penalty if violated."""
    c = x[:2 * N].reshape(N, 2)
    t = x[2 * N]
    if not np.all(np.isfinite(c)):
        return 1e10
    d = _pairwise_distances(c)
    np.fill_diagonal(d, np.inf)
    boundary = 1.0 - np.sqrt(np.sum(c ** 2, axis=1))
    margins = np.concatenate([d.ravel() / 2.0, boundary])
    margins = margins[np.isfinite(margins)]
    if len(margins) == 0:
        return 1e10
    # violations = t - margin (should be <= 0); penalize via soft-max
    viol = t - margins
    viol = viol[viol > 0]
    if len(viol) == 0:
        return -t
    smax = _softmin_logsumexp(-viol, beta)  # = soft-max of viol
    return -t + 10.0 * smax


# ---------------------------------------------------------------------------
# Initializations
# ---------------------------------------------------------------------------
def _ring_to_points(radii, angles_list):
    pts = [np.array([0.0, 0.0])]
    for i in range(1, len(radii)):
        r = radii[i]
        a = angles_list[i]
        pts.append(np.column_stack([r * np.cos(a), r * np.sin(a)]))
    return np.vstack(pts)


def _hexagonal_init(scale=0.12):
    points = []
    for i in range(-12, 13):
        for j in range(-12, 13):
            x = scale * (i + 0.5 * j)
            y = scale * (np.sqrt(3) / 2.0 * j)
            if x * x + y * y <= 0.85:
                points.append([x, y])
    points = np.array(points)
    if len(points) == 0:
        return None
    dists = np.sqrt(np.sum(points ** 2, axis=1))
    idx = np.argsort(dists)
    points = points[idx[:N]]
    if len(points) < N:
        return None
    return points


def _ring_initialization(n, n_rings=5):
    best_points = None
    best_r = 0.0
    for trial in range(20):
        ring_counts = [1]
        remaining = n - 1
        for ring in range(1, n_rings + 1):
            if remaining <= 0:
                break
            rings_left = n_rings - ring
            if rings_left <= 0:
                ring_counts.append(remaining)
                remaining = 0
            else:
                count = max(1, remaining // (rings_left + 1))
                ring_counts.append(count)
                remaining -= count
        while sum(ring_counts) < n:
            ring_counts[-1] += 1
        while sum(ring_counts) > n:
            if len(ring_counts) > 1:
                ring_counts[-1] -= 1
            else:
                break
        points = []
        for ring_idx, count in enumerate(ring_counts):
            if ring_idx == 0:
                points.append([0.0, 0.0])
            else:
                r_ring = 0.15 + 0.85 * (ring_idx / max(n_rings, 1))
                if trial % 3 == 0:
                    r_ring *= 0.9
                elif trial % 3 == 1:
                    r_ring *= 1.0
                else:
                    r_ring *= 1.1
                angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
                for angle in angles:
                    points.append([r_ring * np.cos(angle), r_ring * np.sin(angle)])
        points = np.array(points[:n])
        if len(points) < n:
            continue
        r = _score_vec(points)
        if r > best_r:
            best_r = r
            best_points = points.copy()
    if best_points is None:
        best_points = _hexagonal_init(0.12)
        if best_points is None:
            best_points = np.zeros((n, 2))
            for k in range(n):
                a = 2 * np.pi * k / n
                best_points[k] = [0.5 * np.cos(a), 0.5 * np.sin(a)]
            best_r = _score_vec(best_points)
    return best_points


# ---------------------------------------------------------------------------
# Local optimizers
# ---------------------------------------------------------------------------
def _anneal_points(pts, deadline, max_betas=12):
    current = pts.copy()
    current_score = _score(current)
    betas = [3.0, 8.0, 20.0, 50.0, 120.0, 300.0, 700.0, 1500.0,
             3000.0, 6000.0, 12000.0, 25000.0][:max_betas]
    for beta in betas:
        if time.time() > deadline - 5:
            break
        try:
            res = minimize(
                _smooth_objective,
                current.ravel(),
                args=(beta,),
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-14,
                         'maxfun': 30000}
            )
            new_pts = res.x.reshape(N, 2)
            if not np.all(np.isfinite(new_pts)):
                continue
            s = _score_vec(new_pts)
            if s > current_score:
                current = new_pts
                current_score = s
        except Exception:
            pass
    return current, current_score


def _adam_polish(pts, deadline, lr=1e-3, beta=200.0, max_steps=4000):
    """Vectorized Adam gradient descent on the soft-min objective."""
    best = pts.copy()
    best_score = _score_vec(best)
    x = best.ravel().copy()
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    t = 0
    while t < max_steps and time.time() < deadline - 1:
        c = x.reshape(N, 2)
        d = _pairwise_distances(c)
        np.fill_diagonal(d, np.inf)
        boundary = 1.0 - np.sqrt(np.sum(c ** 2, axis=1))
        pair_half = d / 2.0
        all_vals = np.concatenate([pair_half.ravel(), boundary])
        all_vals = all_vals[np.isfinite(all_vals)]
        if len(all_vals) == 0:
            break
        neg = -all_vals * beta
        neg_max = np.max(neg)
        w = np.exp(neg - neg_max)
        w = w / np.sum(w)
        # gradient of -softmin wrt positions
        grad = np.zeros((N, 2))
        # pair contributions: all_vals[0:N*(N-1)/2] correspond to d.ravel()
        n_pairs = N * (N - 1) // 2
        pair_w = w[:n_pairs] * beta
        D = d.copy()
        D = np.maximum(D, 1e-12)
        # accumulate via broadcasting
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        # grad_pair[i] += sum_j w_ij * diff[i,j] / D_ij ; factor 1/2 from d/2
        W = np.zeros((N, N))
        W[np.triu_indices(N, 1)] = pair_w
        W = W + W.T  # symmetric
        # grad = (1/D) * (W * diff) summed over j, times 0.5
        with np.errstate(divide='ignore', invalid='ignore'):
            G = np.sum(W[:, :, np.newaxis] * diff / D[:, :, np.newaxis], axis=1)
        grad += 0.5 * G
        # boundary contributions: all_vals[n_pairs:] = boundary
        bnd_w = w[n_pairs:] * beta
        r = np.sqrt(np.sum(c ** 2, axis=1))
        r = np.maximum(r, 1e-12)
        grad -= bnd_w[:, np.newaxis] * c / r[:, np.newaxis]
        g = grad.ravel()
        t += 1
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * (g * g)
        mhat = m / (1 - b1 ** t)
        vhat = v / (1 - b2 ** t)
        x = x - lr * mhat / (np.sqrt(vhat) + eps)
        cand = x.reshape(N, 2)
        s = _score_vec(cand)
        if s > best_score:
            best_score = s
            best = cand.copy()
    return best, best_score


def _epigraph_polish(pts, deadline, beta=500.0, maxiter=300):
    """L-BFGS-B on the epigraph objective (auxiliary t)."""
    best = pts.copy()
    best_score = _score_vec(best)
    x0 = np.concatenate([best.ravel(), [best_score]])
    try:
        res = minimize(
            _epigraph_objective,
            x0,
            args=(beta,),
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'ftol': 1e-16, 'gtol': 1e-14}
        )
        if np.all(np.isfinite(res.x)):
            cand = res.x[:2 * N].reshape(N, 2)
            s = _score_vec(cand)
            if s > best_score:
                best = cand
                best_score = s
    except Exception:
        pass
    return best, best_score


# ---------------------------------------------------------------------------
# High-precision polish: constraint push + binary search
# ---------------------------------------------------------------------------
def _constraint_push(pts, r_target, deadline, max_iters=500):
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
    current = pts.copy()
    low = _score_vec(current) - 1e-8
    high = _score_vec(current) + 1e-4
    for _ in range(30):
        if time.time() > deadline - 2:
            break
        mid = (low + high) / 2.0
        polished = _constraint_push(current, mid, deadline, max_iters=200)
        s = _score_vec(polished)
        if s >= mid - 1e-10:
            low = mid
            current = polished
        else:
            high = mid
    return current, _score_vec(current)


# ---------------------------------------------------------------------------
# Simulated annealing (fast, vectorized-ish)
# ---------------------------------------------------------------------------
def _anneal(points, deadline, seed=42):
    rng = np.random.default_rng(seed)
    n = len(points)
    pts = points.copy()
    best_pts = pts.copy()
    best_score = _score_vec(best_pts)
    T = 0.05
    T_min = 0.0001
    cooling = 0.999
    step_size = 0.02
    current_cost = 0.0

    def cost(p):
        rt = best_score
        d = _pairwise_distances(p)
        np.fill_diagonal(d, np.inf)
        e = 0.0
        pm = d / 2.0
        below = pm < 2 * rt
        e += np.sum((2 * rt - pm[below]) ** 2)
        bm = 1.0 - np.linalg.norm(p, axis=1)
        bb = bm < rt
        e += np.sum((rt - bm[bb]) ** 2)
        return e

    current_cost = cost(pts)
    while time.time() < deadline:
        if T < T_min:
            break
        n_moves = int(rng.integers(1, 4))
        new_pts = pts.copy()
        for _ in range(n_moves):
            idx = int(rng.integers(n))
            new_pts[idx] += rng.standard_normal(2) * step_size * (1 + T)
        new_cost = cost(new_pts)
        dc = new_cost - current_cost
        if dc < 0 or rng.random() < np.exp(-dc / T):
            pts = new_pts
            current_cost = new_cost
            s = _score_vec(pts)
            if s > best_score:
                best_score = s
                best_pts = pts.copy()
        T *= cooling
    return best_pts, best_score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE

    # ---- Stage 1: multiple initializations ----
    candidates = []
    ring_pts = _ring_initialization(N)
    rs = _score_vec(ring_pts)
    if rs > 0:
        candidates.append((ring_pts.copy(), rs))
    hex_pts = _hexagonal_init(0.12)
    if hex_pts is not None:
        hs = _score_vec(hex_pts)
        if hs > 0:
            candidates.append((hex_pts.copy(), hs))
    # a few random perturbations of the best init
    base = max(candidates, key=lambda x: x[1])[0] if candidates else ring_pts
    for k in range(3):
        cand = base + np.random.randn(N, 2) * 0.02
        cs = _score_vec(cand)
        if cs > 0:
            candidates.append((cand.copy(), cs))

    best_points = max(candidates, key=lambda x: x[1])[0]
    best_score = _score_vec(best_points)

    # ---- Stage 2: multi-stage soft-min annealing on the best init ----
    if time.time() < deadline - 30:
        c_opt, r_opt = _anneal_points(best_points, deadline - 25, max_betas=12)
        if r_opt > best_score:
            best_score = r_opt
            best_points = c_opt.copy()

    # ---- Stage 3: Adam polish ----
    if time.time() < deadline - 25:
        c_a, r_a = _adam_polish(best_points, deadline - 22, lr=1e-3, beta=200.0)
        if r_a > best_score:
            best_score = r_a
            best_points = c_a.copy()

    # ---- Stage 4: epigraph polish ----
    if time.time() < deadline - 22:
        c_e, r_e = _epigraph_polish(best_points, deadline - 20, beta=500.0)
        if r_e > best_score:
            best_score = r_e
            best_points = c_e.copy()

    # ---- Stage 5: simulated annealing to escape local minima ----
    remaining = deadline - (time.time() - t0)
    if remaining > 35:
        sa_pts, sa_score = _anneal(best_points, time.time() + min(remaining - 20, 35), seed=42)
        if sa_score > best_score:
            best_score = sa_score
            best_points = sa_pts.copy()
        # polish after SA
        if time.time() < deadline - 18:
            c_opt, r_opt = _anneal_points(best_points, deadline - 16, max_betas=8)
            if r_opt > best_score:
                best_score = r_opt
                best_points = c_opt.copy()
        if time.time() < deadline - 15:
            c_a, r_a = _adam_polish(best_points, deadline - 14, lr=1e-3, beta=400.0)
            if r_a > best_score:
                best_score = r_a
                best_points = c_a.copy()

    # ---- Stage 6: multi-start perturbations ----
    n_restarts = 0
    while time.time() < deadline - 12:
        scale = 0.015 / (1.0 + 0.1 * n_restarts)
        x0 = best_points + np.random.randn(N, 2) * scale
        n_restarts += 1
        try:
            c_opt, r_opt = _anneal_points(x0, deadline - 10, max_betas=6)
            if r_opt > best_score:
                best_score = r_opt
                best_points = c_opt.copy()
        except Exception:
            pass
        if n_restarts >= 4:
            break

    # ---- Stage 7: high-precision final polish (constraint push + binary search) ----
    if time.time() < deadline - 5:
        c_final, r_final = _binary_search_polish(best_points, deadline - 2)
        if r_final > best_score:
            best_score = r_final
            best_points = c_final.copy()
        else:
            # still try a direct push at current radius to fix tiny violations
            c_push = _constraint_push(best_points, best_score, deadline - 2, max_iters=300)
            rp = _score_vec(c_push)
            if rp > best_score:
                best_score = rp
                best_points = c_push.copy()

    # ---- Ensure validity ----
    c_final = best_points.reshape(N, 2)
    r2 = np.linalg.norm(c_final, axis=1)
    for i in range(N):
        if r2[i] > 1 - best_score + 1e-10:
            c_final[i] *= (1 - best_score) / r2[i]

    return c_final
