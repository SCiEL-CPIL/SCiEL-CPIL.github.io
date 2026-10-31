import numpy as np
import time
import scipy.spatial as spat
import scipy.optimize as opt
from itertools import combinations

N = 13
_triple_idx = list(combinations(range(N), 3))
_i_arr = np.array([t[0] for t in _triple_idx], dtype=np.int64)
_j_arr = np.array([t[1] for t in _triple_idx], dtype=np.int64)
_k_arr = np.array([t[2] for t in _triple_idx], dtype=np.int64)
_n_triples = len(_triple_idx)

# Precompute base angles for regular 13-gon
_BASE_ANG = np.linspace(0.0, 2.0 * np.pi, N, endpoint=False)


def _triangle_areas(points):
    v1 = points[_j_arr] - points[_i_arr]
    v2 = points[_k_arr] - points[_i_arr]
    det = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(det)


def _hull_area(points):
    try:
        hull = spat.ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0


def calculate_score(points):
    if len(points) < 3:
        return 0.0
    hull_area = _hull_area(points)
    if hull_area <= 1e-9:
        return 0.0
    areas = _triangle_areas(points)
    min_area = np.min(areas)
    return min_area / hull_area


def _polar_to_points(params):
    """params: (2N,) -> [r_0..r_12, dtheta_0..dtheta_12] -> (N,2) points."""
    r = params[:N]
    dth = params[N:]
    theta = _BASE_ANG + dth
    return np.column_stack((r * np.cos(theta), r * np.sin(theta)))


def _polar_jacobian(params):
    """Returns (2N, N, 2) Jacobian: d(point_i)/d(param_j)."""
    r = params[:N]
    dth = params[N:]
    theta = _BASE_ANG + dth
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    J = np.zeros((2 * N, N, 2))
    # dr_i: d(point_i)/dr_i = (cos, sin)
    for i in range(N):
        J[i, i, 0] = cos_t[i]
        J[i, i, 1] = sin_t[i]
    # dtheta_i: d(point_i)/dtheta_i = (-r sin, r cos)
    for i in range(N):
        J[N + i, i, 0] = -r[i] * sin_t[i]
        J[N + i, i, 1] = r[i] * cos_t[i]
    return J


def _soft_min_neg(params, beta=40.0):
    """Negative soft-min of triangle areas / hull area (to minimize)."""
    pts = _polar_to_points(params)
    hull_area = _hull_area(pts)
    if hull_area <= 1e-9:
        return 1e10
    areas = _triangle_areas(pts)
    max_a = np.max(areas)
    exp_neg = np.exp(-beta * (areas - max_a))
    log_sum = np.log(np.sum(exp_neg)) + beta * max_a
    soft_min_val = log_sum / beta
    return -soft_min_val / hull_area


def _soft_min_neg_grad(params, beta=40.0):
    """Gradient of negative soft-min score w.r.t. polar params."""
    pts = _polar_to_points(params)
    hull_area = _hull_area(pts)
    if hull_area <= 1e-9:
        return np.zeros(2 * N)

    areas = _triangle_areas(pts)
    max_a = np.max(areas)
    exp_neg = np.exp(-beta * (areas - max_a))
    weights = exp_neg / np.sum(exp_neg)

    # Signed determinant and signs
    v1 = pts[_j_arr] - pts[_i_arr]
    v2 = pts[_k_arr] - pts[_i_arr]
    det_signed = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    signs = np.sign(det_signed)
    signs[signs == 0] = 1.0

    # Gradient of soft_min w.r.t. point positions (26-dim)
    grad_pts = np.zeros((N, 2))
    for t in range(_n_triples):
        w = weights[t]
        i, j, k = _i_arr[t], _j_arr[t], _k_arr[t]
        s = 0.5 * signs[t]
        # d(area)/dp_i
        grad_pts[i, 0] += w * s * (pts[j, 1] - pts[k, 1])
        grad_pts[i, 1] += w * s * (pts[j, 0] - pts[k, 0])
        # d(area)/dp_j
        grad_pts[j, 0] += w * s * v2[t, 1]
        grad_pts[j, 1] += w * s * (-v2[t, 0])
        # d(area)/dp_k
        grad_pts[k, 0] += w * s * (-v1[t, 1])
        grad_pts[k, 1] += w * s * (v1[t, 0])

    # grad_soft_min / hull_area (approximate, ignoring hull gradient)
    grad_pts = grad_pts / hull_area

    # Chain rule: grad_params = J^T @ grad_pts
    J = _polar_jacobian(params)  # (2N, N, 2)
    # Flatten grad_pts to (N, 2) -> need to contract with J
    grad_params = np.zeros(2 * N)
    for j_param in range(2 * N):
        for i_pt in range(N):
            for c in range(2):
                grad_params[j_param] += J[j_param, i_pt, c] * grad_pts[i_pt, c]

    # We are minimizing negative score, so negate
    return -grad_params


def _de_objective(params):
    """Objective for DE: negative score (minimize)."""
    pts = _polar_to_points(params)
    hull_area = _hull_area(pts)
    if hull_area <= 1e-9:
        return 1e10
    areas = _triangle_areas(pts)
    return -np.min(areas) / hull_area


def _local_optimize(params0, deadline, beta=40.0):
    """Run L-BFGS-B on polar params."""
    best_params = params0.copy()
    best_val = _soft_min_neg(params0, beta)

    try:
        res = opt.minimize(
            _soft_min_neg,
            params0,
            jac=_soft_min_neg_grad,
            args=(beta,),
            method='L-BFGS-B',
            options={'maxiter': 500, 'maxfun': 2000, 'ftol': 1e-14, 'gtol': 1e-12},
        )
        if res.fun < best_val:
            best_val = res.fun
            best_params = res.x
    except Exception:
        pass

    return best_params, -best_val


def _de_global(deadline, seed=42):
    """Differential Evolution in polar space."""
    np.random.seed(seed)
    # Bounds: r in [0.3, 1.5], dtheta in [-0.25, 0.25]
    bounds = [(0.3, 1.5)] * N + [(-0.25, 0.25)] * N

    # Population of 20, maxiter limited by time
    pop = 20
    max_iter = 30

    best_params = None
    best_score = -1e10

    # Seed the population with structured + random
    pop_arr = np.zeros((pop, 2 * N))
    # Structured seeds
    for idx, (r_base, th_base) in enumerate([(1.0, 0.0), (0.9, 0.05), (1.1, -0.05), (1.0, 0.1)]):
        if idx < pop:
            pop_arr[idx, :N] = r_base + np.random.normal(0, 0.05, N)
            pop_arr[idx, N:] = th_base + np.random.normal(0, 0.03, N)
    # Random seeds
    for idx in range(4, pop):
        pop_arr[idx, :N] = np.random.uniform(0.5, 1.3, N)
        pop_arr[idx, N:] = np.random.uniform(-0.2, 0.2, N)

    # Simple DE loop
    F = 0.8
    CR = 0.9
    for it in range(max_iter):
        if time.time() > deadline:
            break
        for i in range(pop):
            if time.time() > deadline:
                break
            # Pick distinct random indices
            a, b, c = np.random.choice(pop, 3, replace=False)
            while a == b or a == c or b == c:
                a, b, c = np.random.choice(pop, 3, replace=False)

            # Mutation
            v = pop_arr[a] + F * (pop_arr[b] - pop_arr[c])
            # Crossover
            trial = np.empty(2 * N)
            j_rand = np.random.randint(0, 2 * N)
            for j in range(2 * N):
                if np.random.rand() < CR or j == j_rand:
                    trial[j] = v[j]
                else:
                    trial[j] = pop_arr[i, j]
            # Clamp to bounds
            for j in range(2 * N):
                trial[j] = np.clip(trial[j], bounds[j][0], bounds[j][1])

            # Evaluate
            trial_score = -_de_objective(trial)
            curr_score = -_de_objective(pop_arr[i])

            if trial_score > curr_score:
                pop_arr[i] = trial
            if trial_score > best_score:
                best_score = trial_score
                best_params = trial.copy()

    if best_params is None:
        # Fallback: regular polygon
        best_params = np.zeros(2 * N)
        best_params[:N] = 1.0
        best_score = calculate_score(_polar_to_points(best_params))

    return best_params, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    best_score = -1.0
    best_points = None

    def _update_best(params):
        nonlocal best_score, best_points
        pts = _polar_to_points(params)
        sc = calculate_score(pts)
        if sc > best_score:
            best_score = sc
            best_points = pts.copy()
        return sc

    # --- Phase 1: Differential Evolution global search ---
    if time.time() < deadline - 60.0:
        de_params, de_score = _de_global(deadline - 60.0, seed=42)
        _update_best(de_params)

    # --- Phase 2: Local L-BFGS-B polish from multiple starts ---
    # Start 1: DE result
    if time.time() < deadline - 30.0:
        p, s = _local_optimize(de_params, deadline - 30.0, beta=40.0)
        _update_best(p)

    # Start 2: Regular polygon
    reg_params = np.zeros(2 * N)
    reg_params[:N] = 1.0
    if time.time() < deadline - 20.0:
        p, s = _local_optimize(reg_params, deadline - 20.0, beta=40.0)
        _update_best(p)

    # Start 3: Slightly perturbed regular
    np.random.seed(7)
    pert_params = np.zeros(2 * N)
    pert_params[:N] = 1.0 + np.random.normal(0, 0.08, N)
    pert_params[N:] = np.random.normal(0, 0.05, N)
    if time.time() < deadline - 15.0:
        p, s = _local_optimize(pert_params, deadline - 15.0, beta=40.0)
        _update_best(p)

    # Start 4: Another random perturbation
    np.random.seed(99)
    pert2 = np.zeros(2 * N)
    pert2[:N] = 1.0 + np.random.normal(0, 0.12, N)
    pert2[N:] = np.random.normal(0, 0.08, N)
    if time.time() < deadline - 10.0:
        p, s = _local_optimize(pert2, deadline - 10.0, beta=40.0)
        _update_best(p)

    # --- Phase 3: Fine polish with higher beta on best ---
    if best_points is not None and time.time() < deadline - 5.0:
        # Convert best_points back to polar-ish params for fine tuning
        # Just re-run L-BFGS from current best with higher beta
        # Reconstruct params from best_points: use nearest regular-gon fit
        best_pts = best_points
        r_est = np.linalg.norm(best_pts, axis=1)
        ang_est = np.arctan2(best_pts[:, 1], best_pts[:, 0])
        dth_est = ang_est - _BASE_ANG
        # Unwrap
        for i in range(1, N):
            while dth_est[i] - dth_est[i-1] > np.pi:
                dth_est[i] -= 2 * np.pi
            while dth_est[i] - dth_est[i-1] < -np.pi:
                dth_est[i] += 2 * np.pi
        fine_params = np.concatenate([r_est, dth_est])
        p, s = _local_optimize(fine_params, deadline - 5.0, beta=80.0)
        _update_best(p)

    if best_points is None:
        # Fallback: regular 13-gon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))

    return best_points
