"""Rigid octahedron-core + cube-shell design with smooth polish."""
import time
import numpy as np
from scipy.optimize import minimize

N = 14
D = 3

# EVOLVE-BLOCK-START
def _pairwise_dists_sq(P):
    """Return matrix of squared pairwise distances (n, n)."""
    diff = P[:, None, :] - P[None, :, :]
    return np.sum(diff * diff, axis=2)

def _score_from_d2(D2):
    """Score = (max/min)^2 from squared-distance matrix."""
    mask = ~np.eye(N, dtype=bool)
    vals = D2[mask]
    if len(vals) == 0:
        return np.inf
    dmin = vals.min()
    dmax = vals.max()
    if dmin <= 1e-12:
        return np.inf
    r = dmax / dmin
    return r * r

def _make_points(s, theta):
    """
    Build 14 points: 6 octahedron core + 8 cube shell.
    Octahedron vertices: (±1,0,0),(0,±1,0),(0,0,±1) scaled by s.
    Cube vertices: (±1,±1,±1) scaled by s, rotated by theta about z-axis.
    """
    # Octahedron core (6 points)
    octa = np.array([
        [1,0,0],[-1,0,0],
        [0,1,0],[0,-1,0],
        [0,0,1],[0,0,-1]
    ], dtype=float) * s

    # Cube shell (8 points), rotated about z by theta
    cube = np.array([
        [ 1, 1, 1],[-1, 1, 1],[ 1,-1, 1],[-1,-1, 1],
        [ 1, 1,-1],[-1, 1,-1],[ 1,-1,-1],[-1,-1,-1]
    ], dtype=float) * s

    ct, st = np.cos(theta), np.sin(theta)
    Rz = np.array([[ct, -st, 0],[st, ct, 0],[0, 0, 1]], dtype=float)
    cube = cube @ Rz.T

    return np.vstack([octa, cube])

def _objective(params):
    s, theta = params
    if s <= 1e-8:
        return 1e10
    P = _make_points(s, theta)
    D2 = _pairwise_dists_sq(P)
    return _score_from_d2(D2)

def _soft_ratio(vals, alpha=30.0):
    """
    Compute soft max/min ratio^2 using log-sum-exp.
    vals: array of positive squared distances.
    Returns smooth approximation of (max/min)^2.
    """
    if np.any(vals <= 1e-14):
        return 1e10
    log_vals = np.log(vals)
    # soft_max = exp(alpha * logaddexp(log_vals)) / alpha
    # soft_min = exp(-alpha * logaddexp(-log_vals)) / alpha
    soft_max = np.exp(np.logaddexp.reduce(log_vals * alpha) / alpha)
    soft_min = np.exp(-np.logaddexp.reduce(-log_vals * alpha) / alpha)
    if soft_min <= 1e-14:
        return 1e10
    r = soft_max / soft_min
    return r * r

def _smooth_obj(x, alpha=30.0):
    """
    Smooth objective: (soft_max / soft_min)^2 of squared pairwise distances.
    x: flat array of shape (N*D,)
    """
    P = x.reshape(N, D)
    D2 = _pairwise_dists_sq(P)
    mask = ~np.eye(N, dtype=bool)
    vals = D2[mask]
    return _soft_ratio(vals, alpha)

def _polish(P0, time_budget, alpha=30.0):
    """Local smooth polish using L-BFGS-B on free coordinates."""
    deadline = time.time() + time_budget
    P = P0.copy()
    best_P = P.copy()
    best_score = _score_from_d2(_pairwise_dists_sq(P))

    def obj(x):
        return _smooth_obj(x, alpha)

    # First polish with moderate alpha
    x0 = P.flatten()
    res = minimize(obj, x0, method='L-BFGS-B', options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10})
    if res.fun < best_score or res.success:
        P = res.x.reshape(N, D)
        s = _score_from_d2(_pairwise_dists_sq(P))
        if s < best_score:
            best_score = s
            best_P = P.copy()

    # Second polish with higher alpha for sharper max/min
    if time.time() < deadline:
        alpha2 = 60.0
        def obj2(x):
            return _smooth_obj(x, alpha2)
        x0 = P.flatten()
        res2 = minimize(obj2, x0, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
        if res2.fun < best_score or res2.success:
            P = res2.x.reshape(N, D)
            s = _score_from_d2(_pairwise_dists_sq(P))
            if s < best_score:
                best_score = s
                best_P = P.copy()

    # Third polish with even higher alpha
    if time.time() < deadline:
        alpha3 = 100.0
        def obj3(x):
            return _smooth_obj(x, alpha3)
        x0 = P.flatten()
        res3 = minimize(obj3, x0, method='L-BFGS-B', options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-10})
        if res3.fun < best_score or res3.success:
            P = res3.x.reshape(N, D)
            s = _score_from_d2(_pairwise_dists_sq(P))
            if s < best_score:
                best_score = s
                best_P = P.copy()

    return best_P, best_score

def construct():
    start_time = time.time()
    DEADLINE = 120.0  # safety margin
    time_budget_total = DEADLINE - 5.0  # 5s margin

    # Phase 1: Grid search over (s, theta)
    best_score = np.inf
    best_params = None

    s_range = np.linspace(0.3, 2.0, 60)
    theta_range = np.linspace(0.0, np.pi/4, 40)

    for s in s_range:
        if time.time() - start_time > time_budget_total * 0.4:
            break
        for theta in theta_range:
            P = _make_points(s, theta)
            D2 = _pairwise_dists_sq(P)
            sc = _score_from_d2(D2)
            if sc < best_score:
                best_score = sc
                best_params = (s, theta)

    if best_params is None:
        # Fallback: simple grid
        side = int(np.ceil(N ** (1 / D)))
        grid = list(__import__('itertools').product(range(side), repeat=D))[:N]
        return np.array(grid, dtype=float)

    P_best = _make_points(best_params[0], best_params[1])

    # Phase 2: Local search around best (s, theta)
    if time.time() - start_time < time_budget_total * 0.7:
        s0, th0 = best_params
        s_fine = np.linspace(max(0.1, s0 - 0.3), s0 + 0.3, 30)
        th_fine = np.linspace(max(0.0, th0 - 0.3), min(np.pi/2, th0 + 0.3), 30)
        for s in s_fine:
            for theta in th_fine:
                P = _make_points(s, theta)
                D2 = _pairwise_dists_sq(P)
                sc = _score_from_d2(D2)
                if sc < best_score:
                    best_score = sc
                    best_params = (s, theta)
                    P_best = P

    # Phase 3: Smooth polish with multiple alpha values
    polish_budget = min(30.0, time_budget_total - (time.time() - start_time) - 5.0)
    if polish_budget > 2.0:
        P_polished, sc_polished = _polish(P_best, polish_budget, alpha=30.0)
        if sc_polished < best_score:
            best_score = sc_polished
            P_best = P_polished

    # Verify distinctness
    D2 = _pairwise_dists_sq(P_best)
    mask = ~np.eye(N, dtype=bool)
    if D2[mask].min() < 1e-10:
        # Perturb if needed
        P_best = P_best + np.random.default_rng(42).normal(0, 1e-6, P_best.shape)

    return P_best
# EVOLVE-BLOCK-END
