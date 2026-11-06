"""
Combined sphere-packing program.
Structure: stable log-sum-exp annealing with multiple crystallographic starts,
followed by a penalty-based equalization polish on the best configuration.
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 100
DEADLINE = 140.0  # Safety margin from 150s limit


class _Timeout(Exception):
    pass


def _all_dists(c):
    """Return (pair_dists_raw, wall_dists, i_idx, j_idx, d)."""
    d = cdist(c, c)
    i_idx, j_idx = np.triu_indices(N, k=1)
    pair_dists_raw = d[i_idx, j_idx]
    wall_dists = np.concatenate([
        0.5 - c[:, 0],
        0.5 + c[:, 0],
        0.5 - c[:, 1],
        0.5 + c[:, 1],
        0.5 - c[:, 2],
        0.5 + c[:, 2],
    ])
    return pair_dists_raw, wall_dists, i_idx, j_idx, d


def _compute_smooth_obj_grad(c, alpha):
    """
    Numerically-stable smooth log-sum-exp objective + gradient.
    f(c) = (1/alpha) * log( sum_s exp(-alpha * s) )
    Minimizing f maximizes a smooth approximation of the packing radius.
    """
    pair_dists_raw, wall_dists, i_idx, j_idx, d = _all_dists(c)
    all_dists = np.concatenate([pair_dists_raw / 2.0, wall_dists])

    exp_arg = -alpha * all_dists
    max_arg = np.max(exp_arg)
    exp_terms = np.exp(exp_arg - max_arg)

    Z = np.sum(exp_terms)
    if Z == 0.0:
        return 0.0, np.zeros_like(c)

    obj_val = (1.0 / alpha) * (np.log(Z) + max_arg)

    grad_c = np.zeros_like(c)

    # Pairwise part
    exp_pair = exp_terms[:len(pair_dists_raw)]
    weights_pair = exp_pair / Z

    pair_dists_raw_safe = np.where(pair_dists_raw < 1e-12, 1e-12, pair_dists_raw)

    ci = c[i_idx]
    cj = c[j_idx]
    diff = ci - cj
    norms = pair_dists_raw_safe[:, np.newaxis]
    u = diff / norms

    w = weights_pair[:, np.newaxis]
    contrib = w * u / 2.0

    np.add.at(grad_c, i_idx, -contrib)
    np.add.at(grad_c, j_idx, contrib)

    # Wall part
    exp_wall = exp_terms[len(pair_dists_raw):]
    weights_wall = exp_wall / Z
    ww = weights_wall.reshape(6, N)

    grad_c[:, 0] += ww[0] - ww[1]
    grad_c[:, 1] += ww[2] - ww[3]
    grad_c[:, 2] += ww[4] - ww[5]

    return obj_val, grad_c


def _compute_penalty_obj_grad(c, target_r):
    """
    Penalty objective: sum of squared violations of the target radius.
    Drives the tightest (critical) constraints toward the common radius.
    """
    pair_dists_raw, wall_dists, i_idx, j_idx, d = _all_dists(c)
    all_dists = np.concatenate([pair_dists_raw / 2.0, wall_dists])

    violations = np.maximum(0.0, target_r - all_dists)
    obj_val = float(np.sum(violations ** 2))

    grad_c = np.zeros_like(c)

    # Pairwise part
    viol_pair = violations[:len(pair_dists_raw)]
    active = viol_pair > 1e-12
    if np.any(active):
        pair_dists_raw_safe = np.where(pair_dists_raw < 1e-12, 1e-12, pair_dists_raw)

        ci = c[i_idx[active]]
        cj = c[j_idx[active]]
        diff = ci - cj
        norms = pair_dists_raw_safe[active, np.newaxis]
        u = diff / norms

        w = 2.0 * viol_pair[active][:, np.newaxis]
        np.add.at(grad_c, i_idx[active], -w * u / 2.0)
        np.add.at(grad_c, j_idx[active], w * u / 2.0)

    # Wall part
    viol_wall = violations[len(pair_dists_raw):]
    ww = viol_wall.reshape(6, N)

    grad_c[:, 0] += 2.0 * ww[0] - 2.0 * ww[1]
    grad_c[:, 1] += 2.0 * ww[2] - 2.0 * ww[3]
    grad_c[:, 2] += 2.0 * ww[4] - 2.0 * ww[5]

    return obj_val, grad_c


def _score(c):
    """Exact packing radius."""
    if c.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(c)):
        return 0.0
    bound = np.min(0.5 - np.abs(c))
    if bound < 0.0:
        return 0.0
    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    pair_min = np.min(d) / 2.0
    return float(min(bound, pair_min))


def _initial_grid():
    """5x5x4 grid."""
    xs = np.linspace(-0.4, 0.4, 5)
    ys = np.linspace(-0.4, 0.4, 5)
    zs = np.linspace(-0.375, 0.375, 4)
    centres = []
    for z in zs:
        for y in ys:
            for x in xs:
                centres.append([x, y, z])
    return np.array(centres[:N])


def _initial_grid_v2():
    """5x5x4 grid with alternating layer offset."""
    xs = np.linspace(-0.4, 0.4, 5)
    ys = np.linspace(-0.4, 0.4, 5)
    zs = np.linspace(-0.375, 0.375, 4)
    centres = []
    for idx, z in enumerate(zs):
        offset = 0.08 if idx % 2 == 1 else 0.0
        for y in ys:
            for x in xs:
                centres.append([x + offset, y, z])
    return np.clip(np.array(centres[:N]), -0.5, 0.5)


def _initial_grid_v3():
    """4x5x5 grid (different axis ordering)."""
    xs = np.linspace(-0.375, 0.375, 4)
    ys = np.linspace(-0.4, 0.4, 5)
    zs = np.linspace(-0.4, 0.4, 5)
    centres = []
    for z in zs:
        for y in ys:
            for x in xs:
                centres.append([x, y, z])
    return np.array(centres[:N])


def _initial_bcc():
    """BCC-inspired: 5x5x4 with alternating diagonal offset."""
    xs = np.linspace(-0.42, 0.42, 5)
    ys = np.linspace(-0.42, 0.42, 5)
    zs = np.linspace(-0.39, 0.39, 4)
    centres = []
    for idx, z in enumerate(zs):
        off = 0.05 if idx % 2 == 1 else 0.0
        for y in ys:
            for x in xs:
                centres.append([x + off, y + off, z])
    return np.clip(np.array(centres[:N]), -0.5, 0.5)


def _initial_fcc():
    """FCC-inspired: 5x5x4 with alternating sign offsets."""
    xs = np.linspace(-0.42, 0.42, 5)
    ys = np.linspace(-0.42, 0.42, 5)
    zs = np.linspace(-0.39, 0.39, 4)
    centres = []
    for idx, z in enumerate(zs):
        off_x = 0.04 if idx % 2 == 0 else -0.04
        off_y = 0.04 if idx % 2 == 1 else -0.04
        for y in ys:
            for x in xs:
                centres.append([x + off_x, y + off_y, z])
    return np.clip(np.array(centres[:N]), -0.5, 0.5)


def _initial_random(seed):
    """Random initialization."""
    rng = np.random.RandomState(seed)
    return rng.uniform(-0.45, 0.45, (N, 3))


def _optimize_smooth(c, alpha, maxiter, deadline, ftol=1e-12, gtol=1e-8):
    """Run one L-BFGS-B step with the smooth log-sum-exp objective."""
    if time.time() > deadline:
        return c

    def obj(x):
        if time.time() > deadline:
            raise _Timeout()
        val, _ = _compute_smooth_obj_grad(x.reshape(N, 3), alpha)
        return val

    def grad(x):
        if time.time() > deadline:
            raise _Timeout()
        _, g = _compute_smooth_obj_grad(x.reshape(N, 3), alpha)
        return g.flatten()

    x0 = c.flatten()
    bounds = [(-0.5, 0.5)] * (3 * N)
    try:
        result = minimize(
            obj, x0, jac=grad, method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol},
        )
        c_new = np.clip(result.x.reshape(N, 3), -0.5, 0.5)
        if np.all(np.isfinite(c_new)) and result.fun < obj(x0) - 1e-13:
            return c_new
    except _Timeout:
        pass
    except Exception:
        pass
    return c


def _optimize_penalty(c, target_r, maxiter, deadline, ftol=1e-14, gtol=1e-10):
    """Run one L-BFGS-B step with the penalty objective."""
    if time.time() > deadline:
        return c

    def obj(x):
        if time.time() > deadline:
            raise _Timeout()
        val, _ = _compute_penalty_obj_grad(x.reshape(N, 3), target_r)
        return val

    def grad(x):
        if time.time() > deadline:
            raise _Timeout()
            _, g = _compute_penalty_obj_grad(x.reshape(N, 3), target_r)
        return g.flatten()

    x0 = c.flatten()
    bounds = [(-0.5, 0.5)] * (3 * N)
    try:
        result = minimize(
            obj, x0, jac=grad, method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol},
        )
        c_new = np.clip(result.x.reshape(N, 3), -0.5, 0.5)
        if np.all(np.isfinite(c_new)):
            return c_new
    except _Timeout:
        pass
    except Exception:
        pass
    return c


def _anneal(c_init, deadline, start_time):
    """
    Multi-alpha annealing from a given start.
    Returns the best configuration found.
    """
    best_c = c_init.copy()
    best_score = _score(c_init)

    # Alpha schedule: smooth to sharp
    alphas = [5, 10, 20, 50, 100, 200, 500, 1000, 2000]

    for alpha in alphas:
        if time.time() > deadline:
            break

        remaining = deadline - time.time()
        if remaining < 5.0:
            break

        # Scale maxiter based on time remaining
        if remaining > 30.0:
            maxiter = 300
        elif remaining > 15.0:
            maxiter = 150
        else:
            maxiter = 50

        c_new = _optimize_smooth(best_c, alpha, maxiter, deadline)
        score_new = _score(c_new)

        if score_new > best_score:
            best_score = score_new
            best_c = c_new.copy()
        else:
            # Even if score didn't improve, the position might be better
            # for the next alpha stage
            if np.all(np.isfinite(c_new)):
                best_c = c_new.copy()

    return best_c, best_score


def _polish(c_init, deadline):
    """
    Final polish: high-alpha smooth refinement + penalty equalization.
    """
    best_c = c_init.copy()
    best_score = _score(c_init)

    # Phase 1: High-alpha smooth refinement
    alphas_polish = [500, 1000, 2000, 5000]
    for alpha in alphas_polish:
        if time.time() > deadline:
            break
        remaining = deadline - time.time()
        if remaining < 3.0:
            break

        maxiter = min(100, int(remaining * 10))
        if maxiter < 10:
            maxiter = 10

        c_new = _optimize_smooth(best_c, alpha, maxiter, deadline,
                                  ftol=1e-14, gtol=1e-10)
        score_new = _score(c_new)
        if score_new > best_score:
            best_score = score_new
            best_c = c_new.copy()

    # Phase 2: Penalty-based equalization
    # Target slightly above current best to push binding constraints
    target_r = best_score * 1.001
    for _ in range(3):
        if time.time() > deadline:
            break
        remaining = deadline - time.time()
        if remaining < 3.0:
            break

        maxiter = min(80, int(remaining * 8))
        if maxiter < 10:
            maxiter = 10

        c_new = _optimize_penalty(best_c, target_r, maxiter, deadline)
        score_new = _score(c_new)
        if score_new > best_score:
            best_score = score_new
            best_c = c_new.copy()
            # Update target to push further
            target_r = best_score * 1.001
        else:
            # If penalty didn't help, try a lower target
            target_r = best_score * 0.999
            c_new = _optimize_penalty(best_c, target_r, maxiter, deadline)
            score_new = _score(c_new)
            if score_new > best_score:
                best_score = score_new
                best_c = c_new.copy()
                target_r = best_score * 1.001

    return best_c, best_score


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    best_c = None
    best_score = 0.0

    # Multi-start: try several initializations
    init_configs = [
        _initial_grid(),
        _initial_grid_v2(),
        _initial_grid_v3(),
        _initial_bcc(),
        _initial_fcc(),
        _initial_random(123),
        _initial_random(456),
    ]

    # Allocate time among starts: give more time to earlier (structured) starts
    n_starts = len(init_configs)
    # Weighted time allocation
    weights = [0.25, 0.15, 0.15, 0.12, 0.12, 0.10, 0.11]
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]

    total_time = DEADLINE - 25.0  # Reserve 25s for polish
    elapsed = 0.0

    for i, c_init in enumerate(init_configs):
        if time.time() > deadline - 25.0:
            break

        # Time budget for this start
        t_budget = total_time * weights[i]
        t_deadline = start_time + elapsed + t_budget

        # Ensure we don't exceed the global deadline minus polish time
        t_deadline = min(t_deadline, deadline - 25.0)

        c_result, s_result = _anneal(c_init, t_deadline, start_time)

        if s_result > best_score:
            best_score = s_result
            best_c = c_result.copy()

        elapsed = time.time() - start_time

    # Final polish on the best configuration
    if best_c is not None and time.time() < deadline - 5.0:
        c_polished, s_polished = _polish(best_c, deadline - 1.0)
        if s_polished > best_score:
            best_score = s_polished
            best_c = c_polished.copy()

    # Fallback
    if best_c is None:
        best_c = _initial_grid()

    # Ensure validity
    best_c = np.clip(best_c, -0.5, 0.5)

    return best_c
