"""Combined program: LSE Annealing + Directed Perturbation + Adaptive Penalty Polish."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 100

# EVOLVE-BLOCK-START

def _all_dists(c):
    """Return (pair_dists, wall_dists, i_idx, j_idx, d)."""
    d = cdist(c, c)
    i_idx, j_idx = np.triu_indices(N, k=1)
    pair_dists = d[i_idx, j_idx]
    wall_dists = np.concatenate([
        0.5 - c[:, 0],
        0.5 + c[:, 0],
        0.5 - c[:, 1],
        0.5 + c[:, 1],
        0.5 - c[:, 2],
        0.5 + c[:, 2],
    ])
    return pair_dists, wall_dists, i_idx, j_idx, d


def _compute_smooth_obj_grad(c, alpha):
    """
    Numerically-stable smooth log-sum-exp objective + gradient.
    f(c) = (1/alpha) * log( sum_s exp(-alpha * s) )
    """
    pair_dists, wall_dists, i_idx, j_idx, d = _all_dists(c)
    all_dists = np.concatenate([pair_dists / 2.0, wall_dists])

    exp_arg = -alpha * all_dists
    max_arg = np.max(exp_arg)
    exp_terms = np.exp(exp_arg - max_arg)

    Z = np.sum(exp_terms)
    if Z == 0.0:
        return 0.0, np.zeros_like(c)

    obj_val = (1.0 / alpha) * (np.log(Z) + max_arg)

    grad_c = np.zeros_like(c)

    # Pairwise part
    exp_pair = exp_terms[:len(pair_dists)]
    weights_pair = exp_pair / Z

    pair_dists_raw = d[i_idx, j_idx]
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
    exp_wall = exp_terms[len(pair_dists):]
    weights_wall = exp_wall / Z
    ww = weights_wall.reshape(6, N)

    grad_c[:, 0] += ww[0] - ww[1]
    grad_c[:, 1] += ww[2] - ww[3]
    grad_c[:, 2] += ww[4] - ww[5]

    return obj_val, grad_c


def _compute_penalty_obj_grad(c, target_r):
    """
    Penalty objective: sum of squared violations of the target radius.
    """
    pair_dists, wall_dists, i_idx, j_idx, d = _all_dists(c)
    all_dists = np.concatenate([pair_dists / 2.0, wall_dists])

    violations = np.maximum(0.0, target_r - all_dists)
    obj_val = float(np.sum(violations ** 2))

    grad_c = np.zeros_like(c)

    # Pairwise part
    viol_pair = violations[:len(pair_dists)]
    active = viol_pair > 1e-12
    if np.any(active):
        pair_dists_raw = d[i_idx, j_idx]
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
    viol_wall = violations[len(pair_dists):]
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
    xs = np.linspace(-0.4, 0.4, 5)
    ys = np.linspace(-0.4, 0.4, 5)
    zs = np.linspace(-0.375, 0.375, 4)
    centres = []
    for idx, z in enumerate(zs):
        offset = (0.08 if idx % 2 == 1 else 0.0)
        for y in ys:
            for x in xs:
                centres.append([x + offset, y, z])
    return np.array(centres[:N])


def _initial_bcc():
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


class _Timeout(Exception):
    pass


def _optimize_smooth(c, alpha, maxiter, deadline, ftol=1e-12, gtol=1e-8):
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


def _directed_perturbation(c, magnitude=0.01):
    """
    Identify the most constrained pair and wall, and push them apart/inwards.
    """
    c_new = c.copy()
    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    min_d = np.min(d)
    i, j = np.unravel_index(np.argmin(d), d.shape)
    
    # Push i and j apart
    diff = c[i] - c[j]
    norm = np.linalg.norm(diff)
    if norm > 1e-12:
        u = diff / norm
        c_new[i] += magnitude * u
        c_new[j] -= magnitude * u
    
    # Check walls
    wall_dists = np.concatenate([
        0.5 - c[:, 0], 0.5 + c[:, 0],
        0.5 - c[:, 1], 0.5 + c[:, 1],
        0.5 - c[:, 2], 0.5 + c[:, 2],
    ])
    min_w = np.min(wall_dists)
    w_idx = np.argmin(wall_dists)
    sphere_idx = w_idx % N
    face_idx = w_idx // N
    
    # Push sphere_idx away from the closest wall
    if face_idx == 0: c_new[sphere_idx, 0] -= magnitude
    elif face_idx == 1: c_new[sphere_idx, 0] += magnitude
    elif face_idx == 2: c_new[sphere_idx, 1] -= magnitude
    elif face_idx == 3: c_new[sphere_idx, 1] += magnitude
    elif face_idx == 4: c_new[sphere_idx, 2] -= magnitude
    elif face_idx == 5: c_new[sphere_idx, 2] += magnitude
    
    return np.clip(c_new, -0.49, 0.49)


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    np.random.seed(42)

    # 1) Pick best initial configuration
    init_configs = [_initial_grid(), _initial_grid_v2(), _initial_bcc(), _initial_fcc()]
    # Add some random starts
    for _ in range(3):
        init_configs.append(np.random.uniform(-0.4, 0.4, (N, 3)))
        
    best_c = None
    best_score = 0.0
    for init_c in init_configs:
        s = _score(init_c)
        if s > best_score:
            best_score = s
            best_c = init_c.copy()

    # 2) Main Search Loop: LSE Optimization + Directed Perturbation
    alphas = [50, 100, 200, 500]
    c = best_c.copy()
    
    def try_update(cand):
        nonlocal best_c, best_score
        s = _score(cand)
        if s > best_score:
            best_score = s
            best_c = cand.copy()
            return True
        return False

    # Phase 1: Coarse to Fine LSE Annealing
    for alpha in alphas:
        if time.time() > deadline - 30.0:
            break
        c_new = _optimize_smooth(c, alpha, 100, deadline)
        if try_update(c_new):
            c = c_new.copy()
        
        # Directed Perturbation to escape local minima
        c_pert = _directed_perturbation(c, 0.02)
        c_pol = _optimize_smooth(c_pert, alpha, 50, deadline)
        if try_update(c_pol):
            c = c_pol.copy()

    # Phase 2: High Alpha Polish with Random Kicks
    while time.time() < deadline - 20.0:
        c_new = _optimize_smooth(c, 500, 100, deadline)
        if try_update(c_new):
            c = c_new.copy()
        else:
            # Random kick
            kick = 0.01
            c_kick = np.clip(c + kick * np.random.randn(N, 3), -0.49, 0.49)
            c_pol = _optimize_smooth(c_kick, 500, 50, deadline)
            if try_update(c_pol):
                c = c_pol.copy()
            else:
                # Try directed perturbation
                c_dir = _directed_perturbation(c, 0.01)
                c_pol = _optimize_smooth(c_dir, 500, 50, deadline)
                if try_update(c_pol):
                    c = c_pol.copy()
                else:
                    break # No improvement in several kicks

    # 3) Adaptive Penalty Polish
    current_r = best_score
    step_r = 0.0005
    
    while time.time() < deadline - 5.0:
        target_r = current_r + step_r
        c_pol = _optimize_penalty(best_c, target_r, 100, deadline)
        s_pol = _score(c_pol)
        
        if s_pol > best_score:
            best_score = s_pol
            best_c = c_pol.copy()
            if s_pol > current_r + 0.5 * step_r:
                step_r = min(step_r * 1.5, 0.001)
            else:
                step_r = max(step_r * 0.8, 0.0001)
            current_r = s_pol
        else:
            step_r = max(step_r * 0.7, 0.00005)
            if step_r < 0.0001:
                break

    # 4) Final Polish
    if time.time() < deadline - 2.0:
        c_final = _optimize_penalty(best_c, best_score + 0.0001, 100, deadline)
        try_update(c_final)

    best_c = np.clip(best_c, -0.5, 0.5)
    return best_c

# EVOLVE-BLOCK-END
