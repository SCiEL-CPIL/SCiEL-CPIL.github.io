import numpy as np
import time
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

N = 100

def _calc_score(pts):
    """Calculate the actual score (radius) for a set of points."""
    if pts is None or pts.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    if np.any(pts < -0.5) or np.any(pts > 0.5):
        return 0.0
    dists = cdist(pts, pts)
    np.fill_diagonal(dists, 1e10)
    min_pair_dist = np.min(dists)
    min_wall_dist = np.min(0.5 - np.abs(pts))
    return min(min_pair_dist / 2.0, min_wall_dist)

def _energy_and_grad(pts, beta):
    """
    Compute soft-min energy and gradient.
    Energy = -(soft_min) so we minimize it.
    soft_min approximates the minimum of all pairwise distances/2 and wall distances.
    """
    # Pairwise distances / 2
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2)) / 2.0
    mask = ~np.eye(N, dtype=bool)
    dists_masked = np.where(mask, dists, 1.0)

    # Wall distances
    wall_dists = np.concatenate([
        (pts + 0.5).flatten(),
        (0.5 - pts).flatten()
    ])

    all_dists = np.concatenate([dists_masked.flatten(), wall_dists])

    d_min = np.min(all_dists)
    exp_terms = np.exp(-beta * (all_dists - d_min))
    log_sum = np.log(np.sum(exp_terms))
    soft_min_val = d_min + log_sum / beta
    energy = -soft_min_val

    weights = exp_terms / np.sum(exp_terms)

    grad = np.zeros_like(pts)

    # Pair contributions
    pair_weights = weights[:N * N].reshape(N, N)
    with np.errstate(divide='ignore', invalid='ignore'):
        factor = np.where(mask, pair_weights / (2.0 * dists_masked), 0.0)
    weighted_diff = diff * factor[:, :, np.newaxis]
    grad += np.sum(weighted_diff, axis=1)

    # Wall contributions
    wall_weights = weights[N * N:]
    grad += wall_weights[:N * 3].reshape(N, 3)
    grad -= wall_weights[N * 3:].reshape(N, 3)

    return energy, -grad  # Return gradient for minimization (negative of ascent grad)

def _initial_jittered_grid():
    """Generate a jittered 5x4x5 grid."""
    np.random.seed(123)
    x_ticks = np.linspace(-0.42, 0.42, 5)
    y_ticks = np.linspace(-0.39, 0.39, 4)
    z_ticks = np.linspace(-0.42, 0.42, 5)
    pts = []
    for x in x_ticks:
        for y in y_ticks:
            for z in z_ticks:
                pts.append([x, y, z])
                if len(pts) >= N:
                    break
            if len(pts) >= N:
                break
        if len(pts) >= N:
            break
    pts = np.array(pts[:N])
    # Add small jitter to break symmetry
    pts += np.random.randn(N, 3) * 0.005
    pts = np.clip(pts, -0.49, 0.49)
    return pts

def _initial_random():
    """Random initial configuration."""
    np.random.seed(456)
    pts = np.random.uniform(-0.45, 0.45, (N, 3))
    return pts

def _optimize_from(pts0, deadline, seed_offset=0):
    """Run a multi-phase optimization from a starting point."""
    pts = pts0.copy()
    best_pts = pts.copy()
    best_score = _calc_score(pts)

    def time_left():
        return deadline - time.time()

    # Phase 1: L-BFGS with low beta
    if time_left() > 40:
        beta = 30.0
        res = minimize(
            _energy_and_grad, pts, args=(beta,),
            method='L-BFGS-B', jac=True,
            bounds=[(-0.499, 0.499)] * 3 * N,
            options={'maxiter': 500, 'gtol': 1e-5}
        )
        if res.success and res.x is not None:
            cand = np.clip(res.x, -0.499, 0.499)
            s = _calc_score(cand)
            if s > best_score:
                best_score = s
                best_pts = cand.copy()
                pts = cand

    if time_left() < 10:
        return best_pts, best_score

    # Phase 2: L-BFGS with medium beta
    if time_left() > 25:
        beta = 80.0
        res = minimize(
            _energy_and_grad, pts, args=(beta,),
            method='L-BFGS-B', jac=True,
            bounds=[(-0.499, 0.499)] * 3 * N,
            options={'maxiter': 400, 'gtol': 1e-6}
        )
        if res.success and res.x is not None:
            cand = np.clip(res.x, -0.499, 0.499)
            s = _calc_score(cand)
            if s > best_score:
                best_score = s
                best_pts = cand.copy()
                pts = cand

    if time_left() < 8:
        return best_pts, best_score

    # Phase 3: L-BFGS with high beta
    if time_left() > 15:
        beta = 200.0
        res = minimize(
            _energy_and_grad, pts, args=(beta,),
            method='L-BFGS-B', jac=True,
            bounds=[(-0.499, 0.499)] * 3 * N,
            options={'maxiter': 300, 'gtol': 1e-7}
        )
        if res.success and res.x is not None:
            cand = np.clip(res.x, -0.499, 0.499)
            s = _calc_score(cand)
            if s > best_score:
                best_score = s
                best_pts = cand.copy()
                pts = cand

    if time_left() < 5:
        return best_pts, best_score

    # Phase 4: Very high beta polish
    if time_left() > 8:
        beta = 500.0
        res = minimize(
            _energy_and_grad, pts, args=(beta,),
            method='L-BFGS-B', jac=True,
            bounds=[(-0.499, 0.499)] * 3 * N,
            options={'maxiter': 200, 'gtol': 1e-8}
        )
        if res.success and res.x is not None:
            cand = np.clip(res.x, -0.499, 0.499)
            s = _calc_score(cand)
            if s > best_score:
                best_score = s
                best_pts = cand.copy()

    return best_pts, best_score

def _perturb(pts, scale, seed):
    """Create a perturbed copy."""
    rng = np.random.RandomState(seed)
    perturbed = pts + rng.randn(N, 3) * scale
    return np.clip(perturbed, -0.49, 0.49)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    global N
    # Initial configurations
    inits = []
    inits.append(_initial_jittered_grid())
    inits.append(_initial_random())

    # Add a few perturbed versions of the grid
    for i in range(2):
        if time.time() < deadline - 30:
            inits.append(_perturb(_initial_jittered_grid(), 0.02 + 0.01 * i, 1000 + i))

    best_pts = inits[0].copy()
    best_score = _calc_score(inits[0])

    # Optimize from each initial configuration
    for idx, init_pts in enumerate(inits):
        if time.time() >= deadline - 20:
            break
        # Allocate time based on how many inits remain
        remaining_inits = len(inits) - idx
        if remaining_inits <= 1:
            t_alloc = deadline - time.time() - 5
        else:
            t_alloc = min(50.0, (deadline - time.time() - 20) / remaining_inits)

        if t_alloc < 5:
            break

        p, s = _optimize_from(init_pts, deadline)
        if s > best_score:
            best_score = s
            best_pts = p

    # If we have time left, do a few more perturbation attempts from the best
    if time.time() < deadline - 25:
        rng = np.random.RandomState(9999)
        for attempt in range(4):
            if time.time() >= deadline - 20:
                break
            scale = 0.015 * (attempt + 1)
            perturbed = _perturb(best_pts, scale, 7000 + attempt)
            t_alloc = min(30.0, deadline - time.time() - 15)
            if t_alloc < 5:
                break
            p, s = _optimize_from(perturbed, deadline)
            if s > best_score:
                best_score = s
                best_pts = p

    # Final validation
    final_pts = np.clip(best_pts, -0.5, 0.5)
    final_score = _calc_score(final_pts)
    if final_score <= 0:
        return np.clip(inits[0], -0.5, 0.5)

    return final_pts
