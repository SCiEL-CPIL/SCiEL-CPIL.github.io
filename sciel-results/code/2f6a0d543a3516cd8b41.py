"""Fifty equal circles in a square – optimised packing."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

N = 50
SIDE = 1.0
HALF = 0.5
DEADLINE = 118.0  # safety margin before 120s kill


def _score_from_centres(c):
    """Compute the radius score for given centres (N,2)."""
    if c.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(c)):
        return 0.0
    # Check inside square
    if np.any(c < -HALF + 1e-12) or np.any(c > HALF - 1e-12):
        return 0.0
    # Pairwise distances using matrix
    diff = c[:, None, :] - c[None, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = dists.min()
    # Distance to walls
    wall_dists = np.stack([
        c[:, 0] + HALF,
        HALF - c[:, 0],
        c[:, 1] + HALF,
        HALF - c[:, 1]
    ], axis=1)
    min_wall = wall_dists.min()
    r = min(min_pair / 2.0, min_wall)
    return r


def _smooth_score(c_flat, beta=50.0):
    """Smooth surrogate for the radius score using log-sum-exp."""
    c = c_flat.reshape(N, 2)
    c = np.clip(c, -HALF + 1e-9, HALF - 1e-9)
    
    # Pairwise distances
    diff = c[:, None, :] - c[None, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Wall distances for each point
    wall_dists = np.stack([
        c[:, 0] + HALF,
        HALF - c[:, 0],
        c[:, 1] + HALF,
        HALF - c[:, 1]
    ], axis=1)  # shape (N, 4)
    
    # Soft-min of pairwise distances
    d_half = dists / 2.0
    min_d_half = d_half.min()
    exp_d = np.exp(-beta * (d_half - min_d_half))
    soft_min_pair = min_d_half - np.log(exp_d.sum()) / beta
    
    # Soft-min of wall distances
    min_wall = wall_dists.min()
    exp_w = np.exp(-beta * (wall_dists - min_wall))
    soft_min_wall = min_wall - np.log(exp_w.sum()) / beta
    
    # Combine
    all_dists = np.concatenate([soft_min_pair.reshape(1), soft_min_wall.reshape(1)])
    min_all = all_dists.min()
    exp_all = np.exp(-beta * (all_dists - min_all))
    r = min_all - np.log(exp_all.sum()) / beta
    
    return r


def _negative_score(c_flat, beta=50.0):
    """Objective to minimise (negative smooth radius score)."""
    return -_smooth_score(c_flat, beta)


def _penalty_function(c_flat, r_target):
    """
    Penalty function for feasibility check at radius r_target.
    Returns 0 if configuration is valid for r_target.
    """
    c = c_flat.reshape(N, 2)
    
    # Pairwise distances
    diff = c[:, None, :] - c[None, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Violations: max(0, 2*r_target - dist)
    viol_pair = np.maximum(0, 2 * r_target - dists)
    penalty_pair = np.sum(viol_pair ** 2)
    
    # Wall distances
    wall_dists = np.stack([
        c[:, 0] + HALF,
        HALF - c[:, 0],
        c[:, 1] + HALF,
        HALF - c[:, 1]
    ], axis=1)
    
    # Violations: max(0, r_target - dist)
    viol_wall = np.maximum(0, r_target - wall_dists)
    penalty_wall = np.sum(viol_wall ** 2)
    
    return penalty_pair + penalty_wall


def _grid_init():
    """Initial grid layout."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N]


def _hex_init():
    """Hexagonal packing initial layout."""
    rows = int(np.ceil(N / 8))
    cols = 8
    points = []
    for i in range(rows):
        for j in range(cols):
            if len(points) >= N:
                break
            x = (j - (cols - 1) / 2) * 0.22
            y = (i - (rows - 1) / 2) * 0.2
            if i % 2 == 1:
                x += 0.11
            points.append([x, y])
    return np.array(points[:N])


def _random_init(rng):
    """Random initial layout."""
    return rng.uniform(-HALF, HALF, size=(N, 2))


def construct():
    t0 = time.time()
    rng = np.random.default_rng(42)
    
    # Phase 1: Global Optimization with Smooth Surrogate
    best_centres = _grid_init()
    best_score = _score_from_centres(best_centres)
    
    inits = [_grid_init(), _hex_init()]
    for init in inits:
        s = _score_from_centres(init)
        if s > best_score:
            best_score = s
            best_centres = init

    n_starts = 20
    for i in range(n_starts):
        if time.time() - t0 > DEADLINE - 40:  # Leave 40s for Phase 2
            break
        
        if i == 0:
            x0 = best_centres.ravel()
        elif i == 1:
            x0 = _hex_init().ravel()
        elif i == 2:
            x0 = _grid_init().ravel()
        else:
            x0 = _random_init(rng).ravel()
            x0 = x0 + rng.normal(0, 0.03, size=x0.shape)

        try:
            # Adaptive beta
            for beta in [15.0, 30.0, 60.0, 120.0]:
                res = minimize(
                    lambda cf, b=beta: _negative_score(cf, b), x0,
                    method='L-BFGS-B',
                    bounds=[(-HALF + 1e-9, HALF - 1e-9)] * (2 * N),
                    options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10, 'iprint': 0}
                )
                c = np.clip(res.x.reshape(N, 2), -HALF + 1e-9, HALF - 1e-9)
                s = _score_from_centres(c)
                if s > best_score:
                    best_score = s
                    best_centres = c
                x0 = c.ravel()
        except Exception:
            pass

    # Phase 2: Radius Refinement via Bisection and Penalty Minimization
    # We try to increase the radius from best_score.
    r_low = best_score
    r_high = best_score + 0.005  # Initial guess for upper bound
    
    # Ensure r_high is not too high (max possible is 0.5/sqrt(50) approx 0.07)
    # Actually 0.0713 is the known best.
    
    x_best = best_centres.copy()
    
    # Bisection loop
    max_bisection_iters = 30
    for _ in range(max_bisection_iters):
        if time.time() - t0 > DEADLINE - 10:
            break
        
        r_mid = (r_low + r_high) / 2.0
        
        # Check feasibility for r_mid
        # Minimize penalty function
        try:
            res = minimize(
                lambda cf: _penalty_function(cf, r_mid),
                x_best.ravel(),
                method='L-BFGS-B',
                bounds=[(-HALF + 1e-9, HALF - 1e-9)] * (2 * N),
                options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-12, 'iprint': 0}
            )
            c_candidate = np.clip(res.x.reshape(N, 2), -HALF + 1e-9, HALF - 1e-9)
            pen_val = res.fun
            
            # If penalty is very small, it's feasible
            if pen_val < 1e-10:
                r_low = r_mid
                x_best = c_candidate
            else:
                r_high = r_mid
        except Exception:
            # If optimization fails, assume infeasible to be safe
            r_high = r_mid

    # After bisection, r_low is the best feasible radius found.
    # We need to get the corresponding centers.
    # The last feasible x_best corresponds to r_low (or close to it).
    # Let's do a final polish on x_best with the soft-min objective to ensure it's optimal for that configuration.
    
    # Phase 3: Final Polish
    if time.time() - t0 < DEADLINE - 2:
        try:
            # Polish with high beta soft-min
            for beta in [200.0, 400.0]:
                res = minimize(
                    lambda cf, b=beta: _negative_score(cf, b), x_best.ravel(),
                    method='L-BFGS-B',
                    bounds=[(-HALF + 1e-9, HALF - 1e-9)] * (2 * N),
                    options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-14, 'iprint': 0}
                )
                c_final = np.clip(res.x.reshape(N, 2), -HALF + 1e-9, HALF - 1e-9)
                s_final = _score_from_centres(c_final)
                if s_final > best_score:
                    best_score = s_final
                    best_centres = c_final
        except Exception:
            pass

    # Ensure the returned centres correspond to the best score
    # It's possible the bisection found a higher r_low, but the final polish on x_best
    # (which might have been from an earlier feasible step) didn't capture it if x_best wasn't updated correctly.
    # Let's re-evaluate the score of x_best and best_centres.
    s_x_best = _score_from_centres(x_best)
    if s_x_best > best_score:
        best_score = s_x_best
        best_centres = x_best

    best_centres = np.clip(best_centres, -HALF + 1e-9, HALF - 1e-9)
    return best_centres
