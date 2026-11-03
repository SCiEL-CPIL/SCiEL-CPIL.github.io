"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import itertools

# Precompute all pairs for efficiency
_PAIRS = list(itertools.combinations(range(100), 2))
_PAIR_IDX = np.array(_PAIRS, dtype=np.int32)  # shape (M, 2)
M_PAIRS = len(_PAIR_IDX)

def _score(c):
    """Compute the packing score (radius) for centers c of shape (100,3)."""
    # Check validity
    if c.shape != (100, 3):
        return 0.0
    if not np.all(np.isfinite(c)):
        return 0.0
    # Centers must be inside the cube [-0.5, 0.5]^3
    if np.any(np.abs(c) > 0.5 + 1e-12):
        return 0.0
    # Compute pairwise distances
    diffs = c[_PAIR_IDX[:, 0]] - c[_PAIR_IDX[:, 1]]
    dists = np.sqrt(np.sum(diffs ** 2, axis=1))
    min_pair_dist = np.min(dists)
    # Distance to walls
    wall_dists = 0.5 - np.abs(c)  # shape (100, 3)
    min_wall_dist = np.min(wall_dists)
    r = min(min_pair_dist / 2.0, min_wall_dist)
    return r

def _energy_and_grad(c, r_target):
    """
    Energy: sum of squared violations for pairs closer than 2*r_target
           + sum of squared violations for centers too close to walls.
    Returns (energy, gradient).
    """
    c = np.asarray(c, dtype=np.float64)
    # Pairwise violations
    diffs = c[_PAIR_IDX[:, 0]] - c[_PAIR_IDX[:, 1]]
    dists_sq = np.sum(diffs ** 2, axis=1)
    min_dist_sq = (2.0 * r_target) ** 2
    violations = np.maximum(0.0, min_dist_sq - dists_sq)
    energy = np.sum(violations)
    
    # Wall violations
    # For each coordinate, violation if c[i,k] > 0.5 - r_target or c[i,k] < -0.5 + r_target
    # Actually, the constraint is |c[i,k]| <= 0.5 - r_target
    # Violation: max(0, |c[i,k]| - (0.5 - r_target))^2
    margin = 0.5 - r_target
    abs_c = np.abs(c)
    wall_viol = np.maximum(0.0, abs_c - margin) ** 2
    energy += np.sum(wall_viol)
    
    # Gradient
    grad = np.zeros_like(c)
    
    # Pairwise gradient
    # For each pair (i,j), if dist_sq < min_dist_sq:
    #   d(energy)/d(c_i) = -2 * (min_dist_sq - dist_sq) * (c_i - c_j) / ... 
    # Actually, energy_pair = (min_dist_sq - dist_sq)^2 when violated
    # d(energy_pair)/d(c_i) = -2*(min_dist_sq - dist_sq) * d(dist_sq)/d(c_i)
    # d(dist_sq)/d(c_i) = 2*(c_i - c_j)
    # So d(energy_pair)/d(c_i) = -4*(min_dist_sq - dist_sq)*(c_i - c_j)
    # But we have sum of violations, where violation = max(0, min_dist_sq - dist_sq)
    # Wait, I defined violations as max(0, min_dist_sq - dist_sq), and energy = sum(violations)
    # So d(energy)/d(c_i) for pair (i,j): if dist_sq < min_dist_sq, d/d(c_i) = -2*(c_i - c_j)
    # Let me recompute: violation = min_dist_sq - dist_sq (when positive)
    # d(violation)/d(c_i) = -d(dist_sq)/d(c_i) = -2*(c_i - c_j)
    # So grad contribution to c_i: -2*(c_i - c_j) * (1 if violated else 0)
    # grad contribution to c_j: -2*(c_j - c_i) * (1 if violated else 0) = 2*(c_i - c_j)
    
    violated = dists_sq < min_dist_sq
    if np.any(violated):
        v_diffs = diffs[violated]  # c_i - c_j for violated pairs
        # grad[i] += -2 * (c_i - c_j) for each violated pair involving i
        # grad[j] += 2 * (c_i - c_j) for each violated pair involving j
        # Use np.add.at for accumulation
        np.add.at(grad, _PAIR_IDX[violated, 0], -2.0 * v_diffs)
        np.add.at(grad, _PAIR_IDX[violated, 1], 2.0 * v_diffs)
    
    # Wall gradient
    # violation = max(0, |c| - margin)^2
    # d(violation)/d(c) = 2*max(0, |c| - margin) * sign(c)
    wall_v = np.maximum(0.0, abs_c - margin)
    if np.any(wall_v > 0):
        grad += 2.0 * wall_v * np.sign(c)
    
    return energy, grad

def _local_optimize(c0, r_target, max_iter=500, tol=1e-10):
    """
    Local optimization to reduce energy for a given r_target.
    Returns (c_final, energy_final).
    """
    c0 = np.asarray(c0, dtype=np.float64).copy()
    
    def obj_and_grad(x):
        e, g = _energy_and_grad(x, r_target)
        return e, g
    
    # Use L-BFGS-B with bounds
    bounds = [(-0.5, 0.5)] * 300
    res = minimize(obj_and_grad, c0, jac=True, method='L-BFGS-B', 
                   bounds=bounds, options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-8})
    return res.x, res.fun

def _try_radius(c0, r_target, max_iter=500):
    """Try to achieve radius r_target from starting config c0."""
    c_opt, energy = _local_optimize(c0, r_target, max_iter=max_iter)
    # Check if we achieved it
    actual_r = _score(c_opt)
    return c_opt, actual_r, energy

def construct():
    """Construct centers for 100 equal spheres in unit cube."""
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Initial configuration: cubic grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    c = grid[:N].copy()
    
    # Initial score
    best_r = _score(c)
    best_c = c.copy()
    
    # Binary search on radius
    # Lower bound: initial score
    # Upper bound: theoretical max ~ 0.5/3 ≈ 0.166, but realistic ~0.12
    r_lo = best_r
    r_hi = 0.15
    
    # First, do a local optimization at the initial radius to improve
    if time.time() < deadline - 10:
        c_opt, e = _local_optimize(c, r_lo, max_iter=200)
        r_opt = _score(c_opt)
        if r_opt > best_r:
            best_r = r_opt
            best_c = c_opt.copy()
            r_lo = r_opt
    
    # Binary search for maximum radius
    max_iters = 12
    for it in range(max_iters):
        if time.time() > deadline - 15:
            break
        
        r_mid = 0.5 * (r_lo + r_hi)
        # Try to achieve r_mid
        c_try, r_achieved, energy = _try_radius(best_c, r_mid, max_iter=300)
        
        if r_achieved >= r_mid - 1e-8:
            # Success
            r_lo = r_mid
            if r_achieved > best_r:
                best_r = r_achieved
                best_c = c_try.copy()
        else:
            # Failure
            r_hi = r_mid
        
        # Check convergence
        if r_hi - r_lo < 1e-6:
            break
    
    # Final polish: try to improve with more iterations at the best radius
    if time.time() < deadline - 5:
        c_polish, e = _local_optimize(best_c, best_r, max_iter=500, tol=1e-12)
        r_polish = _score(c_polish)
        if r_polish > best_r:
            best_r = r_polish
            best_c = c_polish.copy()
    
    # Ensure valid output
    best_c = np.clip(best_c, -0.5 + 1e-12, 0.5 - 1e-12)
    
    return best_c
# EVOLVE-BLOCK-END
