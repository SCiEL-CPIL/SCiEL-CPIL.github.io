"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_radius(centres):
    """Compute the maximum common radius for given centres (vectorized)."""
    if centres.shape[0] != N:
        return 0.0
    if not np.all(np.isfinite(centres)):
        return 0.0
    # Check boundaries: centres must be in [-0.5, 0.5]^2
    if np.any(centres < -0.5 - 1e-12) or np.any(centres > 0.5 + 1e-12):
        return 0.0
    
    # Distance to boundaries
    boundary_dist = np.minimum(
        np.minimum(0.5 - centres[:, 0], 0.5 + centres[:, 0]),
        np.minimum(0.5 - centres[:, 1], 0.5 + centres[:, 1])
    )
    min_boundary = np.min(boundary_dist)
    
    # Pairwise distances
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist) / 2.0
    
    return min(min_boundary, min_pair)

def _smooth_min_penalty(centres, target_r, alpha=50.0):
    """
    Smooth differentiable penalty that approximates the constraint:
    - All pairwise distances >= 2*target_r
    - All boundary distances >= target_r
    Uses soft-min (log-sum-exp) for smoothness.
    """
    # Pairwise distances
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    d2 = np.sum(diff**2, axis=2)
    np.fill_diagonal(d2, np.inf)
    
    # Convert to distances
    dists = np.sqrt(np.maximum(d2, 1e-15))
    
    # Only consider upper triangle to avoid double counting
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]
    
    # Soft minimum of (pair_dist - 2*target_r)
    # We want to penalize when pair_dist < 2*target_r
    # Use: -softmin(pair_dists - 2*target_r) as penalty
    shifted_pairs = pair_dists - 2.0 * target_r
    # softmin via log-sum-exp (negative version for min)
    m = np.min(shifted_pairs)
    softmin_pairs = m - np.log(np.sum(np.exp(shifted_pairs - m), axis=0))
    
    # Boundary distances
    bx = np.minimum(0.5 - centres[:, 0], 0.5 + centres[:, 0])
    by = np.minimum(0.5 - centres[:, 1], 0.5 + centres[:, 1])
    bdist = np.minimum(bx, by)
    shifted_bounds = bdist - target_r
    m_b = np.min(shifted_bounds)
    softmin_bounds = m_b - np.log(np.sum(np.exp(shifted_bounds - m_b), axis=0))
    
    # Combined penalty: we want both to be >= 0
    # Penalty = max(0, -softmin_pairs) + max(0, -softmin_bounds)
    penalty_pairs = np.maximum(0.0, -softmin_pairs)
    penalty_bounds = np.maximum(0.0, -softmin_bounds)
    
    return penalty_pairs + penalty_bounds

def _neg_smooth_objective(x, target_r, alpha=50.0):
    """
    Objective: maximize target_r subject to constraints, using smooth penalty.
    We minimize: -target_r + alpha * penalty
    """
    centres = x.reshape(N, 2)
    penalty = _smooth_min_penalty(centres, target_r, alpha)
    return -target_r + alpha * penalty

def _generate_hex_initial(row_counts, jitter=0.0):
    """Generate hexagonal lattice initial configuration with optional jitter."""
    r = 0.07
    h = 2 * r
    v = np.sqrt(3) * r
    
    centres = []
    for i, count in enumerate(row_counts):
        y = 0.5 - v/2 - i * v
        xs = np.arange(count) * h - (count - 1) * h / 2
        if i % 2 == 1:
            xs += h / 2
        for x in xs:
            centres.append([x, y])
    
    centres = np.array(centres[:N], dtype=np.float64)
    
    if jitter > 0:
        rng = np.random.default_rng(42)
        centres += rng.normal(0, jitter, size=centres.shape)
    
    return centres

def _optimize_with_smooth(centres0, target_r_init, deadline):
    """
    Optimize using smooth penalty objective.
    First phase: find feasible configuration for target_r_init
    Second phase: increase target_r and re-optimize
    """
    def obj(x):
        c = x.reshape(N, 2)
        # Return large penalty if out of bounds
        if np.any(c < -0.5 - 0.01) or np.any(c > 0.5 + 0.01):
            return 1000.0
        return _neg_smooth_objective(x, target_r_init, alpha=100.0)
    
    x0 = centres0.flatten()
    
    # Phase 1: Optimize for initial target radius
    result = minimize(
        obj,
        x0,
        method='L-BFGS-B',
        bounds=[(-0.5, 0.5)] * (2 * N),
        options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
    )
    
    best_c = result.x.reshape(N, 2)
    best_r = _compute_radius(best_c)
    
    # Phase 2: Try to increase radius
    for trial in range(5):
        if time.time() > deadline:
            break
        target_r = best_r * 1.02
        if target_r > 0.075:
            break
        
        def obj2(x, tr=target_r):
            c = x.reshape(N, 2)
            if np.any(c < -0.5 - 0.01) or np.any(c > 0.5 + 0.01):
                return 1000.0
            return _neg_smooth_objective(x, tr, alpha=100.0)
        
        result2 = minimize(
            obj2,
            best_c.flatten(),
            method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (2 * N),
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        
        c2 = result2.x.reshape(N, 2)
        r2 = _compute_radius(c2)
        
        if r2 > best_r:
            best_r = r2
            best_c = c2
        else:
            break
    
    return best_c, best_r

def _direct_optimize(centres0, deadline):
    """Direct optimization of negative radius (non-smooth but simple)."""
    def obj(x):
        c = x.reshape(N, 2)
        return -_compute_radius(c)
    
    x0 = centres0.flatten()
    
    result = minimize(
        obj,
        x0,
        method='L-BFGS-B',
        bounds=[(-0.5, 0.5)] * (2 * N),
        options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
    )
    
    best_c = result.x.reshape(N, 2)
    best_r = _compute_radius(best_c)
    
    return best_c, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    best_centres = None
    best_r = 0.0
    
    # Multiple hexagonal configurations with different row distributions
    row_configs = [
        [7, 6, 7, 6, 7, 6, 7, 4],
        [6, 7, 6, 7, 6, 7, 6, 5],
        [7, 7, 6, 7, 6, 7, 6, 4],
        [8, 7, 6, 7, 6, 6, 7, 3],
        [7, 6, 7, 7, 6, 6, 7, 4],
        [6, 7, 7, 6, 7, 6, 6, 5],
        [5, 7, 7, 7, 7, 7, 6, 4],
        [7, 7, 7, 6, 7, 6, 6, 4],
    ]
    
    # Try each configuration with smooth optimization
    for config in row_configs:
        if sum(config) != N:
            continue
        if time.time() > deadline:
            break
        
        # Generate initial with small jitter
        c0 = _generate_hex_initial(config, jitter=0.005)
        
        # Smooth optimization
        c_opt, r_opt = _optimize_with_smooth(c0, 0.071, deadline)
        
        if r_opt > best_r:
            best_r = r_opt
            best_centres = c_opt.copy()
    
    # Also try direct optimization from best result so far
    if best_centres is not None and time.time() < deadline:
        c_opt, r_opt = _direct_optimize(best_centres, deadline)
        if r_opt > best_r:
            best_r = r_opt
            best_centres = c_opt.copy()
    
    # Try more configurations if time permits
    if time.time() < deadline - 30:
        extra_configs = [
            [6, 6, 7, 7, 6, 7, 7, 4],
            [7, 6, 6, 7, 7, 6, 7, 4],
            [6, 7, 6, 6, 7, 7, 7, 4],
        ]
        for config in extra_configs:
            if sum(config) != N:
                continue
            if time.time() > deadline:
                break
            c0 = _generate_hex_initial(config, jitter=0.01)
            c_opt, r_opt = _optimize_with_smooth(c0, 0.071, deadline)
            if r_opt > best_r:
                best_r = r_opt
                best_centres = c_opt.copy()
    
    # Fallback: grid if nothing worked
    if best_centres is None or best_r < 0.05:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centres = grid[:N]
        best_r = _compute_radius(best_centres)
    
    # Final validity check
    if not np.all(np.isfinite(best_centres)):
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        best_centres = np.array([(x, y) for y in ticks for x in ticks])[:N]
    
    # Clip to valid range
    best_centres = np.clip(best_centres, -0.5, 0.5)
    
    return best_centres
# EVOLVE-BLOCK-END
