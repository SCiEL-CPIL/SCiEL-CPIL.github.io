"""
Solver for packing 50 equal spheres in a unit cube [-0.5, 0.5]^3.
Strategy: Multi-start with diverse seeds, followed by Repulsion Annealing,
Smooth Gradient Polish, Simulated Annealing (Metropolis), and Exact Local Search.
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 50
DEADLINE_MARGIN = 10.0  # Safety margin before 150s kill

# Precompute upper triangle indices for efficient pairwise operations
_iu, _ju = np.triu_indices(N, k=1)
_NUM_PAIRS = len(_iu)

def _compute_radius(c):
    """
    Compute the maximum common radius for centers c (N,3).
    Radius is limited by min(pair_dist/2, wall_dist).
    """
    # Pairwise distances
    diff = c[_iu] - c[_ju]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    min_pair = np.min(dists)
    
    # Wall distances
    # Distance to nearest wall is 0.5 - |coord|
    # We need the min over all coords and all points
    wall_dists = 0.5 - np.abs(c)
    min_wall = np.min(wall_dists)
    
    return min(min_pair / 2.0, min_wall)

def _get_binding_constraints(c):
    """
    Identify the binding constraints:
    1. The pair of points with minimum distance.
    2. The point closest to the wall.
    Returns: (min_pair_idx, min_wall_idx)
    """
    diff = c[_iu] - c[_ju]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    min_pair_idx = np.argmin(dists)
    
    wall_dists = 0.5 - np.abs(c)
    # Find the global minimum wall distance
    min_wall_idx = np.unravel_index(np.argmin(wall_dists), wall_dists.shape)
    
    return min_pair_idx, min_wall_idx

def _softmin_radius_and_grad(c, t):
    """
    Compute a smooth approximation of the minimum radius and its gradient.
    Uses log-sum-exp for numerical stability.
    softmin(v) = -t * log(sum(exp(-v_i / t)))
    """
    # Pair terms
    diff = c[_iu] - c[_ju]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    pair_vals = dists / 2.0
    
    # Wall terms
    wall_vals = 0.5 - np.abs(c)
    
    # Combine all values
    all_vals = np.concatenate([pair_vals, wall_vals.ravel()])
    
    # Numerically stable log-sum-exp
    m = np.min(all_vals)
    # We want -t * log(sum(exp(-v/t)))
    # = -t * log(exp(-m/t) * sum(exp(-(v-m)/t)))
    # = -t * (-m/t + log(sum(exp(-(v-m)/t))))
    # = m - t * log(sum(exp(-(v-m)/t)))
    
    # Calculate weights w_i = exp(-(v_i - m)/t)
    # For pairs:
    pair_weights = np.exp(-(pair_vals - m) / t)
    # For walls:
    wall_weights = np.exp(-(wall_vals - m) / t)
    
    total_sum = np.sum(pair_weights) + np.sum(wall_weights)
    
    if total_sum < 1e-300:
        return m, np.zeros_like(c)
    
    softmin_val = m - t * np.log(total_sum)
    
    # Gradient calculation
    # d(softmin)/dc = sum_k w_k * (dv_k/dc) / sum_k w_k
    # Note: The derivative of -t log(S) where S = sum exp(-v/t) is
    # -t * (1/S) * sum (exp(-v/t) * (-1/t) * dv/dc)
    # = (1/S) * sum w_k * dv/dc
    
    grad_c = np.zeros_like(c)
    
    # Pair gradients
    # v_pair = dist/2
    # d(dist/2)/dc_i = -0.5 * (c_i - c_j) / dist
    # d(dist/2)/dc_j = 0.5 * (c_i - c_j) / dist
    with np.errstate(divide='ignore', invalid='ignore'):
        unit_vecs = diff / dists[:, np.newaxis]
    unit_vecs[np.isnan(unit_vecs)] = 0.0
    
    # Weighted sum of gradients for pairs
    # grad_c[i] -= 0.5 * w_ij * unit_vec_ij
    # grad_c[j] += 0.5 * w_ij * unit_vec_ij
    pair_grad_contrib = 0.5 * pair_weights[:, np.newaxis] * unit_vecs
    for idx in range(_NUM_PAIRS):
        i, j = _iu[idx], _ju[idx]
        grad_c[i] -= pair_grad_contrib[idx]
        grad_c[j] += pair_grad_contrib[idx]
        
    # Wall gradients
    # v_wall = 0.5 - |c_ik|
    # d(v_wall)/dc_ik = -sign(c_ik)
    signs = np.sign(c)
    signs[signs == 0] = 1.0
    # Wall contribution to gradient
    wall_grad_contrib = wall_weights * signs
    grad_c -= wall_grad_contrib
    
    # Normalize by total sum
    grad_c /= total_sum
    
    return softmin_val, grad_c

def _smooth_obj_and_grad(x, t):
    """
    Objective function for L-BFGS-B.
    We want to MAXIMIZE radius, so we MINIMIZE -radius.
    """
    c = x.reshape(N, 3)
    r, grad_r = _softmin_radius_and_grad(c, t)
    
    # Boundary penalty to keep points strictly inside
    # This helps if the softmin gradient isn't strong enough near boundaries
    pen = np.sum(np.maximum(0.0, np.abs(c) - 0.5 + 1e-8) ** 2)
    pen_grad = np.zeros_like(c)
    over = np.maximum(0.0, np.abs(c) - 0.5 + 1e-8)
    if np.any(over > 0):
        signs = np.sign(c)
        signs[signs == 0] = 1.0
        pen_grad = 2.0 * over * signs
        
    val = -r + 1e3 * pen
    grad = -grad_r + 1e3 * pen_grad
    
    return val, grad.flatten()

def _smooth_polish(c, deadline, t_start=0.01, t_end=0.0001, steps=5):
    """
    Polish the configuration using L-BFGS-B on the smooth objective.
    Anneal the temperature t.
    """
    c = c.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    x0 = c.flatten()
    bounds = [(-0.499, 0.499)] * (N * 3)
    
    # Anneal temperature
    for i in range(steps):
        if time.time() > deadline:
            break
        t = t_start * (t_end / t_start) ** (i / max(steps - 1, 1))
        
        try:
            res = minimize(
                lambda x: _smooth_obj_and_grad(x, t)[0],
                x0,
                jac=lambda x: _smooth_obj_and_grad(x, t)[1],
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-9, 'maxls': 50}
            )
            c_new = np.clip(res.x.reshape(N, 3), -0.499, 0.499)
            r_new = _compute_radius(c_new)
            
            if r_new > best_r:
                best_r = r_new
                best_c = c_new.copy()
                x0 = c_new.flatten()
            else:
                # If no improvement, keep previous x0 to avoid drifting
                pass
                
        except Exception:
            pass
            
    return best_c, best_r

def _repulsion_anneal(c, deadline, steps=2000):
    """
    Simple force-directed repulsion to spread points out.
    Useful as an initial pre-processing step.
    """
    c = c.copy()
    for it in range(steps):
        if time.time() > deadline:
            break
        
        # Repulsion force
        diff = c[_iu] - c[_ju]
        dists = np.sqrt(np.sum(diff * diff, axis=-1))
        # Avoid division by zero
        dists = np.maximum(dists, 1e-6)
        inv_d2 = 1.0 / (dists ** 2)
        unit_vecs = diff / dists[:, np.newaxis]
        
        F = np.zeros_like(c)
        for idx in range(_NUM_PAIRS):
            i, j = _iu[idx], _ju[idx]
            F[i] += inv_d2[idx] * unit_vecs[idx]
            F[j] -= inv_d2[idx] * unit_vecs[idx]
            
        # Wall attraction (push away from walls)
        # Force towards center
        F_wall = -c * 0.1
        
        # Update
        lr = 0.01 * (1.0 - it / steps)
        c += lr * (F + F_wall)
        c = np.clip(c, -0.49, 0.49)
        
    return c

def _metropolis_mc(c, deadline, iters=10000, T0=0.005, T_end=0.0001):
    """
    Simulated Annealing / Metropolis Monte Carlo.
    This is the key to escaping local minima.
    """
    c = c.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    cur_r = best_r
    
    rng = np.random.default_rng(42)
    
    for it in range(iters):
        if time.time() > deadline:
            break
            
        # Cooling schedule
        frac = it / max(iters - 1, 1)
        T = T0 * (T_end / T0) ** frac
        
        # Propose a move: move one random sphere
        i = rng.integers(N)
        # Gaussian perturbation
        delta = rng.normal(scale=0.05, size=3)
        trial = c.copy()
        trial[i] += delta
        trial = np.clip(trial, -0.499, 0.499)
        
        trial_r = _compute_radius(trial)
        
        # Acceptance criterion
        if trial_r >= cur_r:
            c = trial
            cur_r = trial_r
            if cur_r > best_r:
                best_r = cur_r
                best_c = c.copy()
        else:
            delta_r = cur_r - trial_r
            # Accept with probability exp(-delta_r / T)
            # If delta_r is large, prob is small
            # If T is large, prob is higher
            if delta_r < 1e-12:
                c = trial
                cur_r = trial_r
            else:
                prob = np.exp(-delta_r / max(T, 1e-8))
                if rng.random() < prob:
                    c = trial
                    cur_r = trial_r
    
    return best_c, best_r

def _local_search(c, deadline, iters=5000):
    """
    Exact local search focusing on binding constraints.
    """
    c = c.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    rng = np.random.default_rng(123)
    
    step_size = 0.005
    
    for it in range(iters):
        if time.time() > deadline:
            break
            
        min_pair_idx, min_wall_idx = _get_binding_constraints(c)
        
        diff = c[_iu] - c[_ju]
        dists = np.sqrt(np.sum(diff * diff, axis=-1))
        min_pair_dist = dists[min_pair_idx]
        
        wall_dists = 0.5 - np.abs(c)
        min_wall_val = wall_dists[min_wall_idx]
        
        improved = False
        
        if min_pair_dist / 2.0 < min_wall_val:
            # Binding constraint is a pair
            i, j = _iu[min_pair_idx], _ju[min_pair_idx]
            direction = c[i] - c[j]
            norm = np.linalg.norm(direction)
            if norm > 1e-12:
                direction /= norm
            else:
                direction = rng.normal(size=3)
                direction /= np.linalg.norm(direction)
                
            # Try moving i and j apart
            for s in (step_size, step_size * 0.5, step_size * 2.0):
                trial = c.copy()
                trial[i] += s * direction
                trial[j] -= s * direction
                trial = np.clip(trial, -0.499, 0.499)
                r_trial = _compute_radius(trial)
                if r_trial > best_r:
                    c = trial
                    best_r = r_trial
                    best_c = c.copy()
                    improved = True
                    break
        else:
            # Binding constraint is a wall
            i, k = min_wall_idx
            sgn = np.sign(c[i, k])
            if sgn == 0:
                sgn = 1.0
            direction = np.zeros(3)
            direction[k] = sgn
            
            for s in (step_size, step_size * 0.5, step_size * 2.0):
                trial = c.copy()
                trial[i] += s * direction
                trial = np.clip(trial, -0.499, 0.499)
                r_trial = _compute_radius(trial)
                if r_trial > best_r:
                    c = trial
                    best_r = r_trial
                    best_c = c.copy()
                    improved = True
                    break
            
        if improved:
            step_size = min(step_size * 1.1, 0.01)
        else:
            # Try random perturbation to escape
            if it % 100 == 0:
                trial = c.copy()
                trial += rng.normal(scale=0.01, size=(N, 3))
                trial = np.clip(trial, -0.499, 0.499)
                r_trial = _compute_radius(trial)
                if r_trial > best_r:
                    c = trial
                    best_r = r_trial
                    best_c = c.copy()
                else:
                    step_size = max(step_size * 0.9, 0.0001)
                    
    return best_c, best_r

def _seed_fcc():
    """FCC lattice points."""
    pts = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                pts.append([i, j, k])
                pts.append([i+0.5, j+0.5, k])
                pts.append([i+0.5, j, k+0.5])
                pts.append([i, j+0.5, k+0.5])
    pts = np.array(pts)
    pts = (pts - 1.0) / 1.0 # Center at 0, range approx [-1, 2]
    # Scale to fit in [-0.45, 0.45]
    max_abs = np.max(np.abs(pts))
    pts = pts / max_abs * 0.45
    
    # Farthest point sampling to get N points
    sel = [int(np.argmin(np.sum(pts**2, axis=1)))]
    rem = list(range(len(pts)))
    rem.remove(sel[0])
    while len(sel) < N and rem:
        dm = np.full(len(rem), np.inf)
        for ri, idx in enumerate(rem):
            dmin = np.min(np.linalg.norm(pts[idx] - pts[sel], axis=1))
            dm[ri] = dmin
        bi = int(np.argmax(dm))
        sel.append(rem[bi])
        rem.pop(bi)
    return pts[sel]

def _seed_bcc():
    """BCC lattice points."""
    pts = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                pts.append([i, j, k])
                pts.append([i+0.5, j+0.5, k+0.5])
    pts = np.array(pts)
    pts = (pts - 1.0)
    max_abs = np.max(np.abs(pts))
    pts = pts / max_abs * 0.45
    
    sel = [int(np.argmin(np.sum(pts**2, axis=1)))]
    rem = list(range(len(pts)))
    rem.remove(sel[0])
    while len(sel) < N and rem:
        dm = np.full(len(rem), np.inf)
        for ri, idx in enumerate(rem):
            dmin = np.min(np.linalg.norm(pts[idx] - pts[sel], axis=1))
            dm[ri] = dmin
        bi = int(np.argmax(dm))
        sel.append(rem[bi])
        rem.pop(bi)
    return pts[sel]

def _seed_random(seed):
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.45, 0.45, size=(N, 3))

def _seed_grid():
    m = int(np.ceil(N ** (1.0/3.0)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    g = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return g[:N]

def construct():
    start_time = time.time()
    deadline = start_time + (150.0 - DEADLINE_MARGIN)
    
    seeds = [
        _seed_fcc(),
        _seed_bcc(),
        _seed_grid(),
        _seed_random(1),
        _seed_random(2),
    ]
    
    best_c = None
    best_r = -1.0
    
    for s in seeds:
        if time.time() > deadline:
            break
            
        s = np.clip(s, -0.49, 0.49)
        
        # Stage 1: Repulsion Annealing
        c = _repulsion_anneal(s, deadline, steps=1000)
        
        # Stage 2: Smooth Polish
        if time.time() < deadline:
            c, r = _smooth_polish(c, deadline, t_start=0.01, t_end=0.001, steps=3)
            if r > best_r:
                best_r = r
                best_c = c.copy()
                
        # Stage 3: Metropolis MC
        if time.time() < deadline:
            c, r = _metropolis_mc(c, deadline, iters=5000)
            if r > best_r:
                best_r = r
                best_c = c.copy()
                
        # Stage 4: Local Search
        if time.time() < deadline:
            c, r = _local_search(c, deadline, iters=2000)
            if r > best_r:
                best_r = r
                best_c = c.copy()
                
    # Final Polish on the global best
    if best_c is not None and time.time() < deadline:
        # More aggressive smooth polish
        c, r = _smooth_polish(best_c, deadline, t_start=0.001, t_end=0.00001, steps=5)
        if r > best_r:
            best_r = r
            best_c = c.copy()
            
        # More Metropolis
        if time.time() < deadline:
            c, r = _metropolis_mc(best_c, deadline, iters=5000, T0=0.002, T_end=0.00005)
            if r > best_r:
                best_r = r
                best_c = c.copy()
                
        # Final Local Search
        if time.time() < deadline:
            c, r = _local_search(best_c, deadline, iters=3000)
            if r > best_r:
                best_r = r
                best_c = c.copy()
                
    if best_c is None:
        best_c = _seed_grid()
        
    best_c = np.clip(best_c, -0.499999, 0.499999)
    return best_c

# EVOLVE-BLOCK-END
