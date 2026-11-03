"""Suite task: pack 100 equal spheres in the unit cube [-1/2,1/2]^3.
Entry point construct() -> (N,3) array of centres.
Score = min( min_{i<j} |c_i-c_j|/2 , min_i min_k (1/2 - |c_ik|) ).
Design: Hybrid optimization using target-radius annealing with L-BFGS-B.
Combines the robust structure of the soft-min approach with the focused
gradient of the target-radius constraint to escape local minima."""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
_DEADLINE = 145.0          # Safety margin vs 150s wall-clock kill
_rng = np.random.default_rng(12345)


def _radius(c):
    """Common radius: min of half min pairwise distance and min wall clearance."""
    c = np.asarray(c, dtype=float)
    n = c.shape[0]
    d = 1e30
    # Vectorized pairwise distance calculation for speed
    # We can do this in chunks or just use a loop if N is small enough
    # For N=100, a simple loop or vectorized block is fine.
    # Let's use a vectorized approach for the min distance to be safe.
    # diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    # dists = np.sqrt(np.sum(diff**2, axis=2))
    # np.fill_diagonal(dists, np.inf)
    # min_pair_dist = np.min(dists)
    
    # Actually, the loop in the original was okay, but let's optimize it slightly
    # by using broadcasting in blocks if needed, but for 100x100 it's small.
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    min_wall_dist = np.min(0.5 - np.abs(c))
    
    return float(min(min_pair_dist / 2.0, min_wall_dist))


def _energy_target(x, r_target):
    """
    Energy function for target radius optimization.
    Penalizes violations of:
    1. Pairwise distance < 2 * r_target
    2. Wall distance < r_target
    Returns (energy, gradient)
    """
    c = x.reshape(N, 3)
    n = N
    grad = np.zeros_like(c)
    energy = 0.0
    
    # 1. Wall Penalty
    # wall_dist = 0.5 - |c|
    # violation = max(0, r_target - wall_dist)
    # energy += sum(violation^2)
    abs_c = np.abs(c)
    wall_dist = 0.5 - abs_c
    wall_viol = np.maximum(0.0, r_target - wall_dist)
    energy += np.sum(wall_viol**2)
    
    # Gradient for wall:
    # d/dx (r_target - (0.5 - |x|))^2 = 2*(r_target - (0.5 - |x|)) * sign(x)
    # Note: sign(0) is 0, so if x=0 and violating, gradient is 0? 
    # Actually if x=0, wall_dist=0.5. If r_target > 0.5, it violates.
    # The derivative of |x| at 0 is undefined, but we can use np.sign which is 0.
    # To be safe, we can push away from 0 if needed, but usually spheres won't be exactly at 0.
    # Let's use a smooth sign approximation or just np.sign.
    # If wall_viol > 0, we want to push c away from the wall.
    # If c > 0, wall is at 0.5, push left (negative grad).
    # If c < 0, wall is at -0.5, push right (positive grad).
    # grad_wall = -2 * wall_viol * sign(c)
    # Wait, energy = (r_t - (0.5 - |c|))^2.
    # dE/dc = 2(r_t - 0.5 + |c|) * d|c|/dc = 2(r_t - wall_dist) * sign(c).
    # We want to minimize E, so we go in direction of -grad.
    # If c > 0 (near right wall), wall_dist is small. r_t - wall_dist > 0.
    # grad = 2 * pos * 1 = pos. -grad = neg. Move left. Correct.
    grad_wall = -2.0 * wall_viol * np.sign(c)
    # Handle zero sign: if c is very close to 0 and violating, sign is 0.
    # We can add a small bias or just let it be.
    grad += grad_wall
    
    # 2. Pairwise Penalty
    # We need to sum over i < j
    # To save memory, we can do it in a loop or vectorized blocks.
    # N=100, so 4950 pairs. Vectorized is fast.
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2) + 1e-12) # avoid div by 0
    np.fill_diagonal(dists, np.inf)
    
    # Only care about pairs where dist < 2 * r_target
    mask = dists < 2.0 * r_target
    if np.any(mask):
        # Get indices
        i_idx, j_idx = np.where(mask)
        # We only want i < j to avoid double counting
        # np.where returns all, so filter
        valid = i_idx < j_idx
        i_idx = i_idx[valid]
        j_idx = j_idx[valid]
        
        if len(i_idx) > 0:
            d_ij = dists[i_idx, j_idx]
            # violation = 2*r_target - d_ij
            viol = 2.0 * r_target - d_ij
            energy += np.sum(viol**2)
            
            # Gradient:
            # E = sum (2r_t - d_ij)^2
            # dE/dc_i = sum_j 2(2r_t - d_ij) * (-1) * (c_i - c_j)/d_ij
            # dE/dc_j = sum_i 2(2r_t - d_ij) * (1) * (c_i - c_j)/d_ij
            # Note: diff = c_i - c_j.
            # Let v_ij = (c_i - c_j) / d_ij.
            # grad_c_i += -2 * viol * v_ij
            # grad_c_j += 2 * viol * v_ij
            
            # We need to accumulate these into the grad array.
            # Using np.add.at for efficient accumulation
            vecs = (c[i_idx] - c[j_idx]) / d_ij[:, np.newaxis]
            # grad_i contribution
            np.add.at(grad, i_idx, -2.0 * viol[:, np.newaxis] * vecs)
            # grad_j contribution
            np.add.at(grad, j_idx, 2.0 * viol[:, np.newaxis] * vecs)
            
    return energy, grad


def _init_grid():
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    g = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return g[:N].copy()


def _init_random():
    return _rng.uniform(-0.49, 0.49, size=(N, 3))


def _init_jittered():
    return _init_grid() + _rng.normal(0, 0.012, size=(N, 3))


def construct():
    t0 = time.time()
    
    # Initial best
    best = _init_grid()
    best_r = _radius(best)
    
    # List of initializations to try
    inits = [
        _init_grid(),
        _init_random(),
        _init_jittered(),
        _init_random()
    ]
    
    # Target radius annealing schedule
    # We start low to untangle, then push higher
    target_schedule = [0.08, 0.09, 0.10, 0.105, 0.108, 0.110, 0.111, 0.1113, 0.11138]
    
    # For each initialization, run the annealing process
    for it, init_c in enumerate(inits):
        if time.time() - t0 > _DEADLINE - 10.0:
            break
            
        c = np.clip(init_c, -0.49, 0.49)
        x0 = c.ravel()
        
        # Run target radius annealing
        current_c = x0
        for r_target in target_schedule:
            if time.time() - t0 > _DEADLINE - 5.0:
                break
            
            # Skip if target is already lower than current best radius
            # (Though we might still want to run it to refine position for higher targets)
            # Actually, if r_target < best_r, we can skip to save time, 
            # but only if we are confident we can't improve. 
            # Let's just run it, it's fast.
            
            try:
                res = minimize(
                    _energy_target, 
                    current_c, 
                    args=(r_target,),
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 200, 'maxls': 20, 'gtol': 1e-8, 'ftol': 1e-12}
                )
                current_c = np.clip(res.x, -0.5, 0.5)
                # Reshape for next iteration
                # current_c is flat, we need to keep it flat for minimize, 
                # but _energy_target expects flat and reshapes internally.
                
            except Exception:
                pass
        
        # After annealing, evaluate
        cand = np.clip(current_c.reshape(N, 3), -0.5, 0.5)
        r = _radius(cand)
        
        if r > best_r:
            best_r = r
            best = cand.copy()
            
    # Final high-precision polish using the soft-min approach from original
    # This helps fine-tune the result
    if time.time() - t0 < _DEADLINE - 10.0:
        c = best.copy()
        # Use a very high target radius for polishing? 
        # Or use the original soft-min energy?
        # Let's use a high target radius to ensure no violations, then polish.
        # Actually, the original soft-min is good for polishing.
        # Let's implement a quick soft-min polish.
        
        def _energy_soft_min(x, c, K=1.0, beta=80.0):
            n = x.size
            c = np.asarray(c, dtype=float)
            pen = np.maximum(c - x, 0.0) + np.maximum(-c - x, 0.0)
            e_wall = beta * float(np.dot(pen, pen))
            g_wall = -2.0 * beta * (np.sign(x - c) * np.maximum(c - x, 0.0)
                                    + np.sign(-c - x) * np.maximum(-c - x, 0.0))
            
            d = np.zeros(n)
            g = np.zeros(n)
            for i in range(n - 1):
                diff = x[i + 1:] - x[i]
                dist = np.sqrt(np.einsum('ij,ij->i', diff, diff))
                d[i + 1:] = dist
                g[i + 1:] = diff / dist
            dmin = float(np.min(d))
            m = -np.logaddexp.reduce(-beta * d)
            e_rep = -K * m
            w = np.exp(-beta * d - m)
            g = g * w[:, None]
            e_rep += K * float(np.sum(w))
            g_rep = K * g
            return e_rep + e_wall, g_rep + g_wall

        try:
            res = minimize(
                _energy_soft_min, 
                best.ravel(), 
                args=(best, 1.0, 80.0),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 1000, 'maxls': 30, 'gtol': 1e-12, 'ftol': 1e-15}
            )
            cand = np.clip(res.x.reshape(N, 3), -0.5, 0.5)
            r = _radius(cand)
            if r > best_r:
                best_r = r
                best = cand.copy()
        except Exception:
            pass

    # Final validity guard
    best = np.clip(best, -0.4999999, 0.4999999)
    if not np.all(np.isfinite(best)):
        best = _init_grid()
        
    return best
