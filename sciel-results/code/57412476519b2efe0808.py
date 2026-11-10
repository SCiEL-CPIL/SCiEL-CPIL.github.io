"""Improved packing of 50 equal spheres in a unit cube using smooth energy optimization."""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START

def _exact_score(centers):
    """Exact minimum radius score."""
    if len(centers) < 2:
        return 0.5
    coords = np.abs(centers)
    wall_dists = 0.5 - coords
    min_wall = np.min(wall_dists)
    
    # Compute all pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    return min(min_pair / 2.0, min_wall)

def _softmin_obj_and_grad(x, beta=100.0):
    """
    Soft-minimum objective: minimize -softmin(pair_dists/2, wall_dists)
    This directly optimizes toward the actual score.
    """
    centers = x.reshape(N, 3)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu] / 2.0  # radii
    
    # Wall distances
    abs_c = np.abs(centers)
    wall_dists = 0.5 - abs_c  # shape (N, 3)
    flat_walls = wall_dists.flatten()
    
    # Combine all constraints
    all_dists = np.concatenate([pair_dists, flat_walls])
    m = np.min(all_dists)
    
    # Soft min: m - log(sum(exp(-beta*(d - m))))/beta
    shifted = -beta * (all_dists - m)
    # Numerical stability
    shifted -= np.max(shifted)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min_val = m - log_sum / beta
    
    # Gradient
    weights = np.exp(shifted) / np.sum(np.exp(shifted))
    
    grad = np.zeros((N, 3))
    
    # Pair gradient
    for idx in range(len(iu[0])):
        i, j = iu[0][idx], iu[1][idx]
        d = dists[i, j]
        if d > 1e-10:
            w = weights[idx] / beta
            f = -w / d  # d(-softmin)/dx = -w * d(d_ij/2)/dx_i
            diff_ij = centers[i] - centers[j]
            grad[i] += f * diff_ij
            grad[j] -= f * diff_ij
    
    # Wall gradient
    wall_weights = weights[len(pair_dists):].reshape(N, 3)
    signs = np.sign(centers)
    # d(-softmin)/dx = -w_wall * d(wall_dist)/dx = -w * (-sign) = w * sign
    grad += wall_weights / beta * signs
    
    return -soft_min_val, grad.flatten()

def _optimize_from_start(x0, time_limit=30.0):
    """Optimize from a starting point using multi-stage L-BFGS-B."""
    start_time = time.time()
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    x = x0.copy()
    
    # Stage 1: broad exploration with low beta
    if time.time() - start_time < time_limit * 0.3:
        try:
            result = minimize(
                _softmin_obj_and_grad, x, args=(20.0,),
                method='L-BFGS-B', bounds=bounds,
                jac=True,
                options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            x = result.x
        except Exception:
            pass
    
    # Stage 2: medium precision
    if time.time() - start_time < time_limit * 0.6:
        try:
            result = minimize(
                _softmin_obj_and_grad, x, args=(50.0,),
                method='L-BFGS-B', bounds=bounds,
                jac=True,
                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            x = result.x
        except Exception:
            pass
    
    # Stage 3: high precision
    if time.time() - start_time < time_limit * 0.85:
        try:
            result = minimize(
                _softmin_obj_and_grad, x, args=(100.0,),
                method='L-BFGS-B', bounds=bounds,
                jac=True,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            x = result.x
        except Exception:
            pass
    
    # Stage 4: final polish
    if time.time() - start_time < time_limit:
        try:
            result = minimize(
                _softmin_obj_and_grad, x, args=(200.0,),
                method='L-BFGS-B', bounds=bounds,
                jac=True,
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            x = result.x
        except Exception:
            pass
    
    return np.clip(x.reshape(N, 3), -0.5, 0.5)

def _initial_grid():
    """4x4x4 grid initialization with perturbations."""
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    centers = grid[:N].copy()
    centers += np.random.uniform(-0.02, 0.02, (N, 3))
    return np.clip(centers, -0.5, 0.5)

def _initial_bcc():
    """BCC lattice initialization."""
    m = 4
    points = []
    for z in range(m):
        for y in range(m):
            for x in range(m):
                points.append(((x + 0.5) / m - 0.5, (y + 0.5) / m - 0.5, (z + 0.5) / m - 0.5))
                points.append(((x + 1.0) / m - 0.5, (y + 1.0) / m - 0.5, (z + 1.0) / m - 0.5))
    points = np.array(points)
    centers = points[:N].copy()
    centers += np.random.uniform(-0.02, 0.02, (N, 3))
    return np.clip(centers, -0.5, 0.5)

def _initial_fcc():
    """FCC-like lattice initialization."""
    m = 3
    points = []
    for z in range(m):
        for y in range(m):
            for x in range(m):
                points.append(((x + 0.5) / m - 0.5, (y + 0.5) / m - 0.5, (z + 0.5) / m - 0.5))
                if x < m - 1:
                    points.append(((x + 1.0) / m - 0.5, (y + 0.5) / m - 0.5, (z + 0.5) / m - 0.5))
                if y < m - 1:
                    points.append(((x + 0.5) / m - 0.5, (y + 1.0) / m - 0.5, (z + 0.5) / m - 0.5))
                if z < m - 1:
                    points.append(((x + 0.5) / m - 0.5, (y + 0.5) / m - 0.5, (z + 1.0) / m - 0.5))
    points = np.array(points)
    np.random.shuffle(points)
    centers = points[:N].copy()
    centers += np.random.uniform(-0.02, 0.02, (N, 3))
    return np.clip(centers, -0.5, 0.5)

def _initial_random():
    """Random initialization."""
    centers = np.random.uniform(-0.4, 0.4, (N, 3))
    return centers

def _lloyd_refine(centers, time_limit=15.0):
    """Lloyd-like refinement: iteratively reposition each sphere to maximize its local clearance."""
    current = centers.copy()
    best = current.copy()
    best_score = _exact_score(best)
    start_time = time.time()
    
    for iteration in range(50):
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        for i in range(N):
            if time.time() - start_time > time_limit:
                break
            
            # Find the best position for sphere i given all others
            # Try several candidate positions
            old_pos = current[i].copy()
            
            # Compute current distances to all other spheres and walls
            best_local = old_pos.copy()
            best_local_score = -1.0
            
            # Candidate positions: current, plus small perturbations
            candidates = [old_pos.copy()]
            for _ in range(20):
                cand = old_pos + np.random.uniform(-0.03, 0.03, 3)
                cand = np.clip(cand, -0.5, 0.5)
                candidates.append(cand)
            
            for cand in candidates:
                # Compute minimum distance from cand to all other spheres
                diffs = cand - np.delete(current, i, axis=0)
                dmin = np.min(np.sqrt(np.sum(diffs**2, axis=1)))
                # Wall distance
                wall = np.min(0.5 - np.abs(cand))
                local_score = min(dmin / 2.0, wall)
                
                if local_score > best_local_score:
                    best_local_score = local_score
                    best_local = cand
            
            current[i] = best_local
            score = _exact_score(current)
            if score > best_score:
                best = current.copy()
                best_score = score
                improved = True
            elif score < best_score - 0.001:
                # Revert if too bad
                current[i] = old_pos
                score = _exact_score(current)
        
        if not improved and iteration > 5:
            break
    
    return best, best_score

def construct():
    start_time = time.time()
    deadline = 142.0  # 8s safety margin
    
    best_centers = None
    best_score = 0.0
    
    # Try multiple initializations - fewer but with more time each
    initializations = [
        _initial_grid(),
        _initial_bcc(),
        _initial_fcc(),
        _initial_random(),
        _initial_random(),
    ]
    
    for init in initializations:
        if time.time() > deadline - 20.0:
            break
        
        # Optimize from this starting point
        opt_time = min(15.0, max(5.0, deadline - 10.0 - (time.time() - start_time)))
        optimized = _optimize_from_start(init, time_limit=opt_time)
        score = _exact_score(optimized)
        
        # Lloyd refinement if time permits
        if time.time() < deadline - 5.0:
            lloyd_time = min(10.0, max(3.0, deadline - 5.0 - (time.time() - start_time)))
            refined, ref_score = _lloyd_refine(optimized, time_limit=lloyd_time)
            if ref_score > score:
                score = ref_score
                optimized = refined
        
        if best_centers is None or score > best_score:
            best_score = score
            best_centers = optimized
    
    # Final polish on the best result
    if best_centers is not None and time.time() < deadline - 3.0:
        final_time = max(2.0, deadline - 3.0 - (time.time() - start_time))
        polished = _optimize_from_start(best_centers, time_limit=final_time)
        final_score = _exact_score(polished)
        if final_score > best_score:
            best_score = final_score
            best_centers = polished
    
    if best_centers is None:
        best_centers = _initial_grid()
    
    return best_centers
# EVOLVE-BLOCK-END
