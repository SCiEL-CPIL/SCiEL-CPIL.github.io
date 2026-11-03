"""Packing 50 equal circles in a unit circle using ring-based parametric optimization + local refinement."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 50
DEADLINE = 140.0  # Safety margin before 150s kill

def _pairwise_indices(n):
    """Return array of (i, j) pairs for i < j."""
    iu = np.triu_indices(n, k=1)
    return np.column_stack((iu[0], iu[1]))

PAIR_IDX = _pairwise_indices(N)

def _compute_score(c):
    """Compute the max common radius for given centers."""
    c = c.reshape(N, 2)
    ci = c[PAIR_IDX[:, 0]]
    cj = c[PAIR_IDX[:, 1]]
    d = np.sqrt(np.sum((ci - cj) ** 2, axis=1))
    min_dist = np.min(d)
    r2 = np.hypot(c[:, 0], c[:, 1])
    min_boundary = np.min(1 - r2)
    return min(min_dist / 2, min_boundary)

def _energy(c, r_target):
    """
    Compute energy: sum of squared overlaps for pair distances < 2*r_target 
    and boundary violations.
    This is a smooth surrogate for the min-distance objective.
    """
    c = c.reshape(N, 2)
    ci = c[PAIR_IDX[:, 0]]
    cj = c[PAIR_IDX[:, 1]]
    d_sq = np.sum((ci - cj) ** 2, axis=1)
    d = np.sqrt(d_sq)
    d_safe = np.where(d < 1e-6, 1e-6, d)
    
    # Overlap amount
    overlap = np.maximum(0, 2 * r_target - d)
    
    # Boundary penalty
    r2 = np.hypot(c[:, 0], c[:, 1])
    boundary_overlap = np.maximum(0, r2 + r_target - 1)
    
    # Energy
    e_pairs = np.sum(overlap ** 2)
    e_bound = np.sum(boundary_overlap ** 2)
    
    return e_pairs + e_bound

def _energy_grad(c, r_target):
    """
    Compute gradient of the energy function.
    """
    c = c.reshape(N, 2)
    grad = np.zeros_like(c)
    
    ci = c[PAIR_IDX[:, 0]]
    cj = c[PAIR_IDX[:, 1]]
    diff = ci - cj
    d_sq = np.sum(diff ** 2, axis=1)
    d = np.sqrt(d_sq)
    d_safe = np.where(d < 1e-6, 1e-6, d)
    
    # overlap = max(0, 2r - d)
    mask = d < 2 * r_target
    if np.any(mask):
        overlap = 2 * r_target - d[mask]
        factor = 2 * overlap / d_safe[mask]
        
        grad_i_contrib = factor[:, np.newaxis] * diff[mask]
        np.add.at(grad, PAIR_IDX[mask, 0], grad_i_contrib)
        np.add.at(grad, PAIR_IDX[mask, 1], -grad_i_contrib)
        
    # Boundary gradient
    r2 = np.hypot(c[:, 0], c[:, 1])
    mask_b = r2 + r_target > 1
    if np.any(mask_b):
        overlap_b = r2[mask_b] + r_target - 1
        factor_b = 2 * overlap_b / r2[mask_b]
        grad_b_contrib = factor_b[:, np.newaxis] * c[mask_b]
        grad[mask_b] += grad_b_contrib
        
    return grad

def _get_ring_initialization(counts, radii=None, angles=None):
    """
    Generate initial positions for a given ring structure.
    counts: list of numbers of circles in each ring.
    radii: list of ring radii (optional, computed if None)
    angles: list of ring angle offsets (optional, default 0)
    """
    n_rings = len(counts)
    if radii is None:
        # Estimate ring radii
        if n_rings == 1:
            radii = [0.0]
        else:
            radii = np.linspace(0.1, 0.9, n_rings).tolist()
    if angles is None:
        angles = [0.0] * n_rings
        
    pts = []
    for k, n in enumerate(counts):
        R = radii[k]
        angle_offset = angles[k]
        if n == 0:
            continue
        if n == 1:
            theta = angle_offset
            pts.append([R * np.cos(theta), R * np.sin(theta)])
        else:
            for i in range(n):
                theta = 2 * np.pi * i / n + angle_offset
                pts.append([R * np.cos(theta), R * np.sin(theta)])
                
    return np.array(pts)

def _optimize_positions(c0, r_target, time_limit=5.0):
    """
    Optimize positions using L-BFGS-B on the energy function.
    """
    t_start = time.time()
    try:
        res = minimize(
            lambda x: _energy(x, r_target),
            c0.flatten(),
            jac=lambda x: _energy_grad(x, r_target),
            method='L-BFGS-B',
            options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8},
            callback=lambda x: None if time.time() - t_start < time_limit else (_ for _ in ()).throw(Exception("timeout"))
        )
        if res.success:
            return res.x.reshape(N, 2)
    except Exception:
        pass
    return c0

def _polish(c0, time_limit=5.0):
    """
    Final polish using Nelder-Mead on the raw score (max radius).
    This helps navigate the non-smooth landscape near the optimum.
    """
    t_start = time.time()
    def neg_score(x):
        return -_compute_score(x)
    
    try:
        res = minimize(
            neg_score,
            c0.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-10},
            callback=lambda x: None if time.time() - t_start < time_limit else (_ for _ in ()).throw(Exception("timeout"))
        )
        if res.success:
            return res.x.reshape(N, 2)
    except Exception:
        pass
    return c0

def _de_ring_optimize(counts, time_limit=20.0):
    """
    Use differential evolution to find good ring parameters.
    Parameters: radii (n_rings) + angles (n_rings)
    """
    n_rings = len(counts)
    n_params = 2 * n_rings
    
    def de_objective(params):
        radii = params[:n_rings]
        angles = params[n_rings:]
        radii = np.maximum(radii, 0.0)
        
        pts = []
        for k in range(n_rings):
            R = radii[k]
            n = counts[k]
            if n == 0:
                continue
            if n == 1:
                theta = angles[k]
                pts.append([R * np.cos(theta), R * np.sin(theta)])
            else:
                for i in range(n):
                    theta = 2 * np.pi * i / n + angles[k]
                    pts.append([R * np.cos(theta), R * np.sin(theta)])
                    
        c = np.array(pts)
        return -_compute_score(c)
        
    bounds = [(0, 1.0)] * n_rings + [(0, 2*np.pi)] * n_rings
    
    t_de_start = time.time()
    
    try:
        res = differential_evolution(
            de_objective,
            bounds,
            maxiter=50,
            popsize=10,
            tol=1e-6,
            seed=42,
            callback=lambda x, convergence: None if time.time() - t_de_start < time_limit else (_ for _ in ()).throw(Exception("timeout"))
        )
        if res.success:
            params_opt = res.x
            radii_opt = np.maximum(params_opt[:n_rings], 0.0)
            angles_opt = params_opt[n_rings:]
            
            pts = []
            for k in range(n_rings):
                R = radii_opt[k]
                n = counts[k]
                if n == 0:
                    continue
                if n == 1:
                    theta = angles_opt[k]
                    pts.append([R * np.cos(theta), R * np.sin(theta)])
                else:
                    for i in range(n):
                        theta = 2 * np.pi * i / n + angles_opt[k]
                        pts.append([R * np.cos(theta), R * np.sin(theta)])
                        
            return np.array(pts)
    except Exception:
        pass
    return None

def _binary_search_polish(c0, r_low, r_high, time_limit=10.0):
    """
    Binary search on radius, using energy minimization to check feasibility.
    """
    t_start = time.time()
    best_c = c0.copy()
    best_r = r_low
    
    for _ in range(10):
        if time.time() - t_start > time_limit:
            break
            
        r_mid = (r_low + r_high) / 2
        c_test = _optimize_positions(c0, r_mid, time_limit=2.0)
        r_test = _compute_score(c_test)
        
        if r_test >= r_mid - 1e-6:
            # Feasible, try larger radius
            r_low = r_mid
            best_c = c_test.copy()
            best_r = r_test
        else:
            # Not feasible, try smaller radius
            r_high = r_mid
            
    # Final polish
    if time.time() - t_start < time_limit - 2:
        c_final = _polish(best_c, time_limit=min(3.0, time_limit - (time.time() - t_start)))
        r_final = _compute_score(c_final)
        if r_final > best_r:
            best_c = c_final
            best_r = r_final
            
    return best_c, best_r

def construct():
    t_start = time.time()
    
    best_c = None
    best_r = 0.0
    
    # Define several ring structures that sum to 50
    ring_structures = [
        [1, 6, 12, 18, 13],
        [1, 6, 12, 14, 17],
        [1, 6, 10, 14, 19],
        [1, 6, 12, 16, 15],
        [1, 6, 8, 14, 21],
        [1, 6, 10, 12, 21],
        [1, 6, 12, 12, 19],
        [1, 6, 10, 10, 23],
        [1, 6, 14, 14, 15],
        [1, 6, 12, 17, 14],
    ]
    
    # Strategy:
    # 1. For each ring structure, use DE to find good ring parameters
    # 2. Polish the result with binary search + local optimization
    
    for counts in ring_structures:
        if time.time() - t_start > DEADLINE - 10:
            break
            
        # Use DE to find good ring parameters
        c_de = _de_ring_optimize(counts, time_limit=min(15.0, DEADLINE - (time.time() - t_start) - 5))
        
        if c_de is not None:
            r_de = _compute_score(c_de)
            if r_de > best_r:
                best_r = r_de
                best_c = c_de.copy()
                
            # Polish with binary search
            if time.time() - t_start < DEADLINE - 10:
                c_polished, r_polished = _binary_search_polish(
                    c_de, max(0.0, r_de - 0.01), min(0.15, r_de + 0.01),
                    time_limit=min(10.0, DEADLINE - (time.time() - t_start) - 5)
                )
                if r_polished > best_r:
                    best_r = r_polished
                    best_c = c_polished.copy()
    
    # If we didn't get a good result, try a simpler approach
    if best_r < 0.12 and time.time() - t_start < DEADLINE - 20:
        # Use progressive energy minimization from ring initializations
        r_targets = [0.05, 0.08, 0.10, 0.11, 0.12, 0.125]
        
        for counts in ring_structures[:3]:
            if time.time() - t_start > DEADLINE - 10:
                break
                
            c0 = _get_ring_initialization(counts)
            c_current = c0.copy()
            
            for rt in r_targets:
                if time.time() - t_start > DEADLINE - 10:
                    break
                c_current = _optimize_positions(c_current, rt, time_limit=3.0)
                
            if time.time() - t_start < DEADLINE - 5:
                c_polished = _polish(c_current, time_limit=min(5.0, DEADLINE - (time.time() - t_start)))
                r_polished = _compute_score(c_polished)
                
                if r_polished > best_r:
                    best_r = r_polished
                    best_c = c_polished.copy()

    # Final final polish if time permits
    if best_c is not None and time.time() - t_start < DEADLINE - 2:
        c_final = _polish(best_c, time_limit=min(3.0, DEADLINE - (time.time() - t_start) - 1))
        r_final = _compute_score(c_final)
        if r_final > best_r:
            best_c = c_final
            best_r = r_final

    # Ensure validity
    if best_c is None:
        # Fallback: simple grid
        best_c = np.zeros((N, 2))
        
    c_final = best_c.reshape(N, 2)
    r2 = np.hypot(c_final[:, 0], c_final[:, 1])
    for i in range(N):
        if r2[i] > 1 - best_r + 1e-10:
            scale = (1 - best_r) / r2[i]
            c_final[i] *= scale
            
    return c_final
