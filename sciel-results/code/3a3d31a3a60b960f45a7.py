"""Packing 50 equal circles in a unit circle using energy minimization and ring structures."""
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

def _compute_r(c):
    """Compute the max common radius for given centers."""
    c = c.reshape(N, 2)
    ci = c[PAIR_IDX[:, 0]]
    cj = c[PAIR_IDX[:, 1]]
    d = np.sqrt(np.sum((ci - cj) ** 2, axis=1))
    min_dist = np.min(d)
    r2 = np.hypot(c[:, 0], c[:, 1])
    min_boundary = np.min(1 - r2)
    return min(min_dist / 2, min_boundary)

def _energy(c, r_target=0.1):
    """
    Compute energy: sum of squared overlaps for pair distances < 2*r_target 
    and boundary violations.
    This is a smooth surrogate for the min-distance objective.
    """
    c = c.reshape(N, 2)
    ci = c[PAIR_IDX[:, 0]]
    cj = c[PAIR_IDX[:, 1]]
    d_sq = np.sum((ci - cj) ** 2, axis=1)
    
    # Penalty for pairs closer than 2*r_target
    # We want d >= 2*r. If d < 2*r, penalty is (2*r - d)^2.
    # To make it smooth and efficient, we can use a soft penalty.
    # Let's use: penalty = max(0, 2*r_target - d)^2
    # But max is not differentiable. We can use a smooth approximation or just rely on the gradient of the distance.
    # Actually, standard approach: minimize sum of (max(0, 2r - d_i))^2.
    # Gradient of d_i is (c_i - c_j) / d_i.
    
    d = np.sqrt(d_sq)
    # Avoid division by zero
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

def _energy_grad(c, r_target=0.1):
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
    # grad overlap wrt d = -1 if d < 2r else 0
    # grad d wrt c_i = (c_i - c_j) / d
    # grad E wrt c_i = sum_j 2 * overlap * (-1) * (c_i - c_j) / d
    
    mask = d < 2 * r_target
    if np.any(mask):
        # For each pair i,j where d < 2r:
        # grad_i += 2 * (2r - d) * (c_i - c_j) / d
        # grad_j += 2 * (2r - d) * (c_j - c_i) / d = -grad_i contribution
        
        overlap = 2 * r_target - d[mask]
        factor = 2 * overlap / d_safe[mask]
        
        # Accumulate gradients
        # This is a bit tricky to vectorize accumulation into grad array.
        # We can use np.add.at
        grad_i_contrib = factor[:, np.newaxis] * diff[mask]
        np.add.at(grad, PAIR_IDX[mask, 0], grad_i_contrib)
        np.add.at(grad, PAIR_IDX[mask, 1], -grad_i_contrib)
        
    # Boundary gradient
    r2 = np.hypot(c[:, 0], c[:, 1])
    mask_b = r2 + r_target > 1
    if np.any(mask_b):
        # overlap = r2 + r - 1
        # grad r2 wrt c = c / r2
        # grad E wrt c = 2 * overlap * c / r2
        overlap_b = r2[mask_b] + r_target - 1
        factor_b = 2 * overlap_b / r2[mask_b]
        grad_b_contrib = factor_b[:, np.newaxis] * c[mask_b]
        grad[mask_b] += grad_b_contrib
        
    return grad

def _optimize_energy(c0, r_target, time_limit=5.0):
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
        return -_compute_r(x)
    
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

def _get_ring_initialization(counts):
    """
    Generate initial positions for a given ring structure.
    counts: list of numbers of circles in each ring.
    """
    pts = []
    n_rings = len(counts)
    # Estimate ring radii. 
    # For a packing of N circles, the outer ring is close to 1.
    # Simple heuristic: distribute radii linearly or based on square root of index.
    radii = np.linspace(0.1, 0.9, n_rings)
    if n_rings == 1:
        radii = np.array([0.0])
        
    for k, n in enumerate(counts):
        R = radii[k]
        if n == 0:
            continue
        if n == 1:
            # Place single circle at center or slightly offset if not first ring
            if k == 0:
                pts.append([0.0, 0.0])
            else:
                pts.append([R, 0.0])
        else:
            for i in range(n):
                theta = 2 * np.pi * i / n
                pts.append([R * np.cos(theta), R * np.sin(theta)])
                
    return np.array(pts)

def construct():
    t_start = time.time()
    
    best_c = None
    best_r = 0.0
    
    # Define several ring structures that sum to 50
    # Common structures for N=50:
    # 1+6+12+18+13 = 50
    # 1+6+12+14+17 = 50
    # 1+6+10+14+19 = 50
    # 1+6+12+16+15 = 50
    # 1+6+8+14+21 = 50
    # 1+6+10+12+21 = 50
    # 1+6+12+12+19 = 50
    # 1+6+10+10+23 = 50 (less likely)
    # 1+6+12+18+13 is a very common pattern for hexagonal-like packings.
    
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
    
    # Also try a random initialization
    np.random.seed(42)
    random_c = np.random.uniform(-0.8, 0.8, (N, 2))
    
    initializations = []
    for counts in ring_structures:
        c0 = _get_ring_initialization(counts)
        if len(c0) == N:
            initializations.append(c0)
            
    initializations.append(random_c)
    
    # Strategy:
    # 1. For each initialization, run a sequence of energy minimizations with increasing r_target.
    #    Start with a small r_target to allow overlaps to resolve, then increase.
    # 2. After energy minimization, run a polish step.
    
    r_targets = [0.05, 0.08, 0.10, 0.11, 0.12, 0.125]
    
    for c0 in initializations:
        if time.time() - t_start > DEADLINE:
            break
            
        c_current = c0.copy()
        
        # Progressive energy minimization
        for rt in r_targets:
            if time.time() - t_start > DEADLINE - 10:
                break
                
            # Limit time per step
            time_left = DEADLINE - (time.time() - t_start)
            step_time = min(3.0, time_left / 10.0)
            
            c_current = _optimize_energy(c_current, rt, time_limit=step_time)
            
        # Polish
        if time.time() - t_start < DEADLINE - 5:
            c_polished = _polish(c_current, time_limit=min(5.0, DEADLINE - (time.time() - t_start)))
            r_polished = _compute_r(c_polished)
            
            if r_polished > best_r:
                best_r = r_polished
                best_c = c_polished.copy()
                
        # Check current best
        r_current = _compute_r(c_current)
        if r_current > best_r:
            best_r = r_current
            best_c = c_current.copy()

    # If we didn't get a good result, try a global search with Differential Evolution
    # on a reduced set of parameters (e.g., ring radii and angles)
    if best_r < 0.12 and time.time() - t_start < DEADLINE - 20:
        # Use DE to find good ring parameters
        # Parameters: radii (5) + angles (5) = 10 parameters
        # We'll fix the counts to the best structure [1, 6, 12, 18, 13]
        counts = [1, 6, 12, 18, 13]
        n_params = 10
        
        def de_objective(params):
            radii = params[:5]
            angles = params[5:]
            # Ensure radii are non-negative and ordered
            radii = np.maximum(radii, 0.0)
            # Sort radii? No, the order matters for the ring structure.
            # But we can enforce radii[0] < radii[1] ... 
            # Actually, just let DE explore.
            
            pts = []
            for k in range(5):
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
            return -_compute_r(c)
            
        bounds = [(0, 1.0)] * 5 + [(0, 2*np.pi)] * 5
        
        t_de_start = time.time()
        time_limit_de = min(20.0, DEADLINE - (time.time() - t_start) - 5)
        
        try:
            res = differential_evolution(
                de_objective,
                bounds,
                maxiter=50,
                popsize=10,
                tol=1e-6,
                seed=42,
                callback=lambda x, convergence: None if time.time() - t_de_start < time_limit_de else (_ for _ in ()).throw(Exception("timeout"))
            )
            if res.success:
                params_opt = res.x
                radii_opt = np.maximum(params_opt[:5], 0.0)
                angles_opt = params_opt[5:]
                
                pts = []
                for k in range(5):
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
                            
                c_de = np.array(pts)
                
                # Polish the DE result
                if time.time() - t_start < DEADLINE - 5:
                    c_de_polished = _polish(c_de, time_limit=min(5.0, DEADLINE - (time.time() - t_start)))
                    r_de_polished = _compute_r(c_de_polished)
                    
                    if r_de_polished > best_r:
                        best_r = r_de_polished
                        best_c = c_de_polished.copy()
                        
        except Exception:
            pass

    # Final final polish if time permits
    if best_c is not None and time.time() - t_start < DEADLINE - 2:
        c_final = _polish(best_c, time_limit=min(3.0, DEADLINE - (time.time() - t_start) - 1))
        r_final = _compute_r(c_final)
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
