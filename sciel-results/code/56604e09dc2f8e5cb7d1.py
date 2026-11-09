# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

def _score(pts):
    """Return (d_min/d_max)^2, or 0 on degenerate input."""
    try:
        d = pdist(pts)
        if d.size == 0:
            return 0.0
        dmin = d.min()
        dmax = d.max()
        if dmax <= 1e-12 or not np.isfinite(dmin) or not np.isfinite(dmax):
            return 0.0
        r = dmin / dmax
        return float(r * r)
    except Exception:
        return 0.0

def _smooth_obj_and_grad(x_flat, beta):
    """
    Compute objective and gradient for smooth ratio maximization.
    Objective to minimize: - (S_min / S_max)^2
    S_min = -1/beta * ln(sum(exp(-beta * d_ij)))
    S_max =  1/beta * ln(sum(exp( beta * d_ij)))
    """
    pts = x_flat.reshape(14, 3)
    n = 14
    
    # Compute pairwise distances and indices
    # Using a loop for clarity and gradient accumulation, though vectorized is faster
    # For 14 points, 91 pairs. Loop is fine.
    d_sum = 0.0
    u_sum = 0.0
    v_sum = 0.0
    grad = np.zeros_like(pts)
    
    # Precompute exponentials? No, need distances first.
    # We will iterate over pairs
    
    # To avoid overflow in exp(beta * d) if d is large, we can scale.
    # But d is usually O(1). Beta might be large.
    # Let's use log-sum-exp trick if necessary, but for 14 points it's small.
    
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            pairs.append((i, j))
            
    # First pass: compute U, V and store weights
    weights_neg = []
    weights_pos = []
    ds = []
    for i, j in pairs:
        d = np.linalg.norm(pts[i] - pts[j])
        if d < 1e-8:
            d = 1e-8
        ds.append(d)
        w_neg = np.exp(-beta * d)
        w_pos = np.exp(beta * d)
        weights_neg.append(w_neg)
        weights_pos.append(w_pos)
        u_sum += w_neg
        v_sum += w_pos
        
    if u_sum == 0 or v_sum == 0:
        return 0.0, grad
        
    S_min = -1.0 / beta * np.log(u_sum)
    S_max =  1.0 / beta * np.log(v_sum)
    
    if S_max == 0:
        return 0.0, grad
        
    ratio = S_min / S_max
    obj = -(ratio * ratio)
    
    # Gradients
    # d S_min / d x = (1/U) * sum_k w_neg_k * (d d_k / d x)
    # d S_max / d x = (1/V) * sum_k w_pos_k * (d d_k / d x)
    
    # We need to accumulate the vector field for each point
    # For a pair (i, j) with distance d:
    # vec = (pts[i] - pts[j]) / d
    # d d / d pts[i] = vec
    # d d / d pts[j] = -vec
    
    grad_S_min = np.zeros_like(pts)
    grad_S_max = np.zeros_like(pts)
    
    for k, (i, j) in enumerate(pairs):
        d = ds[k]
        vec = (pts[i] - pts[j]) / d
        
        # Contribution to grad_S_min
        w_neg = weights_neg[k] / u_sum
        grad_S_min[i] += w_neg * vec
        grad_S_min[j] -= w_neg * vec
        
        # Contribution to grad_S_max
        w_pos = weights_pos[k] / v_sum
        grad_S_max[i] += w_pos * vec
        grad_S_max[j] -= w_pos * vec
        
    # Chain rule for f = -(S_min / S_max)^2
    # df/d S_min = -2 * (S_min / S_max)^2 * (1 / S_max) = -2 * ratio^2 / S_max
    # df/d S_max = -2 * (S_min / S_max)^2 * (-S_min / S_max^2) = 2 * ratio^2 * S_min / S_max^2
    
    d_obj_d_S_min = -2.0 * (ratio * ratio) / S_max
    d_obj_d_S_max = 2.0 * (ratio * ratio) * S_min / (S_max * S_max)
    
    grad = d_obj_d_S_min * grad_S_min + d_obj_d_S_max * grad_S_max
    
    return obj, grad

def _icosahedron_vertices():
    phi = (1.0 + 5.0 ** 0.5) / 2.0
    v = np.array([
        [-1,  phi, 0], [1,  phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1,  phi], [0, 1,  phi], [0, -1, -phi], [0, 1, -phi],
        [ phi, 0, -1], [ phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1],
    ], dtype=float)
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / n

def _initial_configs():
    configs = []
    ico = _icosahedron_vertices()
    
    # Config 1: Icosahedron + 2 points at poles (approx)
    # Add points at (0,0,1.1) and (0,0,-1.1) and optimize
    p1 = np.vstack([ico, [[0.0, 0.0, 1.1]], [[0.0, 0.0, -1.1]]])
    configs.append(p1.copy())
    
    # Config 2: Icosahedron + 2 points at equator
    p2 = np.vstack([ico, [[1.1, 0.0, 0.0]], [[-1.1, 0.0, 0.0]]])
    configs.append(p2.copy())
    
    # Config 3: Random perturbations of Icosahedron + 2 random points
    rng = np.random.default_rng(42)
    for _ in range(5):
        extra1 = rng.normal(0, 0.5, 3)
        extra2 = rng.normal(0, 0.5, 3)
        p = np.vstack([ico, [extra1], [extra2]])
        configs.append(p.copy())
        
    return configs

def min_max_dist_dim3_14() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    best_pts = None
    best_val = -1.0
    
    configs = _initial_configs()
    
    # Annealing beta
    betas = [1.0, 5.0, 20.0, 100.0]
    
    for idx, cfg in enumerate(configs):
        if time.time() > deadline:
            break
            
        current_pts = cfg.copy()
        
        # Try different betas
        for beta in betas:
            if time.time() > deadline:
                break
                
            try:
                # Use L-BFGS-B
                # We need to pass beta to the objective
                # scipy minimize doesn't easily pass extra args to grad, 
                # so we define a closure or use args param.
                
                def obj(x):
                    val, _ = _smooth_obj_and_grad(x, beta)
                    return val
                
                def grad(x):
                    _, g = _smooth_obj_and_grad(x, beta)
                    return g
                
                res = minimize(
                    obj,
                    current_pts.flatten(),
                    jac=grad,
                    method='L-BFGS-B',
                    options={
                        'maxiter': 1000,
                        'ftol': 1e-12,
                        'gtol': 1e-8,
                        'maxls': 50
                    }
                )
                
                if res.success or res.fun < 0: # res.fun is negative score
                    val = -res.fun
                    if val > best_val:
                        best_val = val
                        best_pts = res.x.reshape(14, 3).copy()
                        current_pts = best_pts.copy()
                        
            except Exception:
                pass
        
        # Evaluate the raw score of the current best from this config
        raw_val = _score(current_pts)
        if raw_val > best_val:
            best_val = raw_val
            best_pts = current_pts.copy()

    # Final refinement with a very high beta on the best solution found
    if best_pts is not None and time.time() < deadline - 30:
        beta_final = 500.0
        try:
            def obj_final(x):
                val, _ = _smooth_obj_and_grad(x, beta_final)
                return val
            
            def grad_final(x):
                _, g = _smooth_obj_and_grad(x, beta_final)
                return g
            
            res = minimize(
                obj_final,
                best_pts.flatten(),
                jac=grad_final,
                method='L-BFGS-B',
                options={
                    'maxiter': 2000,
                    'ftol': 1e-14,
                    'gtol': 1e-10,
                    'maxls': 100
                }
            )
            val = -res.fun
            if val > best_val:
                best_val = val
                best_pts = res.x.reshape(14, 3).copy()
        except Exception:
            pass

    # Fallback
    if best_pts is None or not np.all(np.isfinite(best_pts)):
        best_pts = configs[0].copy()
        
    return best_pts

# EVOLVE-BLOCK-END
