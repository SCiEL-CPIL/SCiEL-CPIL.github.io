"""
Optimized Sphere Packing Program
Uses Smooth Log-Sum-Exp objective with L-BFGS-B optimization.
Includes multi-start strategy, jittered grid initialization, and aggressive alpha annealing.
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 100
DEADLINE = 148.0  # Safety margin from 150s limit

def _compute_smooth_obj_grad(c, alpha):
    """
    Computes the smooth log-sum-exp objective and gradient.
    Objective: f(c) = (1/alpha) * log( sum_{s} exp(-alpha * s) )
    Minimizing f(c) maximizes the minimum distance s.
    """
    # Pairwise distances
    d = cdist(c, c)
    i_idx, j_idx = np.triu_indices(N, k=1)
    pair_dists = d[i_idx, j_idx] / 2.0
    
    # Wall distances
    # dist to + wall: 0.5 - c
    # dist to - wall: 0.5 + c
    wall_dists = np.concatenate([
        0.5 - c[:, 0],
        0.5 + c[:, 0],
        0.5 - c[:, 1],
        0.5 + c[:, 1],
        0.5 - c[:, 2],
        0.5 + c[:, 2]
    ])
    
    all_dists = np.concatenate([pair_dists, wall_dists])
    
    # Numerical stability
    m = np.min(all_dists)
    shifted = all_dists - m
    exp_terms = np.exp(-alpha * shifted)
    Z = np.sum(exp_terms)
    
    if Z == 0:
        return 0.0, np.zeros_like(c)
        
    obj_val = (1.0 / alpha) * (np.log(Z) - alpha * m)
    
    # Gradient
    weights = exp_terms / Z  # Sum to 1
    
    grad_c = np.zeros_like(c)
    
    # Pairwise part
    pair_dists_raw = d[i_idx, j_idx]
    pair_dists_raw_safe = np.where(pair_dists_raw < 1e-12, 1e-12, pair_dists_raw)
    
    ci = c[i_idx]
    cj = c[j_idx]
    diff = ci - cj  # (4950, 3)
    norms = pair_dists_raw_safe[:, np.newaxis]
    u = diff / norms  # Unit vector j -> i
    
    w_pair = weights[:len(pair_dists)]
    w_pair = w_pair[:, np.newaxis]
    
    contrib = (alpha * w_pair * u) / 2.0
    
    np.add.at(grad_c, i_idx, -contrib)
    np.add.at(grad_c, j_idx, contrib)
    
    # Wall part
    w_wall = weights[len(pair_dists):]
    ww = w_wall.reshape(6, N)
    
    grad_c[:, 0] += alpha * (ww[0] - ww[1])
    grad_c[:, 1] += alpha * (ww[2] - ww[3])
    grad_c[:, 2] += alpha * (ww[4] - ww[5])
    
    return obj_val, grad_c

def _lj_obj_grad(c, sigma=0.1, epsilon=1.0):
    """
    Lennard-Jones-inspired objective: sum of pairwise repulsive forces.
    This provides strong repulsion that can help escape local minima.
    """
    d = cdist(c, c)
    i_idx, j_idx = np.triu_indices(N, k=1)
    r = d[i_idx, j_idx]
    r_safe = np.where(r < 1e-12, 1e-12, r)
    
    # LJ potential: U = epsilon * ((sigma/r)^12 - 2*(sigma/r)^6)
    # We only use the repulsive part for optimization
    sr6 = (sigma / r_safe) ** 6
    sr12 = sr6 ** 2
    U = epsilon * (sr12 - 2 * sr6)
    
    # Wall penalty: strong repulsion from walls
    wall_pen = 0.0
    for k in range(3):
        wall_pen += np.sum((sigma / np.maximum(0.5 - c[:, k], 1e-12)) ** 12)
        wall_pen += np.sum((sigma / np.maximum(0.5 + c[:, k], 1e-12)) ** 12)
    
    obj_val = np.sum(U) + wall_pen
    
    # Gradient
    grad_c = np.zeros_like(c)
    
    # dU/dr = epsilon * (-12*sigma^12/r^13 + 12*sigma^6/r^7)
    # = 12*epsilon*sigma^6/r^7 * (-sigma^6/r^6 + 1)
    # = -12*epsilon*sigma^6/r^7 * (sigma^6/r^6 - 1)
    dU_dr = 12 * epsilon * (sigma ** 6) / r_safe ** 7 * (1 - (sigma / r_safe) ** 6)
    
    ci = c[i_idx]
    cj = c[j_idx]
    diff = ci - cj
    norms = r_safe[:, np.newaxis]
    u = diff / norms
    
    contrib = (dU_dr / 2.0)[:, np.newaxis] * u
    np.add.at(grad_c, i_idx, -contrib)
    np.add.at(grad_c, j_idx, contrib)
    
    # Wall gradient
    for k in range(3):
        pos_wall = 0.5 - c[:, k]
        neg_wall = 0.5 + c[:, k]
        pos_safe = np.maximum(pos_wall, 1e-12)
        neg_safe = np.maximum(neg_wall, 1e-12)
        grad_c[:, k] -= 12 * (sigma ** 12) / pos_safe ** 13
        grad_c[:, k] += 12 * (sigma ** 12) / neg_safe ** 13
    
    return obj_val, grad_c

def _score(c):
    """Compute the exact packing radius."""
    if c.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(c)):
        return 0.0
    # Check bounds
    if np.any(c > 0.5) or np.any(c < -0.5):
        return 0.0
    
    bound = np.min(0.5 - np.abs(c))
    if bound < 0.0:
        return 0.0
    
    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    pair_min = np.min(d) / 2.0
    return float(min(bound, pair_min))

def _initial_jittered_grid(seed):
    """Create a 5x5x4 grid with random jitter."""
    rng = np.random.RandomState(seed)
    xs = np.linspace(-0.4, 0.4, 5)
    ys = np.linspace(-0.4, 0.4, 5)
    zs = np.linspace(-0.375, 0.375, 4)
    
    centres = []
    for z in zs:
        for y in ys:
            for x in xs:
                # Add small random jitter to break symmetry
                jitter = rng.uniform(-0.02, 0.02, 3)
                centres.append([x, y, z] + jitter)
    
    return np.array(centres[:N])

def _initial_random(seed):
    """Create a random initialization."""
    rng = np.random.RandomState(seed)
    return rng.uniform(-0.45, 0.45, (N, 3))

def _optimize_start(c_init, time_limit, start_time, alphas=None):
    """
    Run the annealing optimization from a given start.
    Returns the best center configuration found.
    """
    if alphas is None:
        alphas = [5, 10, 20, 50, 100, 200, 500, 1000, 2000]
        
    best_c = c_init.copy()
    best_score = _score(c_init)
    
    for alpha in alphas:
        if time.time() - start_time > time_limit:
            break
            
        # Define objective and gradient for L-BFGS-B
        def obj(x):
            c_flat = x.reshape(N, 3)
            val, _ = _compute_smooth_obj_grad(c_flat, alpha)
            return val
        
        def grad(x):
            c_flat = x.reshape(N, 3)
            _, g = _compute_smooth_obj_grad(c_flat, alpha)
            return g.flatten()
        
        x0 = best_c.flatten()
        bounds = [(-0.5, 0.5)] * (3 * N)
        
        try:
            # Adjust maxiter based on remaining time and alpha
            # Higher alpha requires more iterations to converge to the sharp minimum
            remaining_time = time_limit - (time.time() - start_time)
            if remaining_time < 5:
                break
                
            # Heuristic for maxiter: more for higher alpha
            maxiter = int(min(500, max(50, remaining_time * 10)))
            
            result = minimize(
                obj,
                x0,
                jac=grad,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            if result.success or result.fun < obj(x0) - 1e-15:
                c_new = result.x.reshape(N, 3)
                c_new = np.clip(c_new, -0.5, 0.5)
                
                score_new = _score(c_new)
                if score_new > best_score:
                    best_score = score_new
                    best_c = c_new.copy()
                else:
                    # Even if score didn't improve, the position might be better for next alpha
                    # But only update if we made progress in the objective function
                    if result.fun < obj(x0):
                        best_c = c_new.copy()
                        
        except Exception:
            pass
            
    return best_c

def _lj_optimize(c_init, time_limit, start_time, sigma=0.08):
    """
    Lennard-Jones-inspired optimization using gradient descent with adaptive step size.
    """
    best_c = c_init.copy()
    best_score = _score(c_init)
    
    # Adaptive gradient descent with momentum
    lr = 0.001
    momentum = 0.9
    v = np.zeros_like(c_init)
    
    max_iter = 2000
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        obj_val, grad = _lj_obj_grad(best_c, sigma=sigma)
        
        # Adaptive learning rate
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-10:
            break
        
        # Update with momentum
        v = momentum * v - lr * grad
        c_new = best_c + v
        c_new = np.clip(c_new, -0.5, 0.5)
        
        # Check if we improved
        score_new = _score(c_new)
        if score_new > best_score:
            best_score = score_new
            best_c = c_new.copy()
        else:
            # Reduce learning rate if no improvement
            lr *= 0.95
            v *= 0.5
        
        # Early stop if learning rate too small
        if lr < 1e-8:
            break
    
    return best_c

def construct():
    start_time = time.time()
    
    best_c = None
    best_score = 0.0
    
    # Start 1: Jittered Grid (Structured + Symmetry Breaking)
    c_grid = _initial_jittered_grid(1)
    t1_limit = start_time + (DEADLINE - start_time) * 0.4
    c1 = _optimize_start(c_grid, t1_limit, start_time)
    s1 = _score(c1)
    if s1 > best_score:
        best_score = s1
        best_c = c1
    
    # Start 2: Random with L-J optimization
    if time.time() < DEADLINE - 30:
        c_rand1 = _initial_random(123)
        t2_limit = start_time + (DEADLINE - start_time) * 0.6
        c2 = _lj_optimize(c_rand1, t2_limit, start_time, sigma=0.08)
        s2 = _score(c2)
        if s2 > best_score:
            best_score = s2
            best_c = c2
    
    # Start 3: Random with L-J optimization (different seed)
    if time.time() < DEADLINE - 20:
        c_rand2 = _initial_random(456)
        t3_limit = start_time + (DEADLINE - start_time) * 0.75
        c3 = _lj_optimize(c_rand2, t3_limit, start_time, sigma=0.06)
        s3 = _score(c3)
        if s3 > best_score:
            best_score = s3
            best_c = c3
    
    # Start 4: Grid with L-J optimization
    if time.time() < DEADLINE - 15:
        c_grid2 = _initial_jittered_grid(42)
        t4_limit = start_time + (DEADLINE - start_time) * 0.85
        c4 = _lj_optimize(c_grid2, t4_limit, start_time, sigma=0.07)
        s4 = _score(c4)
        if s4 > best_score:
            best_score = s4
            best_c = c4

    # Final Polish:
    # If we have time, try to refine the best_c found with a very high alpha
    if time.time() < DEADLINE - 5 and best_c is not None:
        c_polish = best_c.copy()
        t_polish_limit = DEADLINE - 1
        
        # High alpha optimization for final polish
        alphas_polish = [500, 1000, 2000, 5000]
        
        for alpha in alphas_polish:
            if time.time() > t_polish_limit:
                break
                
            def obj(x):
                c_flat = x.reshape(N, 3)
                val, _ = _compute_smooth_obj_grad(c_flat, alpha)
                return val
                
            def grad(x):
                c_flat = x.reshape(N, 3)
                _, g = _compute_smooth_obj_grad(c_flat, alpha)
                return g.flatten()
                
            try:
                remaining_time = t_polish_limit - time.time()
                if remaining_time < 2:
                    break
                    
                maxiter = int(min(300, max(30, remaining_time * 8)))
                
                result = minimize(
                    obj,
                    c_polish.flatten(),
                    jac=grad,
                    method='L-BFGS-B',
                    bounds=[(-0.5, 0.5)] * (3 * N),
                    options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-12}
                )
                c_new = result.x.reshape(N, 3)
                c_new = np.clip(c_new, -0.5, 0.5)
                score_new = _score(c_new)
                if score_new > best_score:
                    best_score = score_new
                    best_c = c_new
                    c_polish = c_new
            except:
                pass

    if best_c is None:
        # Fallback: return grid if everything failed
        best_c = _initial_jittered_grid(1)

    # Ensure validity
    best_c = np.clip(best_c, -0.5, 0.5)
    
    return best_c
