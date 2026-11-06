"""
Optimized Sphere Packing Program
Combines Smooth Log-Sum-Exp objective with L-BFGS-B, correct gradient derivation,
and a 'kick' mechanism to escape local minima.
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 100
DEADLINE = 145.0  # Safety margin from 150s limit

def _compute_smooth_obj_grad(c, alpha):
    """
    Computes the smooth log-sum-exp objective and gradient.
    Objective: f(c) = (1/alpha) * log( sum_{s} exp(-alpha * s) )
    Minimizing f(c) maximizes the minimum distance s.
    
    Gradient Derivation:
    f(c) = (1/alpha) * log(Z)
    Z = sum_k exp(-alpha * s_k)
    df/dc = (1/alpha) * (1/Z) * dZ/dc
    dZ/dc = sum_k -alpha * exp(-alpha * s_k) * ds_k/dc
    df/dc = (1/alpha) * (1/Z) * sum_k -alpha * exp(-alpha * s_k) * ds_k/dc
    df/dc = - sum_k (exp(-alpha * s_k) / Z) * ds_k/dc
    df/dc = - sum_k w_k * ds_k/dc
    where w_k = exp(-alpha * s_k) / Z
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
    # s_ij = 0.5 * ||c_i - c_j||
    # ds_ij/dc_i = 0.5 * (c_i - c_j) / ||c_i - c_j||
    # Contribution to grad_c: -w_ij * ds_ij/dc_i
    
    pair_dists_raw = d[i_idx, j_idx]
    pair_dists_raw_safe = np.where(pair_dists_raw < 1e-12, 1e-12, pair_dists_raw)
    
    ci = c[i_idx]
    cj = c[j_idx]
    diff = ci - cj  # (4950, 3)
    norms = pair_dists_raw_safe[:, np.newaxis]
    u = diff / norms  # Unit vector j -> i
    
    w_pair = weights[:len(pair_dists)]
    w_pair = w_pair[:, np.newaxis]
    
    # grad_i += -w * 0.5 * u
    # grad_j += +w * 0.5 * u
    contrib = (0.5 * w_pair * u)
    
    np.add.at(grad_c, i_idx, -contrib)
    np.add.at(grad_c, j_idx, contrib)
    
    # Wall part
    # s_wall = 0.5 - c_k (for + wall)
    # ds_wall/dc_k = -1
    # Contribution: -w * (-1) = +w
    # s_wall = 0.5 + c_k (for - wall)
    # ds_wall/dc_k = 1
    # Contribution: -w * (1) = -w
    
    w_wall = weights[len(pair_dists):]
    ww = w_wall.reshape(6, N)
    
    # ww[0] corresponds to 0.5 - c[:, 0] (positive wall x)
    # grad_x += ww[0]
    # ww[1] corresponds to 0.5 + c[:, 0] (negative wall x)
    # grad_x -= ww[1]
    grad_c[:, 0] += (ww[0] - ww[1])
    grad_c[:, 1] += (ww[2] - ww[3])
    grad_c[:, 2] += (ww[4] - ww[5])
    
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
                jitter = rng.uniform(-0.03, 0.03, 3)
                centres.append([x, y, z] + jitter)
    
    return np.array(centres[:N])

def _initial_random(seed):
    """Create a random initialization."""
    rng = np.random.RandomState(seed)
    return rng.uniform(-0.45, 0.45, (N, 3))

def _optimize_with_annealing(c_init, time_limit, start_time, alphas=None):
    """
    Run the annealing optimization from a given start.
    Includes a 'kick' mechanism to escape local minima.
    """
    if alphas is None:
        alphas = [5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
        
    best_c = c_init.copy()
    best_score = _score(c_init)
    current_c = c_init.copy()
    
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
        
        x0 = current_c.flatten()
        bounds = [(-0.5, 0.5)] * (3 * N)
        
        try:
            # Adjust maxiter based on remaining time and alpha
            remaining_time = time_limit - (time.time() - start_time)
            if remaining_time < 5:
                break
                
            # Heuristic for maxiter
            maxiter = int(min(500, max(50, remaining_time * 10)))
            
            result = minimize(
                obj,
                x0,
                jac=grad,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            c_new = result.x.reshape(N, 3)
            c_new = np.clip(c_new, -0.5, 0.5)
            
            score_new = _score(c_new)
            
            if score_new > best_score:
                best_score = score_new
                best_c = c_new.copy()
                current_c = c_new.copy()
            else:
                # If score didn't improve significantly, accept the new position 
                # to allow exploration, but keep best_c safe.
                if result.fun < obj(x0) - 1e-12:
                    current_c = c_new.copy()
                else:
                    # Try a small kick to escape local minimum
                    # Add small noise
                    noise = np.random.normal(0, 0.005, (N, 3))
                    c_kicked = current_c + noise
                    c_kicked = np.clip(c_kicked, -0.5, 0.5)
                    # Check if kick improved score (unlikely but possible)
                    s_kick = _score(c_kicked)
                    if s_kick > best_score:
                        best_score = s_kick
                        best_c = c_kicked.copy()
                        current_c = c_kicked.copy()
                    else:
                        # Keep current_c as is, or use the result if it's structurally similar
                        # For safety, revert to best_c for next alpha if we are stuck
                        if score_new < best_score * 0.9:
                             current_c = best_c.copy()
                        
        except Exception:
            pass
            
    return best_c

def construct():
    start_time = time.time()
    
    best_c = None
    best_score = 0.0
    
    # Strategy: Run multiple diverse starts, allocating time based on potential.
    # Start 1: Jittered Grid (Structured)
    c_grid = _initial_jittered_grid(1)
    t1_limit = start_time + (DEADLINE - start_time) * 0.4
    c1 = _optimize_with_annealing(c_grid, t1_limit, start_time)
    s1 = _score(c1)
    if s1 > best_score:
        best_score = s1
        best_c = c1
    
    # Start 2: Random
    if time.time() < DEADLINE - 30:
        c_rand1 = _initial_random(123)
        t2_limit = start_time + (DEADLINE - start_time) * 0.6
        c2 = _optimize_with_annealing(c_rand1, t2_limit, start_time)
        s2 = _score(c2)
        if s2 > best_score:
            best_score = s2
            best_c = c2
    
    # Start 3: Random (different seed)
    if time.time() < DEADLINE - 20:
        c_rand2 = _initial_random(456)
        t3_limit = start_time + (DEADLINE - start_time) * 0.8
        c3 = _optimize_with_annealing(c_rand2, t3_limit, start_time)
        s3 = _score(c3)
        if s3 > best_score:
            best_score = s3
            best_c = c3

    # Final Polish:
    # If we have time, try to refine the best_c found with a very high alpha
    if time.time() < DEADLINE - 5 and best_c is not None:
        c_polish = best_c.copy()
        t_polish_limit = DEADLINE - 1
        
        # High alpha optimization for final polish
        alphas_polish = [2000, 5000, 10000]
        
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
