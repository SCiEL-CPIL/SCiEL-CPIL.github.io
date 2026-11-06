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
    """
    # Pairwise distances
    d = cdist(c, c)
    i_idx, j_idx = np.triu_indices(N, k=1)
    pair_dists = d[i_idx, j_idx] / 2.0
    
    # Wall distances
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
    
    contrib = (0.5 * w_pair * u)
    
    np.add.at(grad_c, i_idx, -contrib)
    np.add.at(grad_c, j_idx, contrib)
    
    # Wall part
    w_wall = weights[len(pair_dists):]
    ww = w_wall.reshape(6, N)
    
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
    # Slightly tighter grid to encourage closer packing initially?
    # Or keep standard. 5x5x4 is good.
    xs = np.linspace(-0.42, 0.42, 5)
    ys = np.linspace(-0.42, 0.42, 5)
    zs = np.linspace(-0.4, 0.4, 4)
    
    centres = []
    for z in zs:
        for y in ys:
            for x in xs:
                jitter = rng.uniform(-0.02, 0.02, 3)
                centres.append([x, y, z] + jitter)
    
    return np.array(centres[:N])

def _initial_random(seed):
    """Create a random initialization."""
    rng = np.random.RandomState(seed)
    return rng.uniform(-0.45, 0.45, (N, 3))

def _optimize_with_annealing(c_init, time_limit, start_time, alphas=None):
    """
    Run the annealing optimization from a given start.
    """
    if alphas is None:
        # More granular steps at the beginning, coarser at the end
        alphas = [5, 10, 20, 40, 80, 150, 300, 600, 1200, 2500, 5000]
        
    best_c = c_init.copy()
    best_score = _score(c_init)
    current_c = c_init.copy()
    stall_count = 0
    stall_threshold = 3  # If no improvement for 3 alphas, consider a kick
    
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
            remaining_time = time_limit - (time.time() - start_time)
            if remaining_time < 5:
                break
                
            # Heuristic for maxiter: allow more iterations when time is plentiful
            # and alpha is low (smoother landscape).
            if alpha < 100:
                maxiter = int(min(1000, max(100, remaining_time * 15)))
            else:
                maxiter = int(min(500, max(50, remaining_time * 8)))
            
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
                stall_count = 0
            else:
                stall_count += 1
                # If we are stuck, try a kick
                if stall_count >= stall_threshold:
                    # Perform a kick: mix current best with random noise
                    # Or just add larger noise
                    noise_scale = 0.01 + 0.01 * stall_count
                    noise = np.random.normal(0, noise_scale, (N, 3))
                    c_kicked = best_c + noise
                    c_kicked = np.clip(c_kicked, -0.5, 0.5)
                    
                    # Try to optimize from the kicked position immediately with a slightly lower alpha
                    # to see if it escapes the local minimum
                    kick_alpha = max(5, alpha // 2)
                    
                    def obj_kick(x):
                        c_flat = x.reshape(N, 3)
                        val, _ = _compute_smooth_obj_grad(c_flat, kick_alpha)
                        return val
                        
                    def grad_kick(x):
                        c_flat = x.reshape(N, 3)
                        _, g = _compute_smooth_obj_grad(c_flat, kick_alpha)
                        return g.flatten()
                        
                    if time.time() - start_time < time_limit - 10:
                        res_kick = minimize(
                            obj_kick,
                            c_kicked.flatten(),
                            jac=grad_kick,
                            method='L-BFGS-B',
                            bounds=bounds,
                            options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
                        )
                        c_kicked_opt = res_kick.x.reshape(N, 3)
                        c_kicked_opt = np.clip(c_kicked_opt, -0.5, 0.5)
                        score_kicked = _score(c_kicked_opt)
                        
                        if score_kicked > best_score:
                            best_score = score_kicked
                            best_c = c_kicked_opt.copy()
                            current_c = c_kicked_opt.copy()
                            stall_count = 0
                        else:
                            # If kick failed, revert to best_c for next iteration
                            current_c = best_c.copy()
                else:
                    # If not stuck enough, keep the new position if it improved the SLS objective
                    # even if the hard score didn't (it might be a better starting point for higher alpha)
                    if result.fun < obj(x0) - 1e-12:
                        current_c = c_new.copy()
                    else:
                        current_c = best_c.copy()
                        
        except Exception:
            pass
            
    return best_c

def construct():
    start_time = time.time()
    
    best_c = None
    best_score = 0.0
    
    # Strategy: Run multiple diverse starts.
    # Start 1: Jittered Grid (Structured) - usually best for this N
    c_grid = _initial_jittered_grid(1)
    t1_limit = start_time + (DEADLINE - start_time) * 0.5
    c1 = _optimize_with_annealing(c_grid, t1_limit, start_time)
    s1 = _score(c1)
    if s1 > best_score:
        best_score = s1
        best_c = c1
    
    # Start 2: Random
    if time.time() < DEADLINE - 30:
        c_rand1 = _initial_random(123)
        t2_limit = start_time + (DEADLINE - start_time) * 0.7
        c2 = _optimize_with_annealing(c_rand1, t2_limit, start_time)
        s2 = _score(c2)
        if s2 > best_score:
            best_score = s2
            best_c = c2
            
    # Start 3: Different Grid Jitter
    if time.time() < DEADLINE - 20:
        c_grid2 = _initial_jittered_grid(42)
        t3_limit = start_time + (DEADLINE - start_time) * 0.85
        c3 = _optimize_with_annealing(c_grid2, t3_limit, start_time)
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
        alphas_polish = [2000, 5000, 10000, 20000]
        
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
                    
                maxiter = int(min(500, max(50, remaining_time * 10)))
                
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
        best_c = _initial_jittered_grid(1)

    # Ensure validity
    best_c = np.clip(best_c, -0.5, 0.5)
    
    return best_c
