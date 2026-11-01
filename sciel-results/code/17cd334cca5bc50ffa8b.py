import numpy as np
import time
from scipy.optimize import minimize

N = 100
DEADLINE = 145.0  # Safety margin for 150s limit

def _score(c):
    """Compute the largest common radius for centres c (shape (N, 3))."""
    if c.shape != (N, 3):
        return -1.0
    if not np.all(np.isfinite(c)):
        return -1.0
    # distance to walls
    wall_dists = 0.5 - np.abs(c)
    min_wall = np.min(wall_dists)
    
    # pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist) / 2.0
    
    return min(min_wall, min_pair)

def _init_grid():
    """
    Initialize with a 4x5x5 grid. 
    4*5*5 = 100 points exactly.
    """
    nx, ny, nz = 4, 5, 5
    xs = (np.arange(nx) + 0.5) / nx - 0.5
    ys = (np.arange(ny) + 0.5) / ny - 0.5
    zs = (np.arange(nz) + 0.5) / nz - 0.5
    
    grid = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    return grid[:N].copy()

def _soft_min_obj_grad(x, beta):
    """
    Objective: -soft_min(dists, beta)
    We want to maximize the minimum distance, so we minimize the negative soft min.
    soft_min(d, beta) = -1/beta * log(sum(exp(-beta * d)))
    
    x: flattened (N*3,)
    beta: inverse temperature for soft min (higher = sharper, closer to hard min)
    """
    x = x.reshape(N, 3)
    
    # --- Pairwise Distances ---
    # diff (N, N, 3)
    diff = x[:, np.newaxis, :] - x[np.newaxis, :]
    dists_sq = np.sum(diff**2, axis=2)
    # Avoid division by zero in gradient if points coincide (unlikely in SA but safe)
    dists_sq = np.maximum(dists_sq, 1e-12)
    dists = np.sqrt(dists_sq)
    np.fill_diagonal(dists, np.inf)
    
    # Only i < j
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]
    
    # Soft min of pairwise distances
    # To avoid underflow/overflow in exp, subtract the min
    min_d = np.min(pair_dists)
    exp_terms = np.exp(-beta * (pair_dists - min_d))
    sum_exp = np.sum(exp_terms)
    soft_min_val = min_d - (1.0 / beta) * np.log(sum_exp)
    
    # --- Wall Distances ---
    # Distance to nearest wall for each point
    abs_x = np.abs(x)
    wall_dists = 0.5 - abs_x
    # We need the min over all wall distances
    min_wall = np.min(wall_dists)
    
    # Combine into a single soft min? 
    # The objective is min(min_pair, min_wall).
    # We can treat the set of all constraints (pairwise + wall) as a single set of distances.
    # However, wall distances are scalar per point, pairwise are scalar per pair.
    # Let's concatenate them into one vector of "constraints".
    
    # Wall distances vector
    wall_vec = wall_dists.flatten() # shape (300,)
    
    # All distances
    all_dists = np.concatenate([pair_dists, wall_vec])
    
    # Recalculate soft min on the combined set
    # Note: This changes the balance between pair and wall constraints if counts differ.
    # Pair count: 4950. Wall count: 300.
    # The soft min will be dominated by the smaller values.
    # This is actually good, as we want to maximize the global minimum.
    
    min_all = np.min(all_dists)
    # Optimization for stability
    # If beta is large, exp(-beta * (d - min)) is 1 for min, ~0 for others.
    # We can just use the min if beta is very large, but for gradient we need the sum.
    
    # To prevent overflow in exp for large beta and small differences, 
    # we rely on the shift (d - min_all).
    
    exp_terms_all = np.exp(-beta * (all_dists - min_all))
    sum_exp_all = np.sum(exp_terms_all)
    
    # Objective: -soft_min
    obj_val = -(min_all - (1.0 / beta) * np.log(sum_exp_all))
    
    # Gradient
    # d(obj)/dx = - d(soft_min)/dx
    # d(soft_min)/dx_i = (1 / sum_exp) * sum_j ( -beta * exp(-beta*(d_j - min)) * d(d_j)/dx_i )
    # Wait, derivative of -1/beta * log(sum(exp(-beta d))) is:
    # = -1/beta * (1/sum) * sum(-beta exp(-beta d) * dd/dx)
    # = (1/sum) * sum(exp(-beta d) * dd/dx)
    
    # So grad_soft_min = (1/sum_exp) * sum_j w_j * grad_d_j
    # where w_j = exp(-beta * (d_j - min))
    
    grad = np.zeros((N, 3))
    
    # Pairwise part
    # w_pair = exp_terms[:4950]
    # grad_d_ij = -(x_i - x_j) / d_ij
    # Contribution to grad_i: -w_ij/d_ij * (x_i - x_j)
    # Contribution to grad_j: +w_ij/d_ij * (x_i - x_j)
    
    if sum_exp_all > 0:
        inv_sum = 1.0 / sum_exp_all
        
        # Pairwise
        w_pair = exp_terms_all[:len(pair_dists)]
        # We need to accumulate gradients
        # factor = w_pair / pair_dists
        factors = w_pair / pair_dists
        
        active_i, active_j = iu
        pair_vecs = diff[active_i, active_j] # x_i - x_j
        
        grad_i_contrib = -factors[:, np.newaxis] * pair_vecs
        grad_j_contrib = factors[:, np.newaxis] * pair_vecs
        
        np.add.at(grad, active_i, grad_i_contrib)
        np.add.at(grad, active_j, grad_j_contrib)
        
        # Wall part
        w_wall = exp_terms_all[len(pair_dists):]
        # grad of wall_dist = 0.5 - |x|
        # d(0.5 - |x|)/dx = -sign(x)
        # So contribution is w_wall * (-sign(x))
        
        # wall_vec is flattened (N,3).
        wall_grad_contrib = -w_wall.reshape(N, 3) * np.sign(x)
        # Handle x=0 case for sign
        wall_grad_contrib[x == 0] = 0 # Subgradient 0 is fine for SA
        
        grad += wall_grad_contrib * inv_sum
        
        # Total gradient for objective (negative soft min)
        grad = -grad * inv_sum # Wait, the inv_sum was inside the sum for soft_min grad.
        # Let's re-verify.
        # grad_soft_min = (1/sum) * sum(w_j * grad_d_j)
        # grad_obj = -grad_soft_min = -(1/sum) * sum(w_j * grad_d_j)
        # My code above calculated `grad` as sum(w_j * grad_d_j) WITHOUT dividing by sum?
        # No, I applied inv_sum at the end.
        # Let's check the pairwise part:
        # I did: grad += ... (accumulated w * grad_d)
        # Then: grad *= inv_sum
        # Then: grad = -grad
        # This is correct.
        
    return obj_val, grad.flatten()

def _optimize_sa(c_init, time_limit):
    """
    Perform Simulated Annealing optimization.
    """
    start_time = time.time()
    x = c_init.flatten().copy()
    
    # Beta schedule: start low (smooth), end high (sharp)
    # We want to maximize r.
    # beta = 1/T.
    # Start T=1.0 (beta=1), end T=0.01 (beta=100)?
    # Actually, for soft min, beta is the sharpness.
    # Let's use a fixed high beta for the gradient to focus on the min,
    # but use SA to jump out of local minima.
    # Or adapt beta.
    
    # Let's use a fixed beta that is high enough to approximate the hard min
    # but low enough to provide a smooth gradient.
    # r ~ 0.11. Distances ~ 0.22.
    # beta = 50 seems reasonable.
    beta = 50.0
    
    # SA Parameters
    T = 0.01
    T_min = 1e-5
    cooling_rate = 0.99
    
    best_x = x.copy()
    best_obj = _soft_min_obj_grad(x, beta)[0]
    
    # Precompute indices for speed
    iu = np.triu_indices(N, k=1)
    
    while time.time() < start_time + time_limit:
        if T < T_min:
            break
            
        # Generate a random perturbation
        # Gaussian noise
        noise = np.random.normal(0, T, size=x.shape)
        x_new = x + noise
        
        # Clip to bounds
        x_new = np.clip(x_new, -0.5, 0.5)
        
        obj_new = _soft_min_obj_grad(x_new, beta)[0]
        obj_curr = _soft_min_obj_grad(x, beta)[0]
        
        # Accept if better or with probability exp(-delta/T)
        delta = obj_new - obj_curr
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            x = x_new
            if obj_new < best_obj:
                best_obj = obj_new
                best_x = x.copy()
                
        T *= cooling_rate
        
    return best_x.reshape(N, 3)

def _optimize_lbfgs(c_init, time_limit):
    """
    Perform L-BFGS-B optimization for polishing.
    """
    start_time = time.time()
    x0 = c_init.flatten()
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    # Use a high beta for polishing to focus on the true minimum
    beta = 100.0
    
    try:
        res = minimize(
            _soft_min_obj_grad,
            x0,
            args=(beta,),
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
        )
        if res.fun < _soft_min_obj_grad(x0, beta)[0]:
            return res.x.reshape(N, 3)
    except Exception:
        pass
        
    return c_init

def construct():
    start_time = time.time()
    
    # 1. Initial Grid
    c_grid = _init_grid()
    
    best_c = c_grid.copy()
    best_r = _score(best_c)
    
    # 2. Multi-Start SA
    # We will run SA from the grid, and then from perturbed versions of the best found.
    
    num_restarts = 5
    time_per_restart = (DEADLINE - start_time - 10.0) / num_restarts
    
    for i in range(num_restarts):
        if time.time() > DEADLINE - 10.0:
            break
            
        # Determine initial point for this restart
        if i == 0:
            c_init = best_c
        else:
            # Perturb the best found so far to escape local minima
            # "Geometric Kick": Add noise proportional to current radius
            noise_scale = 0.05 * best_r
            c_init = best_c + np.random.normal(0, noise_scale, size=best_c.shape)
            c_init = np.clip(c_init, -0.5, 0.5)
            
        # Allocate time
        t_start = time.time()
        t_limit = min(time_per_restart, DEADLINE - t_start - 10.0)
        
        if t_limit < 1.0:
            continue
            
        # Run SA
        c_res = _optimize_sa(c_init, t_limit)
        
        # Polish with L-BFGS-B if time permits
        if time.time() < DEADLINE - 5.0:
            c_res = _optimize_lbfgs(c_res, 2.0)
            
        # Evaluate
        r_res = _score(c_res)
        if r_res > best_r:
            best_r = r_res
            best_c = c_res
            
    # 3. Final Polish
    if time.time() < DEADLINE - 2.0:
        best_c = _optimize_lbfgs(best_c, DEADLINE - time.time() - 1.0)
        best_r = _score(best_c)
        
    return best_c

if __name__ == "__main__":
    start = time.time()
    res = construct()
    end = time.time()
    print(f"Time: {end - start:.2f}s")
    print(f"Radius: {_score(res):.10f}")
    print(f"Shape: {res.shape}")
