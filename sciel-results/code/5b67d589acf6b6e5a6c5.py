import numpy as np
import time
from scipy.optimize import minimize

N = 100
HALF = 0.5

def compute_min_dist(pos):
    """Compute the minimum radius for a given configuration."""
    if not np.all(np.isfinite(pos)):
        return 0.0
    
    # Vectorized distance calculation
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    dists = np.sqrt(dists_sq)
    np.fill_diagonal(dists, np.inf)
    
    min_pair_val = np.min(dists)
    
    d_walls = HALF - np.abs(pos)
    min_wall_val = np.min(d_walls)
    
    r_pair = min_pair_val / 2.0
    r_wall = min_wall_val
    
    return min(r_pair, r_wall)

def soft_min_objective_and_grad(x, beta):
    """
    Objective: Minimize the soft-min of the constraints.
    Constraints:
    1. Pair distances d_ij >= 2r  =>  violation v_ij = 2r - d_ij (we want v_ij <= 0)
       Actually, we want to maximize r. 
       Let's define the objective as minimizing the "shortfall".
       Alternatively, we can maximize the soft-min of the actual distances (scaled).
       
       Let's define the problem as maximizing r such that:
       d_ij >= 2r for all i<j
       0.5 - |c_ik| >= r for all i,k
       
       This is equivalent to maximizing the minimum of { d_ij/2, 0.5-|c_ik| }.
       
       We can minimize the negative soft-min:
       J(x) = - (1/beta) * log( sum( exp(-beta * d_ij/2) + sum(exp(-beta * (0.5-|c_ik|)) ) ) )
       
       Wait, soft-min of a set {a_i} is approx (1/beta) log(sum(exp(-beta a_i))) as beta->inf?
       No, soft-min(x) = - (1/beta) log(sum(exp(-beta x_i))).
       As beta->inf, this approaches min(x_i).
       
       So we want to MINIMIZE -soft_min(distances) ? No, we want to MAXIMIZE soft_min.
       So we MINIMIZE -soft_min.
       
       Let S = { d_ij/2, w_ik }
       soft_min(S) = - (1/beta) * log( sum_{s in S} exp(-beta * s) )
       
       We want to maximize soft_min(S).
       So we minimize f(x) = -soft_min(S) = (1/beta) * log( sum_{s in S} exp(-beta * s) )
       
       Gradient of f(x) wrt x:
       f(x) = (1/beta) * log( Z ), where Z = sum exp(-beta s_j)
       df/dx = (1/beta) * (1/Z) * sum( d(exp(-beta s_j))/dx )
             = (1/beta) * (1/Z) * sum( -beta * exp(-beta s_j) * ds_j/dx )
             = - (1/Z) * sum( exp(-beta s_j) * ds_j/dx )
             = - sum( p_j * ds_j/dx )
             where p_j = exp(-beta s_j) / Z  (softmax weights)
             
       So grad = - sum_j p_j * grad(s_j)
       
       s_j for pairs: d_ij / 2. 
       grad(d_ij/2) wrt x_i = (x_i - x_j) / (2 * d_ij)
       grad(d_ij/2) wrt x_j = (x_j - x_i) / (2 * d_ij)
       
       s_k for walls: 0.5 - |c_ik|.
       grad(0.5 - |c_ik|) wrt x_ik = -sign(c_ik) if |c_ik| is the max component for that sphere?
       Actually, the constraint is min_k (0.5 - |c_ik|).
       The soft-min sums over ALL wall distances? 
       Yes, S includes all 300 wall distances and 4950 pair distances.
       
       This is a very large sum (5250 terms).
       exp(-beta * s) can underflow if beta is large and s is large.
       We should shift s by the minimum to prevent underflow.
       
       Let s_min = min(S).
       Let s'_j = s_j - s_min.
       Z = exp(beta * s_min) * sum(exp(-beta s'_j))
       log(Z) = beta * s_min + log(sum(exp(-beta s'_j)))
       f(x) = (1/beta) log(Z) = s_min + (1/beta) log(sum(exp(-beta s'_j)))
       
       This is numerically stable.
       
       Gradient:
       p_j = exp(-beta s'_j) / sum(exp(-beta s'_j))
       grad(f) = grad(s_min) - sum_j p_j * grad(s_j)
       
       Wait, s_min is the hard min, which is non-differentiable.
       However, the soft-min formulation is differentiable everywhere.
       The gradient of the soft-min is simply -sum_j p_j * grad(s_j).
       
       Let's re-verify.
       f(x) = - soft_min(S) = (1/beta) log( sum exp(-beta s_j) )
       df/dx = (1/beta) * (1/Z) * sum (-beta exp(-beta s_j) ds_j/dx)
             = - sum (exp(-beta s_j)/Z) * (ds_j/dx)
             = - sum p_j * (ds_j/dx)
             
       This is correct.
       
       We need to compute:
       1. All pair distances d_ij and wall distances w_ik.
       2. s_j values: d_ij/2 and w_ik.
       3. s_min for stability shift.
       4. Weights p_j.
       5. Gradient accumulation.
       
       Complexity: O(N^2) per iteration. N=100 -> 10,000 pairs. Manageable.
    """
    centers = x.reshape(-1, 3)
    n = len(centers)
    
    # 1. Compute distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    dists = np.sqrt(dists_sq + 1e-15)
    np.fill_diagonal(dists, np.inf)
    
    # Wall distances
    abs_centers = np.abs(centers)
    # The constraint is min_k (0.5 - |c_ik|).
    # This is equivalent to 0.5 - max_k |c_ik|.
    # So the "distance" to the wall for sphere i is d_wall_i = 0.5 - max(abs(c_ix), abs(c_iy), abs(c_iz)).
    max_abs = np.max(abs_centers, axis=1)
    d_walls = 0.5 - max_abs
    
    # 2. Construct the list of "s" values
    # We need to be careful with memory. 4950 pairs + 100 walls.
    # We can compute the gradient in parts to save memory if needed, but 5000 floats is tiny.
    
    # Pair terms
    i_idx, j_idx = np.triu_indices(n, k=1)
    d_pairs = dists[i_idx, j_idx]
    s_pairs = d_pairs / 2.0
    
    # Wall terms
    s_walls = d_walls
    
    # Combine for finding s_min
    all_s = np.concatenate([s_pairs, s_walls])
    s_min = np.min(all_s)
    
    # Shifted s values
    s_pairs_shifted = s_pairs - s_min
    s_walls_shifted = s_walls - s_min
    
    # 3. Compute weights p_j
    # p_j = exp(-beta * s_j_shifted) / sum(exp(-beta * s_k_shifted))
    
    # Clip exponents to prevent underflow/overflow issues, though shifted s >= 0 so exp <= 1.
    exp_pairs = np.exp(-beta * s_pairs_shifted)
    exp_walls = np.exp(-beta * s_walls_shifted)
    
    sum_exp = np.sum(exp_pairs) + np.sum(exp_walls)
    
    if sum_exp == 0:
        # Degenerate case, return zero gradient
        return 0.0, np.zeros_like(x)
        
    p_pairs = exp_pairs / sum_exp
    p_walls = exp_walls / sum_exp
    
    # 4. Compute Gradient
    # grad = - sum p_j * grad(s_j)
    
    grad = np.zeros_like(centers)
    
    # Pair gradients
    # s_ij = d_ij / 2
    # d(s_ij)/d(x_i) = (x_i - x_j) / (2 * d_ij)
    # d(s_ij)/d(x_j) = (x_j - x_i) / (2 * d_ij)
    
    vecs = diff[i_idx, j_idx] # x_i - x_j
    # d_ij = d_pairs
    # factor = p_pairs / d_pairs
    # grad_i += - factor * vecs
    # grad_j += - factor * (-vecs) = + factor * vecs
    
    # Handle d_pairs == 0 (should not happen with 1e-15 in sqrt)
    factors = p_pairs / d_pairs
    
    forces_i = -factors[:, np.newaxis] * vecs
    forces_j = factors[:, np.newaxis] * vecs
    
    np.add.at(grad, i_idx, forces_i)
    np.add.at(grad, j_idx, forces_j)
    
    # Wall gradients
    # s_i = 0.5 - max(|c_ix|, |c_iy|, |c_iz|)
    # Let m_i = max component.
    # s_i = 0.5 - |c_{i, m_i}|
    # d(s_i)/d(x_{i, k}) = -sign(x_{i, k}) if k == m_i, else 0.
    # grad_i = - p_i * grad(s_i) = p_i * sign(x_{i, m_i}) * e_{m_i}
    
    # Identify max component for each sphere
    max_abs_flat = abs_centers
    max_indices = np.argmax(max_abs_flat, axis=1)
    
    # Get signs
    signs = np.sign(centers)
    # Handle 0
    signs[signs == 0] = 1.0
    
    # Extract the sign of the max component
    # max_indices shape (N,), centers shape (N,3)
    # We need signs[i, max_indices[i]]
    sign_max = signs[np.arange(n), max_indices]
    
    # grad_i = p_i * sign_max[i] * e_{max_indices[i]}
    # We can do this with advanced indexing or a loop.
    # grad[np.arange(n), max_indices] += p_walls * sign_max
    
    grad[np.arange(n), max_indices] += p_walls * sign_max
    
    # Objective value
    # f(x) = s_min + (1/beta) * log(sum_exp)
    obj_val = s_min + (1.0 / beta) * np.log(sum_exp)
    
    return obj_val, grad.flatten()

def _optimize_soft_min(init_centers, target_beta, max_iter=200, time_limit=10.0):
    """
    Optimize using the soft-min objective.
    """
    def objective_and_grad(x):
        val, g = soft_min_objective_and_grad(x, target_beta)
        return val, g

    bounds = [(-0.5, 0.5)] * (N * 3)
    
    start_time = time.time()
    try:
        res = minimize(
            objective_and_grad,
            init_centers.flatten(),
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'disp': False, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        
        if time.time() > start_time + time_limit:
             pass
             
        return res.x.reshape(-1, 3)
    except Exception:
        return init_centers

def _polish_hard_min(pos, max_iter=500, ftol=1e-12, gtol=1e-10):
    """
    High-precision polish pass using the hard-min objective (maximize min dist).
    """
    def hard_min_obj_and_grad(flat_pos):
        pos = flat_pos.reshape(N, 3)
        
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        min_pair_val = np.min(dists)
        min_pair_idx = np.unravel_index(np.argmin(dists), dists.shape)
        i, j = min_pair_idx
        
        d_walls = HALF - np.abs(pos)
        min_wall_val = np.min(d_walls)
        min_wall_idx = np.unravel_index(np.argmin(d_walls), d_walls.shape)
        wi, wk = min_wall_idx
        
        r_pair = min_pair_val / 2.0
        r_wall = min_wall_val
        
        grad = np.zeros_like(pos)
        
        if r_pair <= r_wall:
            d = min_pair_val
            if d > 1e-12:
                vec = (pos[i] - pos[j]) / d
                grad[i] += 0.5 * vec
                grad[j] += -0.5 * vec
            return -r_pair, -grad.flatten()
        else:
            s = np.sign(pos[wi, wk])
            if s == 0:
                s = 1.0
            grad[wi, wk] = -s
            return -r_wall, -grad.flatten()

    bounds = [(-0.5, 0.5)] * (N * 3)
    try:
        res = minimize(
            hard_min_obj_and_grad,
            pos.flatten(),
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol}
        )
        if res.success or res.fun < 0:
            new_pos = res.x.reshape(N, 3)
            if np.all(np.isfinite(new_pos)):
                return new_pos
    except Exception:
        pass
    return pos

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initial guess: Grid
    # 100 points. 5x4x5 grid.
    def make_grid(nx, ny, nz):
        xs = np.linspace(-0.5 + 0.5/nx, 0.5 - 0.5/nx, nx)
        ys = np.linspace(-0.5 + 0.5/ny, 0.5 - 0.5/ny, ny)
        zs = np.linspace(-0.5 + 0.5/nz, 0.5 - 0.5/nz, nz)
        grid = []
        for z in zs:
            for y in ys:
                for x in xs:
                    grid.append([x, y, z])
        return np.array(grid)
    
    best_pos = make_grid(5, 4, 5)
    best_r = compute_min_dist(best_pos)
    
    rng = np.random.default_rng(42)
    
    # Strategy:
    # 1. Use Soft-Min optimization with increasing beta (sharpness).
    # 2. Start with low beta for global exploration, increase for local exploitation.
    # 3. Periodically polish with hard-min.
    
    # Beta schedule: start low, increase exponentially
    beta = 5.0
    beta_increase = 1.5
    
    attempt_count = 0
    last_improvement_time = start_time
    
    while time.time() < deadline:
        attempt_count += 1
        
        # Check stagnation
        if time.time() - last_improvement_time > 15.0:
             # Reset beta to a lower value to escape local minima?
             # Or just try a random restart with current best pos perturbed
             beta = 5.0
             init_pos = best_pos + rng.normal(0, 0.02, size=(N, 3))
             init_pos = np.clip(init_pos, -0.5, 0.5)
        else:
             # Normal progression: increase beta
             beta *= beta_increase
             # Cap beta to avoid numerical issues
             beta = min(beta, 100.0)
             
             # Use best_pos as init, with very small perturbation if beta is high
             if beta > 50:
                 perturbation = 0.001
             else:
                 perturbation = 0.01
                 
             init_pos = best_pos + rng.normal(0, perturbation, size=(N, 3))
             init_pos = np.clip(init_pos, -0.5, 0.5)
        
        # Run optimization
        time_left = deadline - time.time()
        opt_time = min(10.0, time_left * 0.8)
        
        if opt_time < 1.0:
            break
            
        new_pos = _optimize_soft_min(init_pos, beta, max_iter=100, time_limit=opt_time)
        
        # Check improvement
        achieved_r = compute_min_dist(new_pos)
        
        if achieved_r > best_r:
            best_r = achieved_r
            best_pos = new_pos
            last_improvement_time = time.time()
            
        # If we are close to deadline, do a final polish
        if time.time() > deadline - 20.0:
            break

    # Final Polish
    if time.time() < deadline - 5.0:
        # Try a few hard-min polishes with slightly perturbed starts to escape any flat spots
        for _ in range(3):
            polished_pos = _polish_hard_min(best_pos, max_iter=300)
            final_r = compute_min_dist(polished_pos)
            if final_r > best_r:
                best_r = final_r
                best_pos = polished_pos
            else:
                # Try a tiny perturbation before next polish
                perturbed = best_pos + rng.normal(0, 0.001, size=(N, 3))
                perturbed = np.clip(perturbed, -0.5, 0.5)
                best_pos = perturbed
                
    return best_pos

# EVOLVE-BLOCK-END
