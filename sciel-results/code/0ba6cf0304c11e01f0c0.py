# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

N = 100

def _compute_radius(centers):
    """Compute the largest common radius for given centers in [-0.5, 0.5]^3."""
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Wall distance
    wall = 0.5 - np.max(np.abs(centers), axis=1)
    min_wall = float(np.min(wall))
    if min_wall < 0.0:
        return 0.0
    # Pairwise distances
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = float(np.min(dists)) / 2.0
    return min(min_wall, min_pair)

def _soft_min_obj_and_grad(x, T):
    """
    Compute objective and gradient for maximizing the minimum distance.
    We minimize -soft_min(distances).
    soft_min(y) = -T * log(sum(exp(-y_i / T)))
    obj = -soft_min(y) = T * log(sum(exp(-y_i / T)))
    grad = d obj / d x = - (1/T) * sum(exp(-y_i/T) * grad(y_i)) / sum(exp(-y_i/T))
    Actually, let's stick to standard form:
    We want to maximize r = min(constraints).
    Let f(x) = -min(constraints). We minimize f(x).
    Soft approximation: F(x) = T * log(sum_j exp(-c_j(x) / T)) where c_j are constraints (distances).
    dF/dx = (1/T) * sum_j exp(-c_j/T) * (-dc_j/dx) / sum_j exp(-c_j/T)
    dF/dx = - sum_j w_j * dc_j/dx, where w_j = exp(-c_j/T) / sum_k exp(-c_k/T)
    """
    centers = x.reshape(N, 3)
    # Clip to ensure feasibility during optimization? 
    # L-BFGS-B handles bounds, but numerical stability might need clipping inside gradient if close to bounds.
    # However, for gradient calculation, we should use the raw x if possible, or handle bounds carefully.
    # Let's rely on L-BFGS-B bounds and just compute gradient on raw x.
    
    # 1. Wall constraints: c_i = 0.5 - max(|x_i|, |y_i|, |z_i|)
    # This is non-smooth at axes. 
    # Alternative: Use distance to boundary directly?
    # c_i = min(0.5 - x_i, 0.5 + x_i, 0.5 - y_i, ...) 
    # Actually, distance to cube boundary for a point (x,y,z) in [-0.5, 0.5]^3 is:
    # d_i = min(0.5 - x, 0.5 + x, 0.5 - y, 0.5 + y, 0.5 - z, 0.5 + z)
    # This is equivalent to 0.5 - max(|x|, |y|, |z|).
    # The gradient of max(|x|, |y|, |z|) is the unit vector in the direction of the largest component.
    # If multiple components are equal and largest, gradient is average of their signs? No, subgradient.
    # For soft min, we can use a smooth approximation of the distance to boundary.
    # Or just use the pairwise distances and wall distances as separate terms.
    
    # Let's use the 6 planes explicitly for smoothness.
    # c_{i,1} = 0.5 - x_i
    # c_{i,2} = 0.5 + x_i
    # c_{i,3} = 0.5 - y_i
    # c_{i,4} = 0.5 + y_i
    # c_{i,5} = 0.5 - z_i
    # c_{i,6} = 0.5 + z_i
    
    # Pairwise distances:
    # c_{ij} = ||x_i - x_j|| / 2
    
    # Collect all constraint values c_k
    # Number of wall constraints: N * 6
    # Number of pair constraints: N*(N-1)/2
    
    # For efficiency, we can compute gradients of c_k w.r.t x directly.
    
    # Wall constraints gradients are constant.
    # Pairwise constraints gradients are (x_i - x_j) / ||x_i - x_j|| * 0.5
    
    # Let's build the vector of constraint values C.
    # This might be memory heavy for N=100? 
    # Pairs: ~5000. Walls: 600. Total ~5600. Vector size is fine.
    
    centers = x.reshape(N, 3)
    
    # Wall constraints
    # c = 0.5 - |coord| is non-smooth. 
    # Let's use the 6 planes.
    # c_{i,k} = 0.5 - s_{i,k} * coord_{i,k} where s is sign.
    # Actually, distance to plane x=0.5 is 0.5 - x. Distance to x=-0.5 is 0.5 + x.
    # The minimum of these 6 is the distance to boundary.
    
    # We will accumulate log-sum-exp terms.
    # log_sum_exp(-C/T)
    
    # To avoid overflow, we can compute weights directly.
    # w_k = exp(-C_k / T) / Z
    # obj = T * log(Z)
    # grad = - sum w_k * grad C_k
    
    # Let's compute C and grad C.
    
    # 1. Walls
    # For each sphere i, and each coordinate dim d (0,1,2):
    # c_plus = 0.5 - coord
    # c_minus = 0.5 + coord
    # grad c_plus = -1 * e_d
    # grad c_minus = +1 * e_d
    
    # 2. Pairs
    # For each pair (i, j):
    # diff = centers[i] - centers[j]
    # dist = norm(diff)
    # c = dist / 2
    # grad c = 0.5 * diff / dist * (delta_{ij} - delta_{ji})? 
    # d(c)/d(x_i) = 0.5 * (diff / dist)
    # d(c)/d(x_j) = 0.5 * (-diff / dist)
    
    # We need to sum w_k * grad C_k.
    
    # Let's do this in a vectorized way if possible, or loop if necessary for clarity/speed.
    # N=100 is small enough that a loop over pairs might be okay if optimized, but vectorization is better.
    
    # Let's create a list of constraint values and their gradients?
    # No, let's accumulate the gradient vector directly.
    
    grad = np.zeros_like(x)
    
    # --- Walls ---
    # For each sphere i:
    # c1 = 0.5 - x_i, g1 = -e_x
    # c2 = 0.5 + x_i, g2 = +e_x
    # ...
    
    # We can compute exp(-c/T) for all these.
    # There are 6N wall constraints.
    
    # Let's compute the weights for walls first.
    # coords: (N, 3)
    # c_walls: (N, 6)
    # c_walls[:, 0] = 0.5 - centers[:, 0]
    # c_walls[:, 1] = 0.5 + centers[:, 0]
    # c_walls[:, 2] = 0.5 - centers[:, 1]
    # c_walls[:, 3] = 0.5 + centers[:, 1]
    # c_walls[:, 4] = 0.5 - centers[:, 2]
    # c_walls[:, 5] = 0.5 + centers[:, 2]
    
    c_walls = np.empty((N, 6))
    c_walls[:, 0] = 0.5 - centers[:, 0]
    c_walls[:, 1] = 0.5 + centers[:, 0]
    c_walls[:, 2] = 0.5 - centers[:, 1]
    c_walls[:, 3] = 0.5 + centers[:, 1]
    c_walls[:, 4] = 0.5 - centers[:, 2]
    c_walls[:, 5] = 0.5 + centers[:, 2]
    
    # Flatten for log-sum-exp calculation
    c_flat = c_walls.flatten()
    
    # --- Pairs ---
    # This is harder to vectorize into a single flat array with corresponding gradients easily without huge memory.
    # Let's compute the sum of exp terms for pairs and walls separately?
    # No, log(sum(a+b)) != log(a)+log(b).
    # We need the global sum Z.
    
    # Strategy:
    # 1. Compute all C values.
    # 2. Compute max_C for stability.
    # 3. Compute exp(-(C - max_C)/T).
    # 4. Sum them to get Z.
    # 5. Compute weights w = exp(...) / Z.
    # 6. Accumulate gradient.
    
    # Memory for C:
    # Walls: 600
    # Pairs: 4950
    # Total: 5550 floats. This is tiny.
    
    C_vals = np.empty(6*N + N*(N-1)//2)
    
    # Fill walls
    C_vals[:6*N] = c_flat
    
    # Fill pairs
    # We need to be careful with indices.
    # Let's just loop over pairs to fill C_vals and accumulate gradient contributions?
    # Or fill C_vals and then loop again for gradient?
    # Looping over 5000 pairs in Python is ~ms. Acceptable.
    
    pair_idx = 6*N
    for i in range(N):
        for j in range(i+1, N):
            diff = centers[i] - centers[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-10:
                dist = 1e-10
            c = 0.5 * dist
            C_vals[pair_idx] = c
            pair_idx += 1
            
    # Compute max for stability
    c_max = np.max(C_vals)
    exp_vals = np.exp(-(C_vals - c_max) / T)
    Z = np.sum(exp_vals)
    log_Z = np.log(Z)
    obj = T * (log_Z + (c_max - c_max)/T) # Wait.
    # soft_min = -T log sum exp(-C/T)
    # obj = -soft_min = T log sum exp(-C/T)
    # sum exp(-C/T) = exp(c_max/T) * sum exp(-(C-c_max)/T)
    # log(sum ...) = c_max/T + log(sum exp(-(C-c_max)/T))
    # obj = T * (c_max/T + log_Z) = c_max + T * log_Z
    
    obj = c_max + T * log_Z
    
    # Weights w_k = exp(-(C_k - c_max)/T) / Z
    w = exp_vals / Z
    
    # Gradient accumulation
    # grad = - sum w_k * grad C_k
    
    # 1. Walls
    # w_walls = w[:6*N].reshape(N, 6)
    # For each sphere i:
    # w[0] for 0.5-x -> grad -e_x. Contrib: -w[0] * (-e_x) = w[0] e_x
    # w[1] for 0.5+x -> grad +e_x. Contrib: -w[1] * (+e_x) = -w[1] e_x
    # Net x grad: w[0] - w[1]
    # Net y grad: w[2] - w[3]
    # Net z grad: w[4] - w[5]
    
    w_walls = w[:6*N].reshape(N, 6)
    grad[:, 0] += w_walls[:, 0] - w_walls[:, 1]
    grad[:, 1] += w_walls[:, 2] - w_walls[:, 3]
    grad[:, 2] += w_walls[:, 4] - w_walls[:, 5]
    
    # 2. Pairs
    # w_pairs = w[6*N:]
    # We need to iterate again to apply weights to gradients.
    # This is O(N^2) again.
    
    w_pairs = w[6*N:]
    idx = 0
    for i in range(N):
        for j in range(i+1, N):
            wi = w_pairs[idx]
            idx += 1
            if wi < 1e-12: # Optimization: skip if weight is negligible
                continue
            diff = centers[i] - centers[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-10:
                dist = 1e-10
            # grad C_{ij} w.r.t x_i is 0.5 * diff/dist
            # grad C_{ij} w.r.t x_j is -0.5 * diff/dist
            # d obj / d x = - sum w * grad C
            # d obj / d x_i += - wi * (0.5 * diff/dist)
            # d obj / d x_j += - wi * (-0.5 * diff/dist) = + wi * (0.5 * diff/dist)
            
            factor = -0.5 * wi / dist
            grad[i] += factor * diff
            grad[j] -= factor * diff # Wait. 
            # grad obj = - w * grad C
            # grad C_i = 0.5 * diff/dist
            # grad C_j = -0.5 * diff/dist
            # d obj / d x_i = - w * (0.5 * diff/dist)
            # d obj / d x_j = - w * (-0.5 * diff/dist) = + w * (0.5 * diff/dist)
            
            # My factor was -0.5 * wi / dist.
            # grad[i] += factor * diff  => -0.5 * wi / dist * diff. Correct.
            # grad[j] should be +0.5 * wi / dist * diff.
            # So grad[j] -= factor * diff is correct because factor is negative.
            
    return obj, grad

def _initial_jittered_grid():
    nx, ny, nz = 5, 5, 4
    # Spread points in [-0.4, 0.4] to leave room for radius
    xs = np.linspace(-0.4, 0.4, nx)
    ys = np.linspace(-0.4, 0.4, ny)
    zs = np.linspace(-0.4, 0.4, nz)
    centers = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    centers = centers[:N]
    # Add small jitter
    rng = np.random.default_rng(42)
    centers += rng.normal(0, 0.01, centers.shape)
    return centers

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_centers = _initial_jittered_grid()
    best_score = _compute_radius(best_centers)
    
    # Optimization loop with annealing
    # Start with large T, reduce T
    temperatures = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    
    current_centers = best_centers.copy()
    
    for T in temperatures:
        if time.time() > deadline - 5.0:
            break
            
        x0 = current_centers.flatten()
        bounds = [(-0.5, 0.5)] * (N * 3)
        
        try:
            # Use L-BFGS-B with analytic gradient
            res = minimize(
                _soft_min_obj_and_grad, 
                x0, 
                method='L-BFGS-B', 
                jac=True, # We return (f, g)
                bounds=bounds,
                args=(T,),
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res.success or res.fun < _soft_min_obj_and_grad(x0, T)[0]:
                current_centers = res.x.reshape(N, 3)
                score = _compute_radius(current_centers)
                if score > best_score:
                    best_score = score
                    best_centers = current_centers.copy()
        except Exception:
            pass
            
    # Final polish with very small T and more iterations if time permits
    if time.time() < deadline - 10.0:
        T = 0.0005
        x0 = best_centers.flatten()
        bounds = [(-0.5, 0.5)] * (N * 3)
        try:
            res = minimize(
                _soft_min_obj_and_grad, 
                x0, 
                method='L-BFGS-B', 
                jac=True,
                bounds=bounds,
                args=(T,),
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if res.success:
                current_centers = res.x.reshape(N, 3)
                score = _compute_radius(current_centers)
                if score > best_score:
                    best_score = score
                    best_centers = current_centers.copy()
        except Exception:
            pass
            
    # Ensure validity
    best_centers = np.clip(best_centers, -0.4999, 0.4999)
    return best_centers
# EVOLVE-BLOCK-END
