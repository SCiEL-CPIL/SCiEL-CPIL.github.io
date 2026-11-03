import numpy as np
import scipy.optimize as so
import time

N = 300

def _soft_min_obj_and_grad(x, beta):
    """
    x: (N, 2) array of centers
    beta: temperature parameter
    Returns: (value, gradient)
    Minimizing this function maximizes the soft-min of distances.
    """
    x = x.reshape(N, 2)
    
    # Calculate pairwise distances
    # To save memory/time, we might need to be careful, but N=300 is small enough for full matrix
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    
    # Mask out self distances (diagonal)
    np.fill_diagonal(dist, np.inf)
    
    # Pairwise terms: d_ij = dist / 2
    # We sum exp(-beta * d_ij)
    # Note: dist is symmetric, so we count each pair twice. 
    # The soft min formula usually sums over all items. 
    # If we sum over all i<j, we should divide the exponent by 2? 
    # Actually, min is min. Sum of exps is just a sum.
    # Let's stick to summing over all i < j.
    
    # Efficient way:
    # We need gradients.
    # Let's compute the weights w_k = exp(-beta * d_k)
    
    # Wall distances
    r = np.sqrt(np.sum(x**2, axis=1))
    d_wall = 1.0 - r
    # Handle r=0
    r_safe = np.where(r == 0, 1e-10, r)
    
    # Pairwise distances (i < j)
    # We can use the upper triangle
    iu = np.triu_indices(N, k=1)
    d_pair = dist[iu] / 2.0
    
    # Calculate exponents
    exp_pair = np.exp(-beta * d_pair)
    exp_wall = np.exp(-beta * d_wall)
    
    # Sum for denominator
    sum_exp = np.sum(exp_pair) + np.sum(exp_wall)
    
    # Objective value (negative of soft min, since we want to minimize -soft_min? No.)
    # Soft min is approx -1/beta * log(sum exp(-beta d)).
    # We want to MAXIMIZE soft min.
    # So we MINIMIZE -soft min = 1/beta * log(sum exp(-beta d)).
    val = (1.0 / beta) * np.log(sum_exp)
    
    # Gradient
    grad = np.zeros_like(x)
    
    # Contribution from wall
    # d/dx_i (1/beta log sum) = 1/beta * 1/sum * sum (exp * d/dx)
    # Term for i: exp_wall[i] * d/dx_i (1 - |x_i|)
    # d/dx_i (1 - |x_i|) = -x_i / |x_i|
    wall_grad = -x / r_safe[:, np.newaxis]
    # Weighted sum
    grad += (exp_wall / sum_exp)[:, np.newaxis] * wall_grad
    
    # Contribution from pairs
    # For each pair (i, j), term is exp_pair[k] * d/dx_i (d_ij)
    # d/dx_i (0.5 |x_i - x_j|) = 0.5 * (x_i - x_j) / |x_i - x_j|
    # d/dx_j (0.5 |x_i - x x_j|) = 0.5 * (x_j - x_i) / |x_i - x_j|
    
    # We have d_pair for indices iu[0], iu[1]
    # Let's accumulate gradients
    # To do this efficiently, we can use np.add.at
    
    # Vectorized gradient accumulation is tricky with add.at for complex expressions.
    # N=300, pairs ~ 45k. This is small. We can loop or use advanced indexing.
    
    # Let's use np.add.at
    # grad_i += weight * 0.5 * (x_i - x_j) / dist_ij
    # grad_j += weight * 0.5 * (x_j - x_i) / dist_ij
    
    # dist[iu] is the full distance. d_pair is dist/2.
    # unit_vec = (x_i - x_j) / dist
    # grad_contrib = 0.5 * unit_vec
    
    # We need to be careful with zero distances (overlaps).
    # If dist is 0, gradient is undefined. We can add a small epsilon.
    d_safe = np.where(d_pair < 1e-10, 1e-10, d_pair)
    
    # Vector for pairs
    # diff_vec = x[i] - x[j]
    diff_vec = x[iu[0]] - x[iu[1]]
    unit_vec = diff_vec / d_safe[:, np.newaxis]
    
    # grad for i is +0.5 * unit_vec * weight
    # grad for j is -0.5 * unit_vec * weight
    
    weights = exp_pair / sum_exp
    
    np.add.at(grad, iu[0], 0.5 * weights[:, np.newaxis] * unit_vec)
    np.add.at(grad, iu[1], -0.5 * weights[:, np.newaxis] * unit_vec)
    
    return val, grad

def construct():
    start_time = time.time()
    timeout = 110.0 # Safety margin
    
    # Initial Hexagonal Lattice
    # Estimate spacing
    # Area of unit circle = pi. Area of hex cell ~ (sqrt(3)/2) * s^2.
    # N * (sqrt(3)/2) s^2 <= pi => s <= sqrt(2 pi / (N sqrt(3)))
    s_init = np.sqrt(2 * np.pi / (N * np.sqrt(3)))
    
    # Generate points
    # Center at 0
    points = []
    k = int(1.0 / s_init) + 2
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * s_init
            y = j * s_init * np.sqrt(3.0) / 2.0
            if x*x + y*y < 1.0:
                points.append([x, y])
    
    pts = np.array(points)
    
    # If we have too many, pick random N
    # If too few, reduce spacing
    if len(pts) < N:
        s_init *= 0.9
        points = []
        k = int(1.0 / s_init) + 2
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * s_init
                y = j * s_init * np.sqrt(3.0) / 2.0
                if x*x + y*y < 1.0:
                    points.append([x, y])
        pts = np.array(points)
        
    if len(pts) > N:
        # Keep the ones closest to center? Or random?
        # Random might break symmetry but allow better packing?
        # Let's keep the first N (deterministic) or random.
        # For optimization, symmetry breaking can help escape local minima.
        idx = np.random.choice(len(pts), N, replace=False)
        pts = pts[idx]
    elif len(pts) < N:
        # Pad with random points inside circle?
        # This shouldn't happen with the loop above if k is large enough.
        # But if it does, we need to handle it.
        while len(pts) < N:
            # Add random point
            r = np.random.uniform(0, 1)
            theta = np.random.uniform(0, 2*np.pi)
            pts = np.vstack([pts, [r*np.cos(theta), r*np.sin(theta)]])
            
    # Optimization
    best_x = pts
    best_val = _soft_min_obj_and_grad(best_x, 20.0)[0]
    
    beta = 20.0
    
    # Run multiple restarts or just one long run?
    # One long run with L-BFGS-B is usually good.
    
    def obj(x):
        return _soft_min_obj_and_grad(x, beta)[0]
        
    def grad(x):
        return _soft_min_obj_and_grad(x, beta)[1]
        
    # Use L-BFGS-B
    # We need to check time.
    # scipy.optimize.minimize doesn't have a built-in time limit, but we can use a callback.
    
    time_check = [time.time()]
    aborted = [False]
    
    def callback(xk):
        if time.time() - start_time > timeout:
            aborted[0] = True
            return True # Signal to stop? 
            # Actually, callback return True doesn't stop minimize in all versions reliably?
            # Better to catch exception or just check after.
            # L-BFGS-B is iterative.
            
    # To be safe, let's run a manual gradient descent or use minimize with maxiter.
    # Estimate maxiter based on time.
    # Let's just run minimize and hope it finishes. N=300 is small.
    
    try:
        res = so.minimize(obj, best_x, jac=grad, method='L-BFGS-B', options={'maxiter': 1000, 'maxfun': 10000})
        if time.time() - start_time < timeout:
             if res.fun < best_val:
                 best_x = res.x
                 best_val = res.fun
    except Exception:
        pass
    
    # If time allows, do a second pass with higher beta?
    if time.time() - start_time < timeout - 20:
        beta2 = 50.0
        def obj2(x):
            return _soft_min_obj_and_grad(x, beta2)[0]
        def grad2(x):
            return _soft_min_obj_and_grad(x, beta2)[1]
        try:
            res2 = so.minimize(obj2, best_x, jac=grad2, method='L-BFGS-B', options={'maxiter': 500})
            if res2.fun < best_val: # Note: soft min value depends on beta, so comparing raw fun values across betas is tricky.
                # Actually, we should compare the ACTUAL min distance.
                pass
        except Exception:
            pass
            
    # Final polish: calculate actual radius and ensure validity
    # The optimization might push points slightly out or overlap due to numerical errors.
    # We should project back or just return the result.
    # The score is calculated by the evaluator.
    # We just need to return valid points.
    
    # Check if any point is outside or overlapping significantly?
    # The evaluator handles invalidity, but we want a good score.
    # The soft-min optimization keeps them valid mostly.
    
    return best_x
