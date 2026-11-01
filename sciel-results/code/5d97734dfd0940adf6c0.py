"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import itertools

N = 50
DEADLINE = 110.0  # Safety margin for 120s limit

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _get_initial_hex_lattice(n):
    """Generate a hexagonal lattice and pick n points inside unit circle."""
    # Estimate spacing
    # Area of unit circle is pi. Area per hex point is approx 2 * r^2 * sqrt(3)/2 = r^2 * sqrt(3)
    # We want n points, so density n/pi.
    # r^2 * sqrt(3) * n / pi ~ 1 => r ~ sqrt(pi / (n * sqrt(3)))
    # Spacing s = 2r
    r_est = np.sqrt(np.pi / (n * np.sqrt(3)))
    s = 2 * r_est
    
    points = []
    # Generate a grid large enough
    max_range = 2.0 / s + 2
    for i in range(-int(max_range), int(max_range) + 1):
        for j in range(-int(max_range), int(max_range) + 1):
            x = i * s + (j % 2) * s / 2.0
            y = j * s * np.sqrt(3) / 2.0
            if np.hypot(x, y) <= 1.0:
                points.append([x, y])
                
    points = np.array(points)
    if len(points) < n:
        # If we don't have enough, we just return what we have or pad, 
        # but for N=50 and this estimate, we should have plenty.
        # If not, we might need to adjust s.
        # For robustness, let's just take the first n if available, else resize.
        pass
        
    # Sort by distance to origin to pick the most central ones
    dists = np.linalg.norm(points, axis=1)
    indices = np.argsort(dists)
    return points[indices[:n]]

def _soft_min(values, beta=50.0):
    """Compute the soft minimum of an array of values."""
    # To avoid overflow, subtract max
    v = np.array(values)
    m = np.min(v)
    # If beta is large, this approximates min(v)
    # We want to maximize min(v), so we minimize -soft_min
    # soft_min = -1/beta * log(sum(exp(-beta * v)))
    # This is numerically stable if we do:
    # exp(-beta * (v - m))
    exp_terms = np.exp(-beta * (v - m))
    return m - (1.0 / beta) * np.log(np.sum(exp_terms))

def _objective_and_gradient(params, n, beta):
    """
    Objective to minimize: -soft_min(constraints)
    params: (n, 2) array of centers
    constraints: 
      1. pairwise distances / 2
      2. 1 - norm(center)
    """
    centers = params.reshape(n, 2)
    
    # Calculate pairwise distances
    # dist[i, j] = ||c_i - c_j||
    # We only need i < j
    dists = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            dists[i, j] = d
            dists[j, i] = d
            
    # Constraints values
    # 1. d_ij / 2
    vals = []
    for i in range(n):
        for j in range(i+1, n):
            vals.append(dists[i, j] / 2.0)
    # 2. 1 - ||c_i||
    for i in range(n):
        vals.append(1.0 - np.linalg.norm(centers[i]))
        
    vals = np.array(vals)
    
    # Soft min
    obj = -_soft_min(vals, beta)
    
    # Gradient calculation
    # This is expensive. For N=50, 1225 pairs + 50 bounds = 1275 constraints.
    # Gradient of soft min:
    # d/dx (-1/beta log(sum(exp(-beta v))))
    # = -1/beta * (1/S) * sum( exp(-beta v_k) * (-beta * dv_k/dx) )
    # = (1/S) * sum( exp(-beta v_k) * dv_k/dx )
    # where S = sum(exp(-beta v_k))
    
    # To save time, we might skip analytic gradient and use numerical?
    # BFGS can use numerical gradients, but it's slow.
    # Let's try to compute analytic gradient.
    
    exp_vals = np.exp(-beta * (vals - np.min(vals)))
    S = np.sum(exp_vals)
    weights = exp_vals / S  # Weights for each constraint
    
    grad = np.zeros_like(centers)
    
    # Index mapping for vals
    # vals[0..1224] are pairs
    # vals[1225..1274] are bounds
    
    pair_idx = 0
    for i in range(n):
        for j in range(i+1, n):
            w = weights[pair_idx]
            # v = dist(i,j)/2
            # dv/dc_i = (c_i - c_j) / dist(i,j)  (derivative of dist/2 w.r.t c_i)
            # dist = ||c_i - c_j||
            # d(dist)/d(c_i) = (c_i - c_j) / dist
            # d(dist/2)/d(c_i) = (c_i - c_j) / (2 * dist)
            d = dists[i, j]
            if d > 1e-8:
                vec = (centers[i] - centers[j]) / d
                grad[i] += w * vec / 2.0
                grad[j] -= w * vec / 2.0
            pair_idx += 1
            
    bound_idx = 0
    for i in range(n):
        w = weights[n*(n-1)//2 + bound_idx]
        # v = 1 - ||c_i||
        # dv/dc_i = - c_i / ||c_i||
        norm_i = np.linalg.norm(centers[i])
        if norm_i > 1e-8:
            grad[i] -= w * centers[i] / norm_i
        bound_idx += 1
        
    return obj, -grad  # We want to minimize -soft_min, so we return -obj, -grad? 
    # Wait. 
    # maximize soft_min => minimize -soft_min.
    # f = -soft_min.
    # grad_f = -grad_soft_min.
    # My calculation above was for grad of soft_min.
    # So return obj, -grad.

def construct():
    start_time = time.time()
    best_score = 0.0
    best_centers = None
    
    # Initial guess
    init_centers = _get_initial_hex_lattice(N)
    
    # Run multiple optimizations with different noise levels and beta
    # Beta controls the sharpness of the soft-min. Higher beta -> closer to true min.
    # Start with lower beta for exploration, higher for exploitation.
    
    alphas = [0.05, 0.1, 0.2]
    betas = [10.0, 20.0, 50.0, 100.0]
    
    rng = np.random.default_rng(42)
    
    for alpha in alphas:
        if time.time() - start_time > DEADLINE:
            break
        for beta in betas:
            if time.time() - start_time > DEADLINE:
                break
            
            # Add noise
            noisy_centers = init_centers + rng.normal(0, alpha, init_centers.shape)
            
            # Ensure inside circle roughly? The optimizer will handle it, 
            # but starting inside is better.
            # Project to circle if outside?
            norms = np.linalg.norm(noisy_centers, axis=1)
            mask = norms > 0.9
            noisy_centers[mask] *= 0.9 / norms[mask]
            
            # Optimize
            try:
                res = minimize(
                    _objective_and_gradient, 
                    noisy_centers.flatten(), 
                    args=(N, beta),
                    method='BFGS',
                    jac=True,
                    options={'maxiter': 500, 'gtol': 1e-5}
                )
            except Exception:
                continue
                
            centers = res.x.reshape(N, 2)
            
            # Calculate actual score
            # r = min( min_{i<j} d_ij/2, min_i (1-d_i) )
            min_pair = np.inf
            for i in range(N):
                for j in range(i+1, N):
                    d = np.linalg.norm(centers[i] - centers[j])
                    if d < min_pair:
                        min_pair = d
            min_bound = np.inf
            for i in range(N):
                d = 1.0 - np.linalg.norm(centers[i])
                if d < min_bound:
                    min_bound = d
                    
            r = min(min_pair / 2.0, min_bound)
            
            # Check validity
            if np.any(~np.isfinite(centers)):
                continue
            if r > best_score:
                best_score = r
                best_centers = centers.copy()
                
            # If we got a good result, try to polish with higher beta
            if r > 0.125:
                # Polish step
                for beta_polish in [100.0, 200.0]:
                    if time.time() - start_time > DEADLINE:
                        break
                    try:
                        res_polish = minimize(
                            _objective_and_gradient, 
                            centers.flatten(), 
                            args=(N, beta_polish),
                            method='BFGS',
                            jac=True,
                            options={'maxiter': 200, 'gtol': 1e-6}
                        )
                        centers_polish = res_polish.x.reshape(N, 2)
                        
                        min_pair_p = np.inf
                        for i in range(N):
                            for j in range(i+1, N):
                                d = np.linalg.norm(centers_polish[i] - centers_polish[j])
                                if d < min_pair_p:
                                    min_pair_p = d
                        min_bound_p = np.inf
                        for i in range(N):
                            d = 1.0 - np.linalg.norm(centers_polish[i])
                            if d < min_bound_p:
                                min_bound_p = d
                                
                        r_p = min(min_pair_p / 2.0, min_bound_p)
                        if r_p > best_score:
                            best_score = r_p
                            best_centers = centers_polish.copy()
                    except Exception:
                        pass

    if best_centers is None:
        # Fallback to initial
        return init_centers
        
    return best_centers
# EVOLVE-BLOCK-END
