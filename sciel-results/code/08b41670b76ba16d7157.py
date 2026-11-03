"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
import sys
from scipy.optimize import minimize

# Deadline guard: 110s safety margin (limit is 120s)
START_TIME = time.time()
DEADLINE = START_TIME + 110.0

def _compute_radius(centers):
    """Compute the packing radius for a set of centers efficiently."""
    n = len(centers)
    # Compute all pairwise distances
    # diff[i, j] = centers[i] - centers[j]
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    # Set diagonal to inf to ignore self-distance
    np.fill_diagonal(dists, float('inf'))
    min_dist = np.min(dists)
    
    # Minimum distance to boundary
    norms = np.sqrt(np.sum(centers**2, axis=1))
    min_boundary = np.min(1.0 - norms)
    
    return min(min_dist / 2.0, min_boundary)

def _smooth_objective(x, beta=50.0):
    """
    Smooth objective function using soft-min for distances.
    Minimizing this is equivalent to maximizing the minimum distance.
    """
    centers = x.reshape(-1, 2)
    if not np.all(np.isfinite(centers)):
        return 1e10
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dists_sq, float('inf'))
    
    # Soft-min of distances (approximation of min dist)
    # Using log-sum-exp trick for stability, or just np.exp(-beta * dist)
    # min(d) approx -1/beta * log(sum(exp(-beta * d)))
    # To avoid overflow, we can work with squared distances or just use a penalty.
    # Simpler: Penalty method for overlaps and boundary.
    
    # Boundary penalty: ensure centers are inside unit circle
    norms = np.sqrt(np.sum(centers**2, axis=1))
    boundary_viol = np.maximum(0, norms - 1.0)
    boundary_pen = np.sum(boundary_viol**2)
    
    # Overlap penalty: ensure dist > 2*r. But r is variable.
    # We want to maximize r. Let's use the fact that optimal r is min(min_pair_dist/2, min_boundary).
    # We can define an objective that penalizes if dist_ij < 2 * r_est.
    # However, r_est is not in x.
    # Alternative: Maximize a lower bound on r.
    # Let's use a smooth approximation of the minimum distance.
    
    # min_dist approx = -1/beta * log(sum(exp(-beta * dist)))
    # dist = sqrt(dists_sq)
    # To save time, we can approximate min_dist_sq.
    # min_dist_sq approx = -1/beta * log(sum(exp(-beta * dists_sq)))
    # Then r_approx = 0.5 * sqrt(min_dist_sq_approx)
    
    # But we also have boundary.
    # Let's try a different approach: 
    # Objective = - ( min_pair_dist / 2 ) + penalty.
    # Using soft-min on distances:
    
    # Subtract max dist to prevent overflow in exp
    min_d_sq = np.min(dists_sq)
    # We want to maximize the minimum distance.
    # Let's minimize the soft-min of distances.
    # soft_min(d) = -1/beta * log( sum( exp(-beta * d) ) )
    # d = sqrt(dists_sq)
    
    # For N=300, 45k pairs.
    # exp(-beta * d) might underflow if beta is large and d is large.
    # Let's use a simpler penalty: sum of (max(0, 2*r_target - d_ij))^2 + (max(0, r_i - (1-|c_i|)))^2
    # This requires fixing r_target.
    
    # Better: Use the "Max-Min" formulation with a smooth surrogate.
    # f(x) = - (1/beta) * log( sum_{i<j} exp(-beta * d_ij) )  -- this is min distance
    # But we also need boundary.
    # g(x) = - (1/beta) * log( sum_i exp(-beta * (1 - |c_i|)) ) -- this is min boundary dist
    
    # Total radius r = min(f(x), g(x))
    # We want to maximize r.
    # Equivalent to minimizing -min(f, g).
    # -min(A, B) = max(-A, -B).
    # Soft-max(-A, -B) = -1/beta * log( sum( exp(-beta * (-A)), exp(-beta * (-B)) ) )
    #                  = -1/beta * log( exp(beta * A) + exp(beta * B) )
    #                  = -1/beta * log( exp(beta * (-soft_min_dist)) + exp(beta * (-soft_min_bound)) )
    
    # Let's compute soft_min_dist and soft_min_bound.
    
    # To avoid overflow/underflow issues with exp(beta * val), we can shift.
    # But for N=300, let's just try a moderate beta.
    
    # dists = np.sqrt(dists_sq)
    # This is expensive. Let's stick to squared distances if possible?
    # No, radius is linear in distance.
    
    # Let's use a simpler, robust penalty approach which is often faster to converge in Python.
    # We don't know r. Let's guess r is around 0.05.
    # Actually, let's use the `minimize` on the negative of the smooth min.
    
    # Compute distances
    dists = np.sqrt(dists_sq)
    
    # Soft min of pair distances
    # Shift by max to avoid overflow in exp? No, we want min, so we use -dist.
    # min(d) = - max(-d)
    # max(-d) approx -1/beta * log(sum(exp(-beta * d)))
    # So min(d) approx 1/beta * log(sum(exp(-beta * d)))
    
    # For boundary:
    # min(1 - |c|)
    # approx 1/beta * log(sum(exp(-beta * (1 - |c|))))
    
    # r = min(min_d, min_b)
    # We want to maximize r.
    # Let's define J = -r.
    # We can use a smooth approximation of min(A, B):
    # min(A, B) approx -1/beta * log(exp(-beta A) + exp(-beta B))
    # So -min(A, B) approx 1/beta * log(exp(-beta A) + exp(-beta B))
    
    # A = soft_min_dist = 1/beta * log(sum(exp(-beta * dists)))
    # B = soft_min_bound = 1/beta * log(sum(exp(-beta * (1 - norms))))
    
    # This is getting complex. Let's use a simpler penalty method.
    # We know the best radius is ~0.053.
    # Let's optimize for a fixed r, checking if it's feasible? No, that's hard.
    
    # Let's use the direct smooth objective:
    # Objective = - (1/beta) * log( sum_{i<j} exp(-beta * d_ij) + sum_i exp(-beta * (1 - |c_i|)) )
    # This approximates -min(min_pair_dist, min_boundary_dist).
    # Minimizing this maximizes the min distance.
    
    # Combine all terms into one sum for the log-sum-exp
    # Terms: exp(-beta * d_ij) for all pairs, and exp(-beta * (1 - |c_i|)) for all i.
    
    # To avoid overflow in exp(-beta * val) when val is large negative (which it isn't, dists are positive),
    # we are fine. But if dists are small, exp is close to 1.
    
    # Let's compute the log-sum-exp safely.
    # log_sum_exp(x) = max(x) + log(sum(exp(x - max(x))))
    
    # x_vals = [-beta * d_ij for all pairs] + [-beta * (1 - |c_i|) for all i]
    # We want -min(x) essentially? No.
    # min(d) = -1/beta * log(sum(exp(-beta * d)))
    # Let y = -beta * d.
    # min(d) = 1/beta * log(sum(exp(y)))
    
    # We want to maximize min(d). So minimize -min(d) = -1/beta * log(sum(exp(y)))
    
    # Let's compute y_pairs = -beta * dists (upper triangle)
    # y_bound = -beta * (1 - norms)
    
    # All y values.
    # log_sum_exp(y)
    # Objective = -1/beta * log_sum_exp(y)
    
    # This is a single scalar.
    
    # Get upper triangle indices
    i_idx, j_idx = np.triu_indices(n, k=1)
    d_pairs = dists[i_idx, j_idx]
    y_pairs = -beta * d_pairs
    
    b_vals = 1.0 - norms
    y_bound = -beta * b_vals
    
    all_y = np.concatenate([y_pairs, y_bound])
    
    # Log-sum-exp
    max_y = np.max(all_y)
    log_sum_exp = max_y + np.log(np.sum(np.exp(all_y - max_y)))
    
    obj = -1.0 / beta * log_sum_exp
    
    return obj

def construct():
    global START_TIME, DEADLINE
    
    # Generate initial hexagonal lattice packing
    centers = _hex_initial(N)
    x0 = centers.flatten()
    
    best_centers = centers.copy()
    best_radius = _compute_radius(centers)
    
    # Optimization
    # Use L-BFGS-B with a smooth objective
    # We need to be careful with time.
    
    # Try a few restarts or just one long run?
    # One long run with tight time limit.
    
    try:
        # Callback to check time
        def callback(xk):
            nonlocal best_centers, best_radius
            centers_k = xk.reshape(-1, 2)
            if np.all(np.isfinite(centers_k)):
                r_k = _compute_radius(centers_k)
                if r_k > best_radius:
                    best_radius = r_k
                    best_centers = centers_k.copy()
            return time.time() > DEADLINE - 5
        
        res = minimize(
            _smooth_objective, 
            x0, 
            method='L-BFGS-B', 
            options={'maxiter': 1000, 'ftol': 1e-8, 'gtol': 1e-8},
            callback=callback
        )
        
        # Check result
        if res.success or time.time() > DEADLINE - 5:
             centers_res = res.x.reshape(-1, 2)
             if np.all(np.isfinite(centers_res)):
                 r_res = _compute_radius(centers_res)
                 if r_res > best_radius:
                     best_radius = r_res
                     best_centers = centers_res.copy()
    except Exception:
        pass
    
    # Final check
    if time.time() < DEADLINE - 2:
        # Maybe a quick local polish with higher beta?
        # Or just return best found.
        pass
        
    return best_centers

def _hex_initial(n):
    """Generate initial hexagonal lattice packing for n points."""
    points = []
    # Estimate spacing for 300 points in unit circle.
    # Area of circle = pi. Area per point = pi/300 ~ 0.0104.
    # Hex packing density ~ 0.9069.
    # Effective area per point ~ 0.0115.
    # Spacing s ~ sqrt(0.0115 / (sqrt(3)/2)) ~ sqrt(0.0133) ~ 0.115?
    # Wait, area of hexagon with side s/2? No.
    # Distance between centers in hex lattice is s.
    # Area per point in hex lattice is s^2 * sqrt(3)/2.
    # s^2 * 0.866 = 0.0104 / 0.9069 (density) -> s^2 = 0.01146 / 0.866 = 0.0132 -> s = 0.115.
    # Let's try s = 0.11.
    
    spacing = 0.11
    
    # Generate hexagonal lattice points
    k = int(2.0 / spacing) + 5
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if np.hypot(x, y) <= 1.0 - spacing / 2.0:
                points.append((x, y))
    
    points = np.array(points)
    
    # If we have too many, subsample; if too few, add more
    if len(points) > n:
        # Sort by distance from origin and take closest n
        dists = np.sqrt(np.sum(points**2, axis=1))
        idx = np.argsort(dists)[:n]
        points = points[idx]
    elif len(points) < n:
        # Add random points inside the circle
        while len(points) < n:
            theta = np.random.uniform(0, 2 * np.pi)
            r = np.random.uniform(0, 1.0 - spacing / 2.0)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            points = np.vstack([points, [[x, y]]])
    
    return points[:n]
# EVOLVE-BLOCK-END
