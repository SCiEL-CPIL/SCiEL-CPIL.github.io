import numpy as np
from scipy.optimize import minimize
import time

N = 300

def construct():
    # Deadline guard: 150s limit, keep margin for safety
    start_time = time.time()
    max_time = 140.0
    
    # 1. Initialization: Concentric Rings
    # This structure is often a good starting point for circular packings 
    # because it respects the boundary symmetry and provides a diverse 
    # initial distribution compared to a simple lattice.
    
    # Estimate number of rings
    # Area of unit circle = pi. Area per circle ~ pi * r^2.
    # Approximate r ~ sqrt(pi / N) * packing_efficiency. 
    # For N=300, r ~ 0.053.
    # Let's just generate points in rings.
    
    # Heuristic for number of rings:
    # If we have k rings, the outer ring has ~ 2*pi*r_outer / (2*r) points.
    # Let's try to distribute N points into rings.
    # A common heuristic is to start with 1 in center, then rings of 6, 12, 18...
    # But for N=300, we need more flexibility.
    
    # Let's use a simple area-based allocation to rings.
    # Radius of ring i ~ sqrt(i).
    # Number of points in ring i ~ 2 * pi * r_i / (2 * r_circle) ~ pi * r_i / r_circle.
    
    # Let's just create a grid of potential ring radii and assign points.
    # Or simpler: generate a hex lattice but then "relax" it into rings? No, stick to rings.
    
    # Let's try: 
    # Ring 0: 1 point at center.
    # Ring 1: 6 points.
    # Ring 2: 12 points.
    # ...
    # This grows as 6*k. Sum_{k=1..M} 6k = 3M(M+1) + 1.
    # For N=300, 3M^2 ~ 300 => M ~ 10.
    # 3*10*11 + 1 = 331. So about 10-11 rings.
    
    rings = []
    remaining = N
    k = 0
    # First point
    if remaining > 0:
        rings.append(1)
        remaining -= 1
        k = 1
    
    # Add points to rings
    while remaining > 0:
        # Ideal number for next ring is 6*k
        count = 6 * k
        if remaining < count:
            count = remaining
        rings.append(count)
        remaining -= count
        k += 1
        
    # Total points in rings
    total_points = sum(rings)
    # If we have too many or too few (due to integer steps), adjust the last ring
    # Actually, the loop above handles it by taking 'remaining' if less than 'count'.
    # But if remaining > 0 after k increments, we just continue.
    
    # Calculate radii for these rings
    # The radius of ring k should be such that the points fit.
    # Approximate radius of ring k: r_k ~ k * 2 * r_circle / (2*pi) * something?
    # Let's use the area argument.
    # Area up to ring k is ~ pi * R_k^2.
    # Number of points up to ring k is ~ N_k.
    # N_k / N ~ R_k^2 / 1^2 => R_k ~ sqrt(N_k / N).
    
    radii = []
    cumulative = 0
    for count in rings:
        cumulative += count
        # R_k is the radius of the ring with 'count' points.
        # The "center" of this ring in terms of area index is roughly (cumulative - count/2) / N
        # Let's just use sqrt(cumulative/N) as a target radius.
        r_target = np.sqrt(cumulative / N) * 0.95 # 0.95 to keep inside
        radii.append(r_target)
        
    # Build initial points
    points = []
    for i, (count, r_target) in enumerate(zip(rings, radii)):
        if count == 1 and i == 0:
            points.append([0.0, 0.0])
        else:
            angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
            # Add a small offset to break symmetry slightly? No, L-BFGS will handle it.
            # But perfect symmetry can lead to flat directions.
            # Let's add tiny random noise.
            angles += np.random.uniform(0, 2*np.pi/count, size=count) * 0.1
            for a in angles:
                points.append([r_target * np.cos(a), r_target * np.sin(a)])
                
    x0 = np.array(points)
    
    # Ensure we have exactly N points
    if len(x0) < N:
        # Pad with random points near boundary
        diff = N - len(x0)
        extra = np.random.randn(diff, 2) * 0.1
        extra[:, 0] += 0.9
        x0 = np.vstack([x0, extra])
    elif len(x0) > N:
        x0 = x0[:N]

    # Project initial points inside unit circle
    norms = np.linalg.norm(x0, axis=1)
    out_mask = norms > 1.0
    x0[out_mask] /= norms[out_mask, np.newaxis] * 1.01 # Push slightly in

    # 2. Objective Function: Smooth Min
    # We want to maximize r such that:
    # 1. |xi - xj| >= 2r for all i < j
    # 2. |xi| <= 1 - r for all i
    
    # Let f(x, r) = min( min_{i<j} (|xi - xj|/2 - r), min_i (1 - |xi| - r) )
    # We want to find x, r such that f(x, r) >= 0 and r is max.
    # Alternatively, maximize r subject to constraints.
    # L-BFGS-B minimizes. So we can minimize -r subject to constraints? 
    # No, L-BFGS-B doesn't handle inequality constraints well in the standard API (only bounds).
    # We can use a penalty method or reformulate.
    
    # Reformulation:
    # Let g(x) = min( min_{i<j} |xi - xj|/2, min_i (1 - |xi|) ).
    # We want to maximize g(x).
    # This is equivalent to minimizing -g(x).
    # However, g(x) is non-smooth (min of linear functions).
    # We can use a smooth approximation:
    # smooth_min(a, b) = -1/k * log( exp(-k*a) + exp(-k*b) )
    # Or simply: -1/k * log( sum exp(-k * terms) )
    
    # Let terms be the "violations" or "margins".
    # Margin_ij = |xi - xj|/2
    # Margin_i = 1 - |xi|
    # We want to maximize min(Margins).
    # Let h(x) = -1/k * log( sum_{pairs} exp(-k * Margin_ij) + sum_{i} exp(-k * Margin_i) )
    # Minimizing h(x) is approximately maximizing the minimum margin.
    
    # However, calculating all pairs is O(N^2). N=300 -> 45000 pairs.
    # Exp of 45000 terms might be slow or numerically unstable if k is large.
    # We can use a "soft min" over the top K smallest margins? 
    # No, L-BFGS needs a smooth global objective.
    
    # Alternative: 
    # Define the objective as the sum of "penalties" for being too close, 
    # but we want to maximize the *minimum*.
    
    # Let's try a different formulation suitable for L-BFGS-B:
    # Maximize r.
    # Variables: x (600 vars), r (1 var). Total 601 vars.
    # Objective: -r (to minimize)
    # Constraints:
    # |xi - xj|^2 >= (2r)^2  =>  |xi - xj|^2 - 4r^2 >= 0
    # |xi|^2 <= (1-r)^2      =>  (1-r)^2 - |xi|^2 >= 0
    
    # L-BFGS-B in SciPy does NOT support general inequality constraints.
    # It only supports bounds (lb <= x <= ub).
    # So we must use a penalty method or transform the problem.
    
    # Penalty Method:
    # Minimize P(x, r) = -r + lambda * sum( max(0, 4r^2 - |xi-xj|^2)^2 + max(0, |xi|^2 - (1-r)^2)^2 )
    # This is differentiable almost everywhere (max is not differentiable at 0, but subgradient exists).
    # L-BFGS can handle non-smoothness reasonably well, or we can smooth the max:
    # smooth_max(0, u) = 0.5 * (u + sqrt(u^2 + eps))
    
    def objective(params):
        # params: [x1x, x1y, ..., xNx, xNy, r]
        x = params[:2*N].reshape(N, 2)
        r = params[2*N]
        
        # Time check
        if time.time() - start_time > max_time:
            return np.inf
            
        # Calculate margins
        # Pairwise distances
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dists_sq, np.inf)
        
        # Boundary distances
        norms_sq = np.sum(x**2, axis=1)
        
        # Constraints (violations if negative margin)
        # We want dist >= 2r => dist_sq - 4r^2 >= 0
        # We want norm <= 1-r => (1-r)^2 - norm_sq >= 0
        
        margin_pairs = dists_sq - 4 * r**2
        margin_bound = (1 - r)**2 - norms_sq
        
        # Penalty for violations
        # Smooth max(0, -margin)
        # Let v = -margin. We want max(0, v).
        # smooth_max(0, v) = 0.5 * (v + sqrt(v^2 + eps))
        
        eps = 1e-4
        lambda_p = 100.0 # Penalty weight
        
        # To avoid memory explosion with N^2 exps, we just sum the squares of violations.
        # This is a standard L2 penalty.
        # violation_pairs = np.maximum(0, -margin_pairs)
        # penalty_pairs = np.sum(violation_pairs**2)
        
        # L2 penalty is differentiable.
        # Let's use a smooth L1-like penalty or just L2.
        # L2 is fine.
        
        # To save time, we can subsample pairs if N is large, but N=300 is small enough.
        # 300x300 matrix is 90,000 elements. Fast in NumPy.
        
        v_pairs = -margin_pairs
        v_pairs = np.where(v_pairs > 0, v_pairs, 0)
        penalty_pairs = np.sum(v_pairs**2)
        
        v_bound = -margin_bound
        v_bound = np.where(v_bound > 0, v_bound, 0)
        penalty_bound = np.sum(v_bound**2)
        
        return -r + lambda_p * (penalty_pairs + penalty_bound)
        
    # Initial r guess
    r_init = 0.05
    
    # Initial params
    params0 = np.zeros(2*N + 1)
    params0[:2*N] = x0.flatten()
    params0[2*N] = r_init
    
    # Bounds
    # x_i in [-1, 1], y_i in [-1, 1]
    # r in [0, 1]
    bounds = [(-1.0, 1.0)] * (2*N) + [(0.0, 1.0)]
    
    # Optimization
    best_params = params0.copy()
    best_score = -objective(best_params) # This is -(-r + pen) = r - pen. Not the real score.
    
    # We need to track the *actual* valid score (min margin) for comparison.
    def calc_real_score(x, r):
        # r must be feasible. If penalty is high, r is not feasible.
        # The score is the max r such that constraints are satisfied.
        # In the penalty method, the optimizer tries to increase r while reducing penalty.
        # At the end, we should project r to be feasible.
        # Or simply, the score is min(margins) calculated from x, independent of the parameter r?
        # No, the score is determined by x. The parameter r is just a helper.
        # Score = min( min_{i<j} |xi-xj|/2, min_i (1-|xi|) )
        
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        min_bound = np.min(1.0 - np.linalg.norm(x, axis=1))
        return min(min_pair, min_bound)
        
    current_x = x0
    current_r = r_init
    best_r_real = calc_real_score(current_x, current_r)
    
    # Run L-BFGS-B
    # We will run it in a loop, increasing penalty or just letting it run.
    # Since we don't know the optimal penalty, we can start with a moderate one.
    
    # To make it robust, we can do a few restarts or just one long run.
    # L-BFGS-B is deterministic.
    
    try:
        res = minimize(
            objective, 
            params0, 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 2000, 'maxfun': 20000, 'disp': False}
        )
        
        if time.time() - start_time < max_time:
            # Extract result
            x_opt = res.x[:2*N].reshape(N, 2)
            # Project to ensure inside
            norms = np.linalg.norm(x_opt, axis=1)
            out_mask = norms > 1.0
            if np.any(out_mask):
                x_opt[out_mask] /= norms[out_mask, np.newaxis] * 1.01
                
            r_real = calc_real_score(x_opt, res.x[2*N])
            
            if r_real > best_r_real:
                best_r_real = r_real
                current_x = x_opt
                
    except Exception as e:
        print(f"Optimization error: {e}")
        
    # If time remains, try a local refinement with a different initialization or method
    if time.time() - start_time < max_time - 10:
        # Try a simple gradient ascent on the smooth min for a few steps?
        # Or just return the best found.
        
        # Let's try one more restart with a hex lattice init if the ring init failed or was bad.
        # But let's stick to the result to avoid complexity and bugs.
        
        pass
        
    # Final validation and cleanup
    x_final = current_x
    # Ensure finite
    x_final = np.nan_to_num(x_final)
    
    # Final check: ensure all points are strictly inside
    norms = np.linalg.norm(x_final, axis=1)
    # If any point is on the boundary, it's valid (<= 1). 
    # But to be safe against floating point errors in evaluator:
    # The evaluator computes 1 - |ci|. If |ci| > 1, it's negative.
    # We must ensure |ci| <= 1.
    out_mask = norms > 1.0
    if np.any(out_mask):
        x_final[out_mask] /= norms[out_mask, np.newaxis]
        
    return x_final
