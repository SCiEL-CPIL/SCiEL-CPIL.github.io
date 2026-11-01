"""Packing 200 equal circles in a unit square: L-BFGS-B optimization from diverse starts."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 200
DEADLINE = 145.0  # seconds; leave margin before 150s kill

def _score(pts):
    """Return the common radius r for a valid configuration, or 0.0 if invalid."""
    if pts.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    # Distance to boundary
    ax = np.abs(pts[:, 0])
    ay = np.abs(pts[:, 1])
    boundary = np.minimum(0.5 - ax, 0.5 - ay)
    min_boundary = np.min(boundary)
    if min_boundary < 0:
        return 0.0
    # Pairwise distances
    dist = cdist(pts, pts)
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)
    r = min(min_boundary, min_dist / 2.0)
    return r if r > 0 else 0.0

def _hex_init():
    """Generate a hexagonal lattice initialization with more spacing variants."""
    best_pts = None
    best_r = 0
    # Try a wider range of spacings
    s_est = 1.0 / np.sqrt(N) * 1.2
    for s in np.linspace(s_est * 0.85, s_est * 1.15, 20):
        pts_list = []
        row = 0
        while True:
            y = -0.5 + s/2 + row * s * np.sqrt(3) / 2
            if y > 0.5 - s/2:
                break
            offset = (row % 2) * s / 2
            x = -0.5 + s/2 + offset
            while x <= 0.5 - s/2:
                pts_list.append((x, y))
                x += s
            row += 1
        if len(pts_list) < N:
            continue
        pts = np.array(pts_list[:N])
        r = _score(pts)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()
    if best_pts is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        pts = np.array([(x, y) for y in ticks for x in ticks])
        best_pts = pts[:N].copy()
    return best_pts

def _grid_init():
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    pts = np.array([(x, y) for y in ticks for x in ticks])
    return pts[:N].copy()

def _random_init(seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.45, 0.45, size=(N, 2))

def _soft_min_neg(x, alpha):
    """
    Objective function for L-BFGS-B.
    x: flat array of size 2N.
    alpha: sharpness of the soft-min.
    Returns: -soft_min(distances)
    """
    pts = x.reshape(N, 2)
    
    # Distances to boundary
    d_x = 0.5 - np.abs(pts[:, 0])
    d_y = 0.5 - np.abs(pts[:, 1])
    boundary_dists = np.minimum(d_x, d_y).flatten()
    
    # Pairwise distances
    dist = cdist(pts, pts)
    np.fill_diagonal(dist, np.inf)
    pairwise_dists = dist[~np.isinf(dist)]
    
    # Combine all distances
    all_dists = np.concatenate([boundary_dists, pairwise_dists])
    
    # Soft min: -1/alpha * log(sum(exp(-alpha * d)))
    val = - (1.0 / alpha) * np.log(np.sum(np.exp(-alpha * all_dists)))
    
    # We want to maximize min distance, so minimize -min distance.
    return -val

def _exact_min_neg(x):
    """
    Exact negative minimum distance objective for final polish.
    x: flat array of size 2N.
    Returns: -min(all distances)
    """
    pts = x.reshape(N, 2)
    
    # Distances to boundary
    d_x = 0.5 - np.abs(pts[:, 0])
    d_y = 0.5 - np.abs(pts[:, 1])
    boundary_dists = np.minimum(d_x, d_y)
    
    # Pairwise distances
    dist = cdist(pts, pts)
    np.fill_diagonal(dist, np.inf)
    pairwise_dists = dist[~np.isinf(dist)]
    
    # Find the minimum
    min_boundary = np.min(boundary_dists)
    min_pairwise = np.min(pairwise_dists)
    
    # The radius is min(min_boundary, min_pairwise/2)
    # But for the objective, we want to maximize the minimum "clearance"
    # which is min(min_boundary, min_pairwise/2)
    # Actually, let's work with distances directly.
    # The constraint is that all pairwise distances >= 2r and all boundary distances >= r.
    # So the radius r = min(min_boundary, min_pairwise/2).
    # We want to maximize r, so minimize -r.
    r = min(min_boundary, min_pairwise / 2.0)
    return -r

def _bundle_method(pts, time_budget, max_bundle=50):
    """
    Bundle method for the non-smooth max-min packing problem.
    
    The objective f(x) = min_i g_i(x) where g_i are the clearance functions
    (boundary distances and half pairwise distances). Each g_i is concave
    (or linear), so f is concave. We want to maximize f.
    
    The bundle method maintains a set of subgradients (from past evaluations)
    and solves a quadratic subproblem to find a new trial point.
    
    For max-min of concave functions, we can use the following formulation:
    - At each iteration, we have a bundle of (x_k, g_k, subgrad_k) pairs
    - We solve: min_x 0.5*||x - x_k||^2 - min_i (g_i(x_k) + subgrad_i^T (x - x_k))
    - The subproblem is a QP with linear constraints
    """
    t0 = time.time()
    x = pts.flatten().copy()
    x_best = x.copy()
    
    # Compute current radius and identify active constraints
    def _compute_subgradient(x_flat):
        """
        Compute the radius and a subgradient of the max-min objective at x_flat.
        Returns (radius, subgradient_array_of_size_2N).
        
        The objective is r(x) = min( min_i boundary_dist_i(x_i), min_{i<j} dist(i,j)/2 )
        
        For boundary: if circle i is closest to edge k, subgrad is -e_{ik} (unit vector toward edge)
        For pairwise: if pair (i,j) is closest, subgrad for i is (x_i - x_j)/dist, for j is (x_j - x_i)/dist
        """
        pts = x_flat.reshape(N, 2)
        
        # Boundary distances
        d_x = 0.5 - np.abs(pts[:, 0])
        d_y = 0.5 - np.abs(pts[:, 1])
        boundary_dists = np.minimum(d_x, d_y)
        
        # Pairwise distances
        dist = cdist(pts, pts)
        np.fill_diagonal(dist, np.inf)
        min_pair_idx = np.unravel_index(np.argmin(dist), dist.shape)
        min_pair_dist = dist[min_pair_idx]
        
        # Determine which is the minimum
        min_boundary_idx = np.argmin(boundary_dists)
        min_boundary_val = boundary_dists[min_boundary_idx]
        
        r = min(min_boundary_val, min_pair_dist / 2.0)
        
        # Compute subgradient
        subgrad = np.zeros(2 * N)
        
        if min_boundary_val <= min_pair_dist / 2.0:
            # Boundary constraint is active
            i = min_boundary_idx
            # Determine which boundary is closest
            if d_x[i] <= d_y[i]:
                # Closest to x-boundary
                if pts[i, 0] >= 0:
                    subgrad[2*i] = -1.0  # push left (toward center)
                else:
                    subgrad[2*i] = 1.0   # push right (toward center)
            else:
                # Closest to y-boundary
                if pts[i, 1] >= 0:
                    subgrad[2*i+1] = -1.0  # push down
                else:
                    subgrad[2*i+1] = 1.0   # push up
        else:
            # Pairwise constraint is active
            i, j = min_pair_idx
            dx = pts[i, 0] - pts[j, 0]
            dy = pts[i, 1] - pts[j, 1]
            d = np.sqrt(dx*dx + dy*dy)
            if d > 1e-12:
                # Subgradient for i: direction from j to i (push i away from j)
                subgrad[2*i] = dx / d
                subgrad[2*i+1] = dy / d
                # Subgradient for j: direction from i to j (push j away from i)
                subgrad[2*j] = -dx / d
                subgrad[2*j+1] = -dy / d
        
        return r, subgrad
    
    # Initial evaluation
    r_current, sg_current = _compute_subgradient(x)
    
    # Bundle: list of (x_k, r_k, subgrad_k)
    bundle = [(x.copy(), r_current, sg_current.copy())]
    
    # Parameters
    step_size = 0.01
    min_step = 1e-6
    max_iter = 200
    no_improve_count = 0
    max_no_improve = 10
    
    for iteration in range(max_iter):
        if time.time() - t0 > time_budget:
            break
        
        # Build the bundle model
        # We want to maximize: min over bundle of (r_k + sg_k^T (x - x_k))
        # Equivalent to: find x that maximizes t subject to r_k + sg_k^T(x - x_k) >= t for all k
        # Plus regularization: 0.5 * ||x - x_ref||^2 penalty
        
        # Use the reference point x (current point)
        x_ref = x.copy()
        
        # Solve the subproblem using scipy
        # Variables: x (2N) and t (1)
        # Maximize t - mu * ||x - x_ref||^2
        # Subject to: sg_k^T x - t >= r_k - sg_k^T x_k for all k in bundle
        #             -0.5 <= x_i <= 0.5 for all i
        
        from scipy.optimize import linprog
        
        n_vars = 2 * N + 1  # 2N coordinates + t
        n_constraints = len(bundle)
        
        # Objective: minimize -t + mu * ||x - x_ref||^2
        # Linear part: minimize -t (the quadratic part we handle via regularization)
        # Actually, let's use a simpler approach: solve the LP relaxation
        
        # For the bundle method, we solve:
        # min_x 0.5*||x - x_ref||^2 - min_k(r_k + sg_k^T(x - x_k))
        # This is equivalent to:
        # min_{x,t} 0.5*||x - x_ref||^2 + t
        # s.t. t <= r_k + sg_k^T(x - x_k) for all k
        #      -0.5 <= x_i <= 0.5
        
        # Let's use scipy.optimize.minimize with SLSQP or trust-constr
        # But that's expensive. Instead, let's use a simpler proximal bundle step.
        
        # Simple bundle step: average the subgradients with weights
        # based on how "active" each constraint is
        
        # Compute the model values at x_ref
        model_vals = np.array([r_k + np.dot(sg_k, x_ref - x_k) for x_k, r_k, sg_k in bundle])
        
        # Find the minimum model value (this approximates f(x_ref))
        min_model_idx = np.argmin(model_vals)
        min_model_val = model_vals[min_model_idx]
        
        # The direction is the negative of the subgradient of the active constraint
        # But we want to maximize, so we move in the direction of the subgradient
        active_sg = bundle[min_model_idx][2]
        
        # Normalize the step
        sg_norm = np.linalg.norm(active_sg)
        if sg_norm > 1e-12:
            direction = active_sg / sg_norm
        else:
            direction = np.zeros(2 * N)
        
        # Try a step in the direction of the subgradient
        # With some damping and line search
        trial_x = x + step_size * direction
        trial_x = np.clip(trial_x, -0.5, 0.5)
        
        # Evaluate the trial point
        r_trial, sg_trial = _compute_subgradient(trial_x)
        
        if r_trial > r_current + 1e-10:
            # Accept the step
            x = trial_x
            r_current = r_trial
            no_improve_count = 0
            step_size = min(step_size * 1.2, 0.05)
            
            # Update best
            if r_trial > _score(x_best.reshape(N, 2)):
                x_best = x.copy()
            
            # Add to bundle
            bundle.append((x.copy(), r_current, sg_trial.copy()))
            if len(bundle) > max_bundle:
                # Remove the least useful constraint (one with largest violation)
                violations = np.array([r_k + np.dot(sg_k, x - x_k) - r_current for x_k, r_k, sg_k in bundle])
                # Keep the ones with smallest violations (most active)
                keep_idx = np.argsort(violations)[:max_bundle - 1]
                bundle = [bundle[i] for i in keep_idx]
        else:
            # Reject the step
            no_improve_count += 1
            step_size *= 0.7
            
            if no_improve_count >= max_no_improve:
                # Try a different strategy: random perturbation
                rng = np.random.default_rng(iteration)
                perturbation = rng.normal(0, step_size * 0.5, size=2*N)
                trial_x2 = x + perturbation
                trial_x2 = np.clip(trial_x2, -0.5, 0.5)
                r_trial2, sg_trial2 = _compute_subgradient(trial_x2)
                
                if r_trial2 > r_current:
                    x = trial_x2
                    r_current = r_trial2
                    no_improve_count = 0
                    step_size = min(step_size * 1.5, 0.05)
                    
                    if r_trial2 > _score(x_best.reshape(N, 2)):
                        x_best = x.copy()
                    
                    bundle.append((x.copy(), r_current, sg_trial2.copy()))
                    if len(bundle) > max_bundle:
                        violations = np.array([r_k + np.dot(sg_k, x - x_k) - r_current for x_k, r_k, sg_k in bundle])
                        keep_idx = np.argsort(violations)[:max_bundle - 1]
                        bundle = [bundle[i] for i in keep_idx]
                else:
                    # No improvement even with random perturbation
                    if step_size < min_step:
                        break
                    step_size = min_step
        
        # Periodically clean up the bundle
        if iteration % 20 == 0 and len(bundle) > max_bundle:
            # Keep only the most recent and most active constraints
            violations = np.array([r_k + np.dot(sg_k, x - x_k) - r_current for x_k, r_k, sg_k in bundle])
            keep_idx = np.argsort(violations)[:max_bundle]
            bundle = [bundle[i] for i in keep_idx]
    
    return x_best.reshape(N, 2)

def _optimize(pts, time_budget):
    """
    Run L-BFGS-B optimization on the initial points with adaptive alpha.
    """
    t0 = time.time()
    x0 = pts.flatten()
    
    # Phase 1: Coarse optimization with lower alpha
    # Phase 2: Fine optimization with higher alpha
    # Phase 3: Very fine optimization with even higher alpha
    
    phases = [
        {"alpha": 15.0, "max_iter": 300, "gtol": 1e-4},
        {"alpha": 50.0, "max_iter": 300, "gtol": 1e-5},
        {"alpha": 150.0, "max_iter": 300, "gtol": 1e-6},
    ]
    
    for phase in phases:
        if time.time() - t0 > time_budget:
            break
            
        alpha = phase["alpha"]
        
        def obj(x):
            return _soft_min_neg(x, alpha)
        
        try:
            res = minimize(obj, x0, method='L-BFGS-B', 
                           bounds=[(-0.5, 0.5)] * (2 * N),
                           options={'maxiter': phase["max_iter"], 'gtol': phase["gtol"]})
            x_opt = res.x
            x0 = x_opt
        except Exception:
            pass
            
    # Final polish with exact objective
    remaining = time_budget - (time.time() - t0)
    if remaining > 2.0:
        try:
            res = minimize(_exact_min_neg, x0, method='L-BFGS-B', 
                           bounds=[(-0.5, 0.5)] * (2 * N),
                           options={'maxiter': 200, 'gtol': 1e-8})
            x0 = res.x
        except Exception:
            pass
            
    return x0.reshape(N, 2)

def _local_polish(pts, time_budget):
    """
    Simple local search polish: try small perturbations and keep improvements.
    """
    t0 = time.time()
    best_pts = pts.copy()
    best_r = _score(best_pts)
    
    rng = np.random.default_rng(12345)
    step_size = 0.001
    
    while time.time() - t0 < time_budget:
        # Try random perturbation
        perturb = rng.normal(0, step_size, size=(N, 2))
        trial = best_pts + perturb
        trial = np.clip(trial, -0.5, 0.5)
        
        trial_r = _score(trial)
        if trial_r > best_r:
            best_pts = trial
            best_r = trial_r
            # Reduce step size slightly when we find improvement
            step_size *= 0.95
        else:
            # Increase step size if we're stuck
            step_size *= 1.05
            # Cap step size
            step_size = min(step_size, 0.01)
    
    return best_pts

def construct():
    t_start = time.time()
    
    initializers = [
        ("hex", _hex_init),
        ("grid", _grid_init),
        ("rand0", lambda: _random_init(0)),
        ("rand1", lambda: _random_init(42)),
    ]
    
    best_pts = None
    best_r = 0.0
    
    for name, init_fn in initializers:
        if time.time() > DEADLINE - 15:
            break
            
        pts = init_fn()
        r = _score(pts)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()
            
        # Optimize
        remaining = DEADLINE - time.time()
        if remaining < 8:
            break
            
        # Allocate time: give more to hex and grid
        if name == "hex":
            time_budget = min(remaining * 0.30, 40.0)
        elif name == "grid":
            time_budget = min(remaining * 0.20, 25.0)
        else:
            time_budget = min(remaining * 0.12, 15.0)
            
        opt_pts = _optimize(pts, time_budget)
        
        # Clamp to square
        opt_pts = np.clip(opt_pts, -0.5, 0.5)
        
        opt_r = _score(opt_pts)
        if opt_r > best_r:
            best_r = opt_r
            best_pts = opt_pts.copy()
            
    # Bundle method phase on the best solution
    remaining = DEADLINE - time.time()
    if remaining > 20 and best_pts is not None:
        bundle_budget = min(remaining * 0.4, 35.0)
        bundle_pts = _bundle_method(best_pts, bundle_budget)
        bundle_pts = np.clip(bundle_pts, -0.5, 0.5)
        bundle_r = _score(bundle_pts)
        if bundle_r > best_r:
            best_r = bundle_r
            best_pts = bundle_pts.copy()
    
    # Final polish phases
    remaining = DEADLINE - time.time()
    if remaining > 10 and best_pts is not None:
        # Phase 1: More L-BFGS-B optimization
        time_budget = min(remaining * 0.4, 25.0)
        opt_pts = _optimize(best_pts, time_budget)
        opt_pts = np.clip(opt_pts, -0.5, 0.5)
        opt_r = _score(opt_pts)
        if opt_r > best_r:
            best_r = opt_r
            best_pts = opt_pts.copy()
        
        # Phase 2: Bundle method again with fresh start
        remaining = DEADLINE - time.time()
        if remaining > 10:
            bundle_budget = min(remaining * 0.3, 20.0)
            bundle_pts = _bundle_method(best_pts, bundle_budget)
            bundle_pts = np.clip(bundle_pts, -0.5, 0.5)
            bundle_r = _score(bundle_pts)
            if bundle_r > best_r:
                best_r = bundle_r
                best_pts = bundle_pts.copy()
        
        # Phase 3: Local search polish
        remaining = DEADLINE - time.time()
        if remaining > 5:
            time_budget = min(remaining * 0.7, 15.0)
            polished_pts = _local_polish(best_pts, time_budget)
            polished_r = _score(polished_pts)
            if polished_r > best_r:
                best_r = polished_r
                best_pts = polished_pts.copy()
    
    if best_pts is None:
        best_pts = _hex_init()
        
    return best_pts
