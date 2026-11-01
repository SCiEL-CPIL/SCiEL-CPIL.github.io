import time
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

N = 14
D = 3
DEADLINE = 145.0
_t0 = None

def _true_score(pts):
    """Calculate the squared ratio of max to minimum pairwise distance."""
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    mn = d.min()
    mx = d.max()
    if mn < 1e-9:
        return np.inf
    return (mx / mn) ** 2

def _qcqp_objective(x_flat, R, mu):
    """
    Smooth log-barrier penalized objective for the QCQP:
      maximize t (min squared pairwise distance)
      subject to: all squared distances <= R^2 * t
    
    We reformulate as minimizing:
      -t + mu * sum_i log(1 + (d_i^2 - R^2 * t) / eps)  for violated constraints
      + mu * sum_j log(1 + (d_j^2 - t) / eps)           for min-distance constraints
    
    Actually, cleaner: we want to find points where min d^2 = t and max d^2 <= R^2 * t.
    The objective: minimize (max d^2 / min d^2) via penalty on ratio exceeding R^2.
    
    Better formulation: minimize
      f(x) = max_i d_i^2 / min_i d_i^2  (softened)
    subject to the ratio being close to R^2.
    
    Cleanest: parameterize by R, and minimize the violation:
      g(x, R) = max(0, max_d2 - R^2 * min_d2) + max(0, min_d2 - R^2 * max_d2)
    But this is non-smooth.
    
    Let me use: minimize  (max_d2 + R^2 * min_d2 - 2*R * sqrt(max_d2 * min_d2)) / min_d2
    This is (sqrt(max_d2) - R * sqrt(min_d2))^2 / min_d2, which is smooth-ish.
    
    Actually, the simplest effective approach:
    For a given R, we want to find points where the ratio of max to min distance is exactly R.
    We minimize:  (max_d2 - R^2 * min_d2)^2 / (min_d2 * max_d2)
    This is zero when max_d2 = R^2 * min_d2, and positive otherwise.
    It's smooth (except at zero distances).
    """
    pts = x_flat.reshape(N, D)
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    d2 = d ** 2
    
    mn = d2.min()
    mx = d2.max()
    
    if mn < 1e-8:
        return 1e6
    
    # Objective: (mx - R^2 * mn)^2 / (mn * mx)
    # This is 0 when mx = R^2 * mn, i.e., ratio = R
    # Add a small regularization to prevent collapse
    ratio_term = (mx - R * R * mn) ** 2 / (mn * mx + 1e-12)
    
    # Also penalize if points get too close (mn too small relative to scale)
    scale = np.mean(d2)
    if mn < 1e-4 * scale:
        return 1e4 + ratio_term
    
    return ratio_term

def _solve_for_R(R, x0, time_budget):
    """
    Given a target ratio R, find points that achieve ratio close to R.
    Use L-BFGS-B to minimize the QCQP objective.
    """
    start_time = time.time()
    
    def obj(x):
        if time.time() - start_time > time_budget:
            return 1e10
        return _qcqp_objective(x, R, mu=1.0)
    
    try:
        res = minimize(
            obj,
            x0,
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-6, 'maxfun': 500}
        )
        return res.x.reshape(N, D)
    except Exception:
        return x0.reshape(N, D)

def _fib_sphere(n):
    """Generate n points on a unit sphere using Fibonacci lattice."""
    i = np.arange(n)
    phi = np.arccos(1 - 2 * (i + 0.5) / n)
    theta = np.pi * (1 + 5 ** 0.5) * i
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.column_stack([x, y, z])

def _icosa14():
    """Generate 14 points: 12 icosahedron vertices + 2 extra."""
    phi_ratio = (1 + 5 ** 0.5) / 2
    verts = np.array([
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
        [0, phi_ratio, 1], [0, phi_ratio, -1], [0, -phi_ratio, 1], [0, -phi_ratio, -1],
        [1, 0, phi_ratio], [1, 0, -phi_ratio], [-1, 0, phi_ratio], [-1, 0, -phi_ratio],
        [phi_ratio, 1, 0], [phi_ratio, -1, 0], [-phi_ratio, 1, 0], [-phi_ratio, -1, 0]
    ], dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    # Take 12, add 2 more (opposite pairs)
    extra1 = np.array([0.0, 0.0, 1.0])
    extra2 = np.array([0.0, 0.0, -1.0])
    pts = np.vstack([verts[:12], extra1, extra2])
    return pts

def construct():
    global _t0
    if _t0 is None:
        _t0 = time.time()
    
    rng = np.random.default_rng(12345)
    
    best_pts = None
    best_score = np.inf
    
    # Generate diverse initializations
    inits = []
    inits.append(_fib_sphere(N))
    inits.append(_icosa14())
    
    # Random sphere points
    for _ in range(8):
        r = rng.normal(0, 1, size=(N, 3))
        r = r / np.linalg.norm(r, axis=1, keepdims=True)
        inits.append(r)
    
    # Random Gaussian (centered)
    for _ in range(4):
        r = rng.normal(0, 1, size=(N, 3))
        r -= r.mean(axis=0)
        inits.append(r)
    
    # Bisection search on R (the distance ratio)
    # Theoretical lower bound for ratio is 1.0 (all distances equal - impossible in 3D for N=14)
    # Upper bound: start with a generous bound
    R_lo = 1.0
    R_hi = 5.0
    
    # First, get a baseline from the initializations
    for init in inits[:4]:
        score = _true_score(init)
        if score < best_score:
            best_score = score
            best_pts = init.copy()
    
    # Bisection: for each R, try to find a configuration with ratio close to R
    # If we succeed (objective near 0), R is achievable; tighten the bound
    # We want to find the minimum achievable R
    
    # Do a coarse bisection first
    num_bisections = 12  # 2^12 = 4096, but we'll limit by time
    bisection_time_per = 3.0  # seconds per bisection step
    
    # For each R value, try multiple initializations
    num_init_per_R = 3
    
    # Pre-compute: which initializations to use
    init_pool = inits[:]
    
    # Coarse bisection
    for bi in range(num_bisections):
        if time.time() - _t0 > DEADLINE - 15:
            break
        
        R_mid = 0.5 * (R_lo + R_hi)
        
        # Try to achieve ratio R_mid
        found_good = False
        best_obj_val = np.inf
        best_x_for_R = None
        
        for ii in range(num_init_per_R):
            if time.time() - _t0 > DEADLINE - 15:
                break
            
            # Use different initializations
            init_idx = (bi * num_init_per_R + ii) % len(init_pool)
            x0 = init_pool[init_idx].flatten()
            
            # Add small random perturbation for diversity
            if ii > 0:
                x0 = x0 + rng.normal(0, 0.05, size=x0.shape)
            
            time_budget = min(bisection_time_per, DEADLINE - 15 - (time.time() - _t0))
            if time_budget < 0.5:
                break
            
            x_result = _solve_for_R(R_mid, x0, time_budget)
            
            # Evaluate
            score = _true_score(x_result)
            if score < best_score:
                best_score = score
                best_pts = x_result.copy()
            
            # Check if ratio is close to R_mid
            obj_val = _qcqp_objective(x_result.flatten(), R_mid, 1.0)
            if obj_val < best_obj_val:
                best_obj_val = obj_val
                best_x_for_R = x_result.copy()
            
            # If objective is very small, R_mid is achievable
            if obj_val < 0.01:
                found_good = True
                break
        
        # Decide: if we found a good configuration with ratio ~R_mid,
        # then R_mid is achievable, so we can try smaller R
        # Otherwise, R_mid is not achievable, try larger R
        
        # Use the actual achieved ratio to decide
        if best_x_for_R is not None:
            actual_score = _true_score(best_x_for_R)
            actual_R = np.sqrt(actual_score)
            
            if actual_R <= R_mid * 1.05:
                # Achieved ratio is close to or below R_mid, so R_mid is achievable
                R_hi = R_mid
            else:
                # Couldn't achieve R_mid, need larger R
                R_lo = R_mid
        
        # Print progress (commented out for production)
        # print(f"Bisect {bi}: R_mid={R_mid:.4f}, R_lo={R_lo:.4f}, R_hi={R_hi:.4f}, best_score={best_score:.4f}")
    
    # Final refinement: use the best R found and do more optimization
    if time.time() - _t0 < DEADLINE - 20:
        R_final = R_hi  # The smallest achievable ratio we found
        
        # Do more refinement around R_final
        for refinement in range(5):
            if time.time() - _t0 > DEADLINE - 10:
                break
            
            # Try slightly different R values around R_final
            for R_try in [R_final * 0.99, R_final, R_final * 1.01]:
                if time.time() - _t0 > DEADLINE - 10:
                    break
                
                for ii in range(3):
                    if time.time() - _t0 > DEADLINE - 10:
                        break
                    
                    init_idx = (refinement * 3 + ii) % len(init_pool)
                    x0 = init_pool[init_idx].flatten()
                    x0 = x0 + rng.normal(0, 0.02, size=x0.shape)
                    
                    time_budget = min(3.0, DEADLINE - 10 - (time.time() - _t0))
                    if time_budget < 0.5:
                        break
                    
                    x_result = _solve_for_R(R_try, x0, time_budget)
                    score = _true_score(x_result)
                    if score < best_score:
                        best_score = score
                        best_pts = x_result.copy()
    
    # Final polish: run a few more optimizations from the best point found
    if best_pts is not None and time.time() - _t0 < DEADLINE - 10:
        R_best = np.sqrt(best_score)
        for polish in range(3):
            if time.time() - _t0 > DEADLINE - 10:
                break
            x0 = best_pts.flatten() + rng.normal(0, 0.01, size=best_pts.size)
            time_budget = min(3.0, DEADLINE - 10 - (time.time() - _t0))
            if time_budget < 0.5:
                break
            x_result = _solve_for_R(R_best, x0, time_budget)
            score = _true_score(x_result)
            if score < best_score:
                best_score = score
                best_pts = x_result.copy()
    
    # Fallback
    if best_pts is None:
        best_pts = inits[0]
    
    # Ensure distinct points
    d = cdist(best_pts, best_pts)
    np.fill_diagonal(d, np.inf)
    if d.min() < 1e-6:
        best_pts += rng.normal(0, 1e-4, size=(N, D))
    
    return best_pts
