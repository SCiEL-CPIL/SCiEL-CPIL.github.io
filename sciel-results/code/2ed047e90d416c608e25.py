import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 14
D = 3
DEADLINE = 140.0
_t0 = None

def _true_score(pts):
    """
    Calculate the squared ratio of max to min pairwise distance.
    """
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    mn = d.min()
    mx = d.max()
    if mn < 1e-9:
        return np.inf
    return (mx / mn) ** 2

def _qcqp_objective(v, mu, upper_bound_sq):
    """
    Objective function for the QCQP formulation.
    v: vector of 3N coordinates.
    mu: Lagrange multiplier / penalty weight for the lower bound (min distance).
    upper_bound_sq: fixed upper bound on squared distances.
    
    We want to maximize the minimum squared distance d_min^2.
    This is equivalent to minimizing -d_min^2.
    However, d_min^2 is non-smooth (min).
    
    Alternative standard QCQP form:
    Maximize t
    Subject to:
      ||x_i - x_j||^2 >= t  for all i < j  (t is the min squared dist)
      ||x_i - x_j||^2 <= U   for all i < j  (U is the max squared dist bound)
    
    Using penalty method for the ">= t" constraint:
    We fix U (upper bound) and try to find a configuration where all distances
    are at least some value, or we maximize the minimum distance while keeping
    max distance <= U.
    
    Actually, a simpler penalty formulation for "minimize ratio" is to minimize:
    (Max_D - Min_D)^2 / Min_D^2 is hard.
    
    Let's stick to the directive: "maximize the minimum squared pairwise distance 
    subject to the constraint that all squared pairwise distances are less than 
    or equal to a fixed upper bound".
    
    So, for a fixed U (upper bound on d^2), we want to maximize t such that
    d_ij^2 >= t for all pairs.
    
    We can define the objective as:
    J(v) = - (min_{i<j} d_ij^2) + lambda * sum_{i<j} max(0, d_ij^2 - U)^2
    
    The term -min(d^2) is non-smooth. We can approximate min(d^2) using a soft-min
    or simply optimize the variance of distances which correlates well.
    
    Better: Use the "soft-min" approximation for the minimum distance to make it differentiable.
    soft_min(d^2) = -1/k * log(sum(exp(-d_ij^2 / k)))
    As k -> infinity, this approaches min(d^2).
    
    Objective:
    minimize  - soft_min(d^2) + penalty * sum(max(0, d^2 - U)^2)
    
    Wait, maximizing min(d^2) is the goal. So we minimize -min(d^2).
    
    Let's refine the objective:
    1. Compute all squared distances d2_ij.
    2. soft_min_d2 = -k * log(sum(exp(-d2_ij / k)))  (approximation of min d2)
    3. violation = sum( max(0, d2_ij - U)^2 )
    4. Objective = -soft_min_d2 + mu * violation
    
    We will iteratively decrease U to tighten the upper bound, or keep U fixed 
    and increase the weight on the min-distance term.
    
    Actually, the ratio R = (max_d / min_d)^2.
    If we fix the scale such that max_d^2 <= U, we want to maximize min_d^2.
    The optimal ratio for a given U is U / max_min_d2.
    So we want to maximize max_min_d2.
    
    We can search over U. Or, we can just minimize the objective:
    Obj = (max_d2 / min_d2) approximated.
    
    Let's use the penalty method as suggested:
    Maximize t subject to d_ij^2 >= t and d_ij^2 <= U.
    
    Let's implement the objective to minimize:
    L = -t + mu * sum( max(0, t - d_ij^2)^2 + max(0, d_ij^2 - U)^2 )
    
    This is a function of v and t. We can optimize over v and t jointly?
    L-BFGS-B can handle multiple variables.
    Variables: v (3N) + t (1 scalar).
    
    However, t is coupled. If we fix U, we can just maximize the minimum distance.
    Let's simplify:
    For a fixed U, define:
    f(v) = min_{i<j} ||x_i - x_j||^2
    We want to maximize f(v) subject to g(v) = max_{i<j} ||x_i - x_j||^2 <= U.
    
    Penalty method:
    minimize  -f(v) + mu * (max(0, g(v) - U))^2
    
    f(v) is non-smooth. Use soft-min.
    g(v) is non-smooth. Use soft-max.
    
    soft_min(d2, k) = -k * log(sum(exp(-d2/k)))
    soft_max(d2, k) = k * log(sum(exp(d2/k)))
    
    Objective:
    J(v) = -soft_min(d2, k) + mu * (max(0, soft_max(d2, k) - U))^2
    
    We will run this for several values of U, or adaptively.
    """
    x = v.reshape(N, D)
    
    # Compute squared distances
    # d2_ij = ||x_i - x_j||^2
    # Efficient calculation:
    # d2 = sum(x^2, axis=1)[:, None] + sum(x^2, axis=1)[None, :] - 2 * x @ x.T
    x_sq = np.sum(x**2, axis=1)
    d2 = x_sq[:, None] + x_sq[None, :] - 2 * (x @ x.T)
    
    # Numerical stability
    d2 = np.maximum(d2, 0)
    np.fill_diagonal(d2, 0)
    
    # Get upper triangle
    iu = np.triu_indices(N, k=1)
    d2_flat = d2[iu]
    
    if len(d2_flat) == 0:
        return 0.0
    
    # Soft min and max parameters
    # k controls the sharpness. Smaller k is sharper (closer to true min/max).
    # But too small causes numerical issues.
    k = 0.1
    
    # Soft min
    # To avoid overflow in exp, subtract max (for min, subtract min? No, for min we use -d2)
    # soft_min = -k * log(sum(exp(-d2/k)))
    # Let m = min(d2). exp(-(d2-m)/k) is stable.
    # sum(exp(-d2/k)) = exp(m/k) * sum(exp(-(d2-m)/k))
    # log(sum(...)) = m/k + log(sum(exp(-(d2-m)/k)))
    # soft_min = -k * (m/k + log(...)) = -m - k*log(sum(exp(-(d2-m)/k)))
    # Wait, if d2 are all 10, m=10. exp(0)=1. sum=91 (for N=14, 91 pairs).
    # soft_min = -10 - k*log(91). This is < -10.
    # True min is 10. We want to maximize min, so minimize -min.
    # -soft_min = m + k*log(sum(...)). This is > m.
    # So minimizing -soft_min is similar to minimizing m, but we want to MAXIMIZE m.
    # So we minimize - (m + ...).
    
    m_min = np.min(d2_flat)
    exp_terms_min = np.exp(-(d2_flat - m_min) / k)
    sum_exp_min = np.sum(exp_terms_min)
    soft_min_val = m_min + k * np.log(sum_exp_min)
    
    # Soft max
    m_max = np.max(d2_flat)
    exp_terms_max = np.exp((d2_flat - m_max) / k)
    sum_exp_max = np.sum(exp_terms_max)
    soft_max_val = m_max + k * np.log(sum_exp_max)
    
    # Penalty for exceeding upper bound U
    violation = 0.0
    if soft_max_val > upper_bound_sq:
        violation = (soft_max_val - upper_bound_sq) ** 2
    
    # Objective: Minimize -soft_min_val + mu * violation
    # We want large soft_min_val, so we minimize its negative.
    return -soft_min_val + mu * violation

def construct():
    global _t0
    if _t0 is None:
        _t0 = time.time()
    
    rng = np.random.default_rng(42)
    best_pts = None
    best_score = np.inf
    
    # We will try a few different upper bounds U.
    # A good initial guess for U is the squared diameter of a Fibonacci sphere or similar.
    # For N=14 in 3D, points on unit sphere have max dist <= 4 (diameter 2).
    # Let's try U in a range.
    
    # Initial guess for points: Fibonacci sphere
    def get_fib_sphere():
        i = np.arange(N)
        phi = np.arccos(1 - 2 * (i + 0.5) / N)
        theta = np.pi * (1 + 5 ** 0.5) * i
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        return np.column_stack([x, y, z])
    
    pts_init = get_fib_sphere()
    v0 = pts_init.flatten()
    
    # Calculate initial score to have a baseline
    base_score = _true_score(pts_init)
    if base_score < best_score:
        best_score = base_score
        best_pts = pts_init.copy()
    
    # Upper bounds to try (squared distances)
    # If points are on unit sphere, max d^2 is 4.
    # If we allow them to spread, maybe U=5, 6, etc.
    # But if U is too large, the constraint is loose.
    # If U is too small, it's infeasible or forces points to be close, reducing min dist.
    # The ratio is (sqrt(U) / sqrt(min_d2))^2 = U / min_d2.
    # We want to minimize U / min_d2.
    
    upper_bounds = [3.5, 4.0, 4.5, 5.0]
    mu_values = [10.0, 100.0, 1000.0]
    
    # Time management
    time_per_opt = 10.0
    
    for U in upper_bounds:
        if time.time() - _t0 > DEADLINE - 20:
            break
            
        for mu in mu_values:
            if time.time() - _t0 > DEADLINE - 20:
                break
            
            # Start from the best points found so far, or a random restart
            # Let's try starting from the current best
            v_start = best_pts.flatten().copy() if best_pts is not None else v0
            
            # Add small noise to escape local minima
            v_start = v_start + rng.normal(scale=0.01, size=v_start.shape)
            
            try:
                res = minimize(
                    _qcqp_objective,
                    v_start,
                    args=(mu, U),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-5, 'maxls': 50}
                )
                
                v_opt = res.x
                pts_opt = v_opt.reshape(N, D)
                
                # Center the points
                pts_opt = pts_opt - pts_opt.mean(axis=0)
                
                score_opt = _true_score(pts_opt)
                
                if score_opt < best_score:
                    best_score = score_opt
                    best_pts = pts_opt.copy()
                    
            except Exception:
                pass
    
    # Final polish with local search on the best points found
    if best_pts is not None and time.time() - _t0 < DEADLINE - 5:
        pts = best_pts.copy()
        # Simple coordinate descent / local refinement
        # Just a few steps of gradient descent on the true score approximation
        # We can reuse the QCQP objective with a tight U derived from current best
        d = cdist(pts, pts)
        np.fill_diagonal(d, np.inf)
        current_max_d2 = d.max()**2
        current_min_d2 = d.min()**2
        
        # Try to improve with a tighter U
        U_tight = current_max_d2 * 1.05
        mu_tight = 1000.0
        
        v_start = pts.flatten()
        try:
            res = minimize(
                _qcqp_objective,
                v_start,
                args=(mu_tight, U_tight),
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            v_opt = res.x
            pts_opt = v_opt.reshape(N, D)
            pts_opt = pts_opt - pts_opt.mean(axis=0)
            score_opt = _true_score(pts_opt)
            if score_opt < best_score:
                best_score = score_opt
                best_pts = pts_opt
        except Exception:
            pass

    # Fallback
    if best_pts is None:
        best_pts = get_fib_sphere()

    return best_pts
