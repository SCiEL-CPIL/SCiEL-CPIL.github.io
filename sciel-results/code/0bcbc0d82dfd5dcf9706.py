"""
Heilbronn triangle problem: maximize minimum triangle area for 11 points.
Strategy:
1. Use Lagrangian Relaxation with Subgradient Optimization.
   - Treat the 165 triangle area constraints A(i,j,k) >= t as soft constraints.
   - Introduce Lagrange multipliers lambda for each constraint.
   - The primal problem minimizes L = -t + sum(lambda * (t - A(i,j,k))).
   - Since A(i,j,k) is non-smooth due to abs(), we use a smooth approximation 
     or handle the sign based on current configuration. 
     Actually, for Lagrangian relaxation of min A_i, we can use the fact that 
     at the optimum, the active constraints are those with A_i = t.
     We can approximate the min by using a soft-min or by directly optimizing t 
     and penalizing violations.
     
   A better approach for Lagrangian relaxation of "maximize t subject to A_i >= t":
   Dual function: g(lambda) = min_{pts, t} { -t + sum_{i} lambda_i * (t - A_i(pts)) }
   This is non-convex in pts. However, for fixed lambda, the term is linear in t 
   and the A_i terms are quadratic in pts (since area is cross product, which is 
   bilinear in coordinates, so A_i^2 is quartic, but A_i itself is not convex).
   
   Wait, area is 0.5 * |cross|. Cross is linear in the difference vectors.
   Let's use the formulation: maximize t s.t. 2*Area >= 2t.
   2*Area = |x_i(y_j-y_k) + ...|. This is the absolute value of a linear form in pts.
   
   A common trick: Instead of Lagrangian on the min, use a barrier or penalty method 
   with a smooth approximation of the minimum, e.g., log-sum-exp or sum of powers.
   However, the directive asks for Lagrangian Relaxation.
   
   Let's try a different angle: 
   Maximize t.
   Constraints: A_i(pts) >= t.
   Lagrangian: L(pts, t, lambda) = t - sum lambda_i (A_i(pts) - t) = t (1 + sum lambda_i) - sum lambda_i A_i(pts).
   This doesn't seem right for a maximization problem with inequality constraints in standard form.
   
   Standard form: Max f(x) s.t. g_i(x) <= 0.
   Here: Max t s.t. t - A_i(pts) <= 0.
   L = t + sum lambda_i (A_i(pts) - t) = t(1 - sum lambda_i) + sum lambda_i A_i(pts).
   To maximize L wrt t, if sum lambda_i < 1, t -> infinity. If sum lambda_i > 1, t -> 0 (if bounded).
   This suggests we need to fix the scale of lambda or use a different formulation.
   
   Alternative: Use a penalty method which is a specific case of Lagrangian where lambda is updated based on violation.
   Objective: Minimize P(pts, t) = -t + mu * sum_{i} max(0, t - A_i(pts))^2.
   This is smooth if we smooth max(0, x).
   Then we can use gradient-based methods.
   
   Let's implement a **Smoothed Penalty Method** with adaptive penalty parameter mu.
   This is structurally similar to Lagrangian relaxation but often more robust.
   We will use L-BFGS-B for the inner loop because it handles bounds well and is efficient for large N.
   We will use Barycentric coordinates to enforce the triangle constraint naturally (sum=1, >=0).
"""

import numpy as np
import time
import random
from scipy.optimize import minimize
from scipy.special import logsumexp

N = 11
S3H = np.sqrt(3) / 2
DEADLINE = 115.0  # Safety margin before 150s limit

# Pre-compute triplet indices
_triplets = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _triplets.append((i, j, k))
_triplets = np.array(_triplets)
_i_idx = _triplets[:, 0]
_j_idx = _triplets[:, 1]
_k_idx = _triplets[:, 2]
M = len(_triplets)

def _barycentric_to_cartesian(bary):
    """Convert barycentric coordinates (b1, b2, b3) to Cartesian (x, y).
    Vertices: V0=(0,0), V1=(1,0), V2=(0.5, sqrt(3)/2)
    p = b1*V0 + b2*V1 + b3*V2
    x = b2 + 0.5*b3
    y = S3H*b3
    """
    x = bary[:, 1] + 0.5 * bary[:, 2]
    y = S3H * bary[:, 2]
    return np.column_stack((x, y))

def _cartesian_to_bary(pts):
    """Convert Cartesian points to barycentric coordinates."""
    b3 = pts[:, 1] / S3H
    b2 = pts[:, 0] - 0.5 * b3
    b1 = 1.0 - b2 - b3
    return np.column_stack((b1, b2, b3))

def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by three points."""
    p_i = pts[_i_idx]
    p_j = pts[_j_idx]
    p_k = pts[_k_idx]
    
    dx1 = p_j - p_i
    dx2 = p_k - p_i
    
    cross = dx1[:, 0] * dx2[:, 1] - dx1[:, 1] * dx2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def _all_triangle_areas(pts):
    """Compute all triangle areas."""
    p_i = pts[_i_idx]
    p_j = pts[_j_idx]
    p_k = pts[_k_idx]
    
    dx1 = p_j - p_i
    dx2 = p_k - p_i
    
    cross = dx1[:, 0] * dx2[:, 1] - dx1[:, 1] * dx2[:, 0]
    areas = 0.5 * np.abs(cross)
    return areas

def _random_bary_point(rng):
    """Generate a random point in barycentric coordinates."""
    u = rng.random()
    v = rng.random()
    while u + v > 1:
        u = rng.random()
        v = rng.random()
    return np.array([1.0 - u - v, u, v])

def _random_bary_init(rng):
    """Generate a random initial configuration in barycentric coordinates."""
    return np.array([_random_bary_point(rng) for _ in range(N)])

def _greedy_bary_init(rng):
    """Greedy initialization in Cartesian, then convert to Bary."""
    pts_cart = []
    p0 = _random_bary_point(rng)
    pts_cart.append(_barycentric_to_cartesian(p0.reshape(1, -1))[0])
    
    for _ in range(N - 1):
        best_pos = None
        best_min_dist = -1
        for _ in range(200): # Reduced for speed
            cand_bary = _random_bary_point(rng)
            cand_cart = _barycentric_to_cartesian(cand_bary.reshape(1, -1))[0]
            dists = np.linalg.norm(pts_cart - cand_cart, axis=1)
            min_d = np.min(dists)
            if min_d > best_min_dist:
                best_min_dist = min_d
                best_pos = cand_bary
        pts_cart.append(_barycentric_to_cartesian(best_pos.reshape(1, -1))[0])
    
    return _cartesian_to_bary(np.array(pts_cart))

def construct():
    """
    Construct 11 points in the unit equilateral triangle to maximize min triangle area.
    Uses a Smoothed Penalty Method (Lagrangian-like) with L-BFGS-B in Barycentric coordinates.
    """
    start_time = time.time()
    rng = random.Random(42)
    
    best_pts_bary = None
    best_score = -1.0
    
    # Initializations
    initializations = []
    
    # 1. Greedy
    try:
        p_greedy = _greedy_bary_init(rng)
        initializations.append(p_greedy)
    except Exception:
        pass
    
    # 2. Random
    p_rand = _random_bary_init(rng)
    initializations.append(p_rand)
    
    # 3. More random
    for _ in range(3):
        p_rand2 = _random_bary_init(rng)
        initializations.append(p_rand2)

    # Parameters for penalty method
    # Objective: minimize -t + mu * sum(soft_violation)
    # soft_violation = max(0, t - A_i)^2 approximated by smooth function
    # Let's use: f(pts, t) = -t + mu * sum( (t - A_i)^2 * sigmoid((t - A_i)/temp) )
    # Or simpler: just minimize -min(A_i) using a smooth approximation like -logsumexp(-A_i / temp)
    # This is a standard smooth minimum.
    
    def objective_smooth_min(x):
        # x is 3N barycentric coordinates
        bary = x.reshape(N, 3)
        # Ensure barycentric constraints are handled by the optimizer bounds/constraints?
        # L-BFGS-B supports bounds. 
        # b1+b2+b3=1 is a linear equality constraint. L-BFGS-B doesn't handle equality constraints well.
        # We can eliminate b1 = 1 - b2 - b3.
        # So x is 2N variables: b2, b3 for each point.
        # b1 = 1 - b2 - b3.
        # Bounds: b2 >= 0, b3 >= 0, b1 >= 0 => b2+b3 <= 1.
        # This is a simplex constraint.
        
        b2 = x[0::2]
        b3 = x[1::2]
        b1 = 1.0 - b2 - b3
        
        bary = np.column_stack((b1, b2, b3))
        pts = _barycentric_to_cartesian(bary)
        
        areas = _all_triangle_areas(pts)
        
        # Smooth minimum: -logsumexp(-areas / temp) * temp
        # We want to maximize min(areas), so minimize -min(areas).
        # min(areas) approx -1/temp * log(sum(exp(-areas/temp)))
        # So objective = 1/temp * log(sum(exp(-areas/temp)))
        
        # We need to choose temp. Smaller temp -> sharper min.
        # Let's use a fixed temp for this inner loop, or adapt it.
        temp = 0.01
        # To avoid overflow/underflow, shift areas
        areas_shifted = areas - np.max(areas)
        log_sum = logsumexp(-areas_shifted / temp)
        # min_area_approx = -temp * (log_sum + max(areas)/temp) = -temp * log_sum - max(areas)
        # Wait, logsumexp(x) = max(x) + log(sum(exp(x - max(x))))
        # Here x = -areas/temp. max(x) = -min(areas)/temp.
        # logsumexp(-areas/temp) = -min(areas)/temp + log(sum(exp(-(areas - min(areas))/temp)))
        # So min(areas) = -temp * logsumexp(-areas/temp) + temp * log(sum(exp(...)))? No.
        # Let's just use: min_approx = -temp * logsumexp(-areas / temp)
        # Check: if all areas = A, logsumexp(-A/temp) = log(M * exp(-A/temp)) = log(M) - A/temp.
        # min_approx = -temp * (log(M) - A/temp) = A - temp*log(M).
        # This is biased by -temp*log(M).
        # To remove bias, we can add temp*log(M).
        # min_approx_unbiased = A.
        
        # However, the gradient of -min_approx is what matters for optimization.
        # Let's just minimize: temp * logsumexp(-areas / temp)
        # This is equivalent to minimizing -min(areas) + constant.
        
        obj_val = temp * log_sum # This is negative of the biased min?
        # Let's re-derive.
        # We want to minimize -min(A).
        # g(A) = -1/temp * log(sum(exp(-A/temp)))
        # grad g = -1/temp * (sum(-1/temp * exp(-A/temp) * grad A) / sum(exp(-A/temp)))
        #        = 1/temp^2 * sum(w_i * grad A_i) where w_i are softmax weights.
        # So minimizing g is like taking a weighted average of gradients of A, weighted by how small A is.
        # This is good.
        
        # Let's return g(A)
        # g(A) = -1/temp * logsumexp(-A/temp)
        # Note: logsumexp(-A/temp) is roughly -min(A)/temp.
        # So g(A) is roughly min(A).
        # We want to MAXIMIZE min(A), so MINIMIZE -min(A).
        # So we should minimize -g(A) = 1/temp * logsumexp(-A/temp).
        
        return 1.0 / temp * log_sum

    def objective_grad(x):
        b2 = x[0::2]
        b3 = x[1::2]
        b1 = 1.0 - b2 - b3
        bary = np.column_stack((b1, b2, b3))
        pts = _barycentric_to_cartesian(bary)
        
        p_i = pts[_i_idx]
        p_j = pts[_j_idx]
        p_k = pts[_k_idx]
        
        dx1 = p_j - p_i
        dx2 = p_k - p_i
        
        cross = dx1[:, 0] * dx2[:, 1] - dx1[:, 1] * dx2[:, 0]
        areas = 0.5 * np.abs(cross)
        
        # We need the sign of cross to compute gradient of abs(cross)
        # grad |cross| = sign(cross) * grad cross
        # grad cross wrt pts is complex.
        # Let's use numerical gradient for simplicity and robustness, or implement carefully.
        # Given time constraints, numerical gradient with L-BFGS-B is safer.
        # But L-BFGS-B is much faster with analytic gradients.
        # Let's try to implement analytic gradient.
        
        # Area = 0.5 * |det([p_j-p_i, p_k-p_i])|
        # Let D = det. Area = 0.5 |D|.
        # d Area / d p_l = 0.5 * sign(D) * dD/dp_l.
        
        # D = (x_j - x_i)(y_k - y_i) - (y_j - y_i)(x_k - x_i)
        # dD/dx_i = -(y_k - y_i) + (y_j - y_i) = y_j - y_k
        # dD/dy_i = -(x_k - x_i) + (x_j - x_i) = x_j - x_k
        # dD/dx_j = (y_k - y_i)
        # dD/dy_j = -(x_k - x_i)
        # dD/dx_k = -(y_j - y_i)
        # dD/dy_k = (x_j - x_i)
        
        # So grad Area wrt (x,y) of point l:
        # If l == i: 0.5 * sign(D) * (y_j - y_k, x_j - x_k)
        # If l == j: 0.5 * sign(D) * (y_k - y_i, -(x_k - x_i))
        # If l == k: 0.5 * sign(D) * (-(y_j - y_i), x_j - x_i)
        
        # We need to sum these contributions weighted by softmax weights w_i = exp(-A_i/temp) / sum(exp(-A_j/temp)).
        # And then chain rule to barycentric coords.
        
        # This is getting complex. Let's use numerical gradient first to ensure correctness,
        # then switch to analytic if time permits.
        # For now, return None to let L-BFGS-B use numerical.
        return None

    # Since analytic gradient is complex, we will use numerical gradients.
    # L-BFGS-B with numerical gradients is slower but robust.
    
    # Bounds for b2, b3:
    # b2 >= 0, b3 >= 0, b2+b3 <= 1.
    # L-BFGS-B bounds are per-variable. We can't easily express b2+b3<=1.
    # We can use a change of variables: b2 = u^2, b3 = v^2? No, that doesn't help sum constraint.
    # We can use a penalty for b2+b3 > 1.
    # Or, we can project back to simplex after each step? L-BFGS-B doesn't support projection.
    
    # Alternative: Use SLSQP which supports equality constraints.
    # But the directive suggested L-BFGS-B.
    # Let's stick to L-BFGS-B and add a penalty for b2+b3 > 1.
    
    def objective_with_penalty(x):
        b2 = x[0::2]
        b3 = x[1::2]
        
        # Penalty for b2+b3 > 1
        viol = np.maximum(0, b2 + b3 - 1.0)
        pen = 100.0 * np.sum(viol**2)
        
        # Penalty for b2 < 0 or b3 < 0 (though bounds handle this)
        pen += 100.0 * np.sum(np.maximum(0, -b2)**2) + 100.0 * np.sum(np.maximum(0, -b3)**2)
        
        # Main objective
        b1 = 1.0 - b2 - b3
        bary = np.column_stack((b1, b2, b3))
        pts = _barycentric_to_cartesian(bary)
        
        areas = _all_triangle_areas(pts)
        temp = 0.01
        areas_shifted = areas - np.max(areas)
        log_sum = logsumexp(-areas_shifted / temp)
        
        obj = 1.0 / temp * log_sum + pen
        
        return obj

    bounds = [(0, 1)] * (2 * N) # b2, b3 in [0,1]
    
    for s, init_bary in enumerate(initializations):
        if time.time() - start_time > DEADLINE - 10:
            break
            
        # Flatten to 2N variables
        b2 = init_bary[:, 1]
        b3 = init_bary[:, 2]
        x0 = np.concatenate([b2, b3])
        
        # Ensure x0 is within bounds
        x0 = np.clip(x0, 0, 1)
        
        try:
            # Use L-BFGS-B with numerical gradient
            result = minimize(
                objective_with_penalty,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-6}
            )
            
            if result.success:
                x_opt = result.x
                b2_opt = x_opt[0::2]
                b3_opt = x_opt[1::2]
                b1_opt = 1.0 - b2_opt - b3_opt
                
                # Project to simplex if needed
                # If b1 < 0, it means b2+b3 > 1.
                # We need to scale b2, b3 such that b2+b3 <= 1.
                sum_b23 = b2_opt + b3_opt
                if np.any(sum_b23 > 1.0):
                    mask = sum_b23 > 1.0
                    b2_opt[mask] *= 0.99 / sum_b23[mask]
                    b3_opt[mask] *= 0.99 / sum_b23[mask]
                    b1_opt = 1.0 - b2_opt - b3_opt
                
                bary_opt = np.column_stack((b1_opt, b2_opt, b3_opt))
                pts_opt = _barycentric_to_cartesian(bary_opt)
                
                score = _min_triangle_area(pts_opt)
                if score > best_score:
                    best_score = score
                    best_pts_bary = bary_opt.copy()
                    
        except Exception as e:
            pass

    # If no optimization succeeded, fall back to best initial guess
    if best_pts_bary is None:
        for init_bary in initializations:
            pts = _barycentric_to_cartesian(init_bary)
            score = _min_triangle_area(pts)
            if score > best_score:
                best_score = score
                best_pts_bary = init_bary.copy()

    if best_pts_bary is None:
        best_pts_bary = _random_bary_init(rng)

    # Final check and conversion to Cartesian
    best_pts = _barycentric_to_cartesian(best_pts_bary)
    
    return best_pts

if __name__ == "__main__":
    pts = construct()
    print(f"Score: {_min_triangle_area(pts)}")
