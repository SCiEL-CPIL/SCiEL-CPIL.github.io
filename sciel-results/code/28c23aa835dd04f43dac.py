import numpy as np
import time
from scipy.optimize import minimize

N = 11
SQRT3 = np.sqrt(3)

# Vertices of the equilateral triangle
V1 = np.array([0.0, 0.0])
V2 = np.array([1.0, 0.0])
V3 = np.array([0.5, SQRT3 / 2.0])

# Pre-compute indices for all unique triples (i, j, k) with i < j < k
# This avoids generating them in every objective call
triples = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            triples.append((i, j, k))
triples = np.array(triples)
num_triples = len(triples)

def _bary_to_cart(bary):
    """
    Convert barycentric coordinates (u, v) to Cartesian (x, y).
    Input: (N, 2) array of (u, v).
    Output: (N, 2) array of (x, y).
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    x = v + 0.5 * w
    y = (SQRT3 / 2.0) * w
    
    return np.column_stack((x, y))

def _cart_to_bary(points):
    """
    Convert Cartesian (x, y) to barycentric (u, v).
    Input: (N, 2) array of (x, y).
    Output: (N, 2) array of (u, v).
    """
    x = points[:, 0]
    y = points[:, 1]
    
    # Inverse of the transformation:
    # x = 0.5 + 0.5*v - 0.5*u  => 2x - 1 = v - u
    # y = (sqrt(3)/2) * w      => w = 2y/sqrt(3)
    # u + v + w = 1            => u + v = 1 - w
    
    w = 2.0 * y / SQRT3
    u_plus_v = 1.0 - w
    v_minus_u = 2.0 * x - 1.0
    
    # v = ( (u+v) + (v-u) ) / 2
    v = (u_plus_v + v_minus_u) / 2.0
    # u = ( (u+v) - (v-u) ) / 2
    u = (u_plus_v - v_minus_u) / 2.0
    
    return np.column_stack((u, v))

def _project_to_simplex(bary):
    """
    Project barycentric coordinates to the valid simplex region.
    Input: (N, 2) array of (u, v).
    Output: (N, 2) array of projected (u, v).
    """
    u = bary[:, 0].copy()
    v = bary[:, 1].copy()
    
    u = np.clip(u, 0, 1)
    v = np.clip(v, 0, 1)
    
    s = u + v
    mask = s > 1.0
    if np.any(mask):
        factor = 1.0 / s[mask]
        u[mask] *= factor
        v[mask] *= factor
        
    return np.column_stack((u, v))

def _compute_triangle_areas(points):
    """
    Compute areas of all triangles formed by the points.
    Returns an array of shape (num_triples,).
    """
    # points[i, j, k] for all triples
    # Vectorized cross product calculation
    # i, j, k indices
    i_idx = triples[:, 0]
    j_idx = triples[:, 1]
    k_idx = triples[:, 2]
    
    pi = points[i_idx]
    pj = points[j_idx]
    pk = points[k_idx]
    
    # Vector from i to j
    v_ij = pj - pi
    # Vector from i to k
    v_ik = pk - pi
    
    # Cross product in 2D: x1*y2 - y1*x2
    crosses = v_ij[:, 0] * v_ik[:, 1] - v_ij[:, 1] * v_ik[:, 0]
    areas = 0.5 * np.abs(crosses)
    
    return areas

def _soft_min_surrogate(areas, beta=10.0):
    """
    A smooth approximation of the minimum of the areas.
    Uses the negative power sum: - (sum(areas^(-p)))^(-1/p) as p -> inf gives -min(areas).
    Here we use a fixed p (beta) for smoothness.
    Actually, to maximize min(area), we can minimize -min(area).
    Surrogate for min(x): (sum(x^(-beta)))^(-1/beta) is a lower bound on min(x) for beta>0?
    Let's check: if x = [1, 100], min=1. (1^-2 + 100^-2)^-0.5 = (1 + 0.0001)^-0.5 ≈ 0.99995. Close.
    If we use a larger beta, it gets closer to the min.
    However, gradients can be unstable.
    Alternative: Log-sum-exp trick for min: -log(sum(exp(-beta*x)))/beta.
    This is smooth and has nice gradients.
    f(x) = -1/beta * log(sum(exp(-beta*x)))
    grad f = exp(-beta*x) / sum(exp(-beta*x))  (normalized softmax weights)
    """
    # To avoid overflow/underflow in exp, we shift by min
    min_val = np.min(areas)
    shifted = -beta * (areas - min_val)
    # exp(shifted) is in [0, 1] since shifted <= 0
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    
    # Value: min_val - (1/beta) * log(sum_exp)
    val = min_val - (1.0 / beta) * np.log(sum_exp)
    
    # Gradient: d/dx_i [ -1/beta * log(sum(exp(-beta*x))) ]
    # = -1/beta * (1/sum_exp) * sum( d/dx_i exp(-beta*x_j) )
    # = -1/beta * (1/sum_exp) * exp(-beta*x_i) * (-beta)
    # = exp(-beta*x_i) / sum_exp
    # This is the softmax probability of x_i.
    grad = exp_shifted / sum_exp
    
    return val, grad

def _objective_and_grad(bary_flat):
    """
    Objective function for L-BFGS-B.
    Input: 1D array of length 2*N.
    Returns: objective value, gradient.
    """
    bary = bary_flat.reshape(N, 2)
    
    # We work in Cartesian space for optimization? 
    # No, L-BFGS-B works better with simple box constraints.
    # Barycentric coordinates have a triangular domain. 
    # It is often better to optimize in Cartesian coordinates with projection/penalty,
    # or use a parameterization that maps R^2N to the simplex.
    # However, the current code uses barycentric. 
    # Let's stick to Cartesian for the optimizer to avoid the non-linear constraint u+v<=1 
    # which makes the gradient of the projection tricky.
    # We will convert bary -> cart, but the optimizer variables will be Cartesian?
    # If we use Cartesian, we need to ensure points stay inside.
    # L-BFGS-B supports box constraints. The triangle is not a box.
    # So we must either:
    # 1. Use barycentric and handle the constraint via penalty or projection (hard for gradients).
    # 2. Use a different parameterization.
    # 3. Optimize in Cartesian and use a barrier/penalty for boundary violations.
    
    # Let's try optimizing in Cartesian coordinates.
    # Variables: x1, y1, x2, y2, ... x11, y11
    # Constraints: Points must be inside the triangle.
    # Inside condition for (x,y):
    # y >= 0
    # y <= sqrt(3) * x  (left edge? No, left edge is x=0 to x=0.5? No, V1(0,0) to V3(0.5, sqrt3/2))
    # Line V1-V3: y = sqrt(3) x. So y <= sqrt(3) x => sqrt(3)x - y >= 0.
    # Line V2-V3: V2(1,0) to V3(0.5, sqrt3/2).
    # Slope = (sqrt3/2 - 0) / (0.5 - 1) = (sqrt3/2) / (-0.5) = -sqrt3.
    # Equation: y - 0 = -sqrt3 (x - 1) => y = -sqrt3 x + sqrt3 => sqrt3 x + y - sqrt3 <= 0.
    # So constraints are:
    # y_i >= 0
    # sqrt(3) x_i - y_i >= 0  => -sqrt(3) x_i + y_i <= 0
    # sqrt(3) x_i + y_i - sqrt(3) <= 0
    
    # L-BFGS-B can handle linear inequality constraints? No, only bounds (box).
    # So we must use a penalty method or transform variables.
    
    # Let's revert to Barycentric but handle the constraint by projecting 
    # and using a penalty in the objective if we go out.
    # But gradients of projection are zero almost everywhere, which stops L-BFGS-B.
    
    # Better strategy: 
    # Use a smooth "soft" constraint or just rely on the fact that 
    # the optimal solution is usually strictly inside or on the boundary.
    # If we optimize in Cartesian, we can use a penalty for being outside.
    
    # Let's switch the optimizer to work on Cartesian coordinates.
    # The input bary_flat is actually treated as Cartesian in this new strategy?
    # No, the function signature is fixed? No, I can change the internal representation 
    # as long as construct() returns Cartesian points.
    # The optimizer variables will be Cartesian (x, y) for each point.
    
    points = bary_flat.reshape(N, 2) # Now treating input as Cartesian
    
    # Penalty for being outside the triangle
    # 1. y < 0
    # 2. y > sqrt(3) x  (i.e., x < y/sqrt3)
    # 3. y > -sqrt(3) x + sqrt(3) (i.e., y + sqrt3 x > sqrt3)
    
    penalty = 0.0
    grad_penalty = np.zeros_like(points)
    
    # 1. y < 0
    neg_y = np.clip(-points[:, 1], 0, None)
    penalty += np.sum(neg_y**2)
    grad_penalty[:, 1] += 2 * neg_y * -1 # d/dy (-y^2) = -2y, but we have (y)^2 where y<0? 
    # Let's define violation v = max(0, -y). Penalty v^2. 
    # d/dy v^2 = 2v * dv/dy. If y<0, v=-y, dv/dy=-1. d/dy = 2(-y)(-1) = 2y.
    # If y>=0, v=0, d/dy=0.
    # So grad_y += 2 * y * (y < 0) ? No, if y<0, 2y is negative. 
    # Let's just use a large penalty factor and simple clipping for now, 
    # but L-BFGS-B needs gradients.
    
    # Let's use a simpler approach: 
    # Optimize in Barycentric coordinates but use a smooth approximation of the indicator function?
    # No, let's stick to the previous plan: 
    # The current code uses Barycentric. 
    # I will implement the Soft-Min on Barycentric coordinates, 
    # but I will handle the simplex constraint by using a penalty term 
    # that is smooth: 
    # P = sum( max(0, -u)^2 + max(0, -v)^2 + max(0, u+v-1)^2 )
    # This is C1 continuous.
    
    bary = bary_flat.reshape(N, 2)
    u = bary[:, 0]
    v = bary[:, 1]
    
    # Penalty terms
    neg_u = np.clip(-u, 0, None)
    neg_v = np.clip(-v, 0, None)
    over_sum = np.clip(u + v - 1.0, 0, None)
    
    penalty_val = 1000.0 * (np.sum(neg_u**2) + np.sum(neg_v**2) + np.sum(over_sum**2))
    
    grad_penalty = np.zeros_like(bary)
    # d/du (neg_u^2) = 2 neg_u * (-1) if u<0 else 0 => -2u if u<0
    grad_penalty[:, 0] += 1000.0 * (-2.0 * u * (u < 0))
    grad_penalty[:, 1] += 1000.0 * (-2.0 * v * (v < 0))
    # d/du (over_sum^2) = 2 over_sum * (1) if u+v>1 else 0
    mask_over = (u + v > 1.0)
    grad_penalty[mask_over, 0] += 1000.0 * (2.0 * over_sum[mask_over])
    grad_penalty[mask_over, 1] += 1000.0 * (2.0 * over_sum[mask_over])
    
    # Convert to Cartesian for area calculation
    points = _bary_to_cart(bary)
    
    # Calculate areas
    areas = _compute_triangle_areas(points)
    
    # Soft min surrogate
    beta = 50.0 # High beta for sharpness
    min_val, grad_min = _soft_min_surrogate(areas, beta)
    
    # We want to MAXIMIZE min_val, so MINIMIZE -min_val.
    obj_val = -min_val + penalty_val
    
    # Gradient of -min_val wrt areas is -grad_min
    # Need chain rule: d(obj)/d(bary) = d(obj)/d(points) * d(points)/d(bary)
    # d(obj)/d(points) comes from d(obj)/d(areas) * d(areas)/d(points)
    
    # This chain rule is complex to implement vectorized.
    # Alternative: Use finite differences for gradient? Too slow.
    # Alternative: Use Nelder-Mead on the Smooth Surrogate?
    # If the function is smooth, NM works better than on the raw min.
    # But L-BFGS-B is much faster if we can get gradients.
    
    # Let's try a different approach: 
    # Use the fact that the optimal configuration often has points on the boundary or symmetric.
    # But we need a general solver.
    
    # Let's use `minimize` with method='Nelder-Mead' but on the SMOOTH surrogate.
    # The smooth surrogate makes the landscape less jagged, allowing NM to converge 
    # faster and avoid local minima that trap it on the raw non-smooth function.
    # We will do multiple restarts.
    
    return obj_val, (grad_penalty + np.zeros_like(bary)).flatten() # Placeholder, we will use NM

# Re-evaluating: Implementing the gradient of the soft-min through the triangle area calculation 
# is doable but error-prone. 
# Given the time constraints, a robust alternative is:
# 1. Use a high-quality initialization (e.g., grid search or structured points).
# 2. Use Nelder-Mead on the RAW function but with a "jitter" or multiple restarts?
#    The current code does this but stagnates.
#    
# Let's try the **Smooth Surrogate with Nelder-Mead**.
# The smoothness allows NM to take larger steps and navigate the valley better.

def _objective_smooth(bary_flat, beta=20.0):
    bary = bary_flat.reshape(N, 2)
    u = bary[:, 0]
    v = bary[:, 1]
    
    # Penalty for out of bounds (smooth)
    neg_u = np.clip(-u, 0, None)
    neg_v = np.clip(-v, 0, None)
    over_sum = np.clip(u + v - 1.0, 0, None)
    penalty = 1000.0 * (np.sum(neg_u**2) + np.sum(neg_v**2) + np.sum(over_sum**2))
    
    points = _bary_to_cart(bary)
    areas = _compute_triangle_areas(points)
    
    # Soft min
    min_val, _ = _soft_min_surrogate(areas, beta)
    
    return -min_val + penalty

def _objective_raw(bary_flat):
    bary = bary_flat.reshape(N, 2)
    bary = _project_to_simplex(bary)
    points = _bary_to_cart(bary)
    min_area = np.min(_compute_triangle_areas(points))
    return -min_area

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_score = -np.inf
    best_bary = None
    
    # Strategy: Multiple restarts of Nelder-Mead on the Smooth Surrogate.
    # Then polish the best result with Nelder-Mead on the Raw Function.
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initial points
    # 1. Grid-like initialization
    # 2. Random initialization
    
    inits = []
    
    # Structured init: vertices, midpoints, centroid, etc.
    # Vertices
    inits.append(np.array([
        [0.0, 0.0], # V1
        [1.0, 0.0], # V2
        [0.5, 0.5], # V3 in bary (0.5, 0.5, 0.0) -> wait, V3 is (0.5, 0.5, 0)? 
        # Bary (u,v,w). V1=(1,0,0), V2=(0,1,0), V3=(0,0,1).
        # My bary_to_cart: x = v + 0.5w, y = sqrt3/2 w.
        # If u=1, v=0, w=0 -> x=0, y=0 (V1). Correct.
        # If u=0, v=1, w=0 -> x=1, y=0 (V2). Correct.
        # If u=0, v=0, w=1 -> x=0.5, y=sqrt3/2 (V3). Correct.
        # So V3 is [0,0] in (u,v).
        [0.0, 0.0], # V3
        [1/3.0, 1/3.0], # Centroid
        [0.5, 0.0], # Mid V1-V2
        [0.25, 0.25], # Mid V1-V3? 
        # Mid V1-V3: (V1+V3)/2 = (0.25, sqrt3/4). 
        # Bary: x=0.25, y=sqrt3/4. 
        # w = 2y/sqrt3 = 0.5. v = x - 0.5w = 0.25 - 0.25 = 0. u = 1 - 0.5 - 0 = 0.5.
        # So [0.5, 0.0] is Mid V1-V3? 
        # Let's check [0.5, 0.0]: u=0.5, v=0, w=0.5. 
        # x = 0 + 0.25 = 0.25. y = sqrt3/4. Yes.
        [0.0, 0.5], # Mid V2-V3? 
        # u=0, v=0.5, w=0.5. x=0.5+0.25=0.75. y=sqrt3/4.
        # Mid V2-V3: (1+0.5)/2=0.75, (0+sqrt3/2)/2=sqrt3/4. Yes.
    ]))
    
    # Add 6 more random points to make 11
    for _ in range(6):
        while True:
            u = rng.uniform(0, 1)
            v = rng.uniform(0, 1)
            if u + v <= 1.0:
                inits[0] = np.vstack([inits[0], [[u, v]]])
                break
                
    # Random inits
    for _ in range(20):
        bary = np.zeros((N, 2))
        for i in range(N):
            while True:
                u = rng.uniform(0, 1)
                v = rng.uniform(0, 1)
                if u + v <= 1.0:
                    bary[i] = [u, v]
                    break
        inits.append(bary)
        
    # Run NM on Smooth Surrogate for each init
    for i, init_bary in enumerate(inits):
        if time.time() > deadline - 20.0:
            break
            
        try:
            res = minimize(
                _objective_smooth, 
                init_bary.flatten(), 
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True}
            )
            
            if time.time() > deadline - 10.0:
                break
                
            # Evaluate on RAW function
            final_bary = _project_to_simplex(res.x.reshape(N, 2))
            final_points = _bary_to_cart(final_bary)
            score = np.min(_compute_triangle_areas(final_points))
            
            if score > best_score:
                best_score = score
                best_bary = final_bary.copy()
                
        except Exception:
            pass

    # Polish the best result with NM on RAW function
    if best_bary is not None and time.time() < deadline - 5.0:
        try:
            res_final = minimize(
                _objective_raw, 
                best_bary.flatten(), 
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
            )
            
            final_bary = _project_to_simplex(res_final.x.reshape(N, 2))
            final_points = _bary_to_cart(final_bary)
            score = np.min(_compute_triangle_areas(final_points))
            
            if score > best_score:
                best_score = score
                best_bary = final_bary.copy()
                
        except Exception:
            pass

    # Final return
    if best_bary is None:
        # Fallback
        best_bary = np.array([[1/3.0, 1/3.0]] * N)
        
    final_bary = _project_to_simplex(best_bary)
    final_points = _bary_to_cart(final_bary)
    
    return final_points

if __name__ == "__main__":
    start = time.time()
    pts = construct()
    elapsed = time.time() - start
    area = np.min(_compute_triangle_areas(pts))
    print(f"Time: {elapsed:.2f}s, Min Area: {area:.6f}")
