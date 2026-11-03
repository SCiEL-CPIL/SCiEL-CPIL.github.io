import time
import numpy as np
from scipy.spatial import ConvexHull

N = 14
N_HALF = 7

# Precompute indices for all combinations of 3 points
_indices = np.arange(N)
IDX_I, IDX_J, IDX_K = np.meshgrid(_indices, _indices, _indices, indexing='ij')
mask = (IDX_I < IDX_J) & (IDX_J < IDX_K)
IDX_I = IDX_I[mask].flatten()
IDX_J = IDX_J[mask].flatten()
IDX_K = IDX_K[mask].flatten()
NUM_TRIS = len(IDX_I)

def _all_areas(points):
    pi = points[IDX_I]
    pj = points[IDX_K] # Wait, standard cross product: (B-A) x (C-A). 
    # Let's stick to the original definition for consistency and speed
    pi = points[IDX_I]
    pj = points[IDX_J]
    pk = points[IDX_K]
    v1x = pj[:, 0] - pi[:, 0]
    v1y = pj[:, 1] - pi[:, 1]
    v2x = pk[:, 0] - pi[:, 0]
    v2y = pk[:, 1] - pi[:, 1]
    areas = 0.5 * np.abs(v1x * v2y - v1y * v2x)
    return areas

def _compute_hull_area(points):
    try:
        if np.allclose(points[:, 0], points[0, 0]) or np.allclose(points[:, 1], points[0, 1]):
            return 0.0
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    hull_area = _compute_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = np.min(_all_areas(points))
    return min_area / hull_area

def _soft_min_and_grad(points, beta=100.0):
    """
    Compute soft-min of triangle areas and its gradient w.r.t points.
    Uses log-sum-exp for stability.
    Returns: soft_min_val, grad (N, 2)
    """
    pi = points[IDX_I]
    pj = points[IDX_J]
    pk = points[IDX_K]
    
    # Signed double area: 2 * Area = | (B-A) x (C-A) |
    # Let D = (B-A) x (C-A) = (xj-xi)(yk-yi) - (yj-yi)(xk-xi)
    # Area = 0.5 * |D|
    
    x_i, y_i = pi[:, 0], pi[:, 1]
    x_j, y_j = pj[:, 0], pj[:, 1]
    x_k, y_k = pk[:, 0], pk[:, 1]
    
    # D = (x_j - x_i)*(y_k - y_i) - (y_j - y_i)*(x_k - x_i)
    # Expand for gradient:
    # D = x_j*y_k - x_j*y_i - x_i*y_k + x_i*y_i - y_j*x_k + y_j*x_i + y_i*x_k - y_i*x_i
    # D = x_j*y_k - x_j*y_i - x_i*y_k + y_j*x_i - y_j*x_k + y_i*x_k
    
    # Gradient of D w.r.t x_i: -y_k + y_j
    # Gradient of D w.r.t y_i: x_k - x_j
    # Gradient of D w.r.t x_j: y_k - y_i
    # Gradient of D w.r.t y_j: -x_k + x_i
    # Gradient of D w.r.t x_k: -y_j + y_i
    # Gradient of D w.r.t y_k: x_j - x_i
    
    d_dx_i = -y_k + y_j
    d_dy_i = x_k - x_j
    d_dx_j = y_k - y_i
    d_dy_j = -x_k + x_i
    d_dx_k = -y_j + y_i
    d_dy_k = x_j - x_i
    
    # We want to maximize the minimum area.
    # Let A = 0.5 * |D|.
    # Soft min of A is roughly soft min of |D|.
    # However, |D| is non-differentiable at D=0.
    # Better: Soft min of A^2? Or just use signed D if we assume orientation?
    # No, orientation can flip.
    
    # Let's use the soft-min of the *squared* areas to ensure differentiability and positivity?
    # Or better, just use the standard soft-min of the absolute values?
    # Actually, the gradient of min(A) is the gradient of the smallest A.
    # Soft min approximates this with a weighted average.
    
    # Let's compute weights w_k for each triangle k based on -beta * A_k.
    # A_k = 0.5 * |D_k|
    
    D = d_dx_i * 0 + d_dy_i * 0 # Placeholder, let's compute D explicitly
    D = (x_j - x_i)*(y_k - y_i) - (y_j - y_i)*(x_k - x_i)
    
    A = 0.5 * np.abs(D)
    
    # Soft min of A:
    # S = -1/beta * log( sum( exp(-beta * A_k) ) )
    # dS/dp = sum( w_k * dA_k/dp ) where w_k = exp(-beta * A_k) / sum(exp(-beta * A_j))
    
    z = -beta * A
    max_z = np.max(z)
    # Handle -inf case
    if max_z < -700:
        # All A are huge, gradient is 0
        return 0.0, np.zeros_like(points)
    
    exp_z = np.exp(z - max_z)
    sum_exp = np.sum(exp_z)
    weights = exp_z / sum_exp
    
    soft_min_val = -(max_z + np.log(sum_exp)) / beta
    
    # Gradient of A_k w.r.t points
    # dA_k/dx_i = 0.5 * sign(D_k) * d_dx_i
    # etc.
    
    sign_D = np.sign(D)
    # If D is 0, sign is 0, which is fine (gradient undefined, but weight likely low or spread)
    
    grad_x_i = 0.5 * sign_D * d_dx_i
    grad_y_i = 0.5 * sign_D * d_dy_i
    grad_x_j = 0.5 * sign_D * d_dx_j
    grad_y_j = 0.5 * sign_D * d_dy_j
    grad_x_k = 0.5 * sign_D * d_dx_k
    grad_y_k = 0.5 * sign_D * d_dy_k
    
    # Accumulate gradient
    grad = np.zeros_like(points)
    
    # Scatter add
    np.add.at(grad, IDX_I, np.column_stack([grad_x_i, grad_y_i]) * weights[:, np.newaxis])
    np.add.at(grad, IDX_J, np.column_stack([grad_x_j, grad_y_j]) * weights[:, np.newaxis])
    np.add.at(grad, IDX_K, np.column_stack([grad_x_k, grad_y_k]) * weights[:, np.newaxis])
    
    return soft_min_val, grad

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_points = None
    best_score = -np.inf
    
    def update_best(pts, score):
        nonlocal best_points, best_score
        if score > best_score:
            best_score = score
            best_points = pts.copy()

    def _normalize(pts):
        center = np.mean(pts, axis=0)
        pts = pts - center
        # Scale such that convex hull area is ~1? 
        # Or just normalize to unit circle?
        # The score is invariant to scaling. 
        # Let's normalize to fit in [-1, 1] roughly to keep numbers stable.
        max_norm = np.max(np.linalg.norm(pts, axis=1))
        if max_norm > 0:
            pts = pts / max_norm
        return pts

    def hill_climb(pts_init, max_iter=500, beta=100.0):
        """
        Adaptive radius hill climbing using soft-min gradient.
        """
        pts = pts_init.copy()
        step_size = 0.1
        best_local_score = -np.inf
        best_local_pts = pts.copy()
        
        for i in range(max_iter):
            if time.time() > deadline - 5.0:
                break
                
            # Compute score and gradient
            # We want to maximize min_area / hull_area.
            # Let f = min_area. g = hull_area. Maximize f/g.
            # Gradient of f/g = (grad_f * g - f * grad_g) / g^2.
            # Computing grad_g (hull area) is hard. 
            # Instead, let's just maximize f (min area) and assume hull area is roughly constant 
            # or we normalize points at each step?
            # If we normalize points at each step, the gradient calculation is invalid 
            # because the normalization changes the points.
            
            # Alternative: Work in a fixed scale. 
            # If we start with points in [-1,1], and just push them, they might expand.
            # We can re-normalize every few steps or after a successful step?
            # If we re-normalize, we break the gradient flow.
            
            # Let's try: Maximize soft_min_area. 
            # The ratio min_area/hull_area is scale invariant.
            # If we just maximize min_area, the points will expand to infinity.
            # So we MUST constrain the scale.
            
            # Strategy: 
            # 1. Compute grad of soft_min_area.
            # 2. Compute current hull area.
            # 3. Project gradient to remove radial component? 
            #    Or simpler: After a step, re-normalize to unit hull area?
            #    If we re-normalize, we lose the direction of the step.
            
            # Better Strategy: 
            # Use the ratio directly? No, hull area gradient is discontinuous.
            
            # Let's stick to: 
            # 1. Compute grad of soft_min_area.
            # 2. Take a step: pts_new = pts + step_size * grad.
            # 3. Re-normalize pts_new to have unit convex hull area?
            #    No, that's a non-linear map.
            
            # Let's try a different objective: 
            # Maximize min_area subject to hull_area = 1.
            # Lagrangian: L = min_area - lambda * (hull_area - 1).
            # Still need grad hull_area.
            
            # Practical heuristic:
            # The score is scale invariant. 
            # If we just maximize min_area, the points expand.
            # We can constrain the points to lie on a sphere of radius R?
            # No, they are in a convex region.
            
            # Let's try: 
            # Maximize min_area / (mean distance to center)^2 ?
            # Or just: 
            # 1. Compute grad of soft_min_area.
            # 2. Normalize points to unit norm (centered).
            # 3. Project gradient onto the tangent space of the normalization manifold?
            #    This is getting complicated.
            
            # Simplest robust approach:
            # 1. Compute grad of soft_min_area.
            # 2. Step: pts_new = pts + alpha * grad.
            # 3. If hull_area(pts_new) < hull_area(pts) * 0.9 or > hull_area(pts) * 1.1:
            #    Re-normalize pts_new to match the original hull area scale?
            #    Actually, just re-normalize to unit hull area every step?
            #    If we re-normalize, the "step" is effectively:
            #    pts_new = normalize(pts + alpha * grad)
            #    This is a valid optimization step on the scale-invariant manifold.
            
            hull_area_curr = _compute_hull_area(pts)
            if hull_area_curr < 1e-10:
                return pts, -np.inf
            
            soft_min, grad = _soft_min_and_grad(pts, beta=beta)
            
            # Normalize gradient to unit length to control step size
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-12:
                break
            grad_unit = grad / grad_norm
            
            # Try step
            pts_new = pts + step_size * grad_unit
            
            # Re-normalize to keep scale stable
            # Center
            center = np.mean(pts_new, axis=0)
            pts_new = pts_new - center
            # Scale to match original hull area?
            # Or just scale to unit max norm?
            # Let's scale to unit max norm for stability.
            max_norm_new = np.max(np.linalg.norm(pts_new, axis=1))
            if max_norm_new > 0:
                pts_new = pts_new / max_norm_new
            
            # Evaluate score
            score_new = _score(pts_new)
            
            if score_new > best_local_score:
                best_local_score = score_new
                best_local_pts = pts_new.copy()
                pts = pts_new
                # Increase step size if we improved significantly
                step_size *= 1.1
                if step_size > 0.5:
                    step_size = 0.5
            else:
                # Decrease step size
                step_size *= 0.8
                if step_size < 1e-4:
                    break
                
                # If we fail, revert
                pts = best_local_pts
        
        return best_local_pts, best_local_score

    # --- Phase 1: Generate diverse initial candidates ---
    # 1. Regular 14-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    pts_reg = np.column_stack((np.cos(angles), np.sin(angles)))
    pts_reg = _normalize(pts_reg)
    update_best(pts_reg, _score(pts_reg))
    
    # 2. Two concentric circles (7 inner, 7 outer)
    angles_in = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    angles_out = angles_in + np.pi/7
    r_in, r_out = 0.5, 1.0
    pts_in = r_in * np.column_stack((np.cos(angles_in), np.sin(angles_in)))
    pts_out = r_out * np.column_stack((np.cos(angles_out), np.sin(angles_out)))
    pts_two = np.vstack([pts_in, pts_out])
    pts_two = _normalize(pts_two)
    update_best(pts_two, _score(pts_two))
    
    # 3. Random points
    rng = np.random.default_rng(42)
    for i in range(5):
        pts_rand = rng.uniform(-1, 1, (N, 2))
        pts_rand = _normalize(pts_rand)
        update_best(pts_rand, _score(pts_rand))
        
    # --- Phase 2: Hill Climb from best candidates ---
    # We will run hill climbing from the current best, and also from some random perturbations of it.
    
    if best_points is not None:
        # Refine the best found so far
        refined_pts, refined_score = hill_climb(best_points, max_iter=1000, beta=100.0)
        if refined_score > best_score:
            update_best(refined_pts, refined_score)
            
        # Try different betas
        for beta in [50.0, 200.0, 500.0]:
            if time.time() > deadline - 20.0:
                break
            refined_pts, refined_score = hill_climb(best_points, max_iter=500, beta=beta)
            if refined_score > best_score:
                update_best(refined_pts, refined_score)
                
        # Multi-start: perturb best points and re-climb
        for i in range(10):
            if time.time() > deadline - 10.0:
                break
            noise = rng.normal(0, 0.1, (N, 2))
            pts_perturbed = best_points + noise
            pts_perturbed = _normalize(pts_perturbed)
            refined_pts, refined_score = hill_climb(pts_perturbed, max_iter=300, beta=100.0)
            if refined_score > best_score:
                update_best(refined_pts, refined_score)

    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))

    return best_points
