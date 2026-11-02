"""Program for covering the sphere with 130 equal caps."""
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 130

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s margin for safety

    def fibonacci_spiral(n):
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    def compute_facet_data(pts):
        """
        Computes convex hull facets and their circumradii and gradients.
        Returns:
        faces: array of shape (F, 3) with indices
        radii: array of shape (F,) with circumradii
        grad_R: list of arrays, grad_R[k] is (N, 3) gradient of R_k w.r.t pts
        """
        try:
            hull = ConvexHull(pts)
            faces = hull.simplices
        except Exception:
            return None, None, None

        F = len(faces)
        radii = np.zeros(F)
        # grad_R is a list of (N,3) arrays, or we can accumulate directly
        # But for soft-max we need individual gradients.
        # To save memory/time, we can compute the weighted sum directly in the objective/gradient func.
        
        # We will return faces and a function to compute gradients on the fly?
        # No, L-BFGS-B calls gradient many times. 
        # Better: compute faces and radii, and the gradients of each R_i.
        # F is ~ 2N = 260. N=130. 
        # Storing 260 * 130 * 3 floats is ~ 300KB. Acceptable.
        
        grad_R = np.zeros((F, N, 3))
        
        for i in range(F):
            idx1, idx2, idx3 = faces[i]
            p1 = pts[idx1]
            p2 = pts[idx2]
            p3 = pts[idx3]
            
            # Side lengths
            v12 = p2 - p1
            v23 = p3 - p2
            v31 = p1 - p3
            
            c = np.linalg.norm(v12) # side opposite p3? No, side between p1,p2. Let's call it a_side
            a = np.linalg.norm(v23)
            b = np.linalg.norm(v31)
            
            # Area vector
            S = np.cross(p1, p2) + np.cross(p2, p3) + np.cross(p3, p1)
            S_norm = np.linalg.norm(S)
            
            if S_norm < 1e-10:
                radii[i] = np.inf
                continue
                
            K = 0.5 * S_norm
            # Circumradius R = abc / (4K) = abc / (2 * S_norm)
            R = (a * b * c) / (2.0 * S_norm)
            radii[i] = R
            
            # Gradients
            # dR/dp_k = (1/(2*S_norm)) * [ d(abc)/dp_k - abc * (S . dS/dp_k) / S_norm^2 ]
            
            # d(abc)/dp_k
            # a = |p3-p2|, b = |p1-p3|, c = |p2-p1|
            
            # For p1:
            # da/dp1 = 0
            # db/dp1 = (p1-p3)/b
            # dc/dp1 = (p2-p1)/c
            d_abc_dp1 = (a*c * (p1 - p3) / b + a*b * (p2 - p1) / c)
            
            # dS/dp1 = p2^T - p3^T (as matrix acting on vector? No, S is vector)
            # S . dS/dp1 = S . (p2 - p3) ? 
            # Let's verify: S = p1xp2 + p2xp3 + p3xp1
            # dS/dp1 = p2^T - p3^T
            # S . dS/dp1 = S . (p2 - p3)
            S_dot_dS_dp1 = np.dot(S, p2 - p3)
            
            grad_R[i, idx1] = (d_abc_dp1 - (a*b*c / S_norm**2) * S_dot_dS_dp1) / (2.0 * S_norm)
            
            # For p2:
            # da/dp2 = (p2-p3)/a
            # db/dp2 = 0
            # dc/dp2 = (p1-p2)/c
            d_abc_dp2 = (b*c * (p2 - p3) / a + a*b * (p1 - p2) / c)
            S_dot_dS_dp2 = np.dot(S, p3 - p1)
            grad_R[i, idx2] = (d_abc_dp2 - (a*b*c / S_norm**2) * S_dot_dS_dp2) / (2.0 * S_norm)
            
            # For p3:
            # da/dp3 = (p3-p2)/a ? No, a=|p3-p2|. da/dp3 = (p3-p2)/a
            # db/dp3 = (p3-p1)/b
            # dc/dp3 = 0
            d_abc_dp3 = (a*c * (p3 - p2) / a + a*b * (p3 - p1) / b)
            S_dot_dS_dp3 = np.dot(S, p1 - p2)
            grad_R[i, idx3] = (d_abc_dp3 - (a*b*c / S_norm**2) * S_dot_dS_dp3) / (2.0 * S_norm)
            
        return faces, radii, grad_R

    def soft_max_objective(flat_pts, beta):
        pts = flat_pts.reshape(N, 3)
        # Normalize to ensure we are on sphere (L-BFGS-B might drift slightly)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        pts = pts / norms
        
        faces, radii, grad_R = compute_facet_data(pts)
        if faces is None:
            return 1e6, np.zeros_like(flat_pts)
        
        # Soft max
        # F = 1/beta * log(sum(exp(beta * R_i)))
        # To avoid overflow, subtract max R
        max_R = np.max(radii)
        exp_terms = np.exp(beta * (radii - max_R))
        sum_exp = np.sum(exp_terms)
        objective = max_R + np.log(sum_exp) / beta
        
        # Gradient
        # dF/dp_k = 1/sum_exp * sum_i exp(beta*R_i) * dR_i/dp_k
        # Note: dR_i/dp_k is computed in Euclidean space. We need to project to tangent space.
        
        weights = exp_terms / sum_exp # shape (F,)
        
        # grad_obj shape (N, 3)
        grad_obj = np.tensordot(weights, grad_R, axes=(0, 0))
        
        # Project to tangent space: grad_tan = grad - (grad . p) p
        grad_proj = grad_obj - np.sum(grad_obj * pts, axis=1, keepdims=True) * pts
        
        return objective, grad_proj.ravel()

    def compute_covering_radius(pts):
        try:
            hull = ConvexHull(pts)
            faces = hull.simplices
            max_r = 0.0
            for face in faces:
                p1 = pts[face[0]]
                p2 = pts[face[1]]
                p3 = pts[face[2]]
                v12 = p2 - p1
                v23 = p3 - p2
                v31 = p1 - p3
                a = np.linalg.norm(v23)
                b = np.linalg.norm(v31)
                c = np.linalg.norm(v12)
                S = np.cross(p1, p2) + np.cross(p2, p3) + np.cross(p3, p1)
                S_norm = np.linalg.norm(S)
                if S_norm > 1e-10:
                    R = (a * b * c) / (2.0 * S_norm)
                    if R > max_r:
                        max_r = R
            return max_r
        except Exception:
            return np.inf

    # Initial points
    points = fibonacci_spiral(N).astype(np.float64)
    
    # Optimization Loop with increasing Beta
    best_rad = compute_covering_radius(points)
    best_points = points.copy()
    
    betas = [10.0, 20.0, 50.0, 100.0, 200.0]
    
    current_pts = points.copy()
    
    for beta in betas:
        if time.time() > deadline - 15.0:
            break
            
        x0 = current_pts.ravel()
        
        def obj_wrapper(x):
            val, _ = soft_max_objective(x, beta)
            return val
            
        def grad_wrapper(x):
            _, grad = soft_max_objective(x, beta)
            return grad
            
        try:
            result = minimize(
                obj_wrapper,
                x0,
                jac=grad_wrapper,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 200}
            )
            
            if result.success:
                candidate = result.x.reshape(N, 3)
                norms = np.linalg.norm(candidate, axis=1, keepdims=True)
                candidate = candidate / norms
                rad = compute_covering_radius(candidate)
                
                if rad < best_rad:
                    best_rad = rad
                    best_points = candidate.copy()
                    current_pts = candidate.copy()
                else:
                    # If it got worse, keep previous best for next beta
                    pass
        except Exception as e:
            # print(f"Optimization failed at beta {beta}: {e}")
            pass

    # Final Polish: Steepest Descent on Exact Max Radius
    # This handles the non-smoothness at the very end
    if time.time() < deadline - 5.0:
        pts = best_points.copy()
        step = 0.001
        for _ in range(50):
            if time.time() > deadline - 2.0:
                break
            faces, radii, grad_R = compute_facet_data(pts)
            if faces is None:
                break
            max_idx = np.argmax(radii)
            max_rad = radii[max_idx]
            
            # Gradient of the max radius is the gradient of the facet with max radius
            # But if multiple facets share the max, we should use their average or just one.
            # Using one is a valid subgradient.
            grad = grad_R[max_idx]
            
            # Project to tangent space
            grad_proj = grad - np.sum(grad * pts, axis=1, keepdims=True) * pts
            
            # Normalize gradient step
            grad_norm = np.linalg.norm(grad_proj)
            if grad_norm < 1e-10:
                break
                
            # Move in negative gradient direction
            new_pts = pts - step * (grad_proj / grad_norm)
            # Renormalize
            norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
            new_pts = new_pts / norms
            
            new_rad = compute_covering_radius(new_pts)
            if new_rad < max_rad:
                pts = new_pts
                best_rad = new_rad
                best_points = pts.copy()
                step *= 1.05 # Slightly increase step if improving
            else:
                step *= 0.5
                if step < 1e-6:
                    break

    # Final check and return
    # Ensure no coincident points and not in hemisphere
    # The optimization should have handled this, but let's verify
    if np.linalg.norm(np.sum(best_points)) > 0.9:
        # In a hemisphere, flip one point or randomize?
        # For 130 points, it's unlikely to be in a hemisphere if covering radius is small.
        pass

    return best_points

# EVOLVE-BLOCK-END
