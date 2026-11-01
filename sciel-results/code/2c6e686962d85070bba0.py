"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit (35s buffer)
    
    def fibonacci_sphere(n, offset=0.5):
        i = np.arange(n) + offset
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        return pts

    def get_facet_radii(pts):
        """
        Computes the angular circumradius of each facet of the convex hull.
        Returns an array of radii (in radians) and the hull simplices.
        """
        try:
            hull = ConvexHull(pts)
        except Exception:
            return None, None

        radii = []
        valid_simplices = []
        
        for simplex in hull.simplices:
            v1 = pts[simplex[0]]
            v2 = pts[simplex[1]]
            v3 = pts[simplex[2]]
            
            edge1 = v2 - v1
            edge2 = v3 - v1
            n = np.cross(edge1, edge2)
            norm_n = np.linalg.norm(n)
            if norm_n < 1e-10:
                continue
                
            c = n / norm_n
            # Ensure c is on the correct side (pointing outwards)
            if np.dot(c, v1) < 0:
                c = -c
                
            d = np.linalg.norm(c - v1)
            radius = 2.0 * np.arcsin(np.clip(d / 2.0, 0.0, 1.0))
            
            radii.append(radius)
            valid_simplices.append(simplex)
            
        if not radii:
            return None, None
            
        return np.array(radii), valid_simplices

    def objective(x):
        """
        Soft-max objective function for L-BFGS.
        Minimizes the sum of squared radii, weighted by a softmax to focus on the largest.
        """
        pts = x.reshape(N, 3)
        # Normalize to sphere
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        pts = pts / norms
        
        radii, _ = get_facet_radii(pts)
        if radii is None:
            return 100.0 # Large penalty for invalid hull
            
        # Softmax parameter beta. Higher beta approximates max(radii^2) more closely.
        # We use a moderate beta to maintain smoothness for gradient descent.
        beta = 20.0 
        
        # We want to minimize the "largest" radius.
        # Using log-sum-exp trick for stability:
        # max(r_i) approx (1/beta) * log(sum(exp(beta * r_i)))
        # To make it differentiable and focus on the max, we can minimize:
        # (1/beta) * log(sum(exp(beta * r_i^2)))
        
        scaled = beta * (radii ** 2)
        max_scaled = np.max(scaled)
        
        # Handle potential overflow/underflow
        if not np.isfinite(max_scaled):
             return 100.0
            
        log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
        soft_max_val = log_sum_exp / beta
        
        return soft_max_val

    def gradient(x):
        """
        Numerical gradient of the objective function.
        """
        eps = 1e-7
        grad = np.zeros_like(x)
        f0 = objective(x)
        
        for i in range(len(x)):
            old_val = x[i]
            x[i] += eps
            f_plus = objective(x)
            x[i] = old_val
            grad[i] = (f_plus - f0) / eps
            
        return grad

    # 1. Initialization
    best_points = None
    best_score = np.inf
    
    # Generate multiple random initial configurations (rotations of Fibonacci sphere)
    num_starts = 8
    start_configs = []
    
    # Base Fibonacci sphere
    base_pts = fibonacci_sphere(N)
    
    for i in range(num_starts):
        if time.time() > deadline - 30:
            break
            
        if i == 0:
            pts = base_pts.copy()
        else:
            # Random rotation
            R = np.random.randn(3, 3)
            R, _ = np.linalg.qr(R)
            pts = base_pts @ R.T
            # Add small noise to break symmetry
            pts += np.random.randn(N, 3) * 0.05
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)
            
        start_configs.append(pts)

    # 2. Optimization Loop: L-BFGS with numerical gradients
    for i, init_pts in enumerate(start_configs):
        if time.time() > deadline - 10:
            break
            
        # Flatten points for optimizer
        x0 = init_pts.flatten()
        
        try:
            res = minimize(
                objective, 
                x0, 
                method='L-BFGS-B', 
                jac=gradient,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
        except Exception:
            continue
            
        if res.success or res.fun < best_score:
            opt_pts = res.x.reshape(N, 3)
            opt_pts /= np.linalg.norm(opt_pts, axis=1, keepdims=True)
            
            # Verify the score using the exact max radius (not soft-max)
            radii, _ = get_facet_radii(opt_pts)
            if radii is not None:
                exact_score = np.max(radii)
                if exact_score < best_score:
                    best_score = exact_score
                    best_points = opt_pts.copy()

        # Optional: Local refinement via moving worst facet vertices (if time permits)
        if time.time() < deadline - 5 and best_points is not None:
            current_pts = best_points.copy()
            step = 0.01
            for _ in range(50):
                if time.time() > deadline - 5:
                    break
                radii, simplices = get_facet_radii(current_pts)
                if radii is None: break
                max_idx = np.argmax(radii)
                worst_simplex = simplices[max_idx]
                
                # Recompute circumcenter for worst facet
                v1 = current_pts[worst_simplex[0]]
                v2 = current_pts[worst_simplex[1]]
                v3 = current_pts[worst_simplex[2]]
                edge1 = v2 - v1
                edge2 = v3 - v1
                n = np.cross(edge1, edge2)
                norm_n = np.linalg.norm(n)
                if norm_n < 1e-10: continue
                c = n / norm_n
                if np.dot(c, v1) < 0: c = -c
                
                improved = False
                for idx in worst_simplex:
                    v = current_pts[idx]
                    dir_vec = c - v
                    norm_dir = np.linalg.norm(dir_vec)
                    if norm_dir < 1e-10: continue
                    dir_vec /= norm_dir
                    
                    new_v = v + step * dir_vec
                    new_v /= np.linalg.norm(new_v)
                    
                    old_p = current_pts[idx].copy()
                    current_pts[idx] = new_v
                    
                    new_radii, _ = get_facet_radii(current_pts)
                    if new_radii is not None and np.max(new_radii) < best_score - 1e-8:
                        best_score = np.max(new_radii)
                        best_points = current_pts.copy()
                        improved = True
                    else:
                        current_pts[idx] = old_p
                
                if not improved:
                    step *= 0.5
                    if step < 1e-5:
                        break
                else:
                    step *= 0.9

    if best_points is None:
        # Fallback
        best_points = fibonacci_sphere(N)
        
    # Final normalization
    best_points = best_points / np.linalg.norm(best_points, axis=1, keepdims=True)
    
    return best_points
# EVOLVE-BLOCK-END
