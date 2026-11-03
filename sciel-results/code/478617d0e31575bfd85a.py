from scipy.optimize import minimize
import numpy as np
from scipy.spatial import ConvexHull
import time

N = 50

def construct():
    start_time = time.time()
    deadline = 140.0 # Safety margin
    
    # 1. Initial Configuration: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # 2. Define Objective Function
    def objective(params):
        # params is a flat array of shape (3*N,)
        pts = params.reshape(N, 3)
        # Normalize to unit sphere
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        # Avoid division by zero
        norms[norms == 0] = 1.0
        pts = pts / norms
        
        # Check for coincident points (invalid)
        if np.min(np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=2)) < 1e-6:
            return 1e6
            
        # Compute Convex Hull
        try:
            hull = ConvexHull(pts)
        except Exception:
            return 1e6
            
        # The facets of the convex hull are the Delaunay triangles
        # We need to compute the circumradius for each facet
        max_circumradius = 0.0
        
        for simplex in hull.simplices:
            # simplex is a tuple of 3 indices
            p1, p2, p3 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
            
            # Compute circumcenter of the triangle on the sphere
            # The circumcenter is the intersection of the sphere with the plane
            # perpendicular to the triangle's normal passing through the origin?
            # No, the circumcenter of a spherical triangle is the point on the sphere
            # equidistant from the 3 vertices.
            # It is the normalization of the cross product of (p1 x p2) x (p2 x p3)?
            # Actually, the circumcenter C satisfies C . p1 = C . p2 = C . p3.
            # C is in the span of p1, p2, p3.
            # C = a*p1 + b*p2 + c*p3
            # This is equivalent to finding the null space of a matrix.
            
            # A simpler formula for spherical circumcenter:
            # C is proportional to p1 x p2 + p2 x p3 + p3 x p1 ? No.
            # Let's solve the linear system.
            # We want C such that C . p1 = C . p2 = C . p3 and ||C||=1.
            # This implies C is orthogonal to (p1-p2) and (p2-p3).
            # So C is parallel to (p1-p2) x (p2-p3).
            
            v1 = p1 - p2
            v2 = p2 - p3
            c_vec = np.cross(v1, v2)
            
            if np.linalg.norm(c_vec) < 1e-10:
                # Degenerate triangle, points are collinear or coincident
                # This shouldn't happen often, but if it does, the circumradius is large
                # or undefined. We approximate by the max distance from midpoint.
                mid = (p1 + p2 + p3) / 3.0
                mid /= np.linalg.norm(mid)
                dist = np.arccos(np.clip(np.dot(mid, p1), -1.0, 1.0))
                if dist > max_circumradius:
                    max_circumradius = dist
                continue
                
            c_vec /= np.linalg.norm(c_vec)
            
            # The circumcenter is c_vec or -c_vec. We need the one that is "inside" the triangle?
            # Actually, for a triangle on the sphere, the circumcenter is unique.
            # We check which one is closer to the vertices? No, both are equidistant.
            # The covering radius is the distance from the circumcenter to the vertices.
            # This distance is the same for both c_vec and -c_vec?
            # No, the angular distance is min(arccos(c_vec.p1), arccos((-c_vec).p1)).
            # But the circumcenter is the center of the circle passing through p1, p2, p3.
            # The radius is the angular distance.
            
            # Let's calculate the distance from c_vec to p1.
            cos_dist = np.clip(np.dot(c_vec, p1), -1.0, 1.0)
            dist = np.arccos(cos_dist)
            
            # If the triangle is "large" (contains the antipode?), the circumcenter might be the other way.
            # But for covering, we are looking for the largest empty cap.
            # The center of the empty cap is the circumcenter.
            # If the triangle is acute, the circumcenter is inside.
            # If obtuse, it's outside.
            # In either case, the radius of the circumcircle is the distance.
            
            if dist > max_circumradius:
                max_circumradius = dist
                
        # Convert to degrees for scoring consistency, though minimization is scale-invariant
        return np.degrees(max_circumradius)

    # 3. Optimization
    # We use Nelder-Mead because it doesn't require gradients and handles non-smoothness.
    # We run multiple restarts if time permits.
    
    best_params = points.flatten()
    best_score = objective(best_params)
    
    # Nelder-Mead settings
    # maxiter is limited by time.
    # 150 variables. Initial simplex size is important.
    
    def run_optimization(params, max_time):
        from scipy.optimize import minimize
        start_t = time.time()
        
        # Use a small initial step size
        # The points are on the unit sphere, so coordinates are in [-1, 1].
        # A step of 0.1 is reasonable.
        
        def obj_wrapper(x):
            if time.time() - start_t > max_time:
                raise TimeoutError()
            return objective(x)
            
        try:
            res = minimize(obj_wrapper, params, method='Nelder-Mead', 
                           options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-8, 'initial_simplex': None})
            return res.x, res.fun
        except TimeoutError:
            return None, None
        except Exception:
            return None, None

    # Run optimization
    # We can't easily pass a time limit to minimize directly, so we rely on maxiter
    # or wrap the objective. The wrapper above raises an exception, which minimize catches?
    # No, minimize propagates exceptions. We need to handle it outside or use a timeout mechanism.
    # For simplicity, we limit maxiter. 10000 iterations of Nelder-Mead on 150D might be too slow.
    # Let's reduce maxiter to 2000 and see.
    
    # Actually, a better approach for speed:
    # Use L-BFGS-B on a smooth approximation?
    # Let's stick to Nelder-Mead but with a tighter loop.
    
    # Alternative: Use a simple gradient descent with projection?
    # The gradient of the max circumradius is the gradient of the active constraint.
    
    # Let's try a few restarts with Nelder-Mead.
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        if attempts == 1:
            current_params = points.flatten()
        else:
            # Perturb the best solution found so far
            noise = np.random.normal(0, 0.1, N * 3)
            current_params = best_params + noise
            
        # Run Nelder-Mead
        # We estimate that 1000 iterations is a good balance.
        # If it's too slow, we will hit the deadline in the objective function.
        
        def obj_with_timeout(x):
            if time.time() > deadline - 5:
                raise ValueError("Time out")
            return objective(x)
            
        try:
            res = minimize(obj_with_timeout, current_params, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except ValueError:
            break
        except Exception:
            continue
            
    # Final normalization and return
    final_points = best_params.reshape(N, 3)
    final_points /= np.linalg.norm(final_points, axis=1, keepdims=True)
    
    return final_points
