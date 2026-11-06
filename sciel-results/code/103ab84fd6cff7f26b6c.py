import numpy as np
import time
import sys
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import math

N = 130

def _normalize(v):
    """Normalize vectors to unit length."""
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    # Handle zero norms just in case
    norms[norms == 0] = 1.0
    return v / norms

def _thomson_initialization(n):
    """
    Initialize points using Thomson problem heuristic (electrostatic repulsion).
    This provides a much better starting distribution than Fibonacci spiral
    for covering problems, as it minimizes large gaps early on.
    """
    # Start with random points or Fibonacci, then run a few steps of electrostatic repulsion
    # to settle into a low-energy state.
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts = _normalize(pts)
    
    # Run a few steps of simple gradient descent on electrostatic potential
    # U = sum 1/r_ij
    # Force on i = sum (p_i - p_j) / r_ij^3
    for _ in range(50):
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dist_sq, 1e-6)
        dist = np.sqrt(dist_sq)
        # Force magnitude 1/r^3, direction diff
        # We want to minimize U, so move against gradient.
        # Gradient of 1/r is -1/r^2 * (p_i-p_j)/r = -(p_i-p_j)/r^3
        # So force (negative gradient) is (p_i-p_j)/r^3
        force_mag = 1.0 / (dist_sq * dist)
        np.fill_diagonal(force_mag, 0.0)
        forces = np.sum(diff * force_mag[:, :, np.newaxis], axis=1)
        
        # Project to tangent plane
        dot = np.sum(forces * pts, axis=1, keepdims=True)
        forces_tangent = forces - dot * pts
        
        pts = pts + 0.01 * forces_tangent
        pts = _normalize(pts)
        
    return pts

def _covering_radius_deg(pts):
    """
    Calculate the covering radius in degrees.
    The covering radius is the maximum distance from any point on the sphere
    to the nearest configuration point.
    This is equal to the maximum circumradius of the facets of the convex hull.
    """
    try:
        hull = ConvexHull(pts)
        # Get facet indices
        facets = hull.simplices
        
        max_circumradius_deg = 0.0
        
        # We can speed this up by checking only a subset if time is tight,
        # but N=130 is small enough that checking all facets is fast (~1000 facets)
        for facet in facets:
            # Get the 3 points of the facet
            p1, p2, p3 = pts[facet[0]], pts[facet[1]], pts[facet[2]]
            
            # Calculate circumcenter of the spherical triangle
            # The circumcenter is the point on the sphere equidistant from p1, p2, p3.
            # It lies in the plane defined by p1, p2, p3.
            # We can find the normal to the plane n = (p2-p1) x (p3-p1)
            # The circumcenter c is in the direction of the sum of the points weighted by opposite side lengths?
            # Actually, for a spherical triangle, the circumcenter is the intersection of the perpendicular bisectors.
            # A simpler geometric approach:
            # The circumcenter C satisfies C . p1 = C . p2 = C . p3 = cos(R)
            # C is a linear combination of p1, p2, p3: C = a*p1 + b*p2 + c*p3
            # Due to symmetry, if the triangle is isosceles, etc.
            # General solution:
            # Let M be the matrix with rows p1, p2, p3.
            # We want C such that M C = [cos R, cos R, cos R]^T
            # C = M^{-1} [cos R, cos R, cos R]^T ? No, C is in the span of p1,p2,p3.
            # Let C = x1 p1 + x2 p2 + x3 p3.
            # p1 . C = x1 (p1.p1) + x2 (p1.p2) + x3 (p1.p3) = x1 + x2 d12 + x3 d13
            # p2 . C = x1 d12 + x2 + x3 d23
            # p3 . C = x1 d13 + x2 d23 + x3
            # We want p1.C = p2.C = p3.C = k
            # This is a system of 3 linear equations for x1, x2, x3 in terms of k.
            # Actually, we can just solve for the direction of C.
            # The vector C is orthogonal to (p1-p2) and (p2-p3)? No.
            # (C-p1) is perp to (p1+p2)? No.
            # |C-p1|^2 = |C-p2|^2 => 1 - 2 C.p1 = 1 - 2 C.p2 => C.p1 = C.p2.
            # So C is in the intersection of the hyperplanes p1.C = p2.C and p2.C = p3.C.
            # This means C is orthogonal to (p1-p2) and (p2-p3).
            # So C is parallel to the cross product (p1-p2) x (p2-p3).
            
            v1 = p1 - p2
            v2 = p2 - p3
            c_vec = np.cross(v1, v2)
            
            if np.linalg.norm(c_vec) < 1e-10:
                # Degenerate facet (points nearly collinear or identical)
                # In this case, the circumradius is likely large or the facet is invalid.
                # We can estimate the radius as the max distance to the 3 points.
                # Or just return a large value to penalize.
                # Actually, if points are collinear, the "facet" is not a proper triangle.
                # The convex hull algorithm should handle this, but let's be safe.
                # The covering radius for this region is determined by the largest gap.
                # Approximation: max distance to the 3 points.
                # For a point on the sphere, the max distance to the set {p1,p2,p3} is at the antipode of the centroid?
                # Let's just compute the distance from the centroid of the triangle to the points.
                centroid = _normalize((p1+p2+p3)/3.0)
                d1 = np.arccos(np.clip(np.dot(centroid, p1), -1.0, 1.0))
                d2 = np.arccos(np.clip(np.dot(centroid, p2), -1.0, 1.0))
                d3 = np.arccos(np.clip(np.dot(centroid, p3), -1.0, 1.0))
                r_rad = np.max([d1, d2, d3])
                r_deg = np.degrees(r_rad)
                if r_deg > max_circumradius_deg:
                    max_circumradius_deg = r_deg
                continue
                
            c_vec = _normalize(c_vec)
            
            # There are two possible circumcenters (c_vec and -c_vec).
            # We want the one "inside" the triangle, i.e., closest to the points.
            # Check which one has smaller distance to p1.
            d1 = np.arccos(np.clip(np.dot(c_vec, p1), -1.0, 1.0))
            d2 = np.arccos(np.clip(np.dot(-c_vec, p1), -1.0, 1.0))
            
            if d1 < d2:
                c = c_vec
            else:
                c = -c_vec
            
            # The covering radius for this facet is the angular distance from the circumcenter to any vertex.
            r_rad = d1
            r_deg = np.degrees(r_rad)
            
            if r_deg > max_circumradius_deg:
                max_circumradius_deg = r_deg
                
    except Exception:
        # If hull computation fails (e.g., points in a hemisphere), return a large penalty
        return 180.0
        
    return max_circumradius_deg

def construct():
    """
    Constructs a set of N points on the unit sphere to minimize the covering radius.
    Strategy: 
    1. Initialize with Thomson (electrostatic) configuration for good initial spread.
    2. Use L-BFGS-B to minimize a smooth surrogate for the covering radius.
       The surrogate is the soft-min (log-sum-exp) of the distances to the facet circumcenters.
       This makes the objective differentiable and amenable to gradient-based optimization.
    3. Final polish with a local search on the exact covering radius if time permits.
    """
    start_time = time.time()
    time_limit = 145.0  # Safety margin for 150s kill
    
    # 1. Initialization
    pts = _thomson_initialization(N)
    
    # 2. Optimization
    # We will optimize the positions of the points.
    # To make it compatible with L-BFGS-B, we can optimize in Cartesian coordinates
    # and project back to the sphere after each step.
    # However, L-BFGS-B expects a flat vector.
    
    # Surrogate objective:
    # We want to minimize the max circumradius of the facets.
    # Let R_f be the circumradius of facet f.
    # We approximate max_f R_f by (1/beta) * log(sum_f exp(beta * R_f)).
    # As beta -> infinity, this approaches the max.
    # We need to compute R_f for each facet. This is expensive inside the objective function.
    # Alternative: Optimize a proxy that is cheap to compute but correlated with covering radius.
    # A common proxy is the sum of inverse squared distances to neighbors (packing).
    # But we want covering.
    
    # Let's try a different approach:
    # Since N=130 is small, we can compute the convex hull and facet radii in the objective function.
    # But L-BFGS-B calls the objective many times.
    # ConvexHull for 130 points is fast (~1ms).
    # So we can afford to compute the exact covering radius in the objective,
    # but we need it to be differentiable. The max function is not differentiable.
    # We use the log-sum-exp trick on the radii.
    
    beta = 20.0  # Sharpness of the max approximation
    
    def objective(x):
        """
        x: flat array of shape (N*3,)
        Returns: scalar objective value
        """
        p = x.reshape(N, 3)
        p = _normalize(p)
        
        try:
            hull = ConvexHull(p)
            facets = hull.simplices
            
            radii = []
            for facet in facets:
                p1, p2, p3 = p[facet[0]], p[facet[1]], p[facet[2]]
                v1 = p1 - p2
                v2 = p2 - p3
                c_vec = np.cross(v1, v2)
                norm_c = np.linalg.norm(c_vec)
                if norm_c < 1e-10:
                    # Degenerate
                    centroid = _normalize((p1+p2+p3)/3.0)
                    d1 = np.arccos(np.clip(np.dot(centroid, p1), -1.0, 1.0))
                    d2 = np.arccos(np.clip(np.dot(centroid, p2), -1.0, 1.0))
                    d3 = np.arccos(np.clip(np.dot(centroid, p3), -1.0, 1.0))
                    r_rad = np.max([d1, d2, d3])
                else:
                    c_vec = c_vec / norm_c
                    d1 = np.arccos(np.clip(np.dot(c_vec, p1), -1.0, 1.0))
                    d2 = np.arccos(np.clip(np.dot(-c_vec, p1), -1.0, 1.0))
                    if d1 < d2:
                        r_rad = d1
                    else:
                        r_rad = d2
                radii.append(r_rad)
                
            radii = np.array(radii)
            # Log-sum-exp for stability
            max_r = np.max(radii)
            # If all radii are very small, log-sum-exp might have issues, but they are positive.
            # Objective: (1/beta) * log(sum(exp(beta * r)))
            # To avoid overflow, we shift by max_r
            # log(sum(exp(beta * r))) = beta * max_r + log(sum(exp(beta * (r - max_r))))
            # So (1/beta) * log(sum(exp(beta * r))) = max_r + (1/beta) * log(sum(exp(beta * (r - max_r))))
            # This is a smooth approximation of max_r.
            
            shifted = beta * (radii - max_r)
            # Clip to avoid underflow/overflow in exp
            shifted = np.clip(shifted, -100, 100)
            log_sum_exp = np.log(np.sum(np.exp(shifted)))
            obj_val = max_r + log_sum_exp / beta
            
            return obj_val
            
        except Exception:
            # If hull fails, return a large penalty
            return 10.0  # ~573 degrees, effectively infinite
            
    # Initial objective value
    x0 = pts.flatten()
    
    # Use L-BFGS-B
    # We don't have bounds in Cartesian space that keep points on the sphere,
    # so we project after each step in the objective.
    # However, L-BFGS-B assumes the function is defined on R^d.
    # This "projection inside objective" trick works but the gradient is not exactly correct
    # for the projected manifold. It often works well in practice for sphere constraints.
    
    result = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
    
    best_pts = _normalize(result.x.reshape(N, 3))
    best_score = _covering_radius_deg(best_pts)
    
    # 3. Local polish
    # If time remains, do a few steps of local search to refine.
    if time.time() - start_time < time_limit - 10:
        # Simple local search: try small random perturbations
        step_size = 0.01
        for _ in range(50):
            if time.time() - start_time > time_limit - 5:
                break
            # Random direction in tangent space
            random_vec = np.random.randn(N, 3)
            # Project to tangent plane
            dot = np.sum(random_vec * best_pts, axis=1, keepdims=True)
            tangent_vec = random_vec - dot * best_pts
            tangent_vec = _normalize(tangent_vec)
            
            new_pts = _normalize(best_pts + step_size * tangent_vec)
            new_score = _covering_radius_deg(new_pts)
            
            if new_score < best_score:
                best_pts = new_pts
                best_score = new_score
                # Reduce step size slightly
                step_size *= 0.95
            else:
                # If no improvement, reduce step size
                step_size *= 0.8
                if step_size < 1e-5:
                    break

    # Final check
    return _normalize(best_pts)

if __name__ == "__main__":
    # Quick test
    pts = construct()
    print(f"Shape: {pts.shape}")
    print(f"Norms: {np.linalg.norm(pts, axis=1)[:5]}")
    print(f"Covering Radius: {_covering_radius_deg(pts)}")
