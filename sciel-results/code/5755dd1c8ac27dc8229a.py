"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull


def construct():
    start_time = time.time()
    time_limit = 118.0  # Safety margin for 120s eval / 150s kill

    # Initialize with Fibonacci spiral points
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    def compute_covering_radius(points):
        """Compute covering radius using convex hull facets."""
        try:
            hull = ConvexHull(points)
            max_radius = 0.0
            for simplex in hull.simplices:
                face_points = points[simplex]
                a, b, c = face_points
                ab = b - a
                ac = c - a
                normal = np.cross(ab, ac)
                normal_norm = np.linalg.norm(normal)
                if normal_norm < 1e-12:
                    continue
                normal = normal / normal_norm
                
                d = 2.0 * (a[0] * (b[1] * c[2] - c[1] * b[2]) - a[1] * (b[0] * c[2] - c[0] * b[2]) + a[2] * (b[0] * c[1] - c[0] * b[1]))
                if abs(d) < 1e-12:
                    continue
                
                a2 = np.dot(a, a)
                b2 = np.dot(b, b)
                c2 = np.dot(c, c)
                
                ux = ((a2 * (b[1] * c[2] - c[1] * b[2]) - b2 * (a[1] * c[2] - c[1] * a[2]) + c2 * (a[1] * b[2] - b[1] * a[2])) / d)
                uy = ((a2 * (b[0] * c[2] - c[0] * b[2]) - b2 * (a[0] * c[2] - c[0] * a[2]) + c2 * (a[0] * b[2] - b[0] * a[2])) / d)
                uz = ((a2 * (b[0] * c[1] - c[0] * b[1]) - b2 * (a[0] * c[1] - c[0] * a[1]) + c2 * (a[0] * b[1] - b[0] * a[1])) / d)
                
                circumcenter = np.array([ux, uy, uz])
                cc_norm = np.linalg.norm(circumcenter)
                if cc_norm < 1e-12:
                    continue
                circumcenter = circumcenter / cc_norm
                
                dot_vals = np.clip(np.dot(circumcenter, face_points), -1.0, 1.0)
                angles = np.arccos(dot_vals)
                facet_radius = np.max(angles)
                max_radius = max(max_radius, facet_radius)
            return max_radius
        except Exception:
            return np.pi

    def potential_energy(x):
        """
        Objective function: Sum of inverse squared distances between all pairs.
        This promotes uniform distribution.
        """
        pts_3d = x.reshape(-1, 3)
        # Compute pairwise squared distances
        # dist^2 = 2 - 2 * dot(p_i, p_j)
        dots = np.dot(pts_3d, pts_3d.T)
        sq_dists = 2.0 - 2.0 * dots
        np.fill_diagonal(sq_dists, 1.0) # Avoid division by zero
        
        # Inverse squared distance
        inv_sq_dists = 1.0 / sq_dists
        np.fill_diagonal(inv_sq_dists, 0.0)
        
        return np.sum(inv_sq_dists)

    def project_to_sphere(x):
        """Project points onto unit sphere."""
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        return x / norms

    best_pts = pts.copy()
    best_radius = compute_covering_radius(best_pts)

    # Phase 1: Optimize potential energy with L-BFGS
    # We optimize in 3D space, but project back to sphere after each step?
    # L-BFGS doesn't handle constraints well. We can use a penalty or just project.
    # Better: Parameterize using angles or just project in the callback?
    # Simplest: Minimize in 3D, but the gradient is tricky if we project.
    # Alternative: Use the fact that for fixed radius, moving along tangent plane is best.
    # Let's just use L-BFGS on the 3D coordinates and rely on the fact that
    # the potential energy gradient will push points away, and we project at the end.
    # Actually, L-BFGS assumes the function is defined on the manifold.
    # If we define f(x) on R^3 but x is constrained to S^2, we can't just use standard L-BFGS.
    
    # Let's use a different approach: Repulsion with manual gradient descent but better initialization/steps?
    # The hint says "L-BFGS on smooth surrogate".
    # Maybe the surrogate is defined on the sphere?
    
    # Let's try a simpler robust method: 
    # 1. Run L-BFGS on the potential energy in 3D, but with a strong projection penalty?
    # No, that's unstable.
    
    # Let's stick to the manual gradient descent but improve the force calculation to be smoother?
    # Or use a different initialization?
    
    # Let's try to use `scipy.optimize.minimize` with `method='L-BFGS-B'` on the potential energy
    # but we need to handle the sphere constraint.
    
    # A common trick: Minimize the potential energy in 3D, but add a penalty term 
    # that forces points to stay on the sphere: sum(||p_i||^2 - 1)^2.
    # This makes the problem unconstrained in R^3.
    
    def objective_with_penalty(x):
        pts_3d = x.reshape(-1, 3)
        energy = potential_energy(x)
        penalty = np.sum((np.linalg.norm(pts_3d, axis=1)**2 - 1.0)**2)
        return energy + 1000.0 * penalty

    # This might work. Let's try it.
    x0 = pts.flatten()
    
    try:
        result = minimize(objective_with_penalty, x0, method='L-BFGS', options={'maxiter': 100, 'ftol': 1e-6, 'gtol': 1e-4})
        if result.success:
            new_pts = result.x.reshape(-1, 3)
            new_pts = project_to_sphere(new_pts)
            new_radius = compute_covering_radius(new_pts)
            if new_radius < best_radius:
                best_radius = new_radius
                best_pts = new_pts
    except Exception:
        pass

    # Phase 2: Refine with facet forces if time permits
    # Use the best_pts from L-BFGS as starting point
    pts = best_pts.copy()
    
    def compute_facet_forces(points):
        """
        Compute forces on each vertex based on the facet circumcenters.
        """
        try:
            hull = ConvexHull(points)
            forces = np.zeros_like(points)
            
            for simplex in hull.simplices:
                face_points = points[simplex]
                a, b, c = face_points
                ab = b - a
                ac = c - a
                normal = np.cross(ab, ac)
                normal_norm = np.linalg.norm(normal)
                if normal_norm < 1e-12:
                    continue
                normal = normal / normal_norm
                
                d = 2.0 * (a[0] * (b[1] * c[2] - c[1] * b[2]) - a[1] * (b[0] * c[2] - c[0] * b[2]) + a[2] * (b[0] * c[1] - c[0] * b[1]))
                if abs(d) < 1e-12:
                    continue
                
                a2 = np.dot(a, a)
                b2 = np.dot(b, b)
                c2 = np.dot(c, c)
                
                ux = ((a2 * (b[1] * c[2] - c[1] * b[2]) - b2 * (a[1] * c[2] - c[1] * a[2]) + c2 * (a[1] * b[2] - b[1] * a[2])) / d)
                uy = ((a2 * (b[0] * c[2] - c[0] * b[2]) - b2 * (a[0] * c[2] - c[0] * a[2]) + c2 * (a[0] * b[2] - b[0] * a[2])) / d)
                uz = ((a2 * (b[0] * c[1] - c[0] * b[1]) - b2 * (a[0] * c[1] - c[0] * a[1]) + c2 * (a[0] * b[1] - b[0] * a[1])) / d)
                
                circumcenter = np.array([ux, uy, uz])
                cc_norm = np.linalg.norm(circumcenter)
                if cc_norm < 1e-12:
                    continue
                circumcenter = circumcenter / cc_norm
                
                dot_vals = np.clip(np.dot(circumcenter, face_points), -1.0, 1.0)
                angles = np.arccos(dot_vals)
                facet_radius = np.max(angles)
                
                weight = facet_radius / (np.pi / 6.0)
                
                for idx in simplex:
                    vertex = points[idx]
                    diff = circumcenter - vertex
                    diff_norm = np.linalg.norm(diff)
                    if diff_norm > 1e-10:
                        forces[idx] += weight * diff / diff_norm
            
            return forces
        except Exception:
            return np.zeros_like(points)

    # Phase 2: Fine polish with facet-based forces
    lr = 0.01
    for iteration in range(400):
        if time.time() - start_time > time_limit:
            break
        if best_radius < 0.1971:
            break

        forces = compute_facet_forces(pts)
        
        force_norms = np.linalg.norm(forces, axis=1, keepdims=True)
        force_norms = np.where(force_norms < 1e-10, 1.0, force_norms)
        forces = forces / force_norms
        
        new_pts = pts + lr * forces
        new_pts = project_to_sphere(new_pts)
        
        new_radius = compute_covering_radius(new_pts)
        if new_radius < best_radius:
            best_radius = new_radius
            best_pts = new_pts
            pts = new_pts
            lr *= 0.995
        else:
            lr *= 0.9

    return best_pts
# EVOLVE-BLOCK-END
