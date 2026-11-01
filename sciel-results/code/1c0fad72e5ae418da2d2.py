"""Covering the sphere with 130 equal caps - optimized via repulsive forces and local refinement."""
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 130

def _covering_radius_from_hull(pts):
    """Compute covering radius in radians using convex hull facets."""
    try:
        hull = ConvexHull(pts)
        max_r = 0.0
        for simplex in hull.simplices:
            tri = pts[simplex]
            # Compute circumcenter of the spherical triangle
            # The covering radius for a facet is the angular distance from the origin
            # to the circumcenter of the triangle on the sphere
            # For a triangle on the unit sphere, the circumcenter is the normalized
            # cross product approach or we can use the fact that the farthest point
            # on the facet (as seen from origin) is at the circumcenter
            
            # Compute the plane normal (which is the direction to the circumcenter)
            a, b, c = tri[0], tri[1], tri[2]
            normal = np.cross(b - a, c - a)
            norm = np.linalg.norm(normal)
            if norm < 1e-12:
                continue
            normal = normal / norm
            
            # The circumcenter on the sphere is in the direction of the normal
            # The angular distance from the circumcenter to any vertex is the circumradius
            # cos(theta) = dot(circumcenter, vertex)
            dot_vals = np.dot(tri, normal)
            # The circumradius is arccos(min(dot_vals)) since the circumcenter is equidistant
            # Actually, for a spherical triangle, the circumcenter is the point equidistant
            # from all three vertices. Let's compute it properly.
            
            # The circumcenter of a spherical triangle is the intersection of the sphere
            # with the plane through the origin perpendicular to the facet plane... 
            # No, the circumcenter is the point on the sphere equidistant from all 3 vertices.
            # It lies in the plane spanned by the three vertices.
            
            # Better approach: The circumcenter is the normalized vector:
            # c = a*|a|^2*(b×c) + ... no, let's use a different method.
            
            # The circumcenter is the point p on the unit sphere such that
            # p·a = p·b = p·c = cos(r) for some r.
            # So p is in the null space of [a-b, a-c] intersected with the unit sphere.
            # p is parallel to (a×b)×(a×c) = a((b×c)·a) ... 
            # Actually p is parallel to the cross product of two of the "bisector" directions.
            
            # Simplest: solve for p in span{a,b,c} such that p·a = p·b = p·c
            # p = alpha*a + beta*b + gamma*c
            # p·a = p·b => alpha*(a·a) + beta*(b·a) + gamma*(c·a) = alpha*(a·b) + beta*(b·b) + gamma*(c·b)
            # This is a 2x2 linear system for alpha,beta,gamma (up to scale).
            
            # Alternative: The circumcenter direction is:
            # n = (a*|a|^2) × (b*|b|^2) ... no.
            
            # Let me use: the circumcenter is the normalized sum of the vertices weighted by 
            # the opposite angles, or more simply, it's the intersection of the unit sphere 
            # with the line through the origin in the direction:
            # d = (b-a)×(c-a) ... no that's the normal to the triangle plane.
            
            # Actually, for points on the unit sphere, the circumcenter of the spherical triangle
            # is the normalized vector:
            # p = (a × (b × c)) + (b × (c × a)) + (c × (a × b))
            # No, that's not right either.
            
            # Let me just use: the circumcenter is the point on the unit sphere closest to 
            # the centroid direction... no.
            
            # Correct method: The circumcenter c satisfies c·a = c·b = c·c_ = cos(R)
            # where c_ is the third point. So c is in the intersection of the unit sphere 
            # and the affine plane defined by these dot product constraints.
            # c is parallel to the cross product of (a-b) and (a-c) projected... 
            # c is in the null space of the 2x3 matrix [a-b; a-c]... no, c·(a-b)=0 and c·(a-c)=0
            # So c is parallel to (a-b)×(a-c) = (a-b)×(c-a) = -(a-b)×(a-c)
            # Wait: c·a = c·b means c·(a-b) = 0. Similarly c·(a-c) = 0.
            # So c is parallel to (a-b)×(a-c).
            
            v1 = a - b
            v2 = a - c
            c_dir = np.cross(v1, v2)
            c_norm = np.linalg.norm(c_dir)
            if c_norm < 1e-12:
                continue
            c_dir = c_dir / c_norm
            
            # Check which direction (c_dir or -c_dir) is the circumcenter
            # It should be on the same side as the triangle vertices from origin
            dot_a = np.dot(c_dir, a)
            if dot_a < 0:
                c_dir = -c_dir
            
            # The circumradius is arccos(dot(c_dir, a))
            cos_r = np.clip(np.dot(c_dir, a), -1.0, 1.0)
            r = np.arccos(cos_r)
            if r > max_r:
                max_r = r
        return max_r
    except Exception:
        return np.pi  # Fallback: if hull fails, return worst case

def _covering_radius_grid(pts, n_grid=2000):
    """Compute covering radius using a dense grid approximation."""
    # Generate a dense Fibonacci spiral for evaluation
    i = np.arange(n_grid) + 0.5
    z = 1.0 - 2.0 * i / n_grid
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    grid_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Compute angular distance from each grid point to nearest pts
    # cos(theta) = grid_pt · pt
    dots = grid_pts @ pts.T  # shape (n_grid, N)
    min_cos = np.max(dots, axis=1)  # max dot product = min angular distance
    min_cos = np.clip(min_cos, -1.0, 1.0)
    max_angle = np.arccos(min_cos)
    return np.max(max_angle)

def _repulsive_forces(pts):
    """Compute repulsive forces between points on the sphere."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    weights = 1.0 / dist_sq
    np.fill_diagonal(weights, 0.0)
    forces = np.sum(weights[:, :, np.newaxis] * diff, axis=1)
    return forces

def _project_to_sphere(pts):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return pts / norms

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin before 150s kill
    
    # 1. Initialization: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_pts = pts.copy()
    best_radius = _covering_radius_grid(pts, 5000)
    
    # 2. Repulsive Force Optimization
    step_size = 0.02
    cooling = 0.9995
    
    while time.time() - start_time < time_limit - 20:  # Leave 20s for refinement
        # Calculate forces
        forces = _repulsive_forces(pts)
        
        # Update points
        pts = pts + step_size * forces
        
        # Renormalize to unit sphere
        pts = _project_to_sphere(pts)
        
        # Cool down step size
        step_size *= cooling
        
        # Periodic check
        if step_size < 1e-8:
            break
        
        # Check every ~100 iterations
        if int((step_size / (0.02 * cooling**100)) * 100) % 50 == 0:
            pass  # Placeholder for periodic check
    
    # Update best
    current_radius = _covering_radius_grid(pts, 5000)
    if current_radius < best_radius:
        best_radius = current_radius
        best_pts = pts.copy()
    
    # 3. Local refinement using scipy minimize on the worst facet
    # Try to improve by moving points towards the worst covering region
    remaining_time = time_limit - (time.time() - start_time)
    
    if remaining_time > 10:
        # Find the worst point on the sphere
        n_eval = 10000
        i_eval = np.arange(n_eval) + 0.5
        z_eval = 1.0 - 2.0 * i_eval / n_eval
        phi_eval = np.pi * (1.0 + np.sqrt(5.0)) * i_eval
        r_eval = np.sqrt(1.0 - z_eval * z_eval)
        eval_pts = np.column_stack([r_eval * np.cos(phi_eval), r_eval * np.sin(phi_eval), z_eval])
        
        # Find worst point
        dots_eval = eval_pts @ pts.T
        min_cos_eval = np.max(dots_eval, axis=1)
        min_cos_eval = np.clip(min_cos_eval, -1.0, 1.0)
        angles_eval = np.arccos(min_cos_eval)
        worst_idx = np.argmax(angles_eval)
        worst_point = eval_pts[worst_idx]
        
        # Find the 3 nearest points to the worst point
        dists_to_worst = np.sum((pts - worst_point) ** 2, axis=1)
        nearest_3 = np.argsort(dists_to_worst)[:3]
        
        # Try to move these 3 points slightly towards the worst point
        # to reduce the covering radius
        for _ in range(50):
            if time.time() - start_time > time_limit - 5:
                break
            
            # Move the 3 nearest points towards the worst point
            for idx in nearest_3:
                pts[idx] = pts[idx] + 0.01 * (worst_point - pts[idx])
            pts = _project_to_sphere(pts)
            
            # Recompute
            current_r = _covering_radius_grid(pts, 3000)
            if current_r < best_radius:
                best_radius = current_r
                best_pts = pts.copy()
            else:
                # Revert
                pts = best_pts.copy()
                break
    
    # Final check
    final_radius = _covering_radius_grid(best_pts, 10000)
    if final_radius < best_radius:
        best_radius = final_radius
    
    return best_pts

# EVOLVE-BLOCK-END
