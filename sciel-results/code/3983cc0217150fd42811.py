import time
import numpy as np
from scipy.spatial import ConvexHull

N = 130
_EPS = 1e-12

def _initial_fib():
    """Generate initial points using Fibonacci spiral."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _get_hull_data(pts):
    """
    Compute convex hull and extract facet data.
    Returns:
        faces: array of vertex indices for each facet
        poles: unit normal vector for each facet (outward)
        radii: angular circumradius for each facet (radians)
        vertices_per_facet: list of vertex indices for each facet
    """
    try:
        hull = ConvexHull(pts)
        faces = hull.simplices
        
        A = pts[faces[:, 0]]
        B = pts[faces[:, 1]]
        C = pts[faces[:, 2]]
        
        # Vectors for the triangle edges
        AB = B - A
        AC = C - A
        
        # Cross product gives normal
        cross = np.cross(AB, AC)
        norms = np.linalg.norm(cross, axis=1, keepdims=True)
        
        # Handle degenerate facets
        valid = norms[:, 0] > _EPS
        if not np.any(valid):
            return None, None, None, None
            
        faces_v = faces[valid]
        A_v = A[valid]
        B_v = B[valid]
        C_v = C[valid]
        cross_v = cross[valid]
        norms_v = norms[valid]
        
        # Ensure normals point outward
        centroids = (A_v + B_v + C_v) / 3.0
        signs = np.sum(cross_v * centroids, axis=1, keepdims=True)
        poles = cross_v * signs
        poles = poles / (np.linalg.norm(poles, axis=1, keepdims=True) + _EPS)
        
        # Angular circumradius: distance from pole to any vertex
        # cos(theta) = P @ A
        cos_theta = np.sum(poles * A_v, axis=1)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        radii = np.arccos(cos_theta)
        
        return faces_v, poles, radii, [faces_v[i].tolist() for i in range(len(faces_v))]
    except Exception:
        return None, None, None, None

def _compute_covering_radius(pts):
    """Compute the exact covering radius in radians."""
    _, _, radii, _ = _get_hull_data(pts)
    if radii is None:
        return np.pi
    return np.max(radii)

def _balance_facet(pts, face_indices, pole, max_iters=50, tol=1e-6):
    """
    Adjust vertices of a specific facet to minimize its circumradius.
    We want to move vertices along the great circle towards the pole such that
    the angular distance from the pole to each vertex is equalized and minimized.
    
    Actually, a simpler heuristic: 
    The circumradius is determined by the farthest vertex. 
    Moving a vertex towards the pole reduces the radius. 
    However, moving it might increase the radius of adjacent facets.
    
    A more robust local step:
    For the worst facet, calculate the "ideal" position if we could move vertices freely.
    Since we are constrained to the sphere, we project the update.
    
    Let's use a subgradient-like step:
    For each vertex v in the facet, move it towards the pole by a small step.
    The step size can be adaptive.
    
    Better: Solve for the optimal movement direction.
    The gradient of the angular distance d = arccos(p.v) w.r.t v (on sphere) is 
    roughly -p / sin(d). 
    So moving v in the direction of -grad (which is towards p) decreases d.
    
    We will move all three vertices towards the pole by a step size eta.
    """
    pts_new = pts.copy()
    v_idx = face_indices
    p = pole
    
    # Current distances
    cos_d = np.sum(pts[v_idx] * p, axis=1)
    cos_d = np.clip(cos_d, -1.0, 1.0)
    d = np.arccos(cos_d)
    
    # We want to reduce the max distance.
    # Simple step: move towards pole.
    # The direction on the sphere towards p from v is (p - v).normalized? No, (p - v) is in R3.
    # The tangent vector at v pointing towards p is (p - (p.v)v).
    
    for k in range(max_iters):
        # Calculate current radii for this facet
        cos_d = np.sum(pts_new[v_idx] * p, axis=1)
        cos_d = np.clip(cos_d, -1.0, 1.0)
        d = np.arccos(cos_d)
        
        max_d = np.max(d)
        # If all are equal and small, we might be stuck in a local min for this facet
        # but globally it might be fine.
        
        # Determine step size. 
        # If we move too much, we might overshoot or break other facets.
        # Start with a small step.
        step = 0.1 * (max_d - np.min(d) + 1e-6) # Relative to spread
        # Cap step to avoid large jumps
        step = min(step, 0.05)
        
        if step < 1e-8:
            break
            
        # Move vertices
        for idx in v_idx:
            v = pts_new[idx]
            # Tangent vector towards p
            proj = np.dot(v, p)
            tangent = p - proj * v
            norm_t = np.linalg.norm(tangent)
            if norm_t < _EPS:
                continue
            tangent /= norm_t
            
            # Update
            new_v = v + step * tangent
            new_v /= (np.linalg.norm(new_v) + _EPS)
            pts_new[idx] = new_v
            
    return pts_new

def _local_search(pts, deadline):
    """
    Iteratively find the worst facet and balance it.
    """
    best_pts = pts.copy()
    best_r = _compute_covering_radius(pts)
    
    # We will iterate, always targeting the current worst facet.
    # To avoid oscillation, we might need to be careful.
    # Simple strategy: 
    # 1. Find worst facet.
    # 2. Balance it.
    # 3. Check global radius. If improved, accept. Else, revert or try smaller step.
    
    # For robustness, we'll do a few passes.
    for _ in range(200): # Max iterations
        if time.time() > deadline:
            break
            
        faces, poles, radii, _ = _get_hull_data(pts)
        if faces is None:
            break
            
        curr_r = np.max(radii)
        if curr_r < best_r:
            best_r = curr_r
            best_pts = pts.copy()
            
        # Find worst facet index
        worst_idx = np.argmax(radii)
        worst_face = faces[worst_idx]
        worst_pole = poles[worst_idx]
        
        # Try to improve this facet
        # We use a slightly more aggressive balancing: 
        # Move vertices towards the pole, but also consider that moving them 
        # might make them too close to each other or affect neighbors.
        
        # Let's try a few step sizes
        improved = False
        for step_mult in [1.0, 0.5, 0.2, 0.1]:
            pts_trial = pts.copy()
            v_idx = worst_face
            p = worst_pole
            
            # Calculate current distances
            cos_d = np.sum(pts_trial[v_idx] * p, axis=1)
            cos_d = np.clip(cos_d, -1.0, 1.0)
            d = np.arccos(cos_d)
            max_d = np.max(d)
            
            # Step size proportional to how far we are from ideal (0)
            # Ideally all d=0, but they can't be. 
            # Let's just take a fixed small step towards the pole.
            step = 0.05 * step_mult
            
            for idx in v_idx:
                v = pts_trial[idx]
                proj = np.dot(v, p)
                tangent = p - proj * v
                norm_t = np.linalg.norm(tangent)
                if norm_t < _EPS:
                    continue
                tangent /= norm_t
                new_v = v + step * tangent
                new_v /= (np.linalg.norm(new_v) + _EPS)
                pts_trial[idx] = new_v
                
            # Check if this improves the global covering radius
            trial_r = _compute_covering_radius(pts_trial)
            if trial_r < curr_r - 1e-8:
                pts = pts_trial
                improved = True
                break
                
        if not improved:
            # If we can't improve the worst facet by moving its vertices towards its pole,
            # we might be stuck. 
            # Try moving the *other* vertices of the worst facet away? No, that increases radius.
            # Maybe the hull structure is changing. 
            # If we are stuck, we break to avoid infinite loop.
            # Or we can try a random perturbation of the worst facet vertices.
            
            # Random perturbation escape
            if np.random.rand() < 0.1:
                pts_trial = pts.copy()
                for idx in worst_face:
                    # Small random tangent perturbation
                    rand_vec = np.random.randn(3)
                    proj = np.dot(pts_trial[idx], rand_vec)
                    tangent = rand_vec - proj * pts_trial[idx]
                    norm_t = np.linalg.norm(tangent)
                    if norm_t > _EPS:
                        tangent /= norm_t
                        new_v = pts_trial[idx] + 0.01 * tangent
                        new_v /= (np.linalg.norm(new_v) + _EPS)
                        pts_trial[idx] = new_v
                trial_r = _compute_covering_radius(pts_trial)
                if trial_r < curr_r:
                    pts = pts_trial
                    improved = True
            if not improved:
                break
                
    return best_pts, best_r

def construct():
    deadline = time.time() + 145.0
    
    # Multi-restart strategy
    best_global_pts = None
    best_global_r = np.pi
    
    # We have ~145s. 
    # Each local search iteration involves ConvexHull (expensive) and radius check.
    # Let's estimate cost. ConvexHull on 130 points is fast (~ms). 
    # Let's allow multiple restarts.
    
    # Initial restart: Fibonacci
    pts = _initial_fib()
    
    # Run local search
    deadline_for_phase = deadline - 5.0 # Leave 5s for final checks
    pts, r = _local_search(pts, deadline_for_phase)
    
    if r < best_global_r:
        best_global_r = r
        best_global_pts = pts.copy()
        
    # If we have time, try other initializations
    # Random initializations
    num_restarts = 5
    time_per_restart = (deadline - time.time()) / (num_restarts + 1)
    
    for k in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
            
        # Generate random points on sphere
        pts_init = np.random.randn(N, 3)
        pts_init /= np.linalg.norm(pts_init, axis=1, keepdims=True)
        
        # Run local search with a tighter deadline for this restart
        # We want to finish before the main deadline
        phase_deadline = min(deadline - 5.0, time.time() + time_per_restart * 0.8)
        
        pts_opt, r_opt = _local_search(pts_init, phase_deadline)
        
        if r_opt < best_global_r:
            best_global_r = r_opt
            best_global_pts = pts_opt.copy()
            
    # Final check and normalization
    if best_global_pts is None:
        return _initial_fib()
        
    pts_final = np.array(best_global_pts, dtype=float)
    pts_final /= (np.linalg.norm(pts_final, axis=1, keepdims=True) + _EPS)
    
    # Validity check
    if not np.all(np.isfinite(pts_final)) or pts_final.shape != (N, 3):
        return _initial_fib()
        
    return pts_final
