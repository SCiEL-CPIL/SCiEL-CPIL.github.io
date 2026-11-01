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

def _compute_hull_radius_and_worst(pts):
    """
    Compute the exact covering radius and identify the worst facet.
    Returns:
        radius: float, the maximum angular circumradius in radians.
        worst_facet_indices: np.array of shape (3,) or None if hull fails.
        worst_pole: np.array of shape (3,) or None.
    """
    try:
        hull = ConvexHull(pts)
        faces = hull.simplices
        M = len(faces)
        
        if M == 0:
            return np.pi, None, None

        A = pts[faces[:, 0]]
        B = pts[faces[:, 1]]
        C = pts[faces[:, 2]]
        
        AB = B - A
        AC = C - A
        
        cross = np.cross(AB, AC)
        area = 0.5 * np.linalg.norm(cross, axis=1)
        
        valid = area > _EPS
        if not np.any(valid):
            return np.pi, None, None
        
        normals = cross[valid]
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        
        # Ensure normals point outward
        centroids = (A[valid] + B[valid] + C[valid]) / 3.0
        signs = np.sum(normals * centroids, axis=1, keepdims=True)
        normals = normals * np.sign(signs)
        normals = normals / (norms + _EPS)
        
        # Angular circumradius for each valid facet
        # cos(theta) = normal . vertex
        # We use vertex A for all, but since they are on the sphere and coplanar, 
        # the angle to the pole is the same for all vertices of the facet.
        cos_theta = np.sum(normals * A[valid], axis=1)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        ang_radii = np.arccos(cos_theta)
        
        max_radius = np.max(ang_radii)
        worst_local_idx = np.argmax(ang_radii)
        
        # Map back to original face indices
        valid_indices = np.where(valid)[0]
        worst_face_idx = valid_indices[worst_local_idx]
        worst_facet_indices = faces[worst_face_idx]
        worst_pole = normals[worst_local_idx]
        
        return max_radius, worst_facet_indices, worst_pole
    except Exception:
        return np.pi, None, None

def _push_towards_pole(pts, facet_indices, pole, step_size):
    """
    Move the vertices of the given facet towards the pole.
    This reduces the circumradius of that facet.
    """
    cand = pts.copy()
    for v in facet_indices:
        p = pts[v]
        # Direction from p to pole
        dir_vec = pole - p
        dir_norm = np.linalg.norm(dir_vec)
        if dir_norm > _EPS:
            # Move a fraction of the way
            # We want to move along the great circle towards the pole
            # Simple Euclidean projection then normalize is a decent approximation for small steps
            new_p = p + step_size * dir_vec
            new_p = new_p / (np.linalg.norm(new_p) + _EPS)
            cand[v] = new_p
    return cand

def _random_perturbation(pts, magnitude):
    """
    Apply a small random perturbation to all points to escape local minima.
    """
    cand = pts.copy()
    # Generate random vectors in tangent space
    # Simple approach: add small random noise and re-normalize
    noise = np.random.randn(*cand.shape) * magnitude
    cand += noise
    norms = np.linalg.norm(cand, axis=1, keepdims=True)
    cand = cand / (norms + _EPS)
    return cand

def construct():
    deadline = time.time() + 145.0  # Safety margin
    
    # Initial configuration
    pts = _initial_fib()
    best_pts = pts.copy()
    best_radius, _, _ = _compute_hull_radius_and_worst(pts)
    
    # Simulated Annealing parameters
    # We start with a relatively high temperature and cool down
    # The "energy" is the covering radius.
    # We accept worse solutions with probability exp(-delta_E / T)
    
    # Initial temperature: enough to allow significant moves
    # Radius is in radians. Typical improvement per step is small.
    # Let's scale temperature to the magnitude of the radius.
    T = 0.01 
    T_min = 1e-5
    cooling_rate = 0.9995
    
    # Base step size for the push
    base_step = 0.1
    
    # Loop until deadline
    while time.time() < deadline:
        current_radius, worst_facet, worst_pole = _compute_hull_radius_and_worst(pts)
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_pts = pts.copy()
            
        if worst_facet is None:
            break
            
        # Generate candidate by pushing worst facet
        # Step size might need to be adaptive or constant.
        # Let's use a constant step for the deterministic push.
        cand = _push_towards_pole(pts, worst_facet, worst_pole, base_step)
        
        # Compute candidate radius
        cand_radius, _, _ = _compute_hull_radius_and_worst(cand)
        
        # Simulated Annealing acceptance
        delta_E = cand_radius - current_radius
        if delta_E < 0 or np.random.random() < np.exp(-delta_E / T):
            pts = cand
        else:
            # If rejected, maybe try a random perturbation to escape?
            # Only do this if we are stuck for a while? 
            # For simplicity, just don't move, but cool down.
            # Actually, if we always reject, we might get stuck.
            # Let's add a small probability of random perturbation regardless.
            if np.random.random() < 0.1:
                cand = _random_perturbation(pts, 0.01)
                cand_radius, _, _ = _compute_hull_radius_and_worst(cand)
                delta_E = cand_radius - current_radius
                if delta_E < 0 or np.random.random() < np.exp(-delta_E / T):
                    pts = cand
        
        # Cool down
        T = max(T_min, T * cooling_rate)
        
        # Optional: Re-center or check validity
        # Ensure points are finite
        if not np.all(np.isfinite(pts)):
            pts = best_pts.copy()
            T = T_min # Reset if broken
            
    # Final polish: Run a few pure greedy steps (no SA) to ensure we end in a local minimum
    # This is important because SA might end at a non-optimal local min due to low T.
    for _ in range(100):
        if time.time() > deadline - 2.0:
            break
        current_radius, worst_facet, worst_pole = _compute_hull_radius_and_worst(pts)
        if current_radius < best_radius:
            best_radius = current_radius
            best_pts = pts.copy()
        if worst_facet is None:
            break
        cand = _push_towards_pole(pts, worst_facet, worst_pole, base_step * 0.5)
        cand_radius, _, _ = _compute_hull_radius_and_worst(cand)
        if cand_radius < current_radius:
            pts = cand
        else:
            # Try smaller step
            cand = _push_towards_pole(pts, worst_facet, worst_pole, base_step * 0.1)
            cand_radius, _, _ = _compute_hull_radius_and_worst(cand)
            if cand_radius < current_radius:
                pts = cand
            else:
                # Try even smaller
                cand = _push_towards_pole(pts, worst_facet, worst_pole, base_step * 0.01)
                cand_radius, _, _ = _compute_hull_radius_and_worst(cand)
                if cand_radius < current_radius:
                    pts = cand
                else:
                    break # Stuck in local min

    # Final check: use best_pts found during the process
    final_pts = best_pts
    
    # Final normalization
    final_pts = np.array(final_pts, dtype=float)
    final_pts /= (np.linalg.norm(final_pts, axis=1, keepdims=True) + _EPS)
    
    # Validity check
    if not np.all(np.isfinite(final_pts)) or final_pts.shape != (N, 3):
        return _initial_fib()
        
    return final_pts

if __name__ == "__main__":
    pts = construct()
    r, _, _ = _compute_hull_radius_and_worst(pts)
    print(f"Covering radius: {np.degrees(r)} degrees")
