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

def _random_init(rng):
    """Generate random initial points."""
    pts = rng.standard_normal((N, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts

def _compute_hull_data(pts):
    """
    Compute convex hull and return:
    - faces: indices of vertices for each face
    - normals: outward unit normals
    - circumradii: angular distance from pole to vertices for each face
    - worst_face_idx: index of the face with maximum circumradius
    """
    try:
        hull = ConvexHull(pts)
        faces = hull.simplices
        
        A = pts[faces[:, 0]]
        B = pts[faces[:, 1]]
        C = pts[faces[:, 2]]
        
        # Compute normals for each face
        AB = B - A
        AC = C - A
        cross = np.cross(AB, AC)
        norms = np.linalg.norm(cross, axis=1, keepdims=True)
        
        # Filter degenerate
        valid = norms.flatten() > _EPS
        if not np.any(valid):
            return None, None, None, None, np.pi
            
        faces_valid = faces[valid]
        A_valid = A[valid]
        B_valid = B[valid]
        C_valid = C[valid]
        normals = cross[valid] / (norms[valid] + _EPS)
        
        # Ensure outward direction
        centroids = (A_valid + B_valid + C_valid) / 3.0
        signs = np.sum(normals * centroids, axis=1, keepdims=True)
        normals = normals * np.sign(signs)
        
        # The pole is the normalized normal.
        # Distance is arccos(pole . vertex)
        cos_theta_A = np.sum(normals * A_valid, axis=1)
        cos_theta_B = np.sum(normals * B_valid, axis=1)
        cos_theta_C = np.sum(normals * C_valid, axis=1)
        
        # The circumradius of the spherical triangle is the maximum of these distances
        # because the pole is the circumcenter on the sphere for the triangle's circumcircle
        # Wait, the circumcenter of a spherical triangle is not necessarily the normal pole.
        # The normal pole is the center of the circumsphere of the 3 points in R3.
        # Since the points are on the unit sphere, the circumsphere is the unit sphere itself? No.
        # The 3 points define a plane. The normal to the plane passes through the origin.
        # The intersection of this normal with the sphere is the pole.
        # The distance from the pole to any of the 3 vertices is the same (by symmetry of the plane cut).
        # So arccos(n . A) = arccos(n . B) = arccos(n . C).
        # Let's verify.
        # n is perpendicular to AB and AC.
        # n . A = n . (A + t1 AB + t2 AC) ? No.
        # The plane equation is n . x = n . A.
        # Distance from origin to plane is |n . A| (since n is unit).
        # The intersection of the line n with the sphere is n.
        # Distance from n to A: |n - A|.
        # cos(theta) = n . A.
        # Since A, B, C are on the plane n . x = n . A, n . B = n . A, n . C = n . A.
        # So the angular distance from the pole n to all three vertices is identical.
        # This is the circumradius of the facet.
        
        cos_theta = np.clip(cos_theta_A, -1.0, 1.0)
        circumradii = np.arccos(cos_theta)
        
        max_idx = np.argmax(circumradii)
        max_radius = circumradii[max_idx]
        
        return faces_valid, normals, circumradii, max_idx, max_radius
    except Exception:
        return None, None, None, None, np.pi

def _move_vertex_away_from_pole(pts, v_idx, pole, step_size):
    """
    Move vertex v_idx away from the pole along the great circle.
    This is equivalent to moving in the direction of the gradient of the distance to the pole.
    The direction on the sphere is (pole - pts[v_idx]) projected onto tangent space?
    No, to move AWAY from pole, we move in direction (pts[v_idx] - pole).
    We need to project this to the tangent space of pts[v_idx].
    tangent_dir = (pts[v_idx] - pole) - (pts[v_idx] - pole) . pts[v_idx] * pts[v_idx]
    Then new_pts = pts[v_idx] + step_size * tangent_dir
    Then normalize.
    """
    p = pts[v_idx]
    # Direction away from pole
    d = p - pole
    # Project to tangent space at p
    # p . p = 1
    d_tangent = d - np.dot(d, p) * p
    norm_d = np.linalg.norm(d_tangent)
    if norm_d < _EPS:
        return pts
    
    # Normalize direction
    d_tangent /= norm_d
    
    # Move
    new_p = p + step_size * d_tangent
    new_p /= (np.linalg.norm(new_p) + _EPS)
    
    pts[v_idx] = new_p
    return pts

def _geometric_local_search(pts, n_iters=1000, step_size=0.05, cooling=False):
    """
    Directly minimize the maximum circumradius of the convex hull facets.
    """
    best_pts = pts.copy()
    best_r = np.pi
    
    for i in range(n_iters):
        faces, normals, circumradii, max_idx, curr_r = _compute_hull_data(pts)
        if faces is None:
            break
        
        if curr_r < best_r:
            best_r = curr_r
            best_pts = pts.copy()
        
        # If radius is very small, we might be in a local minimum or stuck
        # We always try to move the worst facet's vertices
        
        # Identify the worst face
        worst_face = faces[max_idx]
        worst_pole = normals[max_idx]
        
        # Move all 3 vertices of the worst face away from the pole
        # To avoid oscillation, we might only move one or use a smaller step
        # Let's move all 3 slightly
        
        # Adaptive step size
        if cooling:
            curr_step = step_size * (1.0 - i / float(n_iters))
        else:
            curr_step = step_size
            
        for v_idx in worst_face:
            pts = _move_vertex_away_from_pole(pts, v_idx, worst_pole, curr_step)
            
    return best_pts, best_r

def _simulated_annealing(pts, start_temp=0.1, end_temp=0.001, n_iters=2000):
    """
    Simulated annealing using geometric local search as the move generator?
    Or just random perturbations + local search.
    """
    best_pts = pts.copy()
    best_r = np.pi
    curr_pts = pts.copy()
    
    # Initial radius
    _, _, _, _, curr_r = _compute_hull_data(curr_pts)
    if np.isnan(curr_r):
        curr_r = np.pi
    
    if curr_r < best_r:
        best_r = curr_r
        best_pts = curr_pts.copy()
    
    for i in range(n_iters):
        # Cooling schedule
        temp = start_temp * (end_temp / start_temp) ** (i / float(n_iters))
        
        # Generate a random perturbation
        # Pick a random vertex and move it randomly
        v_idx = np.random.randint(N)
        # Random direction on sphere
        rand_vec = np.random.normal(size=3)
        rand_vec /= (np.linalg.norm(rand_vec) + _EPS)
        
        # Project to tangent space
        p = curr_pts[v_idx]
        d = rand_vec - np.dot(rand_vec, p) * p
        norm_d = np.linalg.norm(d)
        if norm_d < _EPS:
            continue
        d /= norm_d
        
        # Step size proportional to temperature
        step = temp * 0.5
        new_p = p + step * d
        new_p /= (np.linalg.norm(new_p) + _EPS)
        
        new_pts = curr_pts.copy()
        new_pts[v_idx] = new_p
        
        # Evaluate
        _, _, _, _, new_r = _compute_hull_data(new_pts)
        if np.isnan(new_r):
            continue
            
        # Accept if better, or with probability exp(-delta/T)
        delta = new_r - curr_r
        if delta < 0 or np.random.random() < np.exp(-delta / max(temp, 1e-9)):
            curr_pts = new_pts
            curr_r = new_r
            
            if curr_r < best_r:
                best_r = curr_r
                best_pts = curr_pts.copy()
                
    return best_pts, best_r

def construct():
    deadline = time.time() + 145.0
    rng = np.random.default_rng(42)
    
    best_pts = _initial_fib()
    best_r = np.pi
    
    # Initial evaluation
    _, _, _, _, best_r = _compute_hull_data(best_pts)
    
    n_restarts = 10
    
    for restart in range(n_restarts):
        if time.time() > deadline - 20.0:
            break
            
        if restart == 0:
            pts = _initial_fib()
        else:
            pts = _random_init(rng)
            
        # Phase 1: Geometric Local Search
        # This directly attacks the worst facet
        # We run it for a fixed number of iterations or until time budget
        phase_deadline = time.time() + 5.0
        iters = 0
        while time.time() < phase_deadline:
            pts, curr_r = _geometric_local_search(pts, n_iters=100, step_size=0.02, cooling=False)
            iters += 100
            if curr_r < best_r:
                best_r = curr_r
                best_pts = pts.copy()
                
        # Phase 2: Simulated Annealing to escape local minima
        phase_deadline = time.time() + 10.0
        if time.time() < phase_deadline:
            pts, curr_r = _simulated_annealing(pts, start_temp=0.05, end_temp=0.001, n_iters=500)
            if curr_r < best_r:
                best_r = curr_r
                best_pts = pts.copy()
                
        # Phase 3: Final Polish with finer steps
        phase_deadline = time.time() + 5.0
        if time.time() < phase_deadline:
            pts, curr_r = _geometric_local_search(pts, n_iters=200, step_size=0.005, cooling=True)
            if curr_r < best_r:
                best_r = curr_r
                best_pts = pts.copy()
                
    # Final normalization
    best_pts = np.array(best_pts, dtype=float)
    best_pts /= (np.linalg.norm(best_pts, axis=1, keepdims=True) + _EPS)
    
    if not np.all(np.isfinite(best_pts)) or best_pts.shape != (N, 3):
        return _initial_fib()
        
    return best_pts
