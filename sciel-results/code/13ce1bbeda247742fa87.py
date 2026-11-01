"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import distance_matrix, Delaunay
import math
from scipy.spatial.transform import Rotation

N = 100

# EVOLVE-BLOCK-START
def _log_energy_and_grad(pts):
    """
    Compute logarithmic energy E = sum_{i<j} log(1/|xi - xj|) = -sum_{i<j} log(|xi - xj|)
    and its gradient with respect to pts (N x 3).
    """
    pts = np.asarray(pts, dtype=np.float64)
    n = pts.shape[0]
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
    dist = np.sqrt(np.maximum(dist_sq, 1e-30))
    dist_safe = np.maximum(dist, 1e-15)
    
    iu = np.triu_indices(n, k=1)
    d_upper = dist_safe[iu]
    energy = -math.fsum(np.log(d_upper).tolist())
    
    dist_sq_safe = np.maximum(dist_sq, 1e-30)
    W = np.zeros_like(dist_sq_safe)
    np.fill_diagonal(W, 0.0)
    W = 1.0 / dist_sq_safe
    np.fill_diagonal(W, 0.0)
    
    row_sum = W.sum(axis=1)
    grad = -pts * row_sum[:, np.newaxis] + W @ pts
    
    return energy, grad

def _project_to_sphere(pts):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _build_icosahedral_mesh(n_points=N):
    """
    Build a rigid polyhedral mesh by starting with an icosahedron and subdividing
    faces to get approximately n_points vertices with fixed connectivity.
    Returns: pts (n x 3), edges (list of (i,j) pairs)
    """
    # Icosahedron vertices
    t = (1.0 + np.sqrt(5.0)) / 2.0
    ico_verts = np.array([
        [-1,  t,  0], [ 1,  t,  0], [-1, -t,  0], [ 1, -t,  0],
        [ 0, -1,  t], [ 0,  1,  t], [ 0, -1, -t], [ 0,  1, -t],
        [ t,  0, -1], [ t,  0,  1], [-t,  0, -1], [-t,  0,  1]
    ], dtype=np.float64)
    ico_verts = ico_verts / np.linalg.norm(ico_verts, axis=1, keepdims=True)
    
    # Icosahedron faces (20 triangular faces)
    ico_faces = [
        (0, 1, 5), (0, 5, 11), (0, 11, 10), (0, 10, 7), (0, 7, 1),
        (1, 7, 6), (1, 6, 5), (5, 6, 11), (6, 11, 10), (7, 10, 9),
        (1, 9, 7), (1, 5, 9), (5, 11, 9), (11, 10, 9), (10, 8, 9),
        (7, 9, 8), (6, 8, 10), (6, 11, 8), (8, 12, 10), (8, 9, 12)
    ]
    
    # Correct icosahedron face list
    ico_faces = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)
    ]
    
    # Subdivide faces to get closer to N points
    # Each subdivision of a triangle adds 3 new vertices (midpoints)
    # Start with 12 vertices, 20 faces
    # After k subdivisions: 12 + 3*20*(2^k - 1)/1 vertices... complex
    # Instead, use a simpler approach: start with icosahedron, add midpoints iteratively
    
    pts = ico_verts.copy()
    faces = list(ico_faces)
    
    # Subdivide until we have close to N points
    max_subdivisions = 4
    for sub_div in range(max_subdivisions):
        if len(pts) >= N:
            break
        
        new_faces = []
        midpoint_cache = {}
        
        for face in faces:
            i, j, k = face
            # Get or create midpoints
            edges = [(i, j), (j, k), (k, i)]
            mids = []
            for a, b in edges:
                key = (min(a, b), max(a, b))
                if key not in midpoint_cache:
                    mid = (pts[a] + pts[b]) / 2.0
                    mid = mid / np.linalg.norm(mid)
                    midpoint_cache[key] = len(pts)
                    pts = np.vstack([pts, mid])
                mids.append(midpoint_cache[key])
            
            m1, m2, m3 = mids
            # Create 4 new faces from the original and 3 midpoints
            new_faces.append((i, m1, m3))
            new_faces.append((j, m2, m1))
            new_faces.append((k, m3, m2))
            new_faces.append((m1, m2, m3))
        
        faces = new_faces
    
    # If we have more than N, trim to N points
    # Keep the first N points and rebuild edges from faces
    if len(pts) > N:
        pts = pts[:N]
        # Rebuild faces to only include valid vertices
        valid_faces = []
        for face in faces:
            if all(f < N for f in face):
                valid_faces.append(face)
        faces = valid_faces
    
    # Build edge list from faces
    edges = set()
    for face in faces:
        for i in range(3):
            a, b = face[i], face[(i+1) % 3]
            edges.add((min(a, b), max(a, b)))
    
    # If we have fewer than N points, add random points
    while len(pts) < N:
        random_pt = np.random.normal(size=3)
        random_pt = random_pt / np.linalg.norm(random_pt)
        pts = np.vstack([pts, random_pt])
    
    pts = _project_to_sphere(pts)
    return pts, list(edges)

def _mesh_energy_and_grad(pts, edges):
    """
    Compute energy for a mesh-constrained configuration.
    The energy includes both the logarithmic energy and a constraint
    that maintains the mesh structure (edge lengths stay similar).
    """
    n = len(pts)
    energy, grad = _log_energy_and_grad(pts)
    
    # Add a soft constraint to maintain edge lengths
    # This helps keep the mesh structure while allowing optimization
    edge_grad = np.zeros_like(pts)
    for i, j in edges:
        diff = pts[i] - pts[j]
        dist = np.linalg.norm(diff)
        if dist > 1e-10:
            # Gradient of log(dist) with respect to pts[i] and pts[j]
            # d/dpts[i] log(||pts[i]-pts[j]||) = (pts[i]-pts[j])/dist^2
            grad_i = diff / (dist * dist)
            edge_grad[i] += grad_i
            edge_grad[j] -= grad_i
    
    # The constraint energy is sum over edges of log(dist_edge)
    # We want to minimize the total energy, so we add the edge constraint
    # with a small weight to maintain structure
    edge_weight = 0.1
    total_grad = grad + edge_weight * edge_grad
    
    return energy, total_grad

def _optimize_mesh(pts, edges, max_iter=3000, tol=1e-9):
    """
    Optimize mesh vertices on the sphere using gradient descent with
    adaptive step size and momentum.
    """
    pts = _project_to_sphere(pts.copy())
    current_energy, current_grad = _log_energy_and_grad(pts)
    
    # Project gradient to tangent space
    grad_dot = np.einsum('ij,ij->i', current_grad, pts)
    grad_tangent = current_grad - grad_dot[:, np.newaxis] * pts
    
    # Momentum
    velocity = np.zeros_like(pts)
    beta = 0.9
    
    for it in range(max_iter):
        # Adaptive step size
        alpha = 0.01
        
        # Try a few step sizes
        best_pts = pts
        best_energy = current_energy
        best_grad = current_grad
        
        for try_alpha in [alpha, alpha*0.5, alpha*0.25, alpha*0.1]:
            # Update with momentum
            velocity = beta * velocity - try_alpha * grad_tangent
            
            # Step in tangent space
            pts_new = pts + velocity
            pts_new = _project_to_sphere(pts_new)
            
            new_energy, new_grad = _log_energy_and_grad(pts_new)
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_pts = pts_new
                best_grad = new_grad
                alpha = try_alpha
        
        if best_energy >= current_energy:
            # No improvement, reduce learning rate
            alpha *= 0.5
            if alpha < 1e-12:
                break
            # Try again with smaller step
            velocity = beta * velocity - alpha * grad_tangent
            pts_new = _project_to_sphere(pts + velocity)
            new_energy, new_grad = _log_energy_and_grad(pts_new)
            if new_energy < current_energy:
                pts = pts_new
                current_energy = new_energy
                current_grad = new_grad
            else:
                break
        else:
            pts = best_pts
            current_energy = best_energy
            current_grad = best_grad
        
        # Update tangent gradient
        grad_dot = np.einsum('ij,ij->i', current_grad, pts)
        grad_tangent = current_grad - grad_dot[:, np.newaxis] * pts
        
        # Check convergence
        if np.linalg.norm(grad_tangent) < tol:
            break
    
    return pts, current_energy

def _fibonacci_spiral(n=N, offset=0.5):
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def construct():
    """
    Construct 100 points on the unit sphere using a rigid polyhedral mesh approach.
    Start with an icosahedral mesh, optimize vertex positions on the sphere
    using gradient descent with momentum, exploiting the fixed connectivity
    to guide the optimization toward low-energy configurations.
    """
    np.random.seed(42)
    
    # Build initial mesh
    pts, edges = _build_icosahedral_mesh(N)
    
    # Ensure we have exactly N points
    if len(pts) < N:
        # Add remaining points randomly
        for _ in range(N - len(pts)):
            random_pt = np.random.normal(size=3)
            random_pt = random_pt / np.linalg.norm(random_pt)
            pts = np.vstack([pts, random_pt])
    else:
        pts = pts[:N]
    
    pts = _project_to_sphere(pts)
    
    # Optimize the mesh
    best_energy = np.inf
    best_pts = pts.copy()
    
    # Multiple optimization runs with different initial perturbations
    for run in range(5):
        if run == 0:
            current_pts = pts.copy()
        else:
            # Perturb the previous best
            noise_scale = 0.01 * (run + 1)
            noise = np.random.normal(0, noise_scale, size=(N, 3))
            current_pts = _project_to_sphere(pts + noise)
        
        try:
            res_pts, res_energy = _optimize_mesh(current_pts, edges, max_iter=2000, tol=1e-9)
            if res_energy < best_energy:
                best_energy = res_energy
                best_pts = res_pts.copy()
        except Exception:
            pass
    
    # Additional refinement with smaller perturbations
    for refine_pass in range(3):
        try:
            noise_scale = 0.002 * (refine_pass + 1)
            noise = np.random.normal(0, noise_scale, size=(N, 3))
            perturbed = _project_to_sphere(best_pts + noise)
            res_pts, res_energy = _optimize_mesh(perturbed, edges, max_iter=3000, tol=1e-10)
            if res_energy < best_energy:
                best_energy = res_energy
                best_pts = res_pts.copy()
        except Exception:
            pass
    
    # Final projection and validation
    best_pts = _project_to_sphere(best_pts)
    
    # Verify no points are too close
    dist = distance_matrix(best_pts, best_pts)
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)
    
    if min_dist < 1e-9:
        # Fall back to Fibonacci spiral
        best_pts = _fibonacci_spiral(N)
    
    # Ensure finite values
    if not np.all(np.isfinite(best_pts)):
        best_pts = _fibonacci_spiral(N)
    
    return best_pts
# EVOLVE-BLOCK-END
