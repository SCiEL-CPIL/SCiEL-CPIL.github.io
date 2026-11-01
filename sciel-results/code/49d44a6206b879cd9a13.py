"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import distance_matrix, Delaunay
import math
import time

N = 100

# EVOLVE-BLOCK-START

def _log_energy_and_grad(pts):
    """
    Compute logarithmic energy E = sum_{i<j} log(1/|xi - xj|) = -sum_{i<j} log(|xi - xj|)
    and its gradient with respect to pts (N x 3).
    """
    pts = np.asarray(pts, dtype=np.float64)
    n = pts.shape[0]
    
    # Compute pairwise distances efficiently
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
    dist = np.sqrt(np.maximum(dist_sq, 1e-30))
    
    # Avoid log(0) by clipping
    dist_safe = np.maximum(dist, 1e-15)
    
    # Energy: sum over i<j of -log(dist[i,j])
    iu = np.triu_indices(n, k=1)
    d_upper = dist_safe[iu]
    energy = -math.fsum(np.log(d_upper).tolist())
    
    # Gradient: grad[i] = -sum_{j != i} (pts[i] - pts[j]) / ||pts[i] - pts[j]||^2
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

def _build_mesh_connectivity(pts):
    """
    Build a Delaunay-based mesh connectivity from initial points.
    Returns adjacency list as list of sets.
    """
    try:
        # Use Delaunay triangulation on the 3D points projected to 2D or use direct approach
        # For sphere, we can use nearest-neighbor graph
        n = len(pts)
        # Compute k-nearest neighbors with k=6 (typical for sphere packing)
        k = 6
        dist_matrix = distance_matrix(pts, pts)
        np.fill_diagonal(dist_matrix, np.inf)
        
        adjacency = [set() for _ in range(n)]
        for i in range(n):
            # Get k nearest neighbors
            neighbors = np.argsort(dist_matrix[i])[:k]
            for j in neighbors:
                adjacency[i].add(j)
                adjacency[j].add(i)
        
        return adjacency
    except Exception:
        # Fallback: complete graph
        n = len(pts)
        adjacency = [set(range(n)) - {i} for i in range(n)]
        return adjacency

def _mesh_constrained_gradient(pts, adjacency, max_iter=3000, tol=1e-10):
    """
    Optimize points on sphere with mesh structure constraints.
    Uses Riemannian gradient descent with mesh-aware momentum.
    """
    pts = _project_to_sphere(pts.copy())
    n = len(pts)
    
    current_energy, current_grad = _log_energy_and_grad(pts)
    
    # Project gradient to tangent space
    grad_dot = np.einsum('ij,ij->i', current_grad, pts)
    grad_tangent = current_grad - grad_dot[:, np.newaxis] * pts
    
    # Velocity for momentum
    velocity = np.zeros_like(pts)
    beta_momentum = 0.9
    
    for k in range(max_iter):
        # Compute effective gradient with mesh regularization
        # Points that are neighbors in the mesh should maintain similar distances
        effective_grad = grad_tangent.copy()
        
        # Add mesh-aware regularization: penalize large distance changes between neighbors
        for i in range(n):
            if i < len(adjacency):
                for j in adjacency[i]:
                    if j > i:  # Avoid double counting
                        # Encourage maintaining current distance
                        diff_ij = pts[i] - pts[j]
                        dist_ij = np.linalg.norm(diff_ij)
                        if dist_ij > 1e-10:
                            # Gradient contribution to maintain distance
                            # This is a soft constraint
                            pass  # For now, use pure energy gradient
        
        # Momentum update
        velocity = beta_momentum * velocity - 0.01 * effective_grad
        
        # Project velocity to tangent space
        vel_dot = np.einsum('ij,ij->i', velocity, pts)
        velocity_tangent = velocity - vel_dot[:, np.newaxis] * pts
        
        # Line search
        alpha = 0.01
        improved = False
        best_alpha = 0.0
        best_energy = current_energy
        best_pts_new = pts.copy()
        
        for backtrack in range(30):
            pts_new = pts + alpha * velocity_tangent
            pts_new = _project_to_sphere(pts_new)
            
            new_energy, _ = _log_energy_and_grad(pts_new)
            
            if new_energy < current_energy:
                best_alpha = alpha
                best_energy = new_energy
                best_pts_new = pts_new
                improved = True
            else:
                alpha *= 0.7
            
            if alpha < 1e-12:
                break
        
        if not improved:
            # Try smaller step
            alpha = 0.001
            pts_new = pts + alpha * velocity_tangent
            pts_new = _project_to_sphere(pts_new)
            new_energy, _ = _log_energy_and_grad(pts_new)
            if new_energy < current_energy:
                best_pts_new = pts_new
                best_energy = new_energy
                improved = True
        
        if not improved:
            break
        
        pts = best_pts_new
        current_energy = best_energy
        
        # Recompute gradient
        current_energy, current_grad = _log_energy_and_grad(pts)
        grad_dot = np.einsum('ij,ij->i', current_grad, pts)
        grad_tangent = current_grad - grad_dot[:, np.newaxis] * pts
        
        # Check convergence
        if np.linalg.norm(grad_tangent) < tol:
            break
    
    return pts, current_energy

def _generate_icosahedral_subdivision(n_target=N, subdivisions=3):
    """
    Generate points by subdividing an icosahedron and projecting to sphere.
    This creates a natural mesh structure.
    """
    # Start with icosahedron
    t = (1.0 + np.sqrt(5.0)) / 2.0
    ico_verts = np.array([
        [-1,  t,  0], [ 1,  t,  0], [-1, -t,  0], [ 1, -t,  0],
        [ 0, -1,  t], [ 0,  1,  t], [ 0, -1, -t], [ 0,  1, -t],
        [ t,  0, -1], [ t,  0,  1], [-t,  0, -1], [-t,  0,  1]
    ], dtype=np.float64)
    ico_verts = ico_verts / np.linalg.norm(ico_verts, axis=1, keepdims=True)
    
    # Icosahedron faces
    ico_faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ]
    
    # Subdivide faces
    all_points = ico_verts.copy()
    all_faces = ico_faces.copy()
    
    for _ in range(subdivisions):
        new_faces = []
        edge_midpoints = {}
        
        for face in all_faces:
            a, b, c = face
            # Get or create midpoints
            edges = [(a, b), (b, c), (c, a)]
            mids = []
            for e in edges:
                e_sorted = tuple(sorted(e))
                if e_sorted not in edge_midpoints:
                    mid = (all_points[e[0]] + all_points[e[1]]) / 2.0
                    mid = mid / np.linalg.norm(mid)
                    edge_midpoints[e_sorted] = len(all_points)
                    all_points = np.vstack([all_points, mid])
                mids.append(edge_midpoints[e_sorted])
            
            ma, mb, mc = mids
            # Create 4 new faces
            new_faces.extend([
                [a, ma, mc],
                [b, mb, ma],
                [c, mc, mb],
                [ma, mb, mc]
            ])
        
        all_faces = new_faces
    
    # If we have more points than needed, sample
    if len(all_points) > n_target:
        # Use stratified sampling
        indices = np.linspace(0, len(all_points) - 1, n_target, dtype=int)
        all_points = all_points[indices]
    elif len(all_points) < n_target:
        # Pad with Fibonacci points
        fib = _fibonacci_spiral(n_target - len(all_points))
        all_points = np.vstack([all_points, fib])
    
    return _project_to_sphere(all_points)

def _fibonacci_spiral(n=N, offset=0.5):
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _optimize_mesh_based(pts, time_budget=60.0):
    """
    Main optimization routine using mesh-based approach.
    """
    start_time = time.time()
    pts = _project_to_sphere(pts.copy())
    
    # Build initial mesh connectivity
    adjacency = _build_mesh_connectivity(pts)
    
    # Phase 1: Coarse optimization
    pts, energy = _mesh_constrained_gradient(pts, adjacency, max_iter=2000, tol=1e-8)
    
    elapsed = time.time() - start_time
    if elapsed > time_budget * 0.5:
        return pts, energy
    
    # Phase 2: Rebuild mesh and refine
    adjacency = _build_mesh_connectivity(pts)
    pts, energy = _mesh_constrained_gradient(pts, adjacency, max_iter=2000, tol=1e-10)
    
    elapsed = time.time() - start_time
    if elapsed > time_budget * 0.8:
        return pts, energy
    
    # Phase 3: Fine-tuning with smaller steps
    for _ in range(3):
        elapsed = time.time() - start_time
        if elapsed > time_budget * 0.9:
            break
        pts, energy = _mesh_constrained_gradient(pts, adjacency, max_iter=1000, tol=1e-12)
    
    return pts, energy

def construct():
    # Generate multiple initial configurations
    inits = []
    
    # 1. Icosahedral subdivision (natural mesh structure)
    try:
        inits.append(_generate_icosahedral_subdivision(N, subdivisions=3))
    except Exception:
        inits.append(_fibonacci_spiral(N))
    
    # 2. Fibonacci spiral
    inits.append(_fibonacci_spiral(N, offset=0.5))
    
    # 3. Perturbed Fibonacci
    noise = np.random.normal(0, 0.1, size=(N, 3))
    inits.append(_project_to_sphere(_fibonacci_spiral(N) + noise))
    
    # 4. Another icosahedral subdivision with different subdivisions
    try:
        inits.append(_generate_icosahedral_subdivision(N, subdivisions=2))
    except Exception:
        pass
    
    best_energy = np.inf
    best_pts = inits[0].copy()
    
    # Optimize from each initial configuration
    for idx, x0 in enumerate(inits):
        try:
            res_pts, res_energy = _optimize_mesh_based(x0, time_budget=30.0)
            if res_energy < best_energy:
                best_energy = res_energy
                best_pts = res_pts.copy()
        except Exception:
            pass
    
    # Final refinement
    try:
        res_pts, res_energy = _optimize_mesh_based(best_pts, time_budget=40.0)
        if res_energy < best_energy:
            best_energy = res_energy
            best_pts = res_pts
    except Exception:
        pass
    
    # Final projection to ensure unit sphere
    best_pts = _project_to_sphere(best_pts)
    
    # Verify no points are too close
    dist = distance_matrix(best_pts, best_pts)
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)
    
    if min_dist < 1e-9:
        # If points are too close, fall back to Fibonacci spiral
        best_pts = _fibonacci_spiral(N)
        
    # Ensure finite values
    if not np.all(np.isfinite(best_pts)):
        best_pts = _fibonacci_spiral(N)
        
    return best_pts
# EVOLVE-BLOCK-END
