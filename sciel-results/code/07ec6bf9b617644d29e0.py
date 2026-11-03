"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import math


def _project_to_sphere(points):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return points / norms


def _log_energy_fast(points):
    """Faster computation using vectorized distance matrix."""
    n = len(points)
    # Compute all pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    # Use fsum for the upper triangle
    upper_idx = np.triu_indices(n, k=1)
    vals = np.log(1.0 / dists[upper_idx])
    return math.fsum(vals)


def _compute_gradients(points):
    """Compute the gradient of log energy for all points, vectorized."""
    n = len(points)
    # Compute pairwise differences and distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]  # (n, n, 3)
    d2 = np.sum(diff ** 2, axis=2)  # (n, n)
    
    # Set diagonal to 1 to avoid division by zero
    np.fill_diagonal(d2, 1.0)
    
    # Gradient: grad_i = -sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    # = -sum_{j != i} diff[i,j] / d2[i,j]
    # Scale each diff by 1/d2
    scale = 1.0 / d2  # (n, n)
    scale[range(n), range(n)] = 0.0  # Zero out diagonal
    
    grad = -np.sum(diff * scale[:, :, np.newaxis], axis=1)  # (n, 3)
    
    # Project onto tangent space
    tangent_grad = np.zeros_like(grad)
    for i in range(n):
        p = points[i]
        g = grad[i]
        g_tangent = g - np.dot(g, p) * p
        tangent_grad[i] = g_tangent
    
    return tangent_grad


def _newton_step(points, step_size=0.01, max_iter=500, momentum=0.9):
    """
    Apply gradient descent with momentum on the sphere's tangent space.
    """
    n = len(points)
    points = points.copy()
    velocity = np.zeros_like(points)
    
    for iteration in range(max_iter):
        # Compute gradient in tangent space
        tangent_grad = _compute_gradients(points)
        
        # Update velocity with momentum
        velocity = momentum * velocity - step_size * tangent_grad
        
        # Move in tangent direction and re-project
        new_points = points + velocity
        new_points = _project_to_sphere(new_points)
        
        # Check for minimum distance
        diff = new_points[:, np.newaxis, :] - new_points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_d = np.min(dists)
        
        if min_d < 1e-9:
            # Backtrack
            velocity = -0.5 * velocity
            continue
        
        # Adaptive step size with line search
        best_step = step_size
        best_energy = _log_energy_fast(points)
        
        for trial in range(10):
            step = best_step * (0.7 ** trial)
            trial_points = points - step * tangent_grad
            trial_points = _project_to_sphere(trial_points)
            
            # Check minimum distance
            diff_t = trial_points[:, np.newaxis, :] - trial_points[np.newaxis, :, :]
            dists_t = np.linalg.norm(diff_t, axis=2)
            np.fill_diagonal(dists_t, np.inf)
            if np.min(dists_t) < 1e-9:
                continue
            
            new_energy = _log_energy_fast(trial_points)
            if new_energy < best_energy:
                best_energy = new_energy
                best_step = step
                new_points = trial_points
        
        if best_step < 1e-10:
            break
        
        points = _project_to_sphere(new_points)
        
        # Check convergence
        if iteration > 0 and iteration % 10 == 0:
            current_energy = _log_energy_fast(points)
            if abs(current_energy - best_energy) < 1e-14:
                break
    
    return points


def construct():
    """
    Construct 150 points on the unit sphere using a geodesic dome (subdivided icosahedron)
    as initial positions, then refine with a tangent-space Newton solver.
    """
    # Step 1: Create a high-resolution geodesic dome (subdivided icosahedron)
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    ico_verts = np.array([
        [-1,  phi, 0], [ 1,  phi, 0], [-1, -phi, 0], [ 1, -phi, 0],
        [0, -1,  phi], [0,  1,  phi], [0, -1, -phi], [0,  1, -phi],
        [ phi, 0, -1], [ phi, 0,  1], [-phi, 0, -1], [-phi, 0,  1]
    ], dtype=float)
    ico_verts = ico_verts / np.linalg.norm(ico_verts, axis=1, keepdims=True)
    
    # Icosahedron faces (triangles)
    ico_faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ]
    
    def subdivide(vertices, faces):
        """Subdivide a triangular mesh by adding midpoints and creating 4 new triangles per face."""
        new_vertices = list(vertices)
        new_faces = []
        edge_cache = {}
        
        def get_midpoint(i, j):
            key = (min(i, j), max(i, j))
            if key in edge_cache:
                return edge_cache[key]
            mid = (vertices[i] + vertices[j]) / 2.0
            mid = mid / np.linalg.norm(mid)  # Project to sphere
            new_vertices.append(mid)
            idx = len(new_vertices) - 1
            edge_cache[key] = idx
            return idx
        
        for face in faces:
            a, b, c = face
            ab = get_midpoint(a, b)
            bc = get_midpoint(b, c)
            ca = get_midpoint(c, a)
            new_faces.append([a, ab, ca])
            new_faces.append([b, bc, ab])
            new_faces.append([c, ca, bc])
            new_faces.append([ab, bc, ca])
        
        return np.array(new_vertices), new_faces
    
    # Subdivide twice to get 162 vertices
    verts, faces = subdivide(ico_verts, ico_faces)
    verts, faces = subdivide(verts, faces)
    
    # We have 162 vertices, need 150. Remove the 12 least "important" ones.
    if len(verts) > N:
        # Compute for each vertex the minimum distance to any other vertex
        diff = verts[:, np.newaxis, :] - verts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_dists = np.min(dists, axis=1)
        
        # Remove vertices with smallest min_dist (most redundant)
        remove_indices = np.argsort(min_dists)[:len(verts) - N]
        keep_mask = np.ones(len(verts), dtype=bool)
        keep_mask[remove_indices] = False
        verts = verts[keep_mask]
    
    # Ensure we have exactly N points
    if len(verts) < N:
        n_extra = N - len(verts)
        i = np.arange(n_extra) + 0.5 + len(verts)
        z = 1.0 - 2.0 * i / N
        phi_angle = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        extra = np.column_stack([r * np.cos(phi_angle), r * np.sin(phi_angle), z])
        verts = np.vstack([verts, extra])
    elif len(verts) > N:
        verts = verts[:N]
    
    # Step 2: Refine with tangent-space Newton solver
    # Use multiple restarts with different initial perturbations
    best_points = verts.copy()
    best_energy = _log_energy_fast(best_points)
    
    # Try several iterations of Newton refinement with different step sizes
    step_sizes = [0.01, 0.005, 0.002, 0.001]
    
    for restart in range(8):
        current = best_points.copy()
        if restart > 0:
            # Add small random perturbation
            rng = np.random.RandomState(42 + restart * 7)
            scale = 0.01 + 0.005 * restart
            current = current + rng.normal(0, scale, size=current.shape)
            current = _project_to_sphere(current)
        
        step_size = step_sizes[restart % len(step_sizes)]
        refined = _newton_step(current, step_size=step_size, max_iter=300, momentum=0.8)
        energy = _log_energy_fast(refined)
        
        if energy < best_energy:
            best_energy = energy
            best_points = refined
    
    # Final projection to ensure unit sphere
    best_points = _project_to_sphere(best_points)
    
    return best_points
# EVOLVE-BLOCK-END
