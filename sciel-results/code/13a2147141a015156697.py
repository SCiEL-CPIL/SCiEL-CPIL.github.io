"""New program: Geodesic dome + constrained Newton solver on tangent space."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import math


def _project_to_sphere(points):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return points / norms


def _log_energy(points):
    """Compute logarithmic energy with fsum for accuracy."""
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    upper_idx = np.triu_indices(n, k=1)
    vals = np.log(1.0 / dists[upper_idx])
    return math.fsum(vals)


def _compute_gradient(points):
    """
    Compute the gradient of log energy.
    E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log|x_i - x_j|
    dE/dx_i = -sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    """
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]  # (n, n, 3)
    d2 = np.sum(diff ** 2, axis=2)  # (n, n)
    
    np.fill_diagonal(d2, 1.0)
    
    scale = 1.0 / d2  # (n, n)
    scale[range(n), range(n)] = 0.0
    
    grad = -np.sum(diff * scale[:, :, np.newaxis], axis=1)  # (n, 3)
    return grad


def _compute_hessian_tangent(points):
    """
    Compute the Hessian of log energy projected onto tangent space.
    This is complex, so we use a finite-difference approximation of the 
    gradient in tangent space to estimate the Hessian.
    """
    n = len(points)
    # We'll build the Hessian block by block using finite differences
    # of the gradient in the tangent space
    
    # For each point i, we need the 2x2 Hessian block in tangent space
    # We'll use a small perturbation in tangent directions
    
    # First, compute gradient
    grad = _compute_gradient(points)
    
    # For each point, create two tangent vectors
    hess_blocks = []
    eps = 1e-7
    
    for i in range(n):
        p = points[i]
        # Create orthonormal tangent basis
        # Use Gram-Schmidt
        if abs(p[2]) < 0.9:
            v1 = np.array([1.0, 0.0, 0.0])
        else:
            v1 = np.array([0.0, 1.0, 0.0])
        
        v1 = v1 - np.dot(v1, p) * p
        v1 = v1 / np.linalg.norm(v1)
        
        v2 = np.cross(p, v1)
        v2 = v2 / np.linalg.norm(v2)
        
        # Compute gradient at perturbed points
        # Perturb point i in v1 direction
        p_plus1 = points.copy()
        p_plus1[i] = p + eps * v1
        p_plus1 = _project_to_sphere(p_plus1)
        grad_plus1 = _compute_gradient(p_plus1)
        
        # Perturb point i in -v1 direction
        p_minus1 = points.copy()
        p_minus1[i] = p - eps * v1
        p_minus1 = _project_to_sphere(p_minus1)
        grad_minus1 = _compute_gradient(p_minus1)
        
        # Perturb point i in v2 direction
        p_plus2 = points.copy()
        p_plus2[i] = p + eps * v2
        p_plus2 = _project_to_sphere(p_plus2)
        grad_plus2 = _compute_gradient(p_plus2)
        
        # Perturb point i in -v2 direction
        p_minus2 = points.copy()
        p_minus2[i] = p - eps * v2
        p_minus2 = _project_to_sphere(p_minus2)
        grad_minus2 = _compute_gradient(p_minus2)
        
        # Hessian entries: d(grad_j)/d(theta_i) where theta_i are tangent coords
        # H[0,0] = (grad_plus1[j] - grad_minus1[j]) / (2*eps) for j component
        # But we need to project the gradient change onto the tangent space
        
        # For the diagonal blocks, we approximate the 2x2 Hessian
        H_ii = np.zeros((2, 2))
        
        # d(grad)/d(theta_1) projected onto tangent space
        dg1 = (grad_plus1 - grad_minus1) / (2 * eps)
        # Project dg1 onto tangent space at each point
        for j in range(n):
            dg1[j] = dg1[j] - np.dot(dg1[j], points[j]) * points[j]
        
        # The Hessian block H_ii[0,0] = v1 . (d/dtheta_1 grad)_i
        # But grad is a vector field, so we need to be more careful
        # Actually, for Newton's method on the manifold, we want the 
        # Riemannian Hessian, which is complex. Let's use a simpler approach:
        # treat it as Euclidean Newton with projection.
        
        # For now, use the Euclidean Hessian approximation
        H_ii[0, 0] = np.dot(dg1[i], v1)
        H_ii[0, 1] = np.dot(dg1[i], v2)
        
        dg2 = (grad_plus2 - grad_minus2) / (2 * eps)
        for j in range(n):
            dg2[j] = dg2[j] - np.dot(dg2[j], points[j]) * points[j]
        
        H_ii[1, 0] = np.dot(dg2[i], v1)
        H_ii[1, 1] = np.dot(dg2[i], v2)
        
        hess_blocks.append(H_ii)
    
    return hess_blocks


def _newton_tangent(points, max_iter=500, tol=1e-12):
    """
    Newton's method on the sphere's tangent space.
    Uses finite-difference Hessian approximation.
    """
    n = len(points)
    points = points.copy()
    current_energy = _log_energy(points)
    
    for iteration in range(max_iter):
        # Compute gradient
        grad = _compute_gradient(points)
        
        # Check convergence
        grad_norm = np.linalg.norm(grad)
        if grad_norm < tol:
            break
        
        # Compute Hessian blocks (expensive, so we do it every few iterations)
        if iteration % 5 == 0 or iteration == 0:
            hess_blocks = _compute_hessian_tangent(points)
        
        # Build the full Newton system
        # For each point i, we have 2 tangent coordinates (theta1, theta2)
        # The gradient in tangent space is [grad_i . v1_i, grad_i . v2_i]
        
        # Solve for the step in tangent space
        step_tangent = np.zeros((n, 2))
        
        for i in range(n):
            p = points[i]
            # Create tangent basis
            if abs(p[2]) < 0.9:
                v1 = np.array([1.0, 0.0, 0.0])
            else:
                v1 = np.array([0.0, 1.0, 0.0])
            
            v1 = v1 - np.dot(v1, p) * p
            v1 = v1 / np.linalg.norm(v1)
            
            v2 = np.cross(p, v1)
            v2 = v2 / np.linalg.norm(v2)
            
            # Gradient in tangent coordinates
            g_tangent = np.array([
                np.dot(grad[i], v1),
                np.dot(grad[i], v2)
            ])
            
            # Hessian block for this point
            H_ii = hess_blocks[i]
            
            # Regularize if needed
            det = H_ii[0, 0] * H_ii[1, 1] - H_ii[0, 1] * H_ii[1, 0]
            if abs(det) < 1e-10:
                H_ii = H_ii + 1e-6 * np.eye(2)
            
            try:
                step_tangent[i] = np.linalg.solve(H_ii, -g_tangent)
            except np.linalg.LinAlgError:
                # Fall back to gradient descent step
                step_tangent[i] = -0.01 * g_tangent
        
        # Apply the step with line search
        best_energy = current_energy
        best_points = points
        step_scale = 1.0
        
        for trial in range(20):
            new_points = points.copy()
            for i in range(n):
                p = new_points[i]
                if abs(p[2]) < 0.9:
                    v1 = np.array([1.0, 0.0, 0.0])
                else:
                    v1 = np.array([0.0, 1.0, 0.0])
                
                v1 = v1 - np.dot(v1, p) * p
                v1 = v1 / np.linalg.norm(v1)
                
                v2 = np.cross(p, v1)
                v2 = v2 / np.linalg.norm(v2)
                
                new_points[i] = p + step_scale * (
                    step_tangent[i, 0] * v1 + step_tangent[i, 1] * v2
                )
            
            new_points = _project_to_sphere(new_points)
            
            # Check minimum distance
            diff_t = new_points[:, np.newaxis, :] - new_points[np.newaxis, :, :]
            dists_t = np.linalg.norm(diff_t, axis=2)
            np.fill_diagonal(dists_t, np.inf)
            if np.min(dists_t) < 1e-9:
                step_scale *= 0.5
                continue
            
            new_energy = _log_energy(new_points)
            if new_energy < best_energy:
                best_energy = new_energy
                best_points = new_points
                step_scale *= 1.5
            else:
                step_scale *= 0.5
        
        if best_energy < current_energy:
            current_energy = best_energy
            points = best_points
        else:
            break
    
    return points, current_energy


def construct():
    """
    Construct 150 points on the unit sphere using a geodesic dome as initial positions,
    then refine with constrained Newton solver on the tangent space.
    """
    # Step 1: Create a high-resolution geodesic dome
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
        """Subdivide a triangular mesh by adding midpoints."""
        new_vertices = list(vertices)
        new_faces = []
        edge_cache = {}
        
        def get_midpoint(i, j):
            key = (min(i, j), max(i, j))
            if key in edge_cache:
                return edge_cache[key]
            mid = (vertices[i] + vertices[j]) / 2.0
            mid = mid / np.linalg.norm(mid)
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
    
    # Subdivide multiple times to get a high-resolution dome
    verts, faces = subdivide(ico_verts, ico_faces)  # 42 vertices
    verts, faces = subdivide(verts, faces)  # 162 vertices
    
    # Select N vertices from the geodesic dome using greedy farthest-point sampling
    if len(verts) > N:
        selected = [0]
        remaining = list(range(1, len(verts)))
        
        while len(selected) < N:
            best_idx = -1
            best_min_dist = -1
            
            for idx in remaining:
                dists_to_selected = np.linalg.norm(verts[idx] - verts[selected], axis=1)
                min_dist = np.min(dists_to_selected)
                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_idx = idx
            
            selected.append(best_idx)
            remaining.remove(best_idx)
        
        verts = verts[selected]
    
    # Step 2: Refine with Newton's method, using multiple restarts
    best_points = verts.copy()
    best_energy = _log_energy(best_points)
    
    # Multiple restarts with different perturbations
    for restart in range(8):
        current = best_points.copy()
        if restart > 0:
            rng = np.random.RandomState(42 + restart * 101)
            scale = 0.01 * (1 + restart * 0.5)
            current = current + rng.normal(0, scale, size=current.shape)
            current = _project_to_sphere(current)
        
        refined, energy = _newton_tangent(current, max_iter=300)
        
        if energy < best_energy:
            best_energy = energy
            best_points = refined
    
    # Final refinement
    final_points, final_energy = _newton_tangent(best_points, max_iter=500)
    if final_energy < best_energy:
        best_points = final_points
        best_energy = final_energy
    
    # Final projection
    best_points = _project_to_sphere(best_points)
    
    return best_points
# EVOLVE-BLOCK-END
