"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import math
from scipy.optimize import minimize


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


def _compute_gradient_and_hessian(points):
    """
    Compute the gradient and Hessian of log energy on the tangent space.
    Returns (tangent_grad, hessian_matrix) where hessian is (n, 3, n, 3).
    """
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]  # (n, n, 3)
    d2 = np.sum(diff ** 2, axis=2)  # (n, n)
    d = np.sqrt(d2)
    
    # Avoid division by zero on diagonal
    d2[range(n), range(n)] = 1.0
    d[range(n), range(n)] = 1.0
    
    # Gradient: grad_i = -sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    scale = 1.0 / d2  # (n, n)
    scale[range(n), range(n)] = 0.0
    
    grad = -np.sum(diff * scale[:, :, np.newaxis], axis=1)  # (n, 3)
    
    # Project gradient onto tangent space
    tangent_grad = np.zeros_like(grad)
    for i in range(n):
        p = points[i]
        g = grad[i]
        tangent_grad[i] = g - np.dot(g, p) * p
    
    # Hessian computation (approximate, using finite differences or analytical)
    # For efficiency, we'll use a quasi-Newton approach (BFGS) instead of full Hessian
    return tangent_grad


def _newton_solve(points, max_iter=500, tol=1e-10):
    """
    Newton's method on the tangent space with BFGS Hessian approximation.
    """
    n = len(points)
    points = points.copy()
    current_energy = _log_energy(points)
    
    # BFGS Hessian approximation
    H = np.eye(n * 3)  # Start with identity
    prev_grad = _compute_gradient_and_hessian(points)
    prev_grad_flat = prev_grad.flatten()
    
    for iteration in range(max_iter):
        grad = _compute_gradient_and_hessian(points)
        grad_flat = grad.flatten()
        
        # Check convergence
        if np.linalg.norm(grad_flat) < tol:
            break
        
        # Compute search direction using BFGS
        try:
            direction = -np.linalg.solve(H, grad_flat)
        except np.linalg.LinAlgError:
            direction = -grad_flat
        
        # Project each component back to tangent space
        direction_reshaped = direction.reshape(n, 3)
        tangent_direction = np.zeros_like(direction_reshaped)
        for i in range(n):
            p = points[i]
            d = direction_reshaped[i]
            tangent_direction[i] = d - np.dot(d, p) * p
        
        # Line search
        step = 0.1
        best_energy = current_energy
        best_points = points
        
        for trial in range(30):
            trial_points = points + step * tangent_direction
            trial_points = _project_to_sphere(trial_points)
            
            # Check minimum distance
            diff_t = trial_points[:, np.newaxis, :] - trial_points[np.newaxis, :, :]
            dists_t = np.linalg.norm(diff_t, axis=2)
            np.fill_diagonal(dists_t, np.inf)
            if np.min(dists_t) < 1e-9:
                step *= 0.5
                continue
            
            new_energy = _log_energy(trial_points)
            if new_energy < best_energy:
                best_energy = new_energy
                best_points = trial_points
                step *= 1.2
            else:
                step *= 0.5
        
        if best_energy < current_energy:
            # Update BFGS Hessian
            new_grad = _compute_gradient_and_hessian(best_points)
            new_grad_flat = new_grad.flatten()
            
            s = best_points.reshape(-1) - points.reshape(-1)
            y = new_grad_flat - prev_grad_flat
            
            sy = np.dot(s, y)
            if sy > 1e-10:
                Hy = H @ y
                HyH = np.dot(Hy, y)
                if HyH > 1e-10:
                    H = H - (np.outer(s, s) * (np.dot(H, s) @ s) + np.outer(y, y) * HyH) / (sy * HyH) + (np.outer(s, y) + np.outer(y, s)) * np.dot(H, s) / (sy * HyH)
            
            current_energy = best_energy
            points = best_points
            prev_grad = new_grad
            prev_grad_flat = new_grad_flat
        else:
            break
    
    return points, current_energy


def construct():
    """
    Construct 150 points using a high-resolution geodesic dome and Newton's method.
    """
    # Create icosahedron
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [-1,  phi, 0], [ 1,  phi, 0], [-1, -phi, 0], [ 1, -phi, 0],
        [0, -1,  phi], [0,  1,  phi], [0, -1, -phi], [0,  1, -phi],
        [ phi, 0, -1], [ phi, 0,  1], [-phi, 0, -1], [-phi, 0,  1]
    ], dtype=float)
    ico_verts = ico_verts / np.linalg.norm(ico_verts, axis=1, keepdims=True)
    
    ico_faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ]
    
    def subdivide(vertices, faces):
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
    
    # Subdivide to get ~642 vertices (level 4)
    verts, faces = subdivide(ico_verts, ico_faces)  # 42
    verts, faces = subdivide(verts, faces)  # 162
    verts, faces = subdivide(verts, faces)  # 642
    
    # Select N vertices using greedy farthest-point sampling
    selected = [0]
    remaining = list(range(1, len(verts)))
    min_dists = np.linalg.norm(verts - verts[0], axis=1)
    min_dists[0] = -1
    
    while len(selected) < N:
        best_idx = np.argmax(min_dists[remaining])
        best_idx = remaining[best_idx]
        selected.append(best_idx)
        remaining.remove(best_idx)
        dists_to_new = np.linalg.norm(verts - verts[best_idx], axis=1)
        min_dists = np.minimum(min_dists, dists_to_new)
        min_dists[best_idx] = -1
    
    verts = verts[selected]
    
    # Refine with Newton's method
    best_points = verts.copy()
    best_energy = _log_energy(best_points)
    
    for restart in range(8):
        current = best_points.copy()
        if restart > 0:
            rng = np.random.RandomState(777 + restart * 101)
            scale = 0.003 * (1 + restart * 0.5)
            current = current + rng.normal(0, scale, size=current.shape)
            current = _project_to_sphere(current)
        
        refined, energy = _newton_solve(current, max_iter=300)
        
        if energy < best_energy:
            best_energy = energy
            best_points = refined
    
    # Final refinement
    final_points, final_energy = _newton_solve(best_points, max_iter=500, tol=1e-12)
    if final_energy < best_energy:
        best_points = final_points
    
    best_points = _project_to_sphere(best_points)
    return best_points
# EVOLVE-BLOCK-END
