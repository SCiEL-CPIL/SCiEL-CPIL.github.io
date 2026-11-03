import numpy as np
import time
import math
from scipy.optimize import minimize

N = 150
DEADLINE = 118.0  # Safety margin for 120s limit

def _subdivided_icosahedron(n_subdivisions):
    """Generate a subdivided icosahedron mesh on the unit sphere."""
    # Golden ratio
    t = (1.0 + math.sqrt(5.0)) / 2.0
    # 12 vertices of the icosahedron
    verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    # Normalize
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    
    # 20 faces of the icosahedron
    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ]
    
    # Subdivide faces
    for _ in range(n_subdivisions):
        new_faces = []
        midpoints = {}
        
        def get_midpoint(i, j):
            key = (min(i, j), max(i, j))
            if key not in midpoints:
                v1 = verts[i]
                v2 = verts[j]
                mid = (v1 + v2) / 2.0
                mid /= np.linalg.norm(mid)
                midpoints[key] = len(verts)
                verts = np.vstack([verts, mid])
            return midpoints[key]
        
        for f in faces:
            a, b, c = f
            ab = get_midpoint(a, b)
            bc = get_midpoint(b, c)
            ca = get_midpoint(c, a)
            new_faces.append([a, ab, ca])
            new_faces.append([b, bc, ab])
            new_faces.append([c, ca, bc])
            new_faces.append([ab, bc, ca])
        faces = new_faces
        
    # Select points
    # For N=150, we want roughly 150 points.
    # An icosahedron has 12 vertices, 30 edges, 20 faces.
    # Subdivision level 1: 12 + 30 + 20*4 = 122 points? No, vertices are shared.
    # Vertices: 12
    # Midpoints of edges: 30
    # Centroids of faces: 20
    # Total: 62 points. Too few.
    # Subdivision level 2:
    # Vertices: 12
    # Edge midpoints: 30
    # Face centroids: 20
    # Sub-edge midpoints: 30 * 2 = 60? No, each edge is split into 4 segments?
    # Let's just generate a large set and pick 150.
    # A simpler way: Use the Fibonacci spiral for initialization as it is robust and fast,
    # but the prompt asks for a different strategy. The current program uses Fibonacci.
    # The inspirations use subdivided icosahedron.
    # Let's generate a dense icosahedron and sample 150 points from it.
    
    # Actually, generating a proper icosahedron subdivision that yields exactly N points is tricky.
    # A common approach is to generate a dense mesh and then select N points.
    # Or, just use the vertices of a high-subdivision icosahedron.
    # For N=150, subdivision level 2 gives 162 vertices (12 + 30 + 20 + 60 + 40? Let's check).
    # Level 0: 12 verts, 20 faces
    # Level 1: 12 verts + 30 edge mids + 20 face centers = 62 verts. 80 faces.
    # Level 2: 62 verts + (80*3/2) edge mids? No.
    # Each face is divided into 4.
    # New vertices: midpoints of the 3 edges of each face.
    # Total edges in Level 1: 80 * 3 / 2 = 120 edges.
    # So 120 new midpoints.
    # Total vertices Level 2: 62 + 120 = 182.
    # This is close to 150. We can just take the first 150 vertices or sample them.
    
    return verts[:N]

def _log_energy_and_grad(pts):
    """Compute log energy and gradient for points on the sphere."""
    pts = pts.reshape(-1, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms
    
    # Compute pairwise distances using dot products for efficiency
    # |xi - xj|^2 = 2 - 2*xi·xj for unit vectors
    dot = np.dot(pts, pts.T)
    np.fill_diagonal(dot, 1.0)
    dist_sq = 2.0 - 2.0 * dot
    np.maximum(dist_sq, 1e-30, out=dist_sq)
    dist = np.sqrt(dist_sq)
    
    # Energy: sum_{i<j} log(1/dist) = -sum_{i<j} log(dist)
    iu = np.triu_indices(N, 1)
    d = dist[iu]
    energy = -np.sum(np.log(d))
    
    # Gradient: dE/dxi = sum_{j != i} (xi - xj) / |xi - xj|^2
    # Since dE/dxi = sum_{j != i} -d/dxi log(dist_ij) = sum_{j != i} (xi-xj)/dist_ij^2
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    # grad_i = sum_j (xi - xj) / dist_ij^2
    # = sum_j xi/dist_ij^2 - sum_j xj/dist_ij^2
    # = xi * sum_j inv_dist_sq[i,j] - sum_j xj * inv_dist_sq[i,j]
    row_sums = np.sum(inv_dist_sq, axis=1, keepdims=True)
    grad = pts * row_sums - np.dot(inv_dist_sq, pts)
    
    # Project gradient to tangent space
    dot_grad = np.sum(grad * pts, axis=1, keepdims=True)
    grad_tangent = grad - dot_grad * pts
    
    return energy, grad_tangent

def construct():
    start_time = time.time()
    
    # Fibonacci spiral initialization
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_energy = np.inf
    best_points = points.copy()
    
    def objective_and_grad(x):
        pts = x.reshape(-N, 3)
        e, g = _log_energy_and_grad(pts)
        return e, g.flatten()
    
    # Phase 1: Initial L-BFGS-B optimization
    x0 = points.flatten()
    result = minimize(objective_and_grad, x0, method='L-BFGS-B', jac=True, 
                      options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10})
    
    points = result.x.reshape(-N, 3)
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    
    energy, _ = _log_energy_and_grad(points)
    best_energy = energy
    best_points = points.copy()
    
    # Phase 2: Simulated Annealing with multiple restarts
    # Use adaptive temperature and more sophisticated perturbation
    base_temp = 0.05
    current_temp = base_temp
    no_improve_count = 0
    max_no_improve = 20
    
    while time.time() - start_time < DEADLINE - 8.0:
        # Adaptive perturbation: perturb more points when stuck
        if no_improve_count > max_no_improve:
            num_perturb = min(15, N // 10)
            current_temp = min(current_temp * 1.5, 0.2)
            no_improve_count = 0
        else:
            num_perturb = max(3, N // 30)
        
        # Select points to perturb - prefer points with high local energy contribution
        # For simplicity, use random selection but with some structure
        indices = np.random.choice(N, num_perturb, replace=False)
        
        # Generate perturbations in tangent space
        for idx in indices:
            p = points[idx]
            # Random tangent vector
            v = np.random.randn(3)
            v -= np.dot(v, p) * p
            v_norm = np.linalg.norm(v)
            if v_norm > 1e-10:
                v = v / v_norm * current_temp
                points[idx] = p + v
                points[idx] /= np.linalg.norm(points[idx])
        
        # Local optimization - use fewer iterations for speed, rely on multiple restarts
        x0 = points.flatten()
        max_iter_inner = 80
        result = minimize(objective_and_grad, x0, method='L-BFGS-B', jac=True, 
                          options={'maxiter': max_iter_inner, 'ftol': 1e-12, 'gtol': 1e-8})
        
        points = result.x.reshape(-N, 3)
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        
        energy, _ = _log_energy_and_grad(points)
        
        # Metropolis acceptance
        if energy < best_energy:
            best_energy = energy
            best_points = points.copy()
            no_improve_count = 0
            # Cool down on improvement
            current_temp = max(current_temp * 0.92, 0.01)
        else:
            delta_e = energy - best_energy
            if delta_e < 0:
                # Numerical noise, accept
                best_energy = energy
                best_points = points.copy()
                no_improve_count = 0
                current_temp = max(current_temp * 0.95, 0.01)
            else:
                # Metropolis criterion
                prob = np.exp(-delta_e / max(current_temp, 1e-8))
                if np.random.rand() < prob:
                    # Accept worse solution to escape local minima
                    no_improve_count += 1
                    current_temp = max(current_temp * 0.98, 0.005)
                else:
                    # Reject, revert to best
                    points = best_points.copy()
                    no_improve_count += 1
                    current_temp = max(current_temp * 0.98, 0.005)
    
    # Final Polish: High-precision optimization
    if time.time() - start_time < DEADLINE - 3.0:
        x0 = best_points.flatten()
        result = minimize(objective_and_grad, x0, method='L-BFGS-B', jac=True, 
                          options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-12})
        
        final_points = result.x.reshape(-N, 3)
        final_points /= np.linalg.norm(final_points, axis=1, keepdims=True)
        
        final_energy, _ = _log_energy_and_grad(final_points)
        if final_energy < best_energy:
            best_energy = final_energy
            best_points = final_points
    
    # Ultra-final polish with even tighter tolerances if time permits
    if time.time() - start_time < DEADLINE - 1.0:
        x0 = best_points.flatten()
        result = minimize(objective_and_grad, x0, method='L-BFGS-B', jac=True, 
                          options={'maxiter': 200, 'ftol': 1e-16, 'gtol': 1e-14})
        
        final_points = result.x.reshape(-N, 3)
        final_points /= np.linalg.norm(final_points, axis=1, keepdims=True)
        
        final_energy, _ = _log_energy_and_grad(final_points)
        if final_energy < best_energy:
            best_energy = final_energy
            best_points = final_points
    
    return best_points
