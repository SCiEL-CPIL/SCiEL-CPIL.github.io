"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import math

def construct():
    """
    Construct 150 points on the unit sphere minimizing logarithmic energy.
    Uses a geodesic dome (subdivided icosahedron) as initial configuration,
    then refines via constrained optimization on the sphere.
    """
    # Step 1: Generate initial points using a high-resolution geodesic dome
    # Start with an icosahedron and subdivide
    
    # Icosahedron vertices
    phi = (1 + math.sqrt(5)) / 2  # golden ratio
    ico_vertices = np.array([
        [-1,  phi, 0], [ 1,  phi, 0], [-1, -phi, 0], [ 1, -phi, 0],
        [ 0, -1,  phi], [ 0,  1,  phi], [ 0, -1, -phi], [ 0,  1, -phi],
        [ phi, 0, -1], [ phi, 0,  1], [-phi, 0, -1], [-phi, 0,  1]
    ], dtype=float)
    
    # Normalize
    ico_vertices = ico_vertices / np.linalg.norm(ico_vertices, axis=1, keepdims=True)
    
    # Subdivide the icosahedron to get more points
    # We need approximately 150 points. An icosahedron has 12 vertices, 30 edges, 20 faces.
    # After one subdivision (triangulating each face into 4), we get 42 vertices.
    # After two subdivisions, we get 162 vertices. That's close to 150.
    # Let's do two levels of subdivision and then select 150 points.
    
    def subdivide(vertices, faces):
        """Subdivide each face into 4 smaller triangles."""
        new_vertices = list(vertices)
        new_faces = []
        edge_cache = {}
        
        def midpoint(v1_idx, v2_idx):
            key = (min(v1_idx, v2_idx), max(v1_idx, v2_idx))
            if key in edge_cache:
                return edge_cache[key]
            
            v1 = np.array(new_vertices[v1_idx])
            v2 = np.array(new_vertices[v2_idx])
            mid = (v1 + v2) / 2.0
            mid = mid / np.linalg.norm(mid)  # project back to sphere
            
            new_vertices.append(mid)
            idx = len(new_vertices) - 1
            edge_cache[key] = idx
            return idx
        
        for face in faces:
            v0, v1, v2 = face
            m01 = midpoint(v0, v1)
            m12 = midpoint(v1, v2)
            m02 = midpoint(v0, v2)
            
            new_faces.append((v0, m01, m02))
            new_faces.append((m01, v1, m12))
            new_faces.append((m02, m12, v2))
            new_faces.append((m01, m12, m02))
        
        return np.array(new_vertices), new_faces
    
    # Initial faces of icosahedron
    ico_faces = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)
    ]
    
    # Two levels of subdivision
    verts, faces = subdivide(ico_vertices, ico_faces)
    verts, faces = subdivide(verts, faces)
    
    # Now we have 162 vertices. We need 150.
    # Select 150 points from these 162. We'll use a greedy approach to select
    # points that are well-spread, or just take the first 150.
    # Actually, let's just take all 162 and then optimize 150 of them.
    # Better: randomly select 150 from the 162, or use a deterministic selection.
    
    # For determinism, let's just take the first 150 vertices
    initial_points = verts[:150]
    
    # Step 2: Refine using constrained optimization
    # The logarithmic energy is: E = sum_{i<j} log(1/||x_i - x_j||) = -sum_{i<j} log(||x_i - x_j||)
    
    def log_energy(x):
        """Compute logarithmic energy for points on sphere."""
        # x is a flat array of 3*N values
        pts = x.reshape(-1, 3)
        energy = 0.0
        n = len(pts)
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(pts[i] - pts[j])
                if d < 1e-12:
                    return 1e10  # penalty for coincident points
                energy += math.log(1.0 / d)
        return energy
    
    def log_energy_grad(x):
        """Compute gradient of logarithmic energy."""
        pts = x.reshape(-1, 3)
        n = len(pts)
        grad = np.zeros_like(pts)
        for i in range(n):
            for j in range(i+1, n):
                diff = pts[i] - pts[j]
                d = np.linalg.norm(diff)
                if d < 1e-12:
                    continue
                # d/dx_i log(1/d) = -1/d * d/dx_i d = -1/d * (x_i - x_j)/d = -(x_i - x_j)/d^2
                # d/dx_j log(1/d) = -1/d * d/dx_j d = -1/d * (x_j - x_i)/d = (x_i - x_j)/d^2
                term = diff / (d * d)
                grad[i] -= term
                grad[j] += term
        return grad.flatten()
    
    # Use L-BFGS-B with bounds, but we need to project back to sphere after each step.
    # Better approach: use scipy.optimize.minimize with a custom projection.
    
    # Alternative: use gradient descent with projection onto sphere
    # Let's use a simpler approach: iteratively move points in the negative gradient
    # direction, then project back to the unit sphere.
    
    points = initial_points.copy()
    
    # Gradient descent with adaptive step size
    lr = 0.01
    best_energy = log_energy(points.flatten())
    best_points = points.copy()
    
    for iteration in range(5000):
        # Compute gradient
        pts = points.reshape(-1, 3)
        n = len(pts)
        grad = np.zeros_like(pts)
        for i in range(n):
            for j in range(i+1, n):
                diff = pts[i] - pts[j]
                d = np.linalg.norm(diff)
                if d < 1e-12:
                    continue
                term = diff / (d * d)
                grad[i] -= term
                grad[j] += term
        
        # Project gradient to tangent space: g_tangent = g - (g . x) x
        # For each point, the constraint is ||x_i|| = 1
        for i in range(n):
            g_dot_x = np.dot(grad[i], pts[i])
            grad[i] = grad[i] - g_dot_x * pts[i]
        
        # Update
        new_points = points - lr * grad
        
        # Project back to unit sphere
        for i in range(n):
            norm = np.linalg.norm(new_points[i])
            if norm > 1e-12:
                new_points[i] = new_points[i] / norm
            else:
                new_points[i] = points[i]
        
        new_energy = log_energy(new_points.flatten())
        
        if new_energy < best_energy:
            best_energy = new_energy
            best_points = new_points.copy()
            points = new_points
        else:
            # Reduce learning rate
            lr *= 0.95
        
        if lr < 1e-8:
            break
    
    result = best_points
    
    # Final check: ensure all points are on unit sphere and finite
    for i in range(len(result)):
        norm = np.linalg.norm(result[i])
        if norm > 1e-12:
            result[i] = result[i] / norm
        if not np.all(np.isfinite(result[i])):
            # Fallback to Fibonacci spiral for this point
            idx = i
            z = 1.0 - 2.0 * (idx + 0.5) / N
            phi_val = math.pi * (1.0 + math.sqrt(5.0)) * (idx + 0.5)
            r = math.sqrt(1.0 - z * z)
            result[i] = [r * math.cos(phi_val), r * math.sin(phi_val), z]
    
    return result

# EVOLVE-BLOCK-END
