import numpy as np
import time
import math
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 150
DEADLINE = 115.0  # Safety margin from 120s

def _log_energy(pts):
    """
    Compute logarithmic energy.
    pts: (N, 3) array of points on the unit sphere.
    Returns: energy
    """
    n = len(pts)
    if n < 2:
        return 0.0
    # Compute pairwise distances
    # dist_sq[i,j] = ||x_i - x_j||^2
    dot = pts @ pts.T
    dist_sq = 2.0 - 2.0 * dot
    np.fill_diagonal(dist_sq, 1.0)
    
    # Check for very small distances
    iu = np.triu_indices(n, k=1)
    d_sq = dist_sq[iu]
    if np.any(d_sq < 1e-12):
        return float('inf')
        
    # Energy: sum_{i<j} log(1/d_ij) = -0.5 * sum_{i<j} log(dist_sq_ij)
    log_d_sq = np.log(np.maximum(dist_sq, 1e-300))
    np.fill_diagonal(log_d_sq, 0.0)
    energy = -0.25 * np.sum(log_d_sq)
    return float(energy)

def _objective_and_grad(x_flat):
    """
    Objective and gradient for L-BFGS-B.
    x_flat: flattened (N, 3) array.
    """
    pts = x_flat.reshape(N, 3)
    
    # Normalize points to ensure they stay on the sphere during optimization steps
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    pts = pts / norms
    
    e = _log_energy(pts)
    if not np.isfinite(e):
        return e, np.zeros_like(x_flat)
        
    # Compute Euclidean gradient
    dot = pts @ pts.T
    dist_sq = 2.0 - 2.0 * dot
    np.fill_diagonal(dist_sq, 1.0)
    
    # dE/dx_i = -0.5 sum_{j!=i} (x_i - x_j) / dist_sq_ij
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    weights = -0.5 / dist_sq
    np.fill_diagonal(weights, 0.0)
    
    grad_eucl = np.einsum('ij,ijk->ik', weights, diff)
    
    # Project to tangent space
    proj_factor = np.sum(grad_eucl * pts, axis=1, keepdims=True)
    grad_riem = grad_eucl - proj_factor * pts
    
    return e, grad_riem.flatten()

def _subdivided_icosahedron(n_points):
    """
    Generate a set of points on the sphere using a subdivided icosahedron structure.
    This provides a highly uniform initial distribution.
    """
    # Icosahedron vertices
    phi = (1 + math.sqrt(5)) / 2
    verts = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ])
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Define edges (triangles) of the icosahedron
    edges = [
        [0, 11], [0, 5], [0, 1], [0, 10], [0, 7],
        [0, 3], [0, 4], [1, 5], [5, 11], [11, 10],
        [10, 7], [7, 3], [3, 4], [4, 2], [2, 6],
        [6, 8], [8, 9], [9, 1], [1, 8], [5, 2],
        [11, 4], [10, 2], [7, 6], [3, 9], [4, 8]
    ]
    
    # Subdivide
    # We want roughly n_points. Each subdivision increases faces by 4x.
    # Start with 20 faces.
    # If we subdivide k times, we have 20 * 4^k faces.
    # Points per face are roughly uniform.
    # Let's just generate a large set and downsample if needed, or adjust subdivision level.
    
    # For N=150, a single subdivision (80 faces) gives 12 unique vertices + 80*3/2 = 120 edge midpoints + 80 face centers?
    # Actually, let's just use a simpler method: generate a high-resolution icosahedron and select the best N points.
    # Or, use the fact that a subdivided icosahedron is a good approximation.
    
    # Let's generate a grid on the icosahedron.
    # A simpler approach for "geodesic dome" feel: use a high-resolution Fibonacci sphere but then "snap" or refine?
    # The prompt asks for "vertices of a regular polyhedron... specifically using a high-resolution geodesic dome".
    # I will implement a basic subdivision.
    
    def subdivide(vertices, edges, levels=2):
        for _ in range(levels):
            new_vertices = list(vertices)
            new_edges = []
            mid_cache = {}
            
            def get_midpoint(v1, v2):
                idx = (min(v1, v2), max(v1, v2))
                if idx in mid_cache:
                    return mid_cache[idx]
                mid = (vertices[v1] + vertices[v2]) / 2.0
                mid = mid / np.linalg.norm(mid)
                new_vertices.append(mid)
                mid_cache[idx] = len(new_vertices) - 1
                return mid_cache[idx]
                
            for e in edges:
                v1, v2 = e
                m12 = get_midpoint(v1, v2)
                # We need the third vertex to form triangles, but for just getting points,
                # we can just collect all midpoints and original vertices.
                # However, to maintain the mesh structure for further subdivision, we need triangles.
                # Let's switch to a triangle-based subdivision.
                pass
            
            # This simple edge subdivision is messy. Let's use a different method.
            # Method: Generate points using a recursive subdivision of the icosahedron faces.
            # Since N is small (150), we can generate a dense set and pick the best ones.
            # Or, simply use a "spherical cap" packing approximation.
            
            # Let's try a different "geodesic" initialization:
            # 1. Start with icosahedron vertices.
            # 2. Subdivide each face into 4 triangles (midpoints).
            # 3. Collect all unique vertices.
            # 4. If we have more than N, use a greedy selection or just take the first N (they are uniform).
            # 5. If we have fewer, subdivide again.
            
            break # Abort this complex recursive implementation in favor of a simpler robust one.
            
    # Simpler Robust Geodesic-like Init:
    # Use a high-resolution Fibonacci sphere, but perturb it slightly to break symmetry?
    # Or use the "Ranjan" algorithm?
    
    # Let's stick to the prompt's spirit: "geodesic dome".
    # I will implement a proper icosahedron subdivision.
    
    def _icosa_subdivide(vertices, faces, levels):
        verts = [v.copy() for v in vertices]
        fs = [f.copy() for f in faces]
        
        for _ in range(levels):
            new_faces = []
            mid_cache = {}
            
            def get_midpoint(v1, v2):
                idx = (min(v1, v2), max(v1, v2))
                if idx in mid_cache:
                    return mid_cache[idx]
                mid = (verts[v1] + verts[v2]) / 2.0
                mid = mid / np.linalg.norm(mid)
                verts.append(mid)
                mid_cache[idx] = len(verts) - 1
                return mid_cache[idx]
                
            for f in fs:
                v1, v2, v3 = f
                m12 = get_midpoint(v1, v2)
                m23 = get_midpoint(v2, v3)
                m31 = get_midpoint(v3, v1)
                
                new_faces.append([v1, m12, m31])
                new_faces.append([v2, m23, m12])
                new_faces.append([v3, m31, m23])
                new_faces.append([m12, m23, m31])
                
            fs = new_faces
            
        return np.array(verts), fs

    # Get icosahedron faces
    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 10], [0, 10, 7], [0, 7, 3],
        [0, 3, 4], [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6],
        [7, 3, 9], [3, 4, 8], [4, 2, 6], [2, 6, 8], [6, 8, 9],
        [2, 5, 9], [2, 9, 8], [11, 4, 2], [10, 2, 6], [7, 6, 9]
    ]
    
    # Determine subdivision level
    # Level 0: 12 verts, 20 faces
    # Level 1: 12 + 30 = 42 verts, 80 faces
    # Level 2: 42 + 120 = 162 verts, 320 faces
    # Level 3: 162 + 480 = 642 verts
    
    # For N=150, Level 2 gives 162 points. This is very close.
    level = 2
    verts, _ = _icosa_subdivide(verts, faces, level)
    
    if len(verts) >= n_points:
        return verts[:n_points]
    else:
        # If not enough, subdivide one more time and take first N
        level += 1
        verts, _ = _icosa_subdivide(verts, faces, level)
        return verts[:n_points]

def _fibonacci_sphere(n):
    """Golden-angle Fibonacci spiral points."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_uniform_sphere(n, seed=None):
    if seed is not None:
        np.random.seed(seed)
    theta = np.random.uniform(0, 2 * np.pi, n)
    phi = np.arccos(2 * np.random.uniform(0, 1, n) - 1)
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.column_stack([x, y, z])

def construct():
    global DEADLINE
    start_time = time.time()
    
    best_energy = float('inf')
    best_pts = None
    
    # Initializations
    inits = []
    
    # 1. Geodesic Dome (Subdivided Icosahedron)
    try:
        geo_pts = _subdivided_icosahedron(N)
        inits.append(geo_pts)
    except Exception:
        pass
        
    # 2. Fibonacci Sphere
    inits.append(_fibonacci_sphere(N))
    
    # 3. Random Uniform (multiple seeds)
    for seed in [42, 123, 999]:
        inits.append(_random_uniform_sphere(N, seed=seed))
        
    # Optimization Loop
    for i, init_pts in enumerate(inits):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 20:
            break
            
        try:
            x0 = init_pts.flatten()
            
            # L-BFGS-B optimization
            res = minimize(
                _objective_and_grad,
                x0,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            opt_pts = res.x.reshape(N, 3)
            # Normalize
            norms = np.linalg.norm(opt_pts, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-15)
            opt_pts = opt_pts / norms
            
            e = _log_energy(opt_pts)
            
            if e < best_energy and np.isfinite(e):
                best_energy = e
                best_pts = opt_pts
                
        except Exception as e:
            print(f"Optimization failed for init {i}: {e}")
            continue
            
    # If we have time, do a final polish on the best result
    if best_pts is not None and time.time() - start_time < DEADLINE - 10:
        try:
            x0 = best_pts.flatten()
            res = minimize(
                _objective_and_grad,
                x0,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-10}
            )
            opt_pts = res.x.reshape(N, 3)
            norms = np.linalg.norm(opt_pts, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-15)
            opt_pts = opt_pts / norms
            
            e = _log_energy(opt_pts)
            if e < best_energy and np.isfinite(e):
                best_energy = e
                best_pts = opt_pts
        except Exception:
            pass

    # Fallback
    if best_pts is None:
        best_pts = _fibonacci_sphere(N)

    # Final normalization
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    best_pts = best_pts / norms
    
    return best_pts
