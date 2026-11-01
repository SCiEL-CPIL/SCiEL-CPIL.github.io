import numpy as np
import itertools
from scipy.optimize import minimize
import time

N = 14
D = 3

def _pairwise_distances_squared(points):
    """Compute squared pairwise distances for a set of points."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    return np.sum(diff ** 2, axis=2)

def _exact_objective(points):
    """Compute the squared ratio of max to min pairwise distance."""
    d2 = _pairwise_distances_squared(points)
    mask = ~np.eye(N, dtype=bool)
    d2_masked = d2[mask]
    min_d2 = np.min(d2_masked)
    max_d2 = np.max(d2_masked)
    if min_d2 <= 1e-10:
        return 1e10
    return (max_d2 / min_d2)

def _fibonacci_sphere(n, radius=1.0):
    """Generate n points uniformly distributed on a sphere using Fibonacci lattice."""
    golden_ratio = (1 + np.sqrt(5)) / 2
    indices = np.arange(n)
    z = 1 - 2 * (indices + 0.5) / n
    radius_per_point = np.sqrt(1 - z**2)
    theta = 2 * np.pi * indices / golden_ratio
    x = radius_per_point * np.cos(theta)
    y = radius_per_point * np.sin(theta)
    pts = np.column_stack((x, y, z))
    return pts * radius

def _icosahedron_points(scale=1.0):
    """Return 12 vertices of a regular icosahedron."""
    phi = (1 + np.sqrt(5)) / 2
    pts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    # Normalize to unit sphere
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms
    return pts * scale

def _octahedron_points(scale=1.0):
    """Return 6 vertices of a regular octahedron."""
    pts = np.array([
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1]
    ], dtype=float)
    return pts * scale

def _cube_points(scale=1.0):
    """Return 8 vertices of a cube."""
    pts = np.array(list(itertools.product([-1, 1], repeat=3)), dtype=float)
    return pts * (scale / np.sqrt(3))

def _generate_initializations():
    """Generate diverse initial point configurations."""
    inits = []
    
    # 1. Fibonacci sphere with various radii
    for r in [0.8, 1.0, 1.2]:
        inits.append(_fibonacci_sphere(N, radius=r))
    
    # 2. Icosahedron + 2 extra points
    ico = _icosahedron_points(1.0)
    extra = np.array([[0, 0, 1.5], [0, 0, -1.5]])
    inits.append(np.vstack([ico, extra]))
    
    # 3. Octahedron + 8 cube points
    octa = _octahedron_points(1.0)
    cube = _cube_points(0.7)
    inits.append(np.vstack([octa, cube]))
    
    # 4. Two concentric shells: 7 inner + 7 outer
    inner = _fibonacci_sphere(7, radius=0.5)
    outer = _fibonacci_sphere(7, radius=1.5)
    inits.append(np.vstack([inner, outer]))
    
    # 5. Random perturbations of structured configs
    base_configs = [
        _fibonacci_sphere(N, radius=1.0),
        np.vstack([_icosahedron_points(1.0), np.array([[0, 0, 1.5], [0, 0, -1.5]])])
    ]
    for base in base_configs:
        for _ in range(3):
            perturbed = base + np.random.randn(N, D) * 0.1
            inits.append(perturbed)
    
    # 6. Pure random
    for _ in range(5):
        pts = np.random.randn(N, D)
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        inits.append(pts)
    
    return inits

def _local_refine(points, time_limit=2.0):
    """Local refinement using Nelder-Mead on the exact objective."""
    start = time.time()
    try:
        res = minimize(
            _exact_objective,
            points,
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-9, 'adaptive': True}
        )
        if time.time() - start > time_limit:
            return points
        return res.x
    except Exception:
        return points

def _coordinate_descent(points, max_iter=200):
    """Simple coordinate descent: optimize one point at a time."""
    pts = points.copy()
    for _ in range(max_iter):
        for i in range(N):
            # Optimize point i while keeping others fixed
            def obj_i(xi):
                candidate = pts.copy()
                candidate[i] = xi
                return _exact_objective(candidate)
            
            try:
                res = minimize(obj_i, pts[i], method='Nelder-Mead', 
                             options={'maxiter': 50, 'xatol': 1e-8, 'fatol': 1e-8})
                pts[i] = res.x
            except Exception:
                pass
    return pts

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    best_score = 1e10
    best_points = None
    
    inits = _generate_initializations()
    
    for idx, init in enumerate(inits):
        if time.time() > deadline:
            break
        
        current_points = init.copy()
        
        # Phase 1: Quick Nelder-Mead refinement
        if time.time() < deadline - 10:
            current_points = _local_refine(current_points, time_limit=3.0)
        
        # Phase 2: Coordinate descent for further improvement
        if time.time() < deadline - 5:
            current_points = _coordinate_descent(current_points, max_iter=50)
        
        # Phase 3: Final Nelder-Mead polish
        if time.time() < deadline - 2:
            current_points = _local_refine(current_points, time_limit=2.0)
        
        score = _exact_objective(current_points)
        if score < best_score:
            best_score = score
            best_points = current_points.copy()
        
        # If we're getting good scores, try more refinements on the best
        if best_score < 5.0 and time.time() < deadline - 15:
            # Additional refinement on the current best
            for _ in range(3):
                if time.time() > deadline - 5:
                    break
                refined = _local_refine(best_points, time_limit=2.0)
                s = _exact_objective(refined)
                if s < best_score:
                    best_score = s
                    best_points = refined.copy()
    
    if best_points is None:
        # Fallback
        best_points = _fibonacci_sphere(N, radius=1.0)
    
    return best_points
