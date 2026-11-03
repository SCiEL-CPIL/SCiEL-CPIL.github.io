"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

def _compute_covering_radius(points):
    """Compute the exact covering radius in degrees using Voronoi vertices of convex hull."""
    try:
        points = points / np.linalg.norm(points, axis=1, keepdims=True)
        hull = ConvexHull(points)
        
        voronoi_vertices = []
        for face in hull.simplices:
            a = points[face[0]]
            b = points[face[1]]
            c = points[face[2]]
            
            n = np.cross(b - a, c - a)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-10:
                continue
            
            d = np.dot(n, a)
            t = d / (n_norm ** 2)
            o = t * n
            o_norm = np.linalg.norm(o)
            if o_norm < 1e-10:
                continue
            o = o / o_norm
            voronoi_vertices.append(o)
        
        if not voronoi_vertices:
            return np.pi * 180.0 / np.pi  # 180 degrees fallback
        
        voronoi_vertices = np.array(voronoi_vertices)
        
        max_dist = 0.0
        for p in points:
            dots = np.dot(voronoi_vertices, p)
            dots = np.clip(dots, -1.0, 1.0)
            dists = np.arccos(dots)
            min_dist = np.min(dists)
            if min_dist > max_dist:
                max_dist = min_dist
        
        return np.degrees(max_dist)
    except Exception:
        return 180.0


def _smooth_covering_radius(points, beta=50.0):
    """Compute smooth approximation of covering radius for gradient-based optimization."""
    try:
        points = points / np.linalg.norm(points, axis=1, keepdims=True)
        hull = ConvexHull(points)
        
        voronoi_vertices = []
        for face in hull.simplices:
            a = points[face[0]]
            b = points[face[1]]
            c = points[face[2]]
            
            n = np.cross(b - a, c - a)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-10:
                continue
            
            d = np.dot(n, a)
            t = d / (n_norm ** 2)
            o = t * n
            o_norm = np.linalg.norm(o)
            if o_norm < 1e-10:
                continue
            o = o / o_norm
            voronoi_vertices.append(o)
        
        if not voronoi_vertices:
            return np.pi
        
        voronoi_vertices = np.array(voronoi_vertices)
        
        max_soft = 0.0
        for p in points:
            dots = np.dot(voronoi_vertices, p)
            dots = np.clip(dots, -1.0, 1.0)
            dists = np.arccos(dots)
            # Soft max approximation
            soft_max = -np.log(np.sum(np.exp(-beta * dists))) / beta
            if soft_max > max_soft:
                max_soft = soft_max
        
        return max_soft
    except Exception:
        return np.pi


def _fibonacci_spiral(n):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_points(n, seed=None):
    """Generate random points on unit sphere."""
    if seed is not None:
        np.random.seed(seed)
    pts = np.random.randn(n, 3)
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    return pts


def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill
    
    best_points = None
    best_radius = float('inf')
    
    # Try multiple initial configurations
    initial_configs = []
    
    # Fibonacci spiral
    initial_configs.append(_fibonacci_spiral(N))
    
    # Random configurations with different seeds
    for seed in [42, 123, 456, 789]:
        initial_configs.append(_random_points(N, seed=seed))
    
    # Spherical code-inspired: start from icosahedral symmetry perturbed
    # Create base icosahedron vertices and subdivide
    phi_g = (1.0 + np.sqrt(5.0)) / 2.0
    ico_verts = np.array([
        [0, 1, phi_g], [0, -1, phi_g], [0, 1, -phi_g], [0, -1, -phi_g],
        [1, phi_g, 0], [-1, phi_g, 0], [1, -phi_g, 0], [-1, -phi_g, 0],
        [phi_g, 0, 1], [-phi_g, 0, 1], [phi_g, 0, -1], [-phi_g, 0, -1]
    ])
    ico_verts = ico_verts / np.linalg.norm(ico_verts, axis=1, keepdims=True)
    
    # Generate additional points by jittering and adding random points
    ico_expanded = ico_verts.copy()
    remaining = N - len(ico_verts)
    if remaining > 0:
        extra = _random_points(remaining, seed=999)
        ico_expanded = np.vstack([ico_verts, extra])
    # Shuffle
    perm = np.random.permutation(N)
    ico_expanded = ico_expanded[perm]
    initial_configs.append(ico_expanded)
    
    for init_idx, init_pts in enumerate(initial_configs):
        if time.time() > deadline - 10:
            break
        
        # Check if this initial config is already better
        init_radius = _compute_covering_radius(init_pts)
        if init_radius < best_radius:
            best_radius = init_radius
            best_points = init_pts.copy()
        
        # Optimize from this initial configuration
        try:
            res = minimize(
                _smooth_covering_radius,
                init_pts.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 200, 'maxfun': 2000, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            
            if res.x is not None:
                opt_pts = res.x.reshape(N, 3)
                opt_pts = opt_pts / np.linalg.norm(opt_pts, axis=1, keepdims=True)
                opt_radius = _compute_covering_radius(opt_pts)
                
                if opt_radius < best_radius:
                    best_radius = opt_radius
                    best_points = opt_pts.copy()
        except Exception:
            pass
    
    # Final polish: try small perturbations around best solution
    if best_points is not None and time.time() < deadline - 15:
        for _ in range(3):
            if time.time() > deadline - 5:
                break
            noise = np.random.normal(0, 0.005, (N, 3))
            perturbed = best_points + noise
            perturbed = perturbed / np.linalg.norm(perturbed, axis=1, keepdims=True)
            
            try:
                res = minimize(
                    _smooth_covering_radius,
                    perturbed.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'maxfun': 1000, 'ftol': 1e-9, 'gtol': 1e-7}
                )
                if res.x is not None:
                    final_pts = res.x.reshape(N, 3)
                    final_pts = final_pts / np.linalg.norm(final_pts, axis=1, keepdims=True)
                    final_radius = _compute_covering_radius(final_pts)
                    if final_radius < best_radius:
                        best_radius = final_radius
                        best_points = final_pts.copy()
            except Exception:
                pass
    
    if best_points is None:
        best_points = _fibonacci_spiral(N)
    
    return best_points
# EVOLVE-BLOCK-END
