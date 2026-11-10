# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points.
    
    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    start_time = time.time()
    time_limit = 600  # Leave 30s margin from 630s

    # Precompute indices for all triples of points
    idx = np.array(list((i, j, k) for i in range(n) for j in range(i+1, n) for k in range(j+1, n)))
    
    def min_triangle_area_vectorized(points):
        """
        Calculate the minimum triangle area among all triples of points using vectorization.
        points: (n, 2)
        """
        p1 = points[idx[:, 0]]
        p2 = points[idx[:, 1]]
        p3 = points[idx[:, 2]]
        
        areas = 0.5 * np.abs(p1[:, 0] * (p2[:, 1] - p3[:, 1]) + 
                             p2[:, 0] * (p3[:, 1] - p1[:, 1]) + 
                             p3[:, 0] * (p1[:, 1] - p2[:, 1]))
        
        return np.min(areas)

    def all_triangle_areas(points):
        """
        Calculate all triangle areas among all triples of points using vectorization.
        points: (n, 2)
        Returns: array of shape (286,)
        """
        p1 = points[idx[:, 0]]
        p2 = points[idx[:, 1]]
        p3 = points[idx[:, 2]]
        
        areas = 0.5 * np.abs(p1[:, 0] * (p2[:, 1] - p3[:, 1]) + 
                             p2[:, 0] * (p3[:, 1] - p1[:, 1]) + 
                             p3[:, 0] * (p1[:, 1] - p2[:, 1]))
        
        return areas

    def hull_area(points):
        """Calculate the convex hull area of the points."""
        try:
            hull = ConvexHull(points)
            return hull.volume
        except Exception:
            return 0.0

    def soft_min(areas, beta):
        """
        Smooth approximation of min using log-sum-exp trick.
        soft_min(x) = -1/beta * log(sum(exp(-beta * x)))
        As beta -> inf, soft_min -> min
        """
        shifted = -beta * areas
        max_shifted = np.max(shifted)
        log_sum = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
        return -log_sum / beta

    def objective_smooth(points, beta=50.0):
        """
        Smooth objective using soft-min surrogate for the minimum triangle area.
        """
        areas = all_triangle_areas(points)
        A_min_smooth = soft_min(areas, beta)
        
        if A_min_smooth < 1e-15:
            return 0.0
        
        # Use bounding box area as a fast proxy for hull area
        x_min, y_min = np.min(points, axis=0)
        x_max, y_max = np.max(points, axis=0)
        bbox_area = (x_max - x_min) * (y_max - y_min)
        
        if bbox_area < 1e-15:
            return 0.0
            
        return -A_min_smooth / bbox_area

    def objective_exact(points):
        """
        Exact objective: minimize -A_min / A_hull.
        Uses actual ConvexHull area.
        """
        A_hull = hull_area(points)
        if A_hull <= 1e-10:
            return 0.0
        
        A_min = min_triangle_area_vectorized(points)
        
        if A_min < 1e-12:
            return 0.0
            
        return -A_min / A_hull

    def objective_bbox(points):
        """
        Fast objective using bounding box area as proxy for hull area.
        """
        x_min, y_min = np.min(points, axis=0)
        x_max, y_max = np.max(points, axis=0)
        bbox_area = (x_max - x_min) * (y_max - y_min)
        
        if bbox_area < 1e-15:
            return 0.0
        
        A_min = min_triangle_area_vectorized(points)
        
        if A_min < 1e-12:
            return 0.0
            
        return -A_min / bbox_area

    def score(points):
        """Calculate the actual score: A_min / A_hull"""
        A_hull = hull_area(points)
        if A_hull <= 1e-10:
            return 0.0
        A_min = min_triangle_area_vectorized(points)
        if A_min < 1e-12:
            return 0.0
        return A_min / A_hull

    # Generate diverse initial configurations focusing on rectangular/square shapes
    rng = np.random.default_rng(seed=42)
    
    initial_configs = []
    
    # 1. Square-based configurations (key insight: allow hull to be square/rectangle)
    # Square with interior points
    for aspect_ratio in [1.0, 1.2, 0.8, 1.5, 0.67]:
        w = aspect_ratio
        h = 1.0
        # 4 corners of square
        corners = np.array([[-w/2, -h/2], [w/2, -h/2], [w/2, h/2], [-w/2, h/2]])
        # 9 interior points in 3x3 grid
        interior_x = np.linspace(-w/3, w/3, 3)
        interior_y = np.linspace(-h/3, h/3, 3)
        interior_pts = np.array([[x, y] for x in interior_x for y in interior_y])
        initial_configs.append(np.vstack([corners, interior_pts]))
    
    # 2. Rectangle with different aspect ratios
    for ar in [0.5, 0.7, 1.0, 1.3, 1.5, 2.0]:
        w = ar
        h = 1.0
        # Distribute 13 points in a pattern optimized for rectangles
        # 4 corners + 8 edge midpoints + 1 center
        pts = [
            [-w/2, -h/2], [w/2, -h/2], [w/2, h/2], [-w/2, h/2],  # corners
            [0, -h/2], [w/2, 0], [0, h/2], [-w/2, 0],  # edge midpoints
            [-w/4, 0], [w/4, 0], [0, h/4], [0, -h/4],  # interior
        ]
        initial_configs.append(np.array(pts))
    
    # 3. Hexagonal lattice (known good for Heilbronn)
    hex_configs = []
    for offset in [0.0, 0.1, 0.2]:
        pts = []
        pts.append([0.0, 0.0])
        for angle in np.linspace(0, 2*np.pi, 6, endpoint=False):
            r = 1.0 + offset * rng.random()
            pts.append([r * np.cos(angle), r * np.sin(angle)])
        for angle in np.linspace(0, 2*np.pi, 6, endpoint=False) + np.pi/6:
            r = 1.8 + offset * rng.random()
            pts.append([r * np.cos(angle), r * np.sin(angle)])
        hex_configs.append(np.array(pts[:13]))
    initial_configs.extend(hex_configs)
    
    # 4. Regular polygon (baseline)
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    initial_configs.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # 5. Regular polygon with center point and perturbed outer ring
    center_pt = np.array([0.0, 0.0])
    outer_angles = np.linspace(0, 2*np.pi, n-1, endpoint=False)
    outer_pts = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
    initial_configs.append(np.vstack([outer_pts, [center_pt]]))
    
    # 6. Perturbed regular polygon
    for _ in range(3):
        perturbation = rng.normal(0, 0.15, (n, 2))
        initial_configs.append(np.column_stack([np.cos(angles), np.sin(angles)]) + perturbation)
        
    # 7. Random points in unit circle
    for _ in range(3):
        r = np.sqrt(rng.random(n))
        theta = rng.random(n) * 2 * np.pi
        initial_configs.append(np.column_stack([r * np.cos(theta), r * np.sin(theta)]))
        
    # 8. Grid-like with jitter
    grid_x = [-1.5, -0.5, 0.5, 1.5]
    grid_y = [-1.5, -0.5, 0.5, 1.5]
    for _ in range(3):
        pts = []
        used = set()
        while len(pts) < n:
            i = rng.integers(0, 4)
            j = rng.integers(0, 4)
            if (i, j) not in used:
                used.add((i, j))
                x = grid_x[i] + rng.normal(0, 0.1)
                y = grid_y[j] + rng.normal(0, 0.1)
                pts.append([x, y])
        initial_configs.append(np.array(pts))
    
    # 9. Two concentric rings with alternating angles
    for ring_ratio in [0.5, 0.6, 0.7]:
        inner_angles = np.linspace(0, 2*np.pi, 7, endpoint=False)
        outer_angles = np.linspace(0, 2*np.pi, 6, endpoint=False) + np.pi/6
        inner_pts = np.column_stack([ring_ratio * np.cos(inner_angles), ring_ratio * np.sin(inner_angles)])
        outer_pts = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
        initial_configs.append(np.vstack([inner_pts, outer_pts]))
    
    # 10. Fibonacci spiral pattern
    phi = (1 + np.sqrt(5)) / 2
    fib_angles = 2 * np.pi * np.arange(n) / phi
    fib_radii = np.sqrt(np.arange(n) / n) * 0.9
    fib_pts = np.column_stack([fib_radii * np.cos(fib_angles), fib_radii * np.sin(fib_angles)])
    initial_configs.append(fib_pts)
    
    # 11. Flower pattern: 1 center + 6 inner ring + 6 outer ring
    for r_inner in [0.4, 0.5, 0.55]:
        center = np.array([[0.0, 0.0]])
        inner_angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
        inner_pts = np.column_stack([r_inner * np.cos(inner_angles), r_inner * np.sin(inner_angles)])
        outer_angles = np.linspace(0, 2*np.pi, 6, endpoint=False) + np.pi/6
        outer_pts = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
        initial_configs.append(np.vstack([center, inner_pts, outer_pts]))
    
    # 12. Grid-like 4-3-3-3 pattern (rows with decreasing width)
    for jitter in [0.0, 0.05, 0.1]:
        rows = [
            [(-1.5, -1.5), (-0.5, -1.5), (0.5, -1.5), (1.5, -1.5)],
            [(-1.0, 0.0), (0.0, 0.0), (1.0, 0.0)],
            [(-1.0, 1.5), (0.0, 1.5), (1.0, 1.5)],
        ]
        pts = []
        for row in rows:
            for (x, y) in row:
                pts.append([x + rng.normal(0, jitter), y + rng.normal(0, jitter)])
        initial_configs.append(np.array(pts[:13]))
    
    # 13. Two rings with offset: 5 inner + 8 outer
    for r_in in [0.45, 0.55]:
        inner_angles = np.linspace(0, 2*np.pi, 5, endpoint=False)
        outer_angles = np.linspace(0, 2*np.pi, 8, endpoint=False) + np.pi/8
        inner_pts = np.column_stack([r_in * np.cos(inner_angles), r_in * np.sin(inner_angles)])
        outer_pts = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
        initial_configs.append(np.vstack([inner_pts, outer_pts]))
    
    best_points = None
    best_score = -float('inf')
    
    # Phase 1: Quick optimization with smooth objective (multiple beta values)
    for beta in [20.0, 50.0, 100.0]:
        if time.time() - start_time > time_limit - 150:
            break
        
        for i, init_points in enumerate(initial_configs):
            if time.time() - start_time > time_limit - 130:
                break
                
            center = np.mean(init_points, axis=0)
            centered = init_points - center
            norm = np.linalg.norm(centered)
            if norm > 1e-10:
                centered = centered / norm * 1.0
            
            try:
                result = minimize(
                    lambda pts: objective_smooth(pts, beta),
                    centered,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 3000,
                        'xatol': 1e-9,
                        'fatol': 1e-11,
                        'adaptive': True,
                        'initial_simplex': 0.1
                    }
                )
                
                if result.fun < 0:
                    exact_score = score(result.x)
                    if exact_score > best_score:
                        best_score = exact_score
                        best_points = result.x.copy()
                        
            except Exception:
                pass

    # Phase 1b: Fast optimization with bbox proxy (much faster than ConvexHull)
    if best_points is not None and time.time() - start_time < time_limit - 100:
        for i in range(min(len(initial_configs), 6)):
            if time.time() - start_time > time_limit - 80:
                break
            init_points = initial_configs[i]
            center = np.mean(init_points, axis=0)
            centered = init_points - center
            norm = np.linalg.norm(centered)
            if norm > 1e-10:
                centered = centered / norm * 1.0
            
            try:
                result = minimize(
                    objective_bbox,
                    centered,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 5000,
                        'xatol': 1e-10,
                        'fatol': 1e-12,
                        'adaptive': True,
                        'initial_simplex': 0.05
                    }
                )
                
                if result.fun < 0:
                    exact_score = score(result.x)
                    if exact_score > best_score:
                        best_score = exact_score
                        best_points = result.x.copy()
                        
            except Exception:
                pass

    # Phase 2: Refine with exact objective + multiple restarts from best
    if best_points is not None and time.time() - start_time < time_limit - 60:
        # First, a direct refinement
        try:
            result = minimize(
                objective_exact,
                best_points,
                method='Nelder-Mead',
                options={
                    'maxiter': 5000,
                    'xatol': 1e-11,
                    'fatol': 1e-13,
                    'adaptive': True,
                    'initial_simplex': 0.03
                }
            )
            
            if result.fun < 0:
                exact_score = -result.fun
                if exact_score > best_score:
                    best_score = exact_score
                    best_points = result.x.copy()
        except Exception:
            pass
        
        # Multiple small perturbations to escape local minima
        if time.time() - start_time < time_limit - 40:
            for pert_scale in [0.02, 0.01, 0.005]:
                if time.time() - start_time > time_limit - 35:
                    break
                rng_local = np.random.default_rng(seed=12345)
                for _ in range(3):
                    if time.time() - start_time > time_limit - 35:
                        break
                    perturbed = best_points + rng_local.normal(0, pert_scale, (n, 2))
                    try:
                        result = minimize(
                            objective_bbox,
                            perturbed,
                            method='Nelder-Mead',
                            options={
                                'maxiter': 3000,
                                'xatol': 1e-10,
                                'fatol': 1e-12,
                                'adaptive': True,
                                'initial_simplex': 0.01
                            }
                        )
                        
                        if result.fun < 0:
                            exact_score = score(result.x)
                            if exact_score > best_score:
                                best_score = exact_score
                                best_points = result.x.copy()
                    except Exception:
                        pass

    # Phase 3: High-precision polish with multiple simplex sizes
    if best_points is not None and time.time() - start_time < time_limit - 40:
        for simplex_size in [0.01, 0.005, 0.002, 0.001]:
            if time.time() - start_time > time_limit - 30:
                break
            try:
                result = minimize(
                    objective_bbox,
                    best_points,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 5000,
                        'xatol': 1e-13,
                        'fatol': 1e-15,
                        'adaptive': True,
                        'initial_simplex': simplex_size
                    }
                )
                
                if result.fun < 0:
                    exact_score = score(result.x)
                    if exact_score > best_score:
                        best_score = exact_score
                        best_points = result.x.copy()
            except Exception:
                pass

    # Phase 4: Final ultra-precise polish with exact objective
    if best_points is not None and time.time() - start_time < time_limit - 20:
        try:
            result = minimize(
                objective_exact,
                best_points,
                method='Nelder-Mead',
                options={
                    'maxiter': 3000,
                    'xatol': 1e-14,
                    'fatol': 1e-16,
                    'adaptive': True,
                    'initial_simplex': 0.0005
                }
            )
            
            if result.fun < 0:
                exact_score = -result.fun
                if exact_score > best_score:
                    best_score = exact_score
                    best_points = result.x.copy()
        except Exception:
            pass
        
        # One more bbox polish if time remains
        if best_points is not None and time.time() - start_time < time_limit - 10:
            try:
                result = minimize(
                    objective_bbox,
                    best_points,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 2000,
                        'xatol': 1e-14,
                        'fatol': 1e-16,
                        'adaptive': True,
                        'initial_simplex': 0.0003
                    }
                )
                
                if result.fun < 0:
                    exact_score = score(result.x)
                    if exact_score > best_score:
                        best_score = exact_score
                        best_points = result.x.copy()
            except Exception:
                pass

    # Final fallback
    if best_points is None:
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        
    # Ensure finite values
    if not np.all(np.isfinite(best_points)):
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        
    return best_points

# EVOLVE-BLOCK-END
