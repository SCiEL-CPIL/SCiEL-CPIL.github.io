# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations


def _tri_areas(pts, idx):
    """Vectorized triangle areas for all triples in idx (shape (m,3))."""
    p0 = pts[idx[:, 0]]
    p1 = pts[idx[:, 1]]
    p2 = pts[idx[:, 2]]
    a = 0.5 * np.abs(
        p0[:, 0] * (p1[:, 1] - p2[:, 1])
        + p1[:, 0] * (p2[:, 1] - p0[:, 1])
        + p2[:, 0] * (p0[:, 1] - p1[:, 1])
    )
    return a


def _hull_area(pts):
    """Area of convex polygon given in CCW order (shoelace)."""
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _compute_score_fast(points: np.ndarray) -> float:
    """Compute A_min / A_hull for a set of points using vectorized triangle areas."""
    n = points.shape[0]
    if n < 3:
        return 0.0
    
    if not np.all(np.isfinite(points)):
        return 0.0
    
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        return 0.0
    
    if A_hull <= 0:
        return 0.0
    
    # Vectorized computation of all triangle areas
    idx = np.array(list(combinations(range(n), 3)))
    min_area = float(_tri_areas(points, idx).min())
    
    return float(min_area / A_hull)


def _find_worst_triangle(points, idx):
    """Find the triangle with minimum area and return its index and area."""
    areas = _tri_areas(points, idx)
    min_idx = np.argmin(areas)
    return min_idx, areas[min_idx]


def _local_search(points, idx, t0, deadline, max_iter=500, step_size=0.05):
    """
    Local search that directly targets the worst triangle by adjusting
    the three points involved to increase its area.
    """
    n = points.shape[0]
    pts = points.copy()
    cur_score = _compute_score_fast(pts)
    
    for it in range(max_iter):
        if time.time() - t0 > deadline:
            break
        
        worst_idx, worst_area = _find_worst_triangle(pts, idx)
        if worst_area < 1e-10:
            break
        
        tri = idx[worst_idx]
        i, j, k = tri
        
        # Try to increase the area of the worst triangle
        # by moving one of the three points
        best_move = None
        best_area = worst_area
        
        # For each of the 3 points, try moving it in a few directions
        for pidx in [i, j, k]:
            # Compute the direction that would increase the area
            # The area is proportional to the cross product of (p_j - p_i) and (p_k - p_i)
            # To increase area, move pidx in the direction perpendicular to the opposite side
            
            # Get the other two points
            others = [x for x in [i, j, k] if x != pidx]
            a, b = others
            
            # Vector from a to b
            ab = pts[b] - pts[a]
            # Perpendicular direction (rotate 90 degrees)
            perp = np.array([-ab[1], ab[0]])
            perp = perp / (np.linalg.norm(perp) + 1e-10)
            
            # Determine which direction increases area
            # Compute signed area with current point
            def signed_area(p):
                return 0.5 * np.cross(pts[a] - p, pts[b] - p)
            
            sa = signed_area(pts[pidx])
            if sa > 0:
                direction = perp
            else:
                direction = -perp
            
            # Try a few step sizes
            for ss in [step_size * 0.5, step_size, step_size * 2.0]:
                new_pts = pts.copy()
                new_pts[pidx] = pts[pidx] + ss * direction
                
                # Check if this improves the worst triangle area
                new_areas = _tri_areas(new_pts, idx)
                new_worst = new_areas[worst_idx]
                
                if new_worst > best_area:
                    # Check overall score
                    new_score = _compute_score_fast(new_pts)
                    if new_score > cur_score:
                        best_move = (pidx, ss, direction, new_score)
                        best_area = new_worst
        
        if best_move is not None:
            pidx, ss, direction, new_score = best_move
            pts[pidx] = pts[pidx] + ss * direction
            cur_score = new_score
        else:
            # No improvement, try random perturbation
            pidx = np.random.randint(0, n)
            direction = np.random.randn(2)
            direction = direction / (np.linalg.norm(direction) + 1e-10)
            new_pts = pts.copy()
            new_pts[pidx] = pts[pidx] + step_size * 0.3 * direction
            new_score = _compute_score_fast(new_pts)
            if new_score > cur_score:
                pts = new_pts
                cur_score = new_score
            else:
                break
    
    return pts, cur_score


def _simulated_annealing(points, idx, t0, deadline, max_iter=2000, T0=0.005, cooling=0.99):
    """
    Simulated annealing with focus on improving the worst triangle.
    """
    n = points.shape[0]
    pts = points.copy()
    cur_score = _compute_score_fast(pts)
    best_score = cur_score
    best_pts = pts.copy()
    T = T0
    
    for it in range(max_iter):
        if time.time() - t0 > deadline:
            break
        
        # Generate a perturbation
        # With some probability, target the worst triangle
        if np.random.random() < 0.6:
            worst_idx, worst_area = _find_worst_triangle(pts, idx)
            tri = idx[worst_idx]
            pidx = tri[np.random.randint(0, 3)]
            
            # Move in direction to increase worst triangle area
            others = [x for x in tri if x != pidx]
            a, b = others
            ab = pts[b] - pts[a]
            perp = np.array([-ab[1], ab[0]])
            perp = perp / (np.linalg.norm(perp) + 1e-10)
            
            def signed_area(p):
                return 0.5 * np.cross(pts[a] - p, pts[b] - p)
            
            sa = signed_area(pts[pidx])
            if sa > 0:
                direction = perp
            else:
                direction = -perp
            
            step = 0.03 * np.random.random()
            new_pts = pts.copy()
            new_pts[pidx] = pts[pidx] + step * direction
        else:
            # Random perturbation
            pidx = np.random.randint(0, n)
            direction = np.random.randn(2)
            direction = direction / (np.linalg.norm(direction) + 1e-10)
            step = 0.05 * np.random.random()
            new_pts = pts.copy()
            new_pts[pidx] = pts[pidx] + step * direction
        
        new_score = _compute_score_fast(new_pts)
        
        # Accept if better, or with probability exp((new - cur) / T)
        if new_score > cur_score or np.random.random() < np.exp((new_score - cur_score) / max(T, 1e-9)):
            pts = new_pts
            cur_score = new_score
            
            if cur_score > best_score:
                best_score = cur_score
                best_pts = pts.copy()
        
        T *= cooling
    
    return best_pts, best_score


def _hexagonal_initialization(n=13):
    """
    Initialize points in a hexagonal lattice pattern.
    """
    pts = []
    # Central point
    pts.append([0.0, 0.0])
    
    # Rings around center
    radius = 1.0
    points_per_ring = 6
    ring_num = 1
    
    while len(pts) < n:
        for k in range(points_per_ring):
            if len(pts) >= n:
                break
            angle = 2 * np.pi * k / points_per_ring + (np.pi / points_per_ring if ring_num % 2 == 0 else 0)
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            pts.append([x, y])
        radius += 1.0
        ring_num += 1
    
    return np.array(pts[:n])


def _grid_initialization(n=13):
    """
    Initialize points on a grid.
    """
    # Use a 4x4 grid, pick n points
    grid_size = 4
    spacing = 1.0
    grid_points = []
    for i in range(grid_size):
        for j in range(grid_size):
            grid_points.append([i * spacing, j * spacing])
    
    # Shuffle and pick n
    np.random.shuffle(grid_points)
    return np.array(grid_points[:n])


def _concentric_initialization(n=13):
    """
    Initialize points in concentric circles.
    """
    pts = []
    # Inner circle
    for k in range(6):
        angle = 2 * np.pi * k / 6
        pts.append([0.5 * np.cos(angle), 0.5 * np.sin(angle)])
    
    # Outer circle
    for k in range(7):
        angle = 2 * np.pi * k / 7 + np.pi / 7
        pts.append([1.5 * np.cos(angle), 1.5 * np.sin(angle)])
    
    return np.array(pts[:n])


def _random_initialization(n=13):
    """
    Random initialization within a unit square.
    """
    return np.random.rand(n, 2) * 2 - 1


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points maximizing (smallest triangle area) / (convex hull area).
    
    Uses a hybrid discrete-continuous approach:
    1. Generate diverse initial configurations
    2. Apply local search targeting worst triangles
    3. Apply simulated annealing for global exploration
    4. Return the best configuration found
    """
    n = 13
    t0 = time.time()
    DEADLINE = 620.0  # Safety margin before 630s kill
    
    # Precompute all C(13,3)=286 triangle index triples
    idx = np.array(list(combinations(range(n), 3)))
    
    best_score = -1.0
    best_pts = None
    
    # Generate diverse initial configurations
    initializers = [
        _hexagonal_initialization,
        _grid_initialization,
        _concentric_initialization,
        _random_initialization,
    ]
    
    for init_func in initializers:
        if time.time() - t0 > DEADLINE - 60:
            break
        
        pts = init_func(n)
        
        # Local search
        pts_ls, score_ls = _local_search(pts, idx, t0, DEADLINE - 30, max_iter=300, step_size=0.04)
        if score_ls > best_score:
            best_score = score_ls
            best_pts = pts_ls.copy()
        
        # Simulated annealing
        if time.time() - t0 < DEADLINE - 30:
            pts_sa, score_sa = _simulated_annealing(pts_ls, idx, t0, DEADLINE - 10, max_iter=1500, T0=0.004, cooling=0.992)
            if score_sa > best_score:
                best_score = score_sa
                best_pts = pts_sa.copy()
    
    # Final polish with Nelder-Mead if time permits
    if best_pts is not None and time.time() - t0 < DEADLINE - 20:
        try:
            from scipy.optimize import minimize
            
            def neg_score(pts_flat):
                pts_2d = pts_flat.reshape(n, 2)
                return -_compute_score_fast(pts_2d)
            
            res = minimize(
                neg_score,
                best_pts.flatten(),
                method="Nelder-Mead",
                options={"maxiter": 1000, "xatol": 1e-6, "fatol": 1e-8},
            )
            
            final_pts = res.x.reshape(n, 2)
            final_score = _compute_score_fast(final_pts)
            
            if final_score > best_score:
                best_score = final_score
                best_pts = final_pts
        except Exception:
            pass
    
    if best_pts is None:
        # Fallback: regular 13-gon
        angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        best_pts = np.column_stack((np.cos(angles), np.sin(angles)))
    
    return best_pts


# EVOLVE-BLOCK-END
