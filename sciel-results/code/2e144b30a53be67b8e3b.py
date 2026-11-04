"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull
from itertools import combinations
from scipy.optimize import minimize

_TRIPLETS = np.array(list(combinations(range(N), 3)))

def _triangle_areas(pts):
    """Compute all C(n,3) triangle areas vectorized."""
    i, j, k = _TRIPLETS.T
    a = pts[i]
    b = pts[j]
    c = pts[k]
    areas = 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))
    return areas

def _convex_hull_area(pts):
    """Compute area of convex hull."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _score(pts):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(pts)
    if hull_area <= 0:
        return 0.0
    areas = _triangle_areas(pts)
    if len(areas) == 0:
        return 0.0
    return np.min(areas) / hull_area

def _regular_polygon(n, radius=1.0, offset=0.0):
    """Generate a regular polygon with n vertices."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + offset
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _place_interior_points(hull_pts, n_interior, seed=0):
    """Place interior points optimally given hull points.
    
    Strategy: Use a grid search over the interior, trying to maximize min triangle area.
    We use a two-stage approach: first place on a fine grid, then refine.
    """
    rng = np.random.RandomState(seed)
    
    # Get hull vertices for reference
    hull = ConvexHull(hull_pts)
    hull_area = hull.volume
    
    # Centroid of hull
    centroid = np.mean(hull_pts, axis=0)
    
    # Strategy: place interior points at positions that maximize minimum distance
    # to all existing points (including hull points)
    # Use a greedy placement with local optimization
    
    all_pts = list(hull_pts)
    
    for _ in range(n_interior):
        # Try multiple candidate positions
        best_pos = None
        best_min_dist = -1
        
        # Candidate 1: centroid
        candidates = [centroid]
        
        # Candidate 2: random positions in hull
        for _ in range(200):
            # Random point in hull using barycentric coordinates
            idx = hull.vertices[:3]
            t1 = rng.random()
            t2 = rng.random()
            if t1 + t2 > 1:
                t1, t2 = 1 - t1, 1 - t2
            p = hull_pts[idx[0]] * (1 - t1 - t2) + hull_pts[idx[1]] * t1 + hull_pts[idx[2]] * t2
            candidates.append(p)
            
            # Also try positions along lines from centroid to hull vertices
            v = rng.randint(len(hull_pts))
            for frac in [0.3, 0.5, 0.7]:
                candidates.append(centroid + frac * (hull_pts[v] - centroid))
        
        for cand in candidates:
            # Compute minimum distance to all existing points
            dists = np.linalg.norm(np.array(all_pts) - cand, axis=1)
            min_dist = np.min(dists)
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_pos = cand
        
        all_pts.append(best_pos)
    
    return np.array(all_pts)

def _optimize_interior_given_hull(hull_pts, n_interior, seed=0, max_iter=100):
    """Optimize interior point positions given fixed hull points.
    
    Uses gradient-free optimization to maximize min triangle area.
    """
    rng = np.random.RandomState(seed)
    
    # Initial placement
    all_pts = _place_interior_points(hull_pts, n_interior, seed)
    
    current_score = _score(all_pts)
    best_score = current_score
    best_pts = all_pts.copy()
    
    # Simulated annealing on interior points only
    T = 0.1
    cooling = 0.995
    
    hull_fixed = all_pts[:len(hull_pts)]
    
    for it in range(max_iter):
        # Pick a random interior point to move
        idx = len(hull_pts) + rng.randint(n_interior)
        
        # Propose a small move
        delta = rng.normal(0, T * 0.5, 2)
        new_pt = all_pts[idx] + delta
        
        # Ensure new point is inside hull (rough check)
        # Use winding number or simple bounding box
        if not _is_inside_convex(new_pt, hull_pts):
            continue
        
        new_all = all_pts.copy()
        new_all[idx] = new_pt
        new_score = _score(new_all)
        
        if new_score > current_score or rng.random() < np.exp((new_score - current_score) / max(T, 1e-10)):
            all_pts = new_all
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_pts = all_pts.copy()
        
        T *= cooling
    
    return best_pts

def _is_inside_convex(p, hull_pts):
    """Check if point p is inside convex polygon hull_pts (in CCW order)."""
    n = len(hull_pts)
    if n < 3:
        return True
    
    # Get convex hull vertices in order
    try:
        hull = ConvexHull(hull_pts)
        verts = hull_pts[hull.vertices]
    except:
        return True
    
    n_v = len(verts)
    for i in range(n_v):
        a = verts[i]
        b = verts[(i + 1) % n_v]
        # Cross product to check if p is on the left side
        cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
        if cross < -1e-10:
            return False
    return True

def _optimize_hull_and_interior(hull_n, interior_n, seed=0, time_budget=30):
    """Optimize both hull and interior points jointly.
    
    hull_n: number of points on the convex hull
    interior_n: number of interior points
    """
    start_time = time.time()
    deadline = start_time + time_budget
    
    # Initial hull: regular polygon
    hull_pts = _regular_polygon(hull_n, radius=1.0, offset=seed * 0.1)
    
    # Place interior points
    all_pts = _place_interior_points(hull_pts, interior_n, seed)
    
    current_score = _score(all_pts)
    best_score = current_score
    best_pts = all_pts.copy()
    
    # Simulated annealing on all points
    T = 0.15
    cooling = 0.998
    
    while time.time() < deadline:
        # Pick a random point to move
        idx = np.random.randint(N)
        
        # Propose a move
        step = T * 0.3
        delta = np.random.normal(0, step, 2)
        new_pt = all_pts[idx] + delta
        
        # If moving a hull point, check it stays on hull (roughly)
        new_all = all_pts.copy()
        new_all[idx] = new_pt
        
        # Check if the new configuration is valid (hull area > 0)
        hull_area = _convex_hull_area(new_all)
        if hull_area <= 0:
            continue
        
        new_score = _score(new_all)
        
        if new_score > current_score or np.random.random() < np.exp((new_score - current_score) / max(T, 1e-10)):
            all_pts = new_all
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_pts = all_pts.copy()
        
        T *= cooling
        if T < 0.001:
            break
    
    return best_pts

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    best_score = 0.0
    best_pts = _regular_polygon(N, radius=1.0)
    
    # Try different hull/interior splits
    # For 13 points, typical good splits: 6-7 hull, 6-7 interior
    splits = [
        (6, 7),
        (7, 6),
        (8, 5),
        (5, 8),
        (9, 4),
    ]
    
    for hull_n, interior_n in splits:
        if time.time() > deadline - 30:
            break
        
        # Multiple seeds for diversity
        for seed in range(5):
            if time.time() > deadline - 25:
                break
            
            try:
                pts = _optimize_hull_and_interior(hull_n, interior_n, seed, time_budget=15)
                score = _score(pts)
                
                if score > best_score:
                    best_score = score
                    best_pts = pts.copy()
            except Exception:
                continue
    
    # Final polish: gradient-based optimization from best solution
    if time.time() < deadline - 10:
        def _objective(x):
            pts = x.reshape(-1, 2)
            hull_area = _convex_hull_area(pts)
            if hull_area <= 0:
                return 1e6
            areas = _triangle_areas(pts)
            # Use soft-min for smoothness
            beta = 50.0
            scaled = -areas * beta
            max_scaled = np.max(scaled)
            soft_min = -max_scaled / beta - np.log(np.sum(np.exp(scaled - max_scaled))) / beta
            return -soft_min / hull_area
        
        try:
            x0 = best_pts.flatten()
            result = minimize(
                _objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            pts = result.x.reshape(-1, 2)
            score = _score(pts)
            
            if score > best_score:
                best_score = score
                best_pts = pts.copy()
        except Exception:
            pass
    
    return best_pts
# EVOLVE-BLOCK-END
