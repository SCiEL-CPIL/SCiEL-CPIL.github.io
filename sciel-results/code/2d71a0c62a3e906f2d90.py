# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
import time
import itertools

N = 13
# Precompute all 286 triangle index triples
TRI_IDX = np.array(list(itertools.combinations(range(N), 3)))

def compute_score(points):
    """Hard Heilbronn score: A_min / A_hull."""
    if not np.all(np.isfinite(points)):
        return 0.0
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        return 0.0
    if A_hull <= 1e-12:
        return 0.0
    p1 = points[TRI_IDX[:, 0]]
    p2 = points[TRI_IDX[:, 1]]
    p3 = points[TRI_IDX[:, 2]]
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = np.abs(cross) * 0.5
    return np.min(areas) / A_hull

def compute_all_areas(points):
    """Compute areas of all 286 triangles."""
    p1 = points[TRI_IDX[:, 0]]
    p2 = points[TRI_IDX[:, 1]]
    p3 = points[TRI_IDX[:, 2]]
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return np.abs(cross) * 0.5

def delaunay_repair(points, max_iters=500, time_limit=None):
    """
    Delaunay-based repair: iteratively move vertices incident to minimum-area
    triangles toward the centroid of their adjacent Delaunay triangles.
    """
    pts = points.copy()
    n = len(pts)
    
    if time_limit is None:
        time_limit = time.time() + 30.0
    
    best_score = compute_score(pts)
    best_pts = pts.copy()
    
    for iteration in range(max_iters):
        if time.time() >= time_limit:
            break
        
        # Compute Delaunay triangulation
        try:
            tri = Delaunay(pts)
        except Exception:
            break
        
        # Compute all triangle areas
        areas = compute_all_areas(pts)
        min_area = np.min(areas)
        
        if min_area <= 1e-12:
            # Try to break collinearity
            min_idx = np.argmin(areas)
            i, j, k = TRI_IDX[min_idx]
            # Move one of the collinear points slightly
            mid = pts[(i + j) // 2]
            direction = pts[k] - mid
            norm = np.linalg.norm(direction)
            if norm > 1e-10:
                direction /= norm
            else:
                direction = np.array([1.0, 0.0])
            pts[(i + j) // 2] += 0.01 * direction
            continue
        
        # Find the minimum-area triangle
        min_tri_idx = np.argmin(areas)
        i, j, k = TRI_IDX[min_tri_idx]
        
        # Get Delaunay triangles
        simplices = tri.simplices
        
        # Find centroid of the Delaunay triangles adjacent to this triangle
        # The triangle (i, j, k) is in the Delaunay triangulation
        # Find its neighbors
        neighbors = []
        for s in simplices:
            if set(s) == {i, j, k}:
                # This is our triangle, find neighbors sharing an edge
                edges = [(s[0], s[1]), (s[1], s[2]), (s[0], s[2])]
                for other_s in simplices:
                    if set(other_s) == {i, j, k}:
                        continue
                    for e in edges:
                        if e[0] in other_s and e[1] in other_s:
                            neighbors.append(other_s)
                            break
        
        if len(neighbors) == 0:
            # No neighbors, just move toward the centroid of the hull
            try:
                hull = ConvexHull(pts)
                centroid = np.mean(pts[hull.vertices], axis=0)
            except Exception:
                centroid = np.mean(pts, axis=0)
            # Move the vertex that's most "out of place"
            for v in [i, j, k]:
                dist_to_centroid = np.linalg.norm(pts[v] - centroid)
                pts[v] = 0.9 * pts[v] + 0.1 * centroid
            continue
        
        # Compute centroid of neighboring triangles
        neighbor_centroids = []
        for ns in neighbors:
            nc = np.mean(pts[ns], axis=0)
            neighbor_centroids.append(nc)
        
        if len(neighbor_centroids) > 0:
            target = np.mean(neighbor_centroids, axis=0)
        else:
            target = np.mean(pts, axis=0)
        
        # Move each vertex of the minimum triangle slightly toward the target
        step = 0.05
        for v in [i, j, k]:
            pts[v] = (1 - step) * pts[v] + step * target
        
        # Check if we improved
        new_score = compute_score(pts)
        if new_score > best_score:
            best_score = new_score
            best_pts = pts.copy()
        else:
            # Revert
            pts = best_pts.copy()
            # Try smaller step
            step *= 0.5
            if step < 0.001:
                break
    
    return best_pts, best_score

def local_search(points, time_limit=None, step_size=0.05):
    """
    Local search: iteratively try moving each point in small steps
    in 8 directions, keeping the best improvement.
    """
    pts = points.copy()
    n = len(pts)
    
    if time_limit is None:
        time_limit = time.time() + 30.0
    
    current_score = compute_score(pts)
    best_score = current_score
    best_pts = pts.copy()
    
    directions = np.array([
        [1, 0], [-1, 0], [0, 1], [0, -1],
        [1, 1], [1, -1], [-1, 1], [-1, -1]
    ], dtype=float)
    directions /= np.sqrt(2)  # Normalize diagonals
    
    improved = True
    while improved and time.time() < time_limit:
        improved = False
        for i in range(n):
            if time.time() >= time_limit:
                break
            for d in directions:
                old = pts[i].copy()
                pts[i] = old + step_size * d
                new_score = compute_score(pts)
                if new_score > current_score:
                    current_score = new_score
                    improved = True
                    if new_score > best_score:
                        best_score = new_score
                        best_pts = pts.copy()
                else:
                    pts[i] = old
    
    # Adaptive step size: shrink if no improvement
    if not improved:
        step_size *= 0.5
        if step_size < 0.001:
            return best_pts, best_score
        # Retry with smaller step
        return local_search(pts, time_limit, step_size)
    
    return best_pts, best_score

def simulated_annealing(init_pts, time_limit, rng, T_start=1.0, T_end=0.001, cooling=0.9999):
    """
    Simulated Annealing on the hard objective.
    Accepts worse moves with probability exp(-delta_score / T).
    """
    current_pts = init_pts.copy()
    current_score = compute_score(current_pts)
    best_pts = current_pts.copy()
    best_score = current_score
    
    T = T_start
    n_points = len(current_pts)
    
    while time.time() < time_limit and T > T_end:
        # Pick a random point to move
        idx = rng.integers(n_points)
        
        # Generate a random perturbation (Gaussian)
        step_size = 0.1 * (0.1 + 0.9 * (T / T_start))
        delta = rng.normal(0, step_size, size=2)
        
        old_x, old_y = current_pts[idx, 0], current_pts[idx, 1]
        new_x = old_x + delta[0]
        new_y = old_y + delta[1]
        
        if new_x < -5 or new_x > 5 or new_y < -5 or new_y > 5:
            T *= cooling
            continue
            
        current_pts[idx, 0] = new_x
        current_pts[idx, 1] = new_y
        
        new_score = compute_score(current_pts)
        
        if new_score > current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_pts = current_pts.copy()
        else:
            delta_s = new_score - current_score
            if delta_s < 0:
                if T < 1e-12:
                    prob = 0.0
                else:
                    prob = np.exp(delta_s / T)
                if rng.random() < prob:
                    current_score = new_score
                else:
                    current_pts[idx, 0] = old_x
                    current_pts[idx, 1] = old_y
        
        T *= cooling
        
    return best_pts, best_score

def heilbronn_convex13() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 620.0  # safety margin before 630s kill
    rng = np.random.default_rng(42)

    best_points = None
    best_score = -1.0

    candidates = []
    # 1. Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    candidates.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    # 2. 7 outer, 6 inner
    ang7 = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    p7 = np.column_stack([np.cos(ang7), np.sin(ang7)])
    ang6 = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi / 6
    p6 = np.column_stack([0.55 * np.cos(ang6), 0.55 * np.sin(ang6)])
    candidates.append(np.vstack([p7, p6]))
    # 3. 7 outer, 6 inner (offset)
    ang6b = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi / 12
    p6b = np.column_stack([0.6 * np.cos(ang6b), 0.6 * np.sin(ang6b)])
    candidates.append(np.vstack([p7, p6b]))
    # 4. 5/5/3
    ang5 = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    p5o = np.column_stack([np.cos(ang5), np.sin(ang5)])
    ang5m = np.linspace(0, 2 * np.pi, 5, endpoint=False) + np.pi / 5
    p5m = np.column_stack([0.6 * np.cos(ang5m), 0.6 * np.sin(ang5m)])
    ang3 = np.linspace(0, 2 * np.pi, 3, endpoint=False) + np.pi / 3
    p3i = np.column_stack([0.3 * np.cos(ang3), 0.3 * np.sin(ang3)])
    candidates.append(np.vstack([p5o, p5m, p3i]))
    # 5. 4/5/4
    ang4o = np.linspace(0, 2 * np.pi, 4, endpoint=False) + np.pi / 4
    p4o = np.column_stack([np.cos(ang4o), np.sin(ang4o)])
    ang5m2 = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    p5m2 = np.column_stack([0.65 * np.cos(ang5m2), 0.65 * np.sin(ang5m2)])
    ang4i = np.linspace(0, 2 * np.pi, 4, endpoint=False) + np.pi / 4
    p4i = np.column_stack([0.35 * np.cos(ang4i), 0.35 * np.sin(ang4i)])
    candidates.append(np.vstack([p4o, p5m2, p4i]))
    # 6. 6 outer, 7 inner
    ang6o = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    p6o = np.column_stack([np.cos(ang6o), np.sin(ang6o)])
    ang7i = np.linspace(0, 2 * np.pi, 7, endpoint=False) + np.pi / 7
    p7i = np.column_stack([0.5 * np.cos(ang7i), 0.5 * np.sin(ang7i)])
    candidates.append(np.vstack([p6o, p7i]))
    # 7. Fibonacci spiral
    golden = (1 + np.sqrt(5)) / 2
    fib_angles = 2 * np.pi * np.arange(N) / (golden ** 2)
    fib_radii = np.sqrt(np.arange(N) / N)
    candidates.append(np.column_stack([fib_radii * np.cos(fib_angles),
                                       fib_radii * np.sin(fib_angles)]))
    # 8. 4/5/4
    ang4a = np.linspace(0, 2 * np.pi, 4, endpoint=False)
    p4a = np.column_stack([np.cos(ang4a), np.sin(ang4a)])
    ang5b = np.linspace(0, 2 * np.pi, 5, endpoint=False) + np.pi / 5
    p5b = np.column_stack([0.65 * np.cos(ang5b), 0.65 * np.sin(ang5b)])
    ang4c = np.linspace(0, 2 * np.pi, 4, endpoint=False) + np.pi / 4
    p4c = np.column_stack([0.3 * np.cos(ang4c), 0.3 * np.sin(ang4c)])
    candidates.append(np.vstack([p4a, p5b, p4c]))
    # 9. 5/4/4
    ang5o2 = np.linspace(0, 2 * np.pi, 5, endpoint=False) + np.pi / 10
    p5o2 = np.column_stack([np.cos(ang5o2), np.sin(ang5o2)])
    ang4m2 = np.linspace(0, 2 * np.pi, 4, endpoint=False)
    p4m2 = np.column_stack([0.6 * np.cos(ang4m2), 0.6 * np.sin(ang4m2)])
    ang4i2 = np.linspace(0, 2 * np.pi, 4, endpoint=False) + np.pi / 4
    p4i2 = np.column_stack([0.3 * np.cos(ang4i2), 0.3 * np.sin(ang4i2)])
    candidates.append(np.vstack([p5o2, p4m2, p4i2]))
    # 10. Random
    for _ in range(5):
        candidates.append(rng.random((N, 2)) * 2 - 1)
    # Perturbed structured
    for base_idx in [0, 1, 3, 4]:
        if base_idx < len(candidates):
            candidates.append(candidates[base_idx] + rng.normal(0, 0.03, size=(N, 2)))

    n_cand = len(candidates)
    for i, init_pts in enumerate(candidates):
        if time.time() >= deadline - 5:
            break
        
        init_score = compute_score(init_pts)
        if init_score > best_score:
            best_score = init_score
            best_points = init_pts.copy()

        elapsed = time.time() - start_time
        remaining = deadline - 5 - elapsed
        remaining_cands = n_cand - i
        time_per_cand = min(remaining / max(remaining_cands, 1), 60.0)
        cand_deadline = time.time() + time_per_cand

        # Phase 1: Delaunay repair
        if time.time() < cand_deadline:
            dr_pts, dr_score = delaunay_repair(init_pts, max_iters=200, time_limit=cand_deadline)
            if dr_score > best_score:
                best_score = dr_score
                best_points = dr_pts.copy()

        # Phase 2: Local search from the best of (init, delaunay_repair)
        if time.time() < deadline - 5:
            ls_start = max(dr_score, init_score)
            ls_pts = dr_pts if dr_score >= init_score else init_pts
            ls_pts, ls_score = local_search(ls_pts, time_limit=cand_deadline, step_size=0.05)
            if ls_score > best_score:
                best_score = ls_score
                best_points = ls_pts.copy()

        # Phase 3: Simulated annealing from the best so far
        if time.time() < deadline - 10 and best_points is not None:
            sa_deadline = time.time() + min((deadline - 10 - time.time()) * 0.5, 30.0)
            if sa_deadline > time.time():
                sa_pts, sa_score = simulated_annealing(best_points, sa_deadline, rng)
                if sa_score > best_score:
                    best_score = sa_score
                    best_points = sa_pts.copy()

        # Phase 4: Another round of Delaunay + local search from the improved point
        if time.time() < deadline - 10 and best_points is not None:
            dr_pts2, dr_score2 = delaunay_repair(best_points, max_iters=100, time_limit=time.time() + 15)
            if dr_score2 > best_score:
                best_score = dr_score2
                best_points = dr_pts2.copy()
            
            if time.time() < deadline - 5:
                ls_pts2, ls_score2 = local_search(best_points, time_limit=deadline - 5, step_size=0.03)
                if ls_score2 > best_score:
                    best_score = ls_score2
                    best_points = ls_pts2.copy()

    if best_points is None or not np.all(np.isfinite(best_points)):
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])

    return best_points
# EVOLVE-BLOCK-END
