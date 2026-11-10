# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull

def heilbronn_convex13():
    """
    Maximize the ratio of the smallest triangle area to the convex hull area
    for 13 points using simulated annealing with structural initialization.
    """
    n = 13
    start_time = time.time()
    deadline = start_time + 580  # Safety margin

    # Precompute triangle indices
    tri_indices = []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                tri_indices.append((i, j, k))
    tri_indices = np.array(tri_indices)
    n_tri = len(tri_indices)

    # Vectorized area calculation
    def calc_areas(points):
        p1 = points[tri_indices[:, 0]]
        p2 = points[tri_indices[:, 1]]
        p3 = points[tri_indices[:, 2]]
        v1 = p2 - p1
        v2 = p3 - p1
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        return np.abs(cross) * 0.5

    def calc_hull_area(points):
        try:
            hull = ConvexHull(points)
            return hull.volume
        except Exception:
            return 0.0

    def evaluate(points):
        areas = calc_areas(points)
        min_area = np.min(areas)
        if min_area <= 0:
            return 0.0
        hull_area = calc_hull_area(points)
        if hull_area <= 0:
            return 0.0
        return min_area / hull_area

    # Initialization: regular 13-gon slightly perturbed
    def init_polygon():
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        # Add small random perturbation to angles
        angles += np.random.normal(0, 0.05, n)
        # Vary radii slightly
        radii = 0.45 + np.random.normal(0, 0.05, n)
        x = 0.5 + radii * np.cos(angles)
        y = 0.5 + radii * np.sin(angles)
        return np.column_stack((x, y))

    def init_random():
        return np.random.random((n, 2)) * 0.9 + 0.05

    def init_grid():
        # 4x4 grid, take 13 points
        pts = []
        for i in range(4):
            for j in range(4):
                pts.append([0.2 + i * 0.2, 0.2 + j * 0.2])
        pts = np.array(pts)
        idx = np.random.choice(16, 13, replace=False)
        return pts[idx]

    # Simulated annealing
    def simulated_annealing(points, max_iters=50000, initial_temp=0.1, cooling=0.9995):
        current_points = points.copy()
        current_score = evaluate(current_points)
        best_points = current_points.copy()
        best_score = current_score
        temp = initial_temp
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Pick a random point to move
            i = np.random.randint(n)
            
            # Generate a random perturbation
            # Gaussian with adaptive sigma
            sigma = temp * 0.5
            delta = np.random.normal(0, sigma, 2)
            
            new_points = current_points.copy()
            new_points[i] += delta
            # Keep within bounds
            np.clip(new_points, 0.01, 0.99, out=new_points)
            
            new_score = evaluate(new_points)
            
            # Acceptance criterion
            if new_score > current_score:
                current_points = new_points
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = current_points.copy()
            else:
                # Boltzmann acceptance
                if current_score > 0:
                    delta_e = current_score - new_score
                    prob = np.exp(-delta_e / (temp + 1e-10))
                    if np.random.random() < prob:
                        current_points = new_points
                        current_score = new_score
            
            # Cool down
            temp *= cooling
            if temp < 1e-6:
                temp = 1e-6
        
        return best_points, best_score

    # Multi-start simulated annealing
    best_points = None
    best_score = 0.0
    
    # Generate initial candidates
    candidates = [
        init_polygon(),
        init_polygon(),
        init_polygon(),
        init_random(),
        init_random(),
        init_grid(),
        init_grid()
    ]
    
    # Evaluate and select top candidates
    scored_cands = []
    for cand in candidates:
        if time.time() > deadline - 50:
            break
        if not np.all(np.isfinite(cand)):
            continue
        try:
            s = evaluate(cand)
            if s > 0:
                scored_cands.append((s, cand))
                if s > best_score:
                    best_score = s
                    best_points = cand.copy()
        except Exception:
            pass
    
    # Sort by score
    scored_cands.sort(key=lambda x: -x[0])
    
    # Run simulated annealing on top candidates
    n_sa = min(4, len(scored_cands))
    for s, cand in scored_cands[:n_sa]:
        if time.time() > deadline - 30:
            break
        refined_pts, refined_score = simulated_annealing(cand, max_iters=30000, initial_temp=0.08, cooling=0.9998)
        if refined_score > best_score:
            best_score = refined_score
            best_points = refined_pts.copy()
    
    # Final polish
    if best_points is not None and time.time() < deadline - 15:
        final_pts, final_score = simulated_annealing(best_points, max_iters=20000, initial_temp=0.02, cooling=0.9999)
        if final_score > best_score:
            best_points = final_pts
            best_score = final_score
    
    if best_points is None:
        best_points = init_polygon()
    
    return best_points
# EVOLVE-BLOCK-END
