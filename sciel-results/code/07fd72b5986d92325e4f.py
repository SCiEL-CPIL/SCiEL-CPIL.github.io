"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from itertools import combinations


def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by three points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]))
                if area < min_area:
                    min_area = area
    return min_area


def _min_pairwise_distance(points):
    """Compute the minimum pairwise distance."""
    n = len(points)
    min_d = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(points[i] - points[j])
            if d < min_d:
                min_d = d
    return min_d


def _barycentric_to_cartesian(a, b, c):
    """Convert barycentric coordinates to Cartesian in the equilateral triangle."""
    h = np.sqrt(3) / 2
    x = b + 0.5 * c
    y = h * c
    return np.array([x, y])


def _cartesian_to_barycentric(p):
    """Convert Cartesian coordinates to barycentric in the equilateral triangle."""
    x, y = p
    h = np.sqrt(3) / 2
    c = y / h
    b = x - 0.5 * c
    a = 1.0 - b - c
    return np.array([a, b, c])


def _project_to_triangle(p):
    """Project point p onto the equilateral triangle."""
    x, y = p
    h = np.sqrt(3) / 2
    
    if y < 0:
        y = 0
    if y > h:
        y = h
    
    # Check left boundary: y <= sqrt(3) * x => x >= y/sqrt(3)
    if x < y / np.sqrt(3):
        x = y / np.sqrt(3)
    
    # Check right boundary: y <= sqrt(3) * (1-x) => x >= 1 - y/sqrt(3)
    if x > 1.0 - y / np.sqrt(3):
        x = 1.0 - y / np.sqrt(3)
    
    return np.array([x, y])


def construct():
    """Construct 11 points in the unit equilateral triangle to maximize minimum triangle area."""
    h = np.sqrt(3) / 2
    
    # Step 1: 1D lattice search over barycentric coordinates
    # Use a fine grid and greedy packing to maximize minimum pairwise distance
    best_points = None
    best_min_area = 0.0
    best_min_dist = 0.0
    
    # Try multiple grid resolutions
    for res in [4, 5, 6, 7, 8, 9, 10]:
        # Generate all valid barycentric triples with a+b+c=1
        # a, b, c are multiples of 1/res
        candidates = []
        for a_num in range(res + 1):
            for b_num in range(res + 1 - a_num):
                c_num = res - a_num - b_num
                a = a_num / res
                b = b_num / res
                c = c_num / res
                pt = _barycentric_to_cartesian(a, b, c)
                candidates.append(pt)
        
        n_cand = len(candidates)
        
        # Greedy packing with multiple restarts
        for restart in range(20):
            np.random.seed(42 + restart)
            picked = []
            remaining = list(range(n_cand))
            
            # Start with a random point
            start_idx = np.random.choice(remaining)
            picked.append(start_idx)
            remaining.remove(start_idx)
            
            while len(picked) < N:
                # Find the point in remaining that maximizes the minimum distance to picked
                best_idx = None
                best_min_dist_candidate = -1
                for idx in remaining:
                    min_dist = min(np.linalg.norm(candidates[idx] - candidates[p]) for p in picked)
                    if min_dist > best_min_dist_candidate:
                        best_min_dist_candidate = min_dist
                        best_idx = idx
                picked.append(best_idx)
                remaining.remove(best_idx)
            
            pts = np.array([candidates[i] for i in picked])
            min_area = _min_triangle_area(pts)
            min_dist = _min_pairwise_distance(pts)
            
            # Prefer higher min_area, but also consider min_dist as a tiebreaker
            if min_area > best_min_area + 1e-10 or (abs(min_area - best_min_area) < 1e-10 and min_dist > best_min_dist):
                best_min_area = min_area
                best_min_dist = min_dist
                best_points = pts.copy()
    
    if best_points is None:
        # Fallback: random points
        points = []
        for _ in range(N):
            while True:
                u, v = np.random.random(), np.random.random()
                if u + v < 1:
                    x = u + 0.5 * v
                    y = h * v
                    points.append([x, y])
                    break
        return np.array(points)
    
    # Step 2: Local polish using scipy.optimize
    # Maximize minimum triangle area by minimizing negative min area
    x0 = best_points.flatten()
    
    def neg_min_area(x_flat):
        pts = x_flat.reshape(N, 2)
        # Project each point to triangle to ensure feasibility
        pts_clipped = np.array([_project_to_triangle(p) for p in pts])
        return -_min_triangle_area(pts_clipped)
    
    # Use Nelder-Mead for local optimization with multiple initializations
    best_result = best_points.copy()
    best_result_area = best_min_area
    
    for trial in range(5):
        # Add small random perturbation
        if trial == 0:
            x0_trial = x0.copy()
        else:
            x0_trial = x0 + np.random.normal(0, 0.01, size=x0.shape)
            x0_trial = np.array([_project_to_triangle(x0_trial.reshape(N, 2)[i]).flatten() for i in range(N)]).flatten()
        
        result = minimize(neg_min_area, x0_trial, method='Nelder-Mead', 
                          options={'maxiter': 2000, 'xatol': 1e-9, 'fatol': 1e-11})
        
        if result.success or -result.fun > best_result_area:
            trial_points = result.x.reshape(N, 2)
            trial_points = np.array([_project_to_triangle(p) for p in trial_points])
            trial_area = _min_triangle_area(trial_points)
            if trial_area > best_result_area:
                best_result_area = trial_area
                best_result = trial_points
    
    return best_result

# EVOLVE-BLOCK-END
