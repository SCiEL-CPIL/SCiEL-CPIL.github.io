"""Heilbronn convex regions: place 13 points to maximize min triangle area / hull area."""
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time
import itertools

N = 13

def _min_triangle_area(pts):
    """Compute the minimum area over all triples of points."""
    n = len(pts)
    if n < 3:
        return 0.0
    min_area = float('inf')
    # Vectorized calculation of all triangle areas is possible but memory heavy for large N.
    # For N=13, 13 choose 3 = 286 triangles. This is small.
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                ax = pts[j, 0] - pts[i, 0]
                ay = pts[j, 1] - pts[i, 1]
                bx = pts[k, 0] - pts[i, 0]
                by = pts[k, 1] - pts[i, 1]
                area = 0.5 * abs(ax * by - ay * bx)
                if area < min_area:
                    min_area = area
    return min_area

def _hull_area(pts):
    """Compute the convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _score_points(pts):
    """Compute min triangle area / convex hull area."""
    n = len(pts)
    if n < 3:
        return 0.0
    hull_area = _hull_area(pts)
    if hull_area <= 0:
        return 0.0
    min_area = _min_triangle_area(pts)
    return min_area / hull_area

def _smooth_min_area(pts, alpha=100.0):
    """
    Compute a smooth approximation of the minimum triangle area.
    f(P) = (sum A_i^{-alpha})^{-1/alpha}
    """
    n = len(pts)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                ax = pts[j, 0] - pts[i, 0]
                ay = pts[j, 1] - pts[i, 1]
                bx = pts[k, 0] - pts[i, 0]
                by = pts[k, 1] - pts[i, 1]
                area = 0.5 * abs(ax * by - ay * bx)
                # Avoid division by zero
                if area < 1e-10:
                    return 0.0
                areas.append(area)
    
    areas = np.array(areas)
    # Soft min
    sum_inv_pow = np.sum(areas ** (-alpha))
    if sum_inv_pow == 0:
        return 0.0
    return (sum_inv_pow ** (-1.0 / alpha))

def _neg_smooth_score(params, alpha=100.0):
    """
    Negative smooth score for minimization.
    We maximize SmoothMinArea / HullArea.
    To avoid division by zero or small hull areas, we can maximize SmoothMinArea - lambda * HullArea?
    No, the ratio is scale invariant. 
    Actually, if we scale points by s, MinArea scales by s^2, HullArea scales by s^2. Ratio is invariant.
    So we can fix the scale, e.g., HullArea = 1, and maximize MinArea.
    Or just maximize the ratio directly.
    """
    pts = params.reshape((N, 2))
    hull_area = _hull_area(pts)
    if hull_area < 1e-6:
        return 1e6 # Large penalty
    
    smooth_area = _smooth_min_area(pts, alpha)
    if smooth_area < 1e-6:
        return 1e6
    
    # We want to maximize smooth_area / hull_area
    # Minimize -log(smooth_area / hull_area) for numerical stability
    # Or just minimize -(smooth_area / hull_area)
    score = smooth_area / hull_area
    return -score

def _make_initial_guesses():
    """Generate diverse initial configurations."""
    configs = []
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    pts1 = np.column_stack([np.cos(angles), np.sin(angles)])
    configs.append(pts1)
    
    # 2. Regular 12-gon + Center
    angles12 = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    outer = np.column_stack([np.cos(angles12), np.sin(angles12)])
    center = np.array([[0.0, 0.0]])
    pts2 = np.vstack([outer, center])
    configs.append(pts2)
    
    # 3. Two rings: 7 outer, 6 inner
    angles7 = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    angles6 = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi/7
    outer = np.column_stack([np.cos(angles7), np.sin(angles7)])
    inner = np.column_stack([np.cos(angles6)*0.5, np.sin(angles6)*0.5])
    pts3 = np.vstack([outer, inner])
    configs.append(pts3)
    
    # 4. Random
    rng = np.random.default_rng(42)
    for _ in range(5):
        pts = rng.uniform(-1, 1, (N, 2))
        configs.append(pts)
        
    # 5. Grid-like
    # 4x4 grid minus one
    grid_pts = []
    for i in range(4):
        for j in range(4):
            if i == 3 and j == 3:
                continue
            grid_pts.append([i - 1.5, j - 1.5])
    pts_grid = np.array(grid_pts)
    configs.append(pts_grid)
            
    return configs

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_score = 0.0
    best_pts = None
    
    initial_guesses = _make_initial_guesses()
    
    # Evaluate initial guesses
    for guess in initial_guesses:
        score = _score_points(guess)
        if score > best_score:
            best_score = score
            best_pts = guess.copy()
            
    if time.time() > deadline:
        return best_pts if best_pts is not None else initial_guesses[0]
    
    # Optimization Strategy
    # Use L-BFGS-B on the smooth objective from multiple starts.
    # We will use a high alpha for smoothness initially, then refine.
    
    alpha = 50.0 # Start with moderate smoothness
    
    # Try to optimize from each initial guess
    for i, guess in enumerate(initial_guesses):
        if time.time() > deadline:
            break
            
        # Normalize initial guess to have hull area ~1 to help optimizer?
        # The objective is scale invariant, so it doesn't strictly matter, 
        # but keeping points in a reasonable range helps.
        hull_a = _hull_area(guess)
        if hull_a > 0:
            scale = 1.0 / np.sqrt(hull_a)
            start_params = (guess * scale).flatten()
        else:
            start_params = guess.flatten()
            
        try:
            res = minimize(
                _neg_smooth_score,
                start_params,
                args=(alpha,),
                method='L-BFGS-B',
                bounds=[(-2, 2)] * (2 * N),
                options={'maxiter': 200, 'ftol': 1e-8}
            )
            
            if res.success or res.fun < -best_score:
                candidate_pts = res.x.reshape((N, 2))
                # Calculate EXACT score for the candidate
                candidate_score = _score_points(candidate_pts)
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_pts = candidate_pts.copy()
                    
        except Exception as e:
            pass
            
    # If we have time, do a second pass with higher alpha (sharper minimum) 
    # starting from the best solution found so far, plus some perturbations.
    if time.time() < deadline - 20:
        alpha_refine = 200.0
        rng = np.random.default_rng(123)
        
        # Restart from best
        starts = [best_pts.copy()]
        # Add perturbations
        for _ in range(5):
            noise = rng.normal(0, 0.1, size=(N, 2))
            starts.append(best_pts + noise)
            
        for start_pts in starts:
            if time.time() > deadline - 5:
                break
                
            hull_a = _hull_area(start_pts)
            if hull_a > 0:
                scale = 1.0 / np.sqrt(hull_a)
                start_params = (start_pts * scale).flatten()
            else:
                start_params = start_pts.flatten()
                
            try:
                res = minimize(
                    _neg_smooth_score,
                    start_params,
                    args=(alpha_refine,),
                    method='L-BFGS-B',
                    bounds=[(-2, 2)] * (2 * N),
                    options={'maxiter': 100, 'ftol': 1e-10}
                )
                
                if res.success or res.fun < -best_score:
                    candidate_pts = res.x.reshape((N, 2))
                    candidate_score = _score_points(candidate_pts)
                    if candidate_score > best_score:
                        best_score = candidate_score
                        best_pts = candidate_pts.copy()
                        
            except Exception:
                pass
                
    return best_pts

if __name__ == "__main__":
    pts = construct()
    print(_score_points(pts))
