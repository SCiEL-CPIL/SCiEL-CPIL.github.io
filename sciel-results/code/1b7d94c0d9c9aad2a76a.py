import numpy as np
from scipy.spatial import ConvexHull
import time
import itertools

def _compute_score(points: np.ndarray) -> float:
    """
    Compute the score A_min / A_hull for a set of points.
    """
    n = points.shape[0]
    if n < 3:
        return 0.0
    
    # Check for finite points
    if not np.all(np.isfinite(points)):
        return 0.0
    
    # Compute convex hull
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        return 0.0
        
    if A_hull <= 0:
        return 0.0
        
    # Compute all triangle areas vectorized
    # Points shape: (n, 2)
    # We need all combinations of 3 points
    
    # Get indices for all combinations
    idx = np.arange(n)
    i, j, k = np.meshgrid(idx, idx, idx, indexing='ij')
    
    # Filter to i < j < k to avoid duplicates and self-combinations
    mask = (i < j) & (j < k)
    i, j, k = i[mask], j[mask], k[mask]
    
    # Extract coordinates
    x1, y1 = points[i, 0], points[i, 1]
    x2, y2 = points[j, 0], points[j, 1]
    x3, y3 = points[k, 0], points[k, 1]
    
    # Area formula: 0.5 * |x1(y2 - y3) + x2(y3 - y1) + x3(y1 - y2)|
    areas = 0.5 * np.abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    
    A_min = np.min(areas)
    
    return A_min / A_hull

def _nelder_mead(points: np.ndarray, start_time: float, max_time: float = 600.0, max_iter: int = 5000) -> np.ndarray:
    """
    Perform Nelder-Mead optimization to maximize the score.
    """
    current_points = points.copy()
    current_score = _compute_score(current_points)
    best_points = current_points.copy()
    best_score = current_score
    
    n = len(current_points)
    dim = 2 * n
    
    # Flatten points to 1D vector
    x = current_points.flatten().copy()
    
    # Create initial simplex
    simplex = [x.copy()]
    for i in range(dim):
        trial = x.copy()
        trial[i] += 0.01
        simplex.append(trial)
    
    # Sort simplex by score (ascending)
    def eval_simplex(simplex):
        scores = []
        for pt in simplex:
            pts_2d = pt.reshape(n, 2)
            scores.append(_compute_score(pts_2d))
        return scores
    
    scores = eval_simplex(simplex)
    order = np.argsort(scores)
    simplex = [simplex[i] for i in order]
    scores = [scores[i] for i in order]
    
    alpha = 1.0  # Reflection
    gamma = 2.0  # Expansion
    rho = 0.5    # Contraction
    sigma = 0.5  # Shrink
    
    for iteration in range(max_iter):
        if time.time() - start_time > max_time - 5:
            break
        
        # Get worst point
        worst = simplex[-1]
        worst_score = scores[-1]
        
        # Compute centroid of all points except worst
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        xr = centroid + alpha * (centroid - worst)
        pts_r = xr.reshape(n, 2)
        fr = _compute_score(pts_r)
        
        if fr < scores[0]:
            # Expansion
            xe = centroid + gamma * (centroid - worst)
            pts_e = xe.reshape(n, 2)
            fe = _compute_score(pts_e)
            if fe < fr:
                simplex[-1] = xe
                scores[-1] = fe
            else:
                simplex[-1] = xr
                scores[-1] = fr
        elif fr < scores[-2]:
            # Accept reflection
            simplex[-1] = xr
            scores[-1] = fr
        else:
            # Contraction
            if fr < worst_score:
                # Outside contraction
                xc = centroid + rho * (worst - centroid)
            else:
                # Inside contraction
                xc = centroid - rho * (centroid - worst)
            
            pts_c = xc.reshape(n, 2)
            fc = _compute_score(pts_c)
            
            if fc < min(fr, worst_score):
                simplex[-1] = xc
                scores[-1] = fc
            else:
                # Shrink
                for i in range(1, len(simplex)):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    pts_s = simplex[i].reshape(n, 2)
                    scores[i] = _compute_score(pts_s)
        
        # Re-sort simplex
        scores_arr = np.array(scores)
        order = np.argsort(scores_arr)
        simplex = [simplex[i] for i in order]
        scores = [scores[i] for i in order]
        
        # Update best
        best_in_simplex = simplex[0]
        best_score_simplex = scores[0]
        if best_score_simplex > best_score:
            best_points = best_in_simplex.reshape(n, 2).copy()
            best_score = best_score_simplex
        
        # Check convergence
        if np.max(scores) - np.min(scores) < 1e-10:
            break
    
    return best_points

def _hill_climb(points: np.ndarray, start_time: float, max_time: float = 600.0) -> np.ndarray:
    """
    Perform hill climbing with adaptive step size.
    """
    current_points = points.copy()
    current_score = _compute_score(current_points)
    best_points = current_points.copy()
    best_score = current_score
    
    n = len(current_points)
    rng = np.random.default_rng(seed=42)
    step_size = 0.05
    
    while time.time() - start_time < max_time - 10:
        improved = False
        
        for i in range(n):
            for _ in range(3):
                angle = rng.uniform(0, 2 * np.pi)
                dist = rng.uniform(0, step_size)
                dx = dist * np.cos(angle)
                dy = dist * np.sin(angle)
                
                new_points = current_points.copy()
                new_points[i, 0] += dx
                new_points[i, 1] += dy
                
                new_score = _compute_score(new_points)
                
                if new_score > current_score:
                    current_points = new_points
                    current_score = new_score
                    improved = True
                    if current_score > best_score:
                        best_points = current_points.copy()
                        best_score = current_score
                        step_size *= 0.95
                        if step_size < 0.001:
                            step_size = 0.001
                else:
                    step_size *= 0.98
                    if step_size < 0.0001:
                        step_size = 0.0001
        
        if not improved:
            step_size *= 0.8
            if step_size < 0.00001:
                break
    
    return best_points

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the area of the
    smallest triangle formed by these points.
    """
    start_time = time.time()
    max_time = 600.0
    
    n = 13
    best_points = None
    best_score = -np.inf
    
    # Try multiple initializations
    initializations = []
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    initializations.append(np.column_stack((np.cos(angles), np.sin(angles))))
    
    # 2. Regular 13-gon with slight perturbation
    rng = np.random.default_rng(seed=1)
    perturbed = initializations[0] + rng.normal(0, 0.05, size=(n, 2))
    initializations.append(perturbed)
    
    # 3. Points on a circle with varying radii
    angles2 = np.linspace(0, 2 * np.pi, n, endpoint=False)
    radii = np.ones(n)
    radii[0] = 0.8
    radii[1] = 1.2
    initializations.append(np.column_stack((radii * np.cos(angles2), radii * np.sin(angles2))))
    
    # 4. Grid-like arrangement
    grid_x = [-1, 0, 1]
    grid_y = [-1, 0, 1]
    grid_points = []
    for x in grid_x:
        for y in grid_y:
            grid_points.append([x, y])
    # Add 4 more points to make 13
    for angle in np.linspace(0, 2 * np.pi, 4, endpoint=False):
        grid_points.append([1.5 * np.cos(angle), 1.5 * np.sin(angle)])
    initializations.append(np.array(grid_points[:13]))
    
    # For each initialization, run Nelder-Mead then hill climbing
    for init_idx, init_points in enumerate(initializations):
        remaining_time = max_time - (time.time() - start_time)
        if remaining_time < 30:
            break
        
        # Run Nelder-Mead
        nm_points = _nelder_mead(init_points, start_time, max_time)
        nm_score = _compute_score(nm_points)
        
        # Run hill climbing from Nelder-Mead result
        remaining_time = max_time - (time.time() - start_time)
        if remaining_time > 20:
            hc_points = _hill_climb(nm_points, start_time, max_time)
            hc_score = _compute_score(hc_points)
            
            if hc_score > nm_score:
                candidate_points = hc_points
                candidate_score = hc_score
            else:
                candidate_points = nm_points
                candidate_score = nm_score
        else:
            candidate_points = nm_points
            candidate_score = nm_score
        
        if candidate_score > best_score:
            best_points = candidate_points
            best_score = candidate_score
    
    return best_points
