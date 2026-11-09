# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from itertools import combinations

def _score(points):
    """Compute A_min / A_hull for a set of 13 points."""
    n = len(points)
    if n < 3:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    try:
        hull = ConvexHull(points)
        a_hull = hull.volume
    except Exception:
        return 0.0
    if a_hull <= 1e-12:
        return 0.0
    
    # Compute all triangle areas vectorized
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = abs(
                    points[i,0] * (points[j,1] - points[k,1]) +
                    points[j,0] * (points[k,1] - points[i,1]) +
                    points[k,0] * (points[i,1] - points[j,1])
                ) * 0.5
                if area < min_area:
                    min_area = area
                    if min_area < 1e-10:
                        return 0.0
    
    if min_area <= 1e-12:
        return 0.0
    return min_area / a_hull

def _score_fast(points):
    """Faster scoring using vectorized operations for the 286 triangles."""
    n = len(points)
    if n < 3:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    try:
        hull = ConvexHull(points)
        a_hull = hull.volume
    except Exception:
        return 0.0
    if a_hull <= 1e-12:
        return 0.0
    
    # Precompute all triangle areas
    areas = np.empty(286)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                areas[idx] = abs(
                    points[i,0] * (points[j,1] - points[k,1]) +
                    points[j,0] * (points[k,1] - points[i,1]) +
                    points[k,0] * (points[i,1] - points[j,1])
                ) * 0.5
                idx += 1
    
    min_area = np.min(areas)
    if min_area <= 1e-12:
        return 0.0
    return min_area / a_hull

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the smallest triangle area
    relative to the convex hull area.
    """
    start_time = time.time()
    deadline = start_time + 570  # Safety margin before 630s limit
    
    n = 13
    rng = np.random.default_rng(seed=42)
    
    best_points = None
    best_score = 0.0
    
    def update_best(pts, score):
        nonlocal best_points, best_score
        if score > best_score:
            best_score = score
            best_points = pts.copy()
    
    # ===== Phase 1: Good initializations =====
    initializations = []
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    initializations.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # 2. Hexagonal lattice pattern (center + 2 rings)
    hex_pts = np.zeros((n, 2))
    hex_pts[0] = [0.0, 0.0]
    for i in range(6):
        angle = i * np.pi / 3
        hex_pts[i + 1] = [np.cos(angle), np.sin(angle)]
    for i in range(6):
        angle = i * np.pi / 3 + np.pi / 6
        hex_pts[i + 7] = [np.sqrt(3) * np.cos(angle), np.sqrt(3) * np.sin(angle)]
    initializations.append(hex_pts)
    
    # 3. Grid 3x3 + 4 center points
    grid_pts = np.array([
        [0.0, 0.0], [1.0, 0.0], [2.0, 0.0],
        [0.0, 1.0], [1.0, 1.0], [2.0, 1.0],
        [0.0, 2.0], [1.0, 2.0], [2.0, 2.0],
        [0.5, 0.5], [1.5, 0.5], [0.5, 1.5], [1.5, 1.5],
    ], dtype=float)
    initializations.append(grid_pts)
    
    # 4. Concentric circles: 1 center + 4 inner + 8 outer
    conc_pts = np.zeros((n, 2))
    conc_pts[0] = [0.0, 0.0]
    for i in range(4):
        angle = i * np.pi / 2 + np.pi / 4
        conc_pts[i + 1] = [0.5 * np.cos(angle), 0.5 * np.sin(angle)]
    for i in range(8):
        angle = i * np.pi / 4
        conc_pts[i + 5] = [np.cos(angle), np.sin(angle)]
    initializations.append(conc_pts)
    
    # 5. Several random points in unit square
    for _ in range(5):
        initializations.append(rng.random((n, 2)))
    
    # 6. Random points in hexagon
    for _ in range(3):
        pts = []
        for _ in range(n):
            r = rng.random() ** 0.5
            theta = rng.random() * 2 * np.pi
            pts.append([r * np.cos(theta), r * np.sin(theta)])
        initializations.append(np.array(pts))
    
    # Evaluate all initializations
    for init_pts in initializations:
        if time.time() > deadline - 60:
            break
        s = _score_fast(init_pts)
        update_best(init_pts, s)
    
    # ===== Phase 2: Multi-scale coordinate descent =====
    current_points = best_points.copy()
    current_score = best_score
    
    # Define step sizes from coarse to fine
    step_schedule = [0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002]
    
    for base_step in step_schedule:
        if time.time() > deadline - 60:
            break
        
        # Multiple passes at this step size
        for pass_num in range(3):
            if time.time() > deadline - 60:
                break
            
            improved = True
            while improved:
                improved = False
                for i in range(n):
                    if time.time() > deadline - 60:
                        break
                    
                    # Try many random directions
                    for _ in range(40):
                        direction = rng.normal(size=2)
                        norm = np.linalg.norm(direction)
                        if norm < 1e-10:
                            continue
                        direction /= norm
                        
                        # Try a few step multipliers
                        for mult in [1.0, 0.5, 0.25, 2.0, 0.1]:
                            step = base_step * mult
                            trial = current_points.copy()
                            trial[i] = current_points[i] + step * direction
                            s = _score_fast(trial)
                            if s > current_score:
                                current_points = trial
                                current_score = s
                                improved = True
                                break
                        if improved:
                            break
                    if time.time() > deadline - 60:
                        break
            
            update_best(current_points, current_score)
    
    # ===== Phase 3: Simulated annealing =====
    if time.time() < deadline - 90:
        current = best_points.copy()
        cur_score = best_score
        T = 0.1
        T_min = 0.001
        cooling = 0.995
        n_sa_steps = 5000
        
        for step in range(n_sa_steps):
            if time.time() > deadline - 60:
                break
            
            # Pick a random point and perturb it
            i = rng.integers(n)
            perturbation = rng.normal(size=2) * T
            trial = current.copy()
            trial[i] = current[i] + perturbation
            
            s = _score_fast(trial)
            
            if s > cur_score:
                current = trial
                cur_score = s
                update_best(current, cur_score)
            else:
                # Accept with probability exp((s - cur_score) / T)
                if s > 0 and T > T_min:
                    ratio = (s - cur_score) / T
                    if np.log(rng.random()) < ratio:
                        current = trial
                        cur_score = s
            
            T *= cooling
            if T < T_min:
                T = T_min
    
    # ===== Phase 4: More random restarts with local optimization =====
    for restart in range(8):
        if time.time() > deadline - 60:
            break
        
        # Start from a perturbed version of best or a new random point
        if restart % 2 == 0:
            start = best_points + rng.normal(size=(n, 2)) * 0.15
        else:
            start = rng.random((n, 2))
        
        s = _score_fast(start)
        if s <= 0:
            continue
        
        current = start.copy()
        cur_score = s
        
        # Quick local optimization
        for base_step in [0.1, 0.05, 0.02, 0.01, 0.005]:
            if time.time() > deadline - 60:
                break
            for i in range(n):
                if time.time() > deadline - 60:
                    break
                for _ in range(20):
                    direction = rng.normal(size=2)
                    norm = np.linalg.norm(direction)
                    if norm < 1e-10:
                        continue
                    direction /= norm
                    for mult in [1.0, 0.5, 2.0]:
                        step = base_step * mult
                        trial = current.copy()
                        trial[i] = current[i] + step * direction
                        s = _score_fast(trial)
                        if s > cur_score:
                            current = trial
                            cur_score = s
                            break
        if cur_score > best_score:
            best_points = current.copy()
            best_score = cur_score
    
    # ===== Phase 5: Fine-tuning with very small steps =====
    if time.time() < deadline - 30:
        current = best_points.copy()
        cur_score = best_score
        for base_step in [0.001, 0.0005, 0.0002, 0.0001]:
            if time.time() > deadline - 30:
                break
            for i in range(n):
                if time.time() > deadline - 30:
                    break
                for _ in range(30):
                    direction = rng.normal(size=2)
                    norm = np.linalg.norm(direction)
                    if norm < 1e-10:
                        continue
                    direction /= norm
                    trial = current.copy()
                    trial[i] = current[i] + base_step * direction
                    s = _score_fast(trial)
                    if s > cur_score:
                        current = trial
                        cur_score = s
        if cur_score > best_score:
            best_points = current.copy()
            best_score = cur_score
    
    # Final validation
    final_score = _score_fast(best_points)
    if final_score <= 0 or not np.all(np.isfinite(best_points)):
        # Fallback to a known valid configuration
        fallback = np.array([
            [0.0, 0.0], [1.0, 0.0], [2.0, 0.0],
            [0.0, 1.0], [1.0, 1.0], [2.0, 1.0],
            [0.0, 2.0], [1.0, 2.0], [2.0, 2.0],
            [0.5, 0.5], [1.5, 0.5], [0.5, 1.5], [1.5, 1.5],
        ], dtype=float)
        return fallback
    
    return best_points

# EVOLVE-BLOCK-END
