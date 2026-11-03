"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize


def _min_triangle_area_fast(points):
    """Fully vectorized: compute min area of any triangle formed by the n points.
    points: (n, 2) array. Returns scalar min area."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            d = points[j] - points[i]  # (2,)
            diff_k = points - points[i]  # (n, 2)
            cross = d[0] * diff_k[:, 1] - d[1] * diff_k[:, 0]  # (n,)
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            mask[j] = False
            areas = 0.5 * np.abs(cross[mask])
            local_min = np.min(areas)
            if local_min < min_area:
                min_area = local_min
    return min_area


def _is_inside_triangle(points):
    """Check if all points are inside the equilateral triangle (0,0),(1,0),(0.5,sqrt(3)/2)."""
    x = points[:, 0]
    y = points[:, 1]
    eps = 1e-9
    # y >= 0
    if np.any(y < -eps):
        return False
    # y <= sqrt(3)*x  (left edge)
    if np.any(y > np.sqrt(3) * x + eps):
        return False
    # y <= sqrt(3)*(1-x)  (right edge)
    if np.any(y > np.sqrt(3) * (1 - x) + eps):
        return False
    return True


def _clip_to_triangle(points):
    """Clip points to be inside the equilateral triangle."""
    points = points.copy()
    x = points[:, 0]
    y = points[:, 1]

    y = np.clip(y, 0, np.sqrt(3) / 2)

    x_min = y / np.sqrt(3)
    x_max = 1 - y / np.sqrt(3)

    x = np.clip(x, x_min, x_max)
    x = np.clip(x, 0, 1)

    points[:, 0] = x
    points[:, 1] = y
    return points


def _objective_de(flat):
    """Objective for differential evolution: minimize negative min triangle area.
    flat: (2*N,) array of cartesian coordinates.
    Returns: -min_area + penalty if any point is outside triangle."""
    points = flat.reshape(N, 2)

    x = points[:, 0]
    y = points[:, 1]

    penalty = 0.0
    eps = 1e-8
    penalty += np.sum(np.maximum(0, -y - eps)) * 50
    penalty += np.sum(np.maximum(0, y - np.sqrt(3) * x + eps)) * 50
    penalty += np.sum(np.maximum(0, y - np.sqrt(3) * (1 - x) + eps)) * 50

    min_area = _min_triangle_area_fast(points)
    return -min_area + penalty


def _local_refine(points, max_iter=200, tol=1e-10):
    """Steepest descent with adaptive step size to maximize min triangle area."""
    points = points.copy()
    current_area = _min_triangle_area_fast(points)
    
    for it in range(max_iter):
        improved = False
        best_step = 0
        best_area = current_area
        
        # Try several step sizes
        for step in [0.05, 0.02, 0.01, 0.005, 0.001]:
            trial = points.copy()
            # For each point, compute gradient-like direction by perturbing
            for i in range(N):
                # Perturb point i in both x and y directions
                for dx in [-step, 0, step]:
                    for dy in [-step, 0, step]:
                        if dx == 0 and dy == 0:
                            continue
                        trial[i, 0] += dx
                        trial[i, 1] += dy
                        trial = _clip_to_triangle(trial)
                        area = _min_triangle_area_fast(trial)
                        if area > best_area + tol:
                            best_area = area
                            best_step = step
                            improved = True
                        # Reset for next trial
                        trial[i, 0] -= dx
                        trial[i, 1] -= dy
            
            if improved and best_step == step:
                break
        
        if not improved:
            break
        
        # Apply the best improvement found
        # Re-find the best single-point move
        best_move = None
        best_move_area = current_area
        for i in range(N):
            for dx in [-best_step, best_step]:
                for dy in [-best_step, best_step]:
                    if dx == 0 and dy == 0:
                        continue
                    trial = points.copy()
                    trial[i, 0] += dx
                    trial[i, 1] += dy
                    trial = _clip_to_triangle(trial)
                    area = _min_triangle_area_fast(trial)
                    if area > best_move_area + tol:
                        best_move_area = area
                        best_move = (i, dx, dy)
        
        if best_move is None:
            break
        
        i, dx, dy = best_move
        points[i, 0] += dx
        points[i, 1] += dy
        points = _clip_to_triangle(points)
        current_area = _min_triangle_area_fast(points)
    
    return points


def _structured_init(seed_idx):
    """Create diverse structured initial configurations based on symmetry."""
    sqrt3 = np.sqrt(3)
    h = sqrt3 / 2  # height of triangle

    configs = []

    # Config 0: Symmetric hexagonal-like arrangement
    configs.append(np.array([
        [0.05, 0.05],
        [0.95, 0.05],
        [0.5, h - 0.05],
        [0.5, 0.05],
        [0.25, h / 2],
        [0.75, h / 2],
        [0.5, h / 3],
        [0.3, 0.3],
        [0.7, 0.3],
        [0.5, 0.45],
        [0.4, 0.35],
    ]))

    # Config 1: Three rows of 4-4-3
    configs.append(np.array([
        [0.1, 0.08], [0.35, 0.08], [0.65, 0.08], [0.9, 0.08],
        [0.2, 0.35], [0.4, 0.35], [0.6, 0.35], [0.8, 0.35],
        [0.3, 0.6], [0.5, 0.6], [0.7, 0.6],
    ]))

    # Config 2: Concentric rings
    configs.append(np.array([
        [0.5, 0.05],
        [0.2, 0.25], [0.8, 0.25],
        [0.5, 0.45],
        [0.3, 0.55], [0.7, 0.55],
        [0.1, 0.15], [0.9, 0.15],
        [0.4, 0.7], [0.6, 0.7],
        [0.5, 0.85],
    ]))

    # Config 3: Grid-based
    configs.append(np.array([
        [0.15, 0.1], [0.4, 0.1], [0.65, 0.1], [0.85, 0.1],
        [0.25, 0.35], [0.5, 0.35], [0.75, 0.35],
        [0.35, 0.6], [0.55, 0.6], [0.75, 0.6],
        [0.45, 0.8],
    ]))

    # Config 4: Random but structured
    rng = np.random.default_rng(seed_idx * 1000 + 42)
    pts = []
    # Place 3 near corners
    pts.append([0.08, 0.08])
    pts.append([0.92, 0.08])
    pts.append([0.5, h - 0.08])
    # Place 8 in interior with some structure
    for i in range(8):
        angle = i * 2 * np.pi / 8
        r = 0.2 + 0.1 * (i % 2)
        cx = 0.5 + r * np.cos(angle)
        cy = h / 3 + r * 0.8 * np.sin(angle)
        pts.append([cx, cy])
    configs.append(np.array(pts))

    return configs


def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s

    rng = np.random.default_rng(42)

    best_points = None
    best_score = -float('inf')

    # Get structured initial configurations
    inits = _structured_init(0)
    
    # Add some random inits
    for _ in range(5):
        pts = np.random.uniform(0.1, 0.9, size=(N, 2))
        pts[:, 1] = np.random.uniform(0, np.sqrt(3)/2 * 0.8, size=N)
        pts = _clip_to_triangle(pts)
        inits.append(pts)

    # First, evaluate all initial configs and keep the top ones
    scored_inits = []
    for points_init in inits:
        points_init = _clip_to_triangle(points_init)
        score = _min_triangle_area_fast(points_init)
        scored_inits.append((score, points_init))
    
    scored_inits.sort(key=lambda x: -x[0])
    top_inits = [pts for _, pts in scored_inits[:8]]

    # Phase 1: Differential Evolution global search
    if time.time() < deadline - 30:
        try:
            # Bounds for each coordinate
            bounds = []
            for i in range(N):
                bounds.append((0, 1))  # x
                bounds.append((0, np.sqrt(3)/2))  # y
            
            de_result = differential_evolution(
                _objective_de,
                bounds,
                seed=42,
                maxiter=100,
                popsize=20,
                tol=1e-6,
                polish=False,
                init='latinhypercube',
                workers=-1
            )
            
            points_de = _clip_to_triangle(de_result.x.reshape(N, 2))
            score_de = _min_triangle_area_fast(points_de)
            
            if score_de > best_score:
                best_score = score_de
                best_points = points_de.copy()
                
        except Exception:
            pass

    # Phase 2: Local refinement from multiple starting points
    for points_init in top_inits:
        if time.time() > deadline - 15:
            break
        
        try:
            refined = _local_refine(points_init, max_iter=100)
            score = _min_triangle_area_fast(refined)
            
            if score > best_score:
                best_score = score
                best_points = refined.copy()
                
        except Exception:
            pass

    # If we found a good DE result, refine it too
    if best_points is not None and time.time() < deadline - 10:
        try:
            refined = _local_refine(best_points, max_iter=150)
            score = _min_triangle_area_fast(refined)
            if score > best_score:
                best_score = score
                best_points = refined.copy()
        except Exception:
            pass

    # If no valid result, use a fallback
    if best_points is None:
        best_points = _clip_to_triangle(np.array([
            [0.05, 0.05], [0.95, 0.05], [0.5, np.sqrt(3) / 2 - 0.05],
            [0.5, 0.05], [0.25, np.sqrt(3) / 4], [0.75, np.sqrt(3) / 4],
            [0.5, np.sqrt(3) / 6], [0.3, 0.3], [0.7, 0.3],
            [0.5, 0.5], [0.4, 0.4]
        ]))
        best_score = _min_triangle_area_fast(best_points)

    return best_points
# EVOLVE-BLOCK-END
