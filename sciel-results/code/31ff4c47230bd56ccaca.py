import numpy as np
import time
from collections import Counter

# Precompute triple indices for N=30
def _precompute_indices(n):
    indices = []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                indices.append((i, j, k))
    return np.array(indices, dtype=np.int32)

_TRIPLE_INDICES = _precompute_indices(30)

def _eval(points):
    """
    Fast evaluation of max multiplicity of positive doubled area.
    """
    pts = np.array(points, dtype=np.int64)
    if len(pts) != 30:
        return 0
    
    i, j, k = _TRIPLE_INDICES[:, 0], _TRIPLE_INDICES[:, 1], _TRIPLE_INDICES[:, 2]
    xi, yi = pts[i, 0], pts[i, 1]
    xj, yj = pts[j, 0], pts[j, 1]
    xk, yk = pts[k, 0], pts[k, 1]
    
    # Doubled area = |det|
    areas = np.abs(xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj))
    
    # Filter zero areas
    mask = areas > 0
    if not np.any(mask):
        return 0
    areas = areas[mask]
    
    # Find max count
    # np.unique is usually faster than Counter for large arrays of ints
    _, counts = np.unique(areas, return_counts=True)
    return int(np.max(counts))

def _generate_grid(w, h):
    """Generate a w x h grid of points, centered or at origin."""
    pts = []
    # Center the grid in [-30, 30]
    # Range is 61x61. 
    # Let's place it at 0,0 for simplicity, or shift it.
    # The problem allows [-30, 30].
    # A 6x5 grid fits easily.
    x_start = -w // 2
    y_start = -h // 2
    for x in range(x_start, x_start + w):
        for y in range(y_start, y_start + h):
            pts.append((x, y))
    return pts

def _random_points():
    """Generate 30 distinct random points in [-30, 30]^2."""
    pts = set()
    while len(pts) < 30:
        x = np.random.randint(-30, 31)
        y = np.random.randint(-30, 31)
        pts.add((x, y))
    return list(pts)

def _hill_climb(points, time_limit=45.0):
    """
    Hill climbing with random moves and occasional kicks.
    """
    start_time = time.time()
    current = list(points)
    current_score = _eval(current)
    best = list(current)
    best_score = current_score
    
    n = len(current)
    
    while time.time() - start_time < time_limit:
        # Decide move type
        # 80% single point move, 20% kick (2-3 points)
        if np.random.rand() < 0.8:
            move_count = 1
        else:
            move_count = np.random.randint(2, 4)
            
        trial = list(current)
        used_coords = set(trial)
        
        valid_move = True
        for _ in range(move_count):
            idx = np.random.randint(n)
            old_x, old_y = trial[idx]
            
            # Remove old from used
            used_coords.discard((old_x, old_y))
            
            # Pick new position
            # Strategy: sometimes random, sometimes local perturbation
            if np.random.rand() < 0.5:
                # Local perturbation
                new_x = old_x + np.random.randint(-3, 4)
                new_y = old_y + np.random.randint(-3, 4)
                # Clamp
                new_x = max(-30, min(30, new_x))
                new_y = max(-30, min(30, new_y))
            else:
                # Random
                new_x = np.random.randint(-30, 31)
                new_y = np.random.randint(-30, 31)
            
            if (new_x, new_y) in used_coords:
                # Collision, revert this specific move attempt or just skip?
                # If we skip, we might not move enough points.
                # Let's just try a random valid one if collision.
                # For simplicity, if collision, mark invalid and break?
                # Better: just retry picking a new spot for this index.
                # To keep it simple, if collision, we abort this whole trial.
                valid_move = False
                break
            
            trial[idx] = (new_x, new_y)
            used_coords.add((new_x, new_y))
        
        if not valid_move:
            continue
            
        # Evaluate
        s = _eval(trial)
        
        if s > current_score:
            current = trial
            current_score = s
            if s > best_score:
                best = list(current)
                best_score = s
        else:
            # Simulated annealing style: accept worse moves occasionally?
            # For now, strict hill climbing.
            pass
            
    return best, best_score

def construct():
    deadline = time.time() + 85.0  # 5s safety margin for 90s limit
    
    best_points = None
    best_score = 0
    
    # Initial candidates
    candidates = [
        _generate_grid(6, 5),
        _generate_grid(5, 6),
        _generate_grid(10, 3),
        _generate_grid(3, 10),
        _generate_grid(15, 2),
        _generate_grid(2, 15),
        _generate_grid(5, 5), # 25 points, add 5 random? No, must be 30 distinct.
        # Let's just use the grids that have exactly 30 points.
        # 6x5=30, 5x6=30, 10x3=30, 3x10=30, 15x2=30, 2x15=30.
        # Also 1x30 is a line (all zero area), bad.
    ]
    
    # Filter candidates to ensure they are valid (30 distinct points in range)
    valid_candidates = []
    for c in candidates:
        if len(set(c)) == 30:
            # Check bounds
            if all(-30 <= x <= 30 and -30 <= y <= 30 for x, y in c):
                valid_candidates.append(c)
                
    # Evaluate initial candidates
    for c in valid_candidates:
        s = _eval(c)
        if s > best_score:
            best_score = s
            best_points = c
            
    # If no valid candidate, use random
    if best_points is None:
        best_points = _random_points()
        best_score = _eval(best_points)
        
    # Main search loop
    # We will do several restarts or just one long climb from the best start?
    # Given the time, a long climb from the best start is usually best.
    # But random restarts can find better basins.
    
    # Let's do a few short restarts first to find a good basin, then one long climb.
    
    # Phase 1: Quick exploration
    np.random.seed(42)
    for i in range(5):
        if time.time() > deadline - 10.0: break
        pts = _random_points()
        s = _eval(pts)
        if s > best_score:
            best_score = s
            best_points = pts
        # Short climb
        hc_pts, hc_score = _hill_climb(pts, time_limit=2.0)
        if hc_score > best_score:
            best_score = hc_score
            best_points = hc_pts
            
    # Phase 2: Long climb from best found
    remaining_time = deadline - time.time()
    if remaining_time > 5.0:
        hc_pts, hc_score = _hill_climb(best_points, time_limit=remaining_time - 3.0)
        if hc_score > best_score:
            best_score = hc_score
            best_points = hc_pts

    return best_points

# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import Counter

# Precompute triple indices for N=30
def _precompute_indices(n):
    indices = []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                indices.append((i, j, k))
    return np.array(indices, dtype=np.int32)

_TRIPLE_INDICES = _precompute_indices(30)

def _eval(points):
    """
    Fast evaluation of max multiplicity of positive doubled area.
    """
    pts = np.array(points, dtype=np.int64)
    if len(pts) != 30:
        return 0
    
    i, j, k = _TRIPLE_INDICES[:, 0], _TRIPLE_INDICES[:, 1], _TRIPLE_INDICES[:, 2]
    xi, yi = pts[i, 0], pts[i, 1]
    xj, yj = pts[j, 0], pts[j, 1]
    xk, yk = pts[k, 0], pts[k, 1]
    
    # Doubled area = |det|
    areas = np.abs(xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj))
    
    # Filter zero areas
    mask = areas > 0
    if not np.any(mask):
        return 0
    areas = areas[mask]
    
    # Find max count
    _, counts = np.unique(areas, return_counts=True)
    return int(np.max(counts))

def _generate_grid(w, h):
    """Generate a w x h grid of points, centered at origin."""
    pts = []
    x_start = -w // 2
    y_start = -h // 2
    for x in range(x_start, x_start + w):
        for y in range(y_start, y_start + h):
            pts.append((x, y))
    return pts

def _random_points():
    """Generate 30 distinct random points in [-30, 30]^2."""
    pts = set()
    while len(pts) < 30:
        x = np.random.randint(-30, 31)
        y = np.random.randint(-30, 31)
        pts.add((x, y))
    return list(pts)

def _hill_climb(points, time_limit=45.0):
    """
    Hill climbing with random moves and occasional kicks.
    """
    start_time = time.time()
    current = list(points)
    current_score = _eval(current)
    best = list(current)
    best_score = current_score
    
    n = len(current)
    
    while time.time() - start_time < time_limit:
        # Decide move type
        if np.random.rand() < 0.8:
            move_count = 1
        else:
            move_count = np.random.randint(2, 4)
            
        trial = list(current)
        used_coords = set(trial)
        
        valid_move = True
        for _ in range(move_count):
            idx = np.random.randint(n)
            old_x, old_y = trial[idx]
            
            # Remove old from used
            used_coords.discard((old_x, old_y))
            
            # Pick new position
            if np.random.rand() < 0.5:
                # Local perturbation
                new_x = old_x + np.random.randint(-3, 4)
                new_y = old_y + np.random.randint(-3, 4)
                new_x = max(-30, min(30, new_x))
                new_y = max(-30, min(30, new_y))
            else:
                # Random
                new_x = np.random.randint(-30, 31)
                new_y = np.random.randint(-30, 31)
            
            if (new_x, new_y) in used_coords:
                valid_move = False
                break
            
            trial[idx] = (new_x, new_y)
            used_coords.add((new_x, new_y))
        
        if not valid_move:
            continue
            
        # Evaluate
        s = _eval(trial)
        
        if s > current_score:
            current = trial
            current_score = s
            if s > best_score:
                best = list(current)
                best_score = s
                
    return best, best_score

def construct():
    deadline = time.time() + 85.0  # 5s safety margin for 90s limit
    
    best_points = None
    best_score = 0
    
    # Initial candidates
    candidates = [
        _generate_grid(6, 5),
        _generate_grid(5, 6),
        _generate_grid(10, 3),
        _generate_grid(3, 10),
        _generate_grid(15, 2),
        _generate_grid(2, 15),
    ]
    
    # Filter candidates to ensure they are valid (30 distinct points in range)
    valid_candidates = []
    for c in candidates:
        if len(set(c)) == 30:
            if all(-30 <= x <= 30 and -30 <= y <= 30 for x, y in c):
                valid_candidates.append(c)
                
    # Evaluate initial candidates
    for c in valid_candidates:
        s = _eval(c)
        if s > best_score:
            best_score = s
            best_points = c
            
    # If no valid candidate, use random
    if best_points is None:
        best_points = _random_points()
        best_score = _eval(best_points)
        
    # Main search loop
    
    # Phase 1: Quick exploration
    np.random.seed(42)
    for i in range(5):
        if time.time() > deadline - 10.0: break
        pts = _random_points()
        s = _eval(pts)
        if s > best_score:
            best_score = s
            best_points = pts
        # Short climb
        hc_pts, hc_score = _hill_climb(pts, time_limit=2.0)
        if hc_score > best_score:
            best_score = hc_score
            best_points = hc_pts
            
    # Phase 2: Long climb from best found
    remaining_time = deadline - time.time()
    if remaining_time > 5.0:
        hc_pts, hc_score = _hill_climb(best_points, time_limit=remaining_time - 3.0)
        if hc_score > best_score:
            best_score = hc_score
            best_points = hc_pts

    return best_points
# EVOLVE-BLOCK-END
