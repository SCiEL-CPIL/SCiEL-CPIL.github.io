"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

def _precompute_triples(n):
    """Precompute all C(n,3) triple indices as a flat array of shape (T, 3)."""
    triples = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                triples.append((i, j, k))
    return np.array(triples, dtype=np.int32)

def _compute_areas(pts, triples):
    """Compute doubled areas for all triples. pts is (n,2) array, triples is (T,3) array."""
    p0 = pts[triples[:, 0]]
    p1 = pts[triples[:, 1]]
    p2 = pts[triples[:, 2]]
    # doubled area = |det(p1-p0, p2-p0)|
    d1 = p1 - p0
    d2 = p2 - p0
    areas = np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0])
    return areas

def _max_multiplicity(areas):
    """Return the maximum multiplicity among positive areas."""
    pos = areas[areas > 0]
    if len(pos) == 0:
        return 0
    vals, counts = np.unique(pos, return_counts=True)
    return int(counts.max())

def _score(pts):
    """Score = max multiplicity of a single positive doubled area."""
    areas = _compute_areas(pts, _TRIPLES)
    return _max_multiplicity(areas)

def _generate_grid_config(n, x_range, y_range):
    """Generate a grid configuration with n points."""
    # Try to find good grid dimensions
    best_config = None
    best_score = -1
    for nx in range(2, 12):
        for ny in range(2, 12):
            if nx * ny != n:
                continue
            # Center the grid
            xs = np.linspace(-x_range, x_range, nx, dtype=int)
            ys = np.linspace(-y_range, y_range, ny, dtype=int)
            pts = np.array([(x, y) for x in xs for y in ys], dtype=int)
            if np.any(np.abs(pts) > 30):
                continue
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_config = pts.copy()
    return best_config, best_score

def _local_search(pts, triples, max_steps=5000, deadline=None):
    """Local search: try moving individual points to improve max multiplicity."""
    current_score = _score(pts)
    pts = pts.copy()
    n = len(pts)
    
    for step in range(max_steps):
        if deadline is not None and time.time() > deadline:
            break
        
        # Pick a random point to move
        idx = np.random.randint(n)
        orig_x, orig_y = pts[idx]
        
        # Try all possible moves within [-30, 30]
        best_move = None
        best_new_score = current_score
        
        # Try nearby moves first (±1, ±2, ±3)
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = orig_x + dx, orig_y + dy
                if nx < -30 or nx > 30 or ny < -30 or ny > 30:
                    continue
                # Check no duplicate
                if np.any((pts[:, 0] == nx) & (pts[:, 1] == ny)):
                    continue
                pts[idx] = [nx, ny]
                s = _score(pts)
                if s > best_new_score:
                    best_new_score = s
                    best_move = (nx, ny)
                pts[idx] = [orig_x, orig_y]
        
        if best_move is not None:
            pts[idx] = best_move
            current_score = best_new_score
        else:
            # Try a random larger move
            for _ in range(5):
                nx = np.random.randint(-30, 31)
                ny = np.random.randint(-30, 31)
                if np.any((pts[:, 0] == nx) & (pts[:, 1] == ny)):
                    continue
                pts[idx] = [nx, ny]
                s = _score(pts)
                if s > best_new_score:
                    best_new_score = s
                    best_move = (nx, ny)
                pts[idx] = [orig_x, orig_y]
                if best_move is not None:
                    break
            if best_move is not None:
                pts[idx] = best_move
                current_score = best_new_score
    
    return pts, current_score

# Precompute triples at module level
_TRIPLES = _precompute_triples(30)

def construct():
    deadline = time.time() + 80  # 80s deadline, 10s safety margin
    np.random.seed(42)
    
    best_pts = None
    best_score = 0
    
    # Try various initial configurations
    configs = []
    
    # Standard grids
    for nx, ny in [(6,5), (5,6), (10,3), (3,10), (15,2), (2,15)]:
        if nx * ny == 30:
            xs = np.linspace(-30, 30, nx, dtype=int)
            ys = np.linspace(-30, 30, ny, dtype=int)
            pts = np.array([(x, y) for x in xs for y in ys], dtype=int)
            if np.any(np.abs(pts) > 30):
                continue
            configs.append(pts)
    
    # Smaller centered grids
    for cx, cy in [(0,0)]:
        for nx, ny in [(6,5), (5,6)]:
            xs = np.arange(-2, 4)  # -2,-1,0,1,2,3 -> shift
            xs = np.arange(cx - (nx-1)//2, cx - (nx-1)//2 + nx)
            ys = np.arange(cy - (ny-1)//2, cy - (ny-1)//2 + ny)
            pts = np.array([(x, y) for x in xs for y in ys], dtype=int)
            configs.append(pts)
    
    # Scaled grids
    for scale in [1, 2, 3, 4, 5]:
        for nx, ny in [(6,5), (5,6)]:
            xs = np.arange(-2, 4) * scale
            ys = np.arange(-2, 4) * scale if ny == 6 else np.arange(-2, 3) * scale
            if len(xs) * len(ys) != 30:
                continue
            pts = np.array([(x, y) for x in xs for y in ys], dtype=int)
            if np.any(np.abs(pts) > 30):
                continue
            configs.append(pts)
    
    # Random initial configs
    for _ in range(10):
        pts = np.random.randint(-30, 31, size=(30, 2))
        # Ensure uniqueness
        unique = set(map(tuple, pts))
        if len(unique) == 30:
            configs.append(pts)
    
    # Evaluate all configs and pick best few for local search
    scored_configs = []
    for pts in configs:
        if len(set(map(tuple, pts))) != 30:
            continue
        if np.any(np.abs(pts) > 30):
            continue
        s = _score(pts)
        scored_configs.append((s, pts))
    
    scored_configs.sort(key=lambda x: -x[0])
    
    # Local search on top configs
    for s, pts in scored_configs[:5]:
        if time.time() > deadline:
            break
        if s > best_score:
            best_score = s
            best_pts = pts.copy()
        
        pts_ls, s_ls = _local_search(pts, _TRIPLES, max_steps=3000, deadline=deadline)
        if s_ls > best_score:
            best_score = s_ls
            best_pts = pts_ls.copy()
    
    # Polish: exhaustive ±2 moves on best result
    if best_pts is not None and time.time() < deadline - 5:
        pts_polish = best_pts.copy()
        current_score = best_score
        n = 30
        for idx in range(n):
            if time.time() > deadline:
                break
            orig = pts_polish[idx].copy()
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = orig[0] + dx, orig[1] + dy
                    if nx < -30 or nx > 30 or ny < -30 or ny > 30:
                        continue
                    if np.any((pts_polish[:, 0] == nx) & (pts_polish[:, 1] == ny)):
                        continue
                    pts_polish[idx] = [nx, ny]
                    s = _score(pts_polish)
                    if s > current_score:
                        current_score = s
                        best_pts = pts_polish.copy()
                        best_score = s
                    pts_polish[idx] = orig
            # Restore
            pts_polish[idx] = orig
    
    if best_pts is None:
        # Fallback: seed config
        return [(x, y) for x in range(6) for y in range(5)]
    
    return [tuple(p) for p in best_pts]
# EVOLVE-BLOCK-END

