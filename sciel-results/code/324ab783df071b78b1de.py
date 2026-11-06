"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
from collections import Counter

def _doubled_areas(pts):
    """Compute absolute doubled areas for all triples of 30 points."""
    n = len(pts)
    pts = np.array(pts, dtype=np.int64)
    areas = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                d = (pts[j,0]-pts[i,0])*(pts[k,1]-pts[i,1]) - (pts[k,0]-pts[i,0])*(pts[j,1]-pts[i,1])
                a = abs(d)
                if a > 0:
                    areas.append(a)
    return areas

def _max_multiplicity(pts):
    """Return the largest multiplicity of a single positive doubled area."""
    areas = _doubled_areas(pts)
    if not areas:
        return 0
    c = Counter(areas)
    return max(c.values())

def _score(pts):
    """Score = max multiplicity of a single positive doubled area."""
    return _max_multiplicity(pts)

def _generate_grid_config(rows, cols, shear=0, offset_x=0, offset_y=0):
    """Generate a sheared grid configuration of rows*cols points."""
    pts = []
    for r in range(rows):
        for c in range(cols):
            x = c + shear * r + offset_x
            y = r + offset_y
            pts.append((x, y))
    return pts

def _local_search_swap(pts, max_iters=5000, deadline=None):
    """Improve configuration via swap-based local search."""
    pts = list(pts)
    best_score = _score(pts)
    n = len(pts)
    
    # Precompute areas for current config
    current_areas = _doubled_areas(pts)
    current_counter = Counter(current_areas)
    
    iter_count = 0
    while iter_count < max_iters:
        if deadline is not None and time.time() > deadline:
            break
        iter_count += 1
        
        # Try swapping a random point with a candidate from the pool
        idx = np.random.randint(n)
        # Generate candidate points around the current point
        cx, cy = pts[idx]
        candidates = []
        for dx in range(-5, 6):
            for dy in range(-5, 6):
                nx, ny = cx + dx, cy + dy
                if -30 <= nx <= 30 and -30 <= ny <= 30:
                    # Check if already in pts
                    if (nx, ny) not in pts:
                        candidates.append((nx, ny))
        
        if not candidates:
            continue
        
        # Try a few random candidates
        for _ in range(3):
            cand = candidates[np.random.randint(len(candidates))]
            new_pts = list(pts)
            new_pts[idx] = cand
            
            new_score = _score(new_pts)
            if new_score > best_score:
                pts = new_pts
                best_score = new_score
                current_areas = _doubled_areas(pts)
                current_counter = Counter(current_areas)
                break
    
    return pts, best_score

def _target_area_construction(target_A, num_points=30, deadline=None):
    """
    Try to construct a configuration where many triples have doubled area = target_A.
    Uses greedy approach: start with a base, then add/swap points to maximize
    triples with area target_A.
    """
    # Start with a simple grid
    pts = _generate_grid_config(6, 5)
    pts = list(pts)
    
    best_score = _score(pts)
    
    # Try to improve by swapping points
    for iteration in range(200):
        if deadline is not None and time.time() > deadline:
            break
            
        # Randomly select a point to replace
        idx = np.random.randint(len(pts))
        cx, cy = pts[idx]
        
        # Generate candidates
        best_new_score = best_score
        best_cand = None
        
        for _ in range(20):
            nx = np.random.randint(-30, 31)
            ny = np.random.randint(-30, 31)
            if (nx, ny) in pts:
                continue
            
            new_pts = list(pts)
            new_pts[idx] = (nx, ny)
            
            s = _score(new_pts)
            if s > best_new_score:
                best_new_score = s
                best_cand = (nx, ny)
        
        if best_cand is not None:
            pts[idx] = best_cand
            best_score = best_new_score
    
    return pts, best_score

def _multi_start_search(deadline):
    """Run multiple initial configurations and local searches."""
    best_pts = None
    best_score = 0
    
    # Generate diverse initial configurations
    configs = []
    
    # Standard grid
    configs.append(_generate_grid_config(6, 5))
    # Sheared grids
    for shear in [-3, -2, -1, 1, 2, 3]:
        configs.append(_generate_grid_config(6, 5, shear=shear))
    # Different grid shapes
    configs.append(_generate_grid_config(5, 6))
    configs.append(_generate_grid_config(5, 6, shear=1))
    configs.append(_generate_grid_config(5, 6, shear=-1))
    configs.append(_generate_grid_config(10, 3))
    configs.append(_generate_grid_config(3, 10))
    configs.append(_generate_grid_config(10, 3, shear=1))
    
    # Random configurations
    for _ in range(5):
        random_pts = set()
        while len(random_pts) < 30:
            x = np.random.randint(-30, 31)
            y = np.random.randint(-30, 31)
            random_pts.add((x, y))
        configs.append(list(random_pts))
    
    # Evaluate each config and do local search
    for cfg in configs:
        if deadline is not None and time.time() > deadline:
            break
        
        if len(cfg) != 30:
            continue
        
        # Check validity
        if len(set(cfg)) != 30:
            continue
        if any(not (-30 <= p[0] <= 30 and -30 <= p[1] <= 30) for p in cfg):
            continue
        
        s = _score(cfg)
        if s > best_score:
            best_score = s
            best_pts = list(cfg)
        
        # Local search from this config
        improved_pts, improved_score = _local_search_swap(cfg, max_iters=2000, deadline=deadline)
        if improved_score > best_score:
            best_score = improved_score
            best_pts = improved_pts
    
    return best_pts, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 75  # 75s deadline with 15s safety margin
    
    np.random.seed(42)
    
    best_pts = None
    best_score = 0
    
    # Phase 1: Multi-start with local search
    if best_pts is None:
        best_pts, best_score = _multi_start_search(deadline)
    
    # Phase 2: Additional local search from the best found
    if best_pts is not None and time.time() < deadline - 5:
        improved_pts, improved_score = _local_search_swap(best_pts, max_iters=10000, deadline=deadline)
        if improved_score > best_score:
            best_score = improved_score
            best_pts = improved_pts
    
    # Phase 3: Target-area guided construction
    if time.time() < deadline - 10:
        # Try some common target areas
        for target_A in [1, 2, 3, 4, 5, 6, 7, 8, 10, 12]:
            if time.time() > deadline - 5:
                break
            ta_pts, ta_score = _target_area_construction(target_A, deadline=deadline)
            if ta_score > best_score:
                best_score = ta_score
                best_pts = ta_pts
    
    # Final safety: ensure we have a valid 30-point configuration
    if best_pts is None or len(best_pts) != 30 or len(set(best_pts)) != 30:
        # Fallback to simple grid
        best_pts = _generate_grid_config(6, 5)
        best_score = _score(best_pts)
    
    return best_pts
# EVOLVE-BLOCK-END
