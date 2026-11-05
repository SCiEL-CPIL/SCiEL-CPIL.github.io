"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _collinear3(p1, p2, p3):
    """Check if three points are collinear."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    return (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1)

def _count_collinear_triples(points):
    """Count the number of collinear triples among the points."""
    n = len(points)
    triples = 0
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if _collinear3(points[i], points[j], points[k]):
                    triples += 1
    return triples

def _has_four_collinear(points):
    """Check if any four points are collinear."""
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    if _collinear3(points[i], points[j], points[k]) and \
                       _collinear3(points[i], points[j], points[l]):
                        return True
    return False

def _max_3free_subset(points, deadline=None):
    """Branch and bound to find the largest subset with no three collinear."""
    n = len(points)
    best = [0]
    
    # Precompute collinearity pairs: for each pair (i,j), store which k's are collinear
    collinear_with = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if _collinear3(points[i], points[j], points[k]):
                    collinear_with[i].add((j, k))
                    collinear_with[j].add((i, k))
                    collinear_with[k].add((i, j))
    
    def dfs(idx, current, current_size):
        if deadline and time.time() > deadline:
            return
        if idx >= n:
            if current_size > best[0]:
                best[0] = current_size
            return
        
        # Pruning
        if current_size + (n - idx) <= best[0]:
            return
        
        # Try excluding idx
        dfs(idx + 1, current, current_size)
        
        # Try including idx
        can_add = True
        for ci in current:
            if (ci, idx) in collinear_with[idx] or (idx, ci) in collinear_with[idx]:
                # Check if there's another point cj in current such that ci, cj, idx are collinear
                for cj in current:
                    if cj != ci and _collinear3(points[idx], points[ci], points[cj]):
                        can_add = False
                        break
            if not can_add:
                break
        
        if can_add:
            current.append(idx)
            dfs(idx + 1, current, current_size + 1)
            current.pop()
    
    dfs(0, [], 0)
    return best[0]

def _greedy_construct(seed, max_triples_target=15):
    """
    Greedy construction: build 20 points trying to create 3-collinear triples
    while avoiding 4-collinear. Uses a structured approach with lines.
    """
    rng = random.Random(seed)
    
    # Start with a base set of points
    points = []
    seen = set()
    
    # Strategy: place points in groups of 3 on lines, but perturb to avoid 4-collinear
    # Use a grid-like structure with controlled perturbations
    
    # Generate candidate lines (directions)
    directions = []
    for dx in range(-10, 11):
        for dy in range(-10, 11):
            if dx == 0 and dy == 0:
                continue
            # Normalize
            g = abs(dx)
            while g > 0 and dy % g == 0:
                g = abs(g // (dy // g)) if dy != 0 else abs(dx)
                break
            directions.append((dx, dy))
    
    # Simplified: use a few key directions
    key_directions = [(1, 0), (0, 1), (1, 1), (1, -1), (2, 1), (1, 2), (2, -1), (-1, 2)]
    
    # Place 20 points with controlled collinearity
    # Use a 5x4 grid base with perturbations
    base_grid = []
    for row in range(4):
        for col in range(5):
            base_grid.append((col * 18, row * 18))
    
    # Shuffle and assign perturbations
    shuffled = base_grid[:]
    rng.shuffle(shuffled)
    
    for i, (bx, by) in enumerate(shuffled):
        # Perturb to break 4-collinear but keep some 3-collinear
        px = bx + rng.randint(-4, 4)
        py = by + rng.randint(-4, 4)
        px = max(-100, min(100, px))
        py = max(-100, min(100, py))
        p = (px, py)
        if p not in seen:
            seen.add(p)
            points.append(list(p))
    
    # Fill remaining if needed
    while len(points) < 20:
        x = rng.randint(-100, 100)
        y = rng.randint(-100, 100)
        p = (x, y)
        if p not in seen:
            seen.add(p)
            points.append(list(p))
    
    return points[:20]

def _local_search_3free(points, deadline):
    """
    Local search that directly minimizes the 3-free subset size.
    Moves points to create more collinear triples.
    """
    points = [p[:] for p in points]
    
    # Initial evaluation
    initial_score = _max_3free_subset(points, deadline - 10)
    current_score = initial_score
    
    while time.time() < deadline - 5:
        improved = False
        
        for i in range(20):
            if time.time() > deadline - 5:
                break
            
            orig = points[i][:]
            
            # Try several random moves
            for attempt in range(50):
                if time.time() > deadline - 5:
                    break
                
                # Random perturbation
                dx = rng.randint(-8, 8) if 'rng' in dir() else random.randint(-8, 8)
                dy = random.randint(-8, 8)
                new_x = max(-100, min(100, orig[0] + dx))
                new_y = max(-100, min(100, orig[1] + dy))
                points[i] = [new_x, new_y]
                
                # Check validity
                if len(set(tuple(p) for p in points)) != 20:
                    points[i] = orig[:]
                    continue
                if _has_four_collinear(points):
                    points[i] = orig[:]
                    continue
                
                # Evaluate
                new_score = _max_3free_subset(points, deadline - 5)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                    break
                else:
                    points[i] = orig[:]
            
            if improved:
                break
        
        if not improved:
            break
    
    return points, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    best_points = None
    best_score = 20  # Worst case (all 20 can be 3-free)
    
    # Phase 1: Greedy construction with multiple seeds
    for seed in range(200):
        if time.time() > deadline - 40:
            break
        
        points = _greedy_construct(seed)
        
        # Check validity
        if len(set(tuple(p) for p in points)) != 20:
            continue
        if _has_four_collinear(points):
            continue
        
        # Quick evaluation using triple count as proxy
        triples = _count_collinear_triples(points)
        
        # Keep track of best by triple count initially
        if best_points is None or triples > _count_collinear_triples(best_points):
            best_points = [p[:] for p in points]
    
    # Phase 2: Local search on best candidates
    if best_points is not None and time.time() < deadline - 10:
        improved_points, improved_score = _local_search_3free(best_points, deadline - 5)
        if len(set(tuple(p) for p in improved_points)) == 20 and not _has_four_collinear(improved_points):
            best_points = improved_points
            best_score = improved_score
    
    # Phase 3: Final exact evaluation
    if best_points is not None and time.time() < deadline - 3:
        final_score = _max_3free_subset(best_points, deadline - 1)
        best_score = final_score
    
    # Return best points found
    if best_points is not None:
        return best_points
    else:
        # Fallback: parabola (no three collinear, score = 20)
        return [[x, x * x] for x in range(-10, 10)]
# EVOLVE-BLOCK-END
