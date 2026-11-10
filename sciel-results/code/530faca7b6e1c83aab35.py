"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import combinations
from collections import Counter

# EVOLVE-BLOCK-START

def _compute_areas(points):
    """Compute all positive doubled areas for triples of points."""
    n = len(points)
    pts = np.array(points, dtype=np.int64)
    areas = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                d = (pts[j][0] - pts[i][0]) * (pts[k][1] - pts[i][1]) - \
                    (pts[j][1] - pts[i][1]) * (pts[k][0] - pts[i][0])
                ad = abs(d)
                if ad > 0:
                    areas.append(ad)
    return areas

def _max_multiplicity(points):
    """Return the largest multiplicity of a single positive doubled area."""
    areas = _compute_areas(points)
    if not areas:
        return 0
    c = Counter(areas)
    return max(c.values())

def _random_points(n=30, lo=-30, hi=30):
    """Generate n distinct random integer points in [lo, hi]^2."""
    pts = set()
    while len(pts) < n:
        x = random.randint(lo, hi)
        y = random.randint(lo, hi)
        pts.add((x, y))
    return list(pts)

def _hill_climb(points, max_steps=500, lo=-30, hi=30):
    """Local search: try moving one point to a random valid location."""
    current = list(points)
    current_score = _max_multiplicity(current)
    n = len(current)
    
    for step in range(max_steps):
        # Pick a random point to move
        idx = random.randrange(n)
        old_pt = current[idx]
        
        # Try a few random new positions
        improved = False
        for _ in range(20):
            new_x = random.randint(lo, hi)
            new_y = random.randint(lo, hi)
            new_pt = (new_x, new_y)
            if new_pt == old_pt:
                continue
            # Check distinctness
            if any(new_pt == p for i, p in enumerate(current) if i != idx):
                continue
            
            candidate = list(current)
            candidate[idx] = new_pt
            new_score = _max_multiplicity(candidate)
            if new_score > current_score:
                current = candidate
                current_score = new_score
                improved = True
                break
            elif new_score == current_score and random.random() < 0.3:
                # Accept equal score moves with some probability (simulated annealing-like)
                current = candidate
                improved = True
                break
        
        if not improved:
            # Try a small perturbation: swap two points
            if random.random() < 0.5:
                i = random.randrange(n)
                j = random.randrange(n)
                if i != j:
                    candidate = list(current)
                    candidate[i], candidate[j] = current[j], current[i]
                    new_score = _max_multiplicity(candidate)
                    if new_score >= current_score:
                        current = candidate
                        current_score = new_score
    
    return current, current_score

def construct():
    """Construct 30 distinct integer points in [-30,30]^2 maximizing max area multiplicity."""
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    best_points = None
    best_score = 0
    
    # Strategy 1: Start with structured grids and refine
    # Try several grid-based initial configurations
    random.seed(42)
    
    # Grid 6x5
    grid1 = [(x, y) for x in range(6) for y in range(5)]
    # Grid 5x6
    grid2 = [(x, y) for x in range(5) for y in range(6)]
    # Grid with offset
    grid3 = [(x, y) for x in range(-2, 4) for y in range(-2, 3)]
    # Different structured pattern
    grid4 = [(x, y) for x in range(0, 6) for y in range(0, 5)]
    grid5 = [(x*2, y*2) for x in range(4) for y in range(4)] + [(7, 0), (7, 7)]
    
    initial_candidates = [grid1, grid2, grid3, grid4, grid5]
    
    for init in initial_candidates:
        if time.time() > deadline:
            break
        if len(set(init)) != 30:
            continue
        if any(abs(x) > 30 or abs(y) > 30 for x, y in init):
            continue
        pts, score = _hill_climb(init, max_steps=200)
        if score > best_score:
            best_score = score
            best_points = pts
    
    # Strategy 2: Random restarts with hill climbing
    num_restarts = 50
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        random_pts = _random_points(30, -30, 30)
        pts, score = _hill_climb(random_pts, max_steps=100)
        if score > best_score:
            best_score = score
            best_points = pts
    
    # Strategy 3: More aggressive local search on best found
    if best_points is not None and time.time() < deadline:
        remaining_steps = 300
        pts, score = _hill_climb(best_points, max_steps=remaining_steps)
        if score > best_score:
            best_score = score
            best_points = pts
    
    if best_points is None:
        # Fallback: return a valid grid
        best_points = [(x, y) for x in range(6) for y in range(5)]
    
    return best_points

# EVOLVE-BLOCK-END
