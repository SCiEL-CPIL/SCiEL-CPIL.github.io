import time
from collections import Counter
import itertools
import random
import numpy as np

def _doubled_area(p1, p2, p3):
    """Compute 2x area (absolute determinant) of triangle p1, p2, p3."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    return abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))

def _score(points):
    """Compute the max multiplicity of any single positive doubled area."""
    areas = []
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                a = _doubled_area(points[i], points[j], points[k])
                if a > 0:
                    areas.append(a)
    if not areas:
        return 0
    return Counter(areas).most_common(1)[0][1]

def _score_fast(points):
    """Faster scoring using numpy for the area computation."""
    n = len(points)
    pts = np.array(points, dtype=np.int64)
    max_mult = 0
    # For 30 points, C(30,3) = 4060, which is small enough
    areas = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                a = abs(pts[i,0]*(pts[j,1]-pts[k,1]) + pts[j,0]*(pts[k,1]-pts[i,1]) + pts[k,0]*(pts[i,1]-pts[j,1]))
                if a > 0:
                    areas.append(a)
    if not areas:
        return 0
    return Counter(areas).most_common(1)[0][1]

def construct():
    start_time = time.time()
    deadline = start_time + 75  # Safety margin
    
    # Initial good seed: a grid-like structure
    # The seed gives 6x5=30 points in [0,5]x[0,4]
    # Let's try to find something better
    
    # Strategy: Use local search (swap-based) to improve a configuration
    # Start with a few different initial configurations
    
    best_points = None
    best_score = 0
    
    # Candidate initial configurations
    inits = []
    
    # 1) The original seed
    inits.append([(x, y) for x in range(6) for y in range(5)])
    
    # 2) A 5x6 grid (transposed)
    inits.append([(x, y) for x in range(5) for y in range(6)])
    
    # 3) A more spread out grid
    inits.append([(x, y) for x in range(6) for y in range(5)])  # same as 1
    
    # 4) Points on a line with perturbations
    inits.append([(i, 0) for i in range(15)] + [(i, 1) for i in range(15)])
    
    # 5) A 10x3 grid
    inits.append([(x, y) for x in range(10) for y in range(3)])
    
    # 6) A 3x10 grid
    inits.append([(x, y) for x in range(3) for y in range(10)])
    
    # 7) Points forming a parallelogram-like structure
    # Take 6x5 grid but shift some rows
    pts = [(x, y) for x in range(6) for y in range(5)]
    inits.append(pts[:])
    
    # 8) Try a configuration where many points share the same x or y
    inits.append([(0, y) for y in range(10)] + [(1, y) for y in range(10)] + [(2, y) for y in range(10)])
    
    # 9) Try clustered points
    inits.append([(x, y) for x in range(5) for y in range(6)])
    
    # 10) Random initial
    random.seed(42)
    pts = set()
    while len(pts) < 30:
        pts.add((random.randint(-30, 30), random.randint(-30, 30)))
    inits.append(list(pts))
    
    # Score all initial configurations
    for init in inits:
        if len(init) != 30:
            continue
        if len(set(init)) != 30:
            continue
        if any(abs(x) > 30 or abs(y) > 30 for x, y in init):
            continue
        s = _score_fast(init)
        if s > best_score:
            best_score = s
            best_points = init[:]
    
    # Local search: try swapping points
    random.seed(123)
    if best_points is None:
        best_points = [(x, y) for x in range(6) for y in range(5)]
        best_score = _score_fast(best_points)
    
    current = best_points[:]
    current_score = best_score
    
    # Precompute area matrix for current configuration
    n = len(current)
    
    def compute_areas(points):
        """Return list of all positive doubled areas."""
        areas = []
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    a = _doubled_area(points[i], points[j], points[k])
                    if a > 0:
                        areas.append(a)
        return areas
    
    # Local search with swaps
    no_improve = 0
    max_no_improve = 200
    
    while time.time() < deadline and no_improve < max_no_improve:
        # Pick a random point to move
        idx = random.randint(0, n-1)
        old_x, old_y = current[idx]
        
        # Try several new positions
        found_improvement = False
        for _ in range(20):
            new_x = random.randint(-30, 30)
            new_y = random.randint(-30, 30)
            if (new_x, new_y) == (old_x, old_y):
                continue
            # Check if (new_x, new_y) is already used
            if (new_x, new_y) in current:
                continue
            
            new_current = current[:]
            new_current[idx] = (new_x, new_y)
            s = _score_fast(new_current)
            
            if s > current_score:
                current = new_current
                current_score = s
                if s > best_score:
                    best_score = s
                    best_points = current[:]
                found_improvement = True
                break
            elif s == current_score and random.random() < 0.1:
                # Accept equal to escape local optima
                current = new_current
                found_improvement = True
                break
        
        if found_improvement:
            no_improve = 0
        else:
            no_improve += 1
    
    # Final polish: try all single-point moves from best configuration
    if best_points is not None and time.time() < deadline:
        final = best_points[:]
        final_score = best_score
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for idx in range(n):
                if time.time() > deadline:
                    break
                old_x, old_y = final[idx]
                for dx in range(-5, 6):
                    for dy in range(-5, 6):
                        if time.time() > deadline:
                            break
                        new_x = old_x + dx
                        new_y = old_y + dy
                        if new_x < -30 or new_x > 30 or new_y < -30 or new_y > 30:
                            continue
                        if (new_x, new_y) == (old_x, old_y):
                            continue
                        if (new_x, new_y) in final:
                            continue
                        new_final = final[:]
                        new_final[idx] = (new_x, new_y)
                        s = _score_fast(new_final)
                        if s > final_score:
                            final = new_final
                            final_score = s
                            improved = True
    
    if final_score > best_score:
        best_points = final
        best_score = final_score
    
    # Verify the result
    if best_points is None:
        return [(x, y) for x in range(6) for y in range(5)]
    
    return best_points
