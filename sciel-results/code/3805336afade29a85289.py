import random
import itertools
import numpy as np
import time
import math

def _check_no_four_collinear(points):
    """Check no four points are collinear."""
    n = len(points)
    # Use a set of lines represented by (slope, intercept) or normalized vectors
    # But for small N, triple check is fine.
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                xi, yi = points[i]
                xj, yj = points[j]
                xk, yk = points[k]
                a = yj - yi
                b = xi - xj
                c = xj*yi - xi*yj
                for l in range(n):
                    if l in (i, j, k):
                        continue
                    xl, yl = points[l]
                    if a * xl + b * yl + c == 0:
                        return False
    return True

def _score(points):
    """Largest subset with no three collinear (exact branch and bound)."""
    n = len(points)
    if n == 0:
        return 0
        
    # Precompute collinearity masks
    # line_mask[i][j] is a bitmask of points k (k != i, j) such that i, j, k are collinear
    line_mask = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            xi, yi = points[i]
            xj, yj = points[j]
            mask = 0
            for k in range(n):
                if k == i or k == j:
                    continue
                xk, yk = points[k]
                if (xj - xi) * (yk - yi) == (xk - xi) * (yj - yi):
                    mask |= (1 << k)
            line_mask[i][j] = mask
            line_mask[j][i] = mask

    best = [0]
    
    def dfs(idx, current_mask, current_size):
        if current_size + (n - idx) <= best[0]:
            return
        if idx == n:
            best[0] = max(best[0], current_size)
            return
        
        # Option 1: Skip idx
        dfs(idx + 1, current_mask, current_size)
        
        # Option 2: Take idx
        can_take = True
        m = current_mask
        while m:
            lsb = m & -m
            a_idx = lsb.bit_length() - 1
            if line_mask[a_idx][idx] & current_mask:
                can_take = False
                break
            m ^= lsb
            
        if can_take:
            dfs(idx + 1, current_mask | (1 << idx), current_size + 1)

    dfs(0, 0, 0)
    return best[0]

def _count_triples(points):
    """Count number of 3-collinear triples."""
    n = len(points)
    count = 0
    for i in range(n):
        for j in range(i+1, n):
            xi, yi = points[i]
            xj, yj = points[j]
            for k in range(j+1, n):
                xk, yk = points[k]
                if (xj - xi) * (yk - yi) == (xk - xi) * (yj - yi):
                    count += 1
    return count

def construct():
    # Greedy Construction
    # Candidate pool: Points in a reasonable range
    # We want to maximize triples, so we should pick points that lie on lines with many other points.
    # A good strategy is to pick points from a set of lines.
    
    # Let's try a structured approach:
    # Pick a few lines, and put points on them.
    # But we must avoid 4-collinear. So max 3 points per line.
    
    # Strategy:
    # 1. Generate a set of candidate points.
    # 2. Greedily select 20 points.
    #    At each step, for each candidate not yet selected:
    #       - Check if adding it creates a 4-collinear set with existing points.
    #       - If valid, count how many new 3-collinear triples it forms.
    #       - Pick the one with the highest count.
    #       - Tie-break randomly or by position.
    
    # To make this work, we need a good candidate pool.
    # A pool of ~100-200 points.
    
    random.seed(42)
    
    # Generate candidate pool
    # Use a grid but with some noise to allow flexibility
    candidates = []
    for x in range(-20, 21, 2):
        for y in range(-20, 21, 2):
            candidates.append((x, y))
            
    # Shuffle candidates to break symmetry
    random.shuffle(candidates)
    
    points = []
    used = set()
    
    for _ in range(20):
        best_candidate = None
        best_score = -1
        best_triples = -1
        
        for c in candidates:
            if c in used:
                continue
                
            new_points = points + [c]
            
            # Check 4-collinear
            if not _check_no_four_collinear(new_points):
                continue
            
            # Count new triples
            # Triples involving c and two points in 'points'
            # Plus triples among 'points' (unchanged)
            # We only need to count new ones to compare
            new_triples = 0
            n = len(points)
            for i in range(n):
                for j in range(i+1, n):
                    xi, yi = points[i]
                    xj, yj = points[j]
                    xc, yc = c
                    if (xj - xi) * (yc - yi) == (xc - xi) * (yj - yi):
                        new_triples += 1
            
            if new_triples > best_triples:
                best_triples = new_triples
                best_candidate = c
                best_score = new_triples # Just use triples as score for selection
                
        if best_candidate is None:
            # Fallback: pick any valid point
            for c in candidates:
                if c in used:
                    continue
                new_points = points + [c]
                if _check_no_four_collinear(new_points):
                    best_candidate = c
                    break
            
        if best_candidate is None:
            # Should not happen with large pool
            return [[x, x*x] for x in range(-10, 10)]
            
        points.append(best_candidate)
        used.add(best_candidate)
        
    current_score = _score(points)
    best_points = [p[:] for p in points]
    best_score = current_score
    
    # Local Search: Hill Climbing
    start_time = time.time()
    time_limit = 110.0
    
    # Define neighborhood: move a point to a nearby location
    # Or swap with a candidate from the pool
    
    while time.time() - start_time < time_limit:
        improved = False
        for i in range(20):
            # Try to improve point i
            old_point = points[i]
            # Try nearby points
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if dx == 0 and dy == 0:
                        continue
                    new_x = old_point[0] + dx
                    new_y = old_point[1] + dy
                    if abs(new_x) > 100 or abs(new_y) > 100:
                        continue
                    
                    new_point = (new_x, new_y)
                    if new_point in used:
                        # Check if it's the same point
                        if new_point == old_point:
                            continue
                        # If it's another point, we are swapping? No, just moving.
                        # If new_point is in used, it means it's occupied by another point.
                        # We are moving i to new_point. If new_point is occupied by j, 
                        # we would need to swap or move j. Let's just skip for simplicity in this step.
                        # Or we can check if new_point is old_point (already handled).
                        # Actually, 'used' contains all current points.
                        # If new_point is in used, it's a collision.
                        continue
                    
                    new_points = points[:]
                    new_points[i] = new_point
                    
                    if not _check_no_four_collinear(new_points):
                        continue
                        
                    new_score = _score(new_points)
                    
                    if new_score < best_score:
                        points = new_points
                        best_score = new_score
                        best_points = [p[:] for p in points]
                        used.discard(old_point)
                        used.add(new_point)
                        improved = True
                        break
                    elif new_score == best_score:
                        # Tie-break: prefer more triples
                        old_triples = _count_triples(points)
                        new_triples = _count_triples(new_points)
                        if new_triples > old_triples:
                            points = new_points
                            best_points = [p[:] for p in points]
                            used.discard(old_point)
                            used.add(new_point)
                            improved = True
                            break
                            
            if improved:
                break
                
        if not improved:
            # Try larger moves or random restarts?
            # Let's try swapping two points
            for i in range(20):
                for j in range(i+1, 20):
                    # Try moving i to j's position and j to i's position?
                    # Or just move i to a random valid spot?
                    pass # Keep it simple for now
                    
    return best_points
