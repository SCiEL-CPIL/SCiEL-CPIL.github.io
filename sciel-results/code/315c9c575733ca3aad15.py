import time
import itertools
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Start with current best: 6 lines of 3 + 2 isolated = score 14
    base_pts = [
        [10, 0], [11, 0], [12, 0],
        [-12, 2], [-11, 2], [-10, 2],
        [10, 4], [11, 4], [12, 4],
        [-12, 6], [-11, 6], [-10, 6],
        [10, 8], [11, 8], [12, 8],
        [-12, 10], [-11, 10], [-10, 10]
    ]
    base_pts.append([50, 50])
    base_pts.append([-50, -50])
    
    def collinear(p1, p2, p3):
        return (p2[0]-p1[0])*(p3[1]-p1[1]) == (p3[0]-p1[0])*(p2[1]-p1[1])
    
    def has_four_collinear(points):
        for comb in itertools.combinations(points, 4):
            if collinear(comb[0], comb[1], comb[2]):
                if collinear(comb[0], comb[1], comb[3]):
                    return True
        return False
    
    def score(points):
        """Compute largest subset with no 3 collinear using branch and bound."""
        n = len(points)
        # Precompute collinearity
        col = [[False]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if collinear(points[i], points[j], points[k]):
                        col[i][j] = col[j][i] = True
                        col[j][k] = col[k][j] = True
                        col[i][k] = col[k][i] = True
        
        # Branch and bound to find max independent set in 3-uniform hypergraph
        # where hyperedges are collinear triples
        # This is NP-hard, so we use a greedy + local search approach
        
        # Greedy: repeatedly pick point that is in fewest collinear triples with remaining
        best = 0
        remaining = set(range(n))
        
        # Try multiple random orderings
        for attempt in range(50):
            if time.time() > deadline:
                break
            order = list(range(n))
            random.shuffle(order)
            selected = []
            for i in order:
                # Check if adding i creates a collinear triple with any two already selected
                can_add = True
                for a in selected:
                    for b in selected:
                        if a < b and collinear(points[a], points[b], points[i]):
                            can_add = False
                            break
                    if not can_add:
                        break
                if can_add:
                    selected.append(i)
            if len(selected) > best:
                best = len(selected)
        
        return best
    
    # Verify base is valid
    if has_four_collinear(base_pts):
        # Fallback
        return [[x, x*x] for x in range(-10, 10)]
    
    base_score = score(base_pts)
    best_pts = base_pts[:]
    best_s = base_score
    
    # Try to improve by local search: perturb points
    # Strategy: try replacing isolated points or moving points to create more 3-collinearities
    
    # Generate candidate configurations to try
    candidates = []
    
    # Try different line arrangements
    # 5 lines of 4 points is invalid (4 collinear)
    # Try 7 lines of 3 + 1 isolated = 22, too many
    # Try 6 lines of 3 + 2 isolated (current)
    # Try configurations with overlapping 3-point lines
    
    # Let's try a systematic search over small perturbations
    # For each point, try moving it to nearby integer positions
    
    def try_improve(points):
        nonlocal best_pts, best_s
        if time.time() > deadline:
            return
        s = score(points)
        if s < best_s:
            best_s = s
            best_pts = points[:]
    
    # Local search: try moving each point to a nearby position
    for iteration in range(500):
        if time.time() > deadline:
            break
        
        # Pick a random point to move
        idx = random.randint(0, 19)
        old_x, old_y = best_pts[idx]
        
        # Try nearby positions
        moved = False
        for dx in range(-5, 6):
            for dy in range(-5, 6):
                if dx == 0 and dy == 0:
                    continue
                new_x = old_x + dx
                new_y = old_y + dy
                if abs(new_x) > 100 or abs(new_y) > 100:
                    continue
                
                new_pts = best_pts[:]
                new_pts[idx] = [new_x, new_y]
                
                # Check validity
                if len(set(map(tuple, new_pts))) != 20:
                    continue
                if has_four_collinear(new_pts):
                    continue
                
                s = score(new_pts)
                if s < best_s:
                    best_s = s
                    best_pts = new_pts
                    moved = True
                    break
            if moved or time.time() > deadline:
                break
        
        # If no improvement, try a larger jump
        if not moved and random.random() < 0.1:
            idx = random.randint(0, 19)
            new_x = random.randint(-100, 100)
            new_y = random.randint(-100, 100)
            if abs(new_x) <= 100 and abs(new_y) <= 100:
                new_pts = best_pts[:]
                new_pts[idx] = [new_x, new_y]
                if len(set(map(tuple, new_pts))) == 20 and not has_four_collinear(new_pts):
                    s = score(new_pts)
                    if s < best_s:
                        best_s = s
                        best_pts = new_pts
    
    return best_pts
