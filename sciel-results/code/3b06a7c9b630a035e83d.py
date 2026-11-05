import time
import random
import math

DEADLINE = 115.0

def construct():
    start_time = time.time()
    random.seed(42)
    
    N = 20
    
    def are_collinear(p1, p2, p3):
        return (p2[0]-p1[0])*(p3[1]-p1[1]) == (p3[0]-p1[0])*(p2[1]-p1[1])
    
    def has_four_collinear(points):
        n = len(points)
        for i in range(n):
            slopes = {}
            xi, yi = points[i]
            for j in range(n):
                if i == j:
                    continue
                xj, yj = points[j]
                dx = xj - xi
                dy = yj - yi
                if dx == 0 and dy == 0:
                    continue
                if dx == 0:
                    slope = (0, 1)
                elif dy == 0:
                    slope = (1, 0)
                else:
                    g = math.gcd(abs(dx), abs(dy))
                    dx, dy = dx // g, dy // g
                    if dx < 0:
                        dx, dy = -dx, -dy
                    slope = (dx, dy)
                
                if slope in slopes:
                    slopes[slope] += 1
                    if slopes[slope] >= 3:
                        return True
                else:
                    slopes[slope] = 1
        return False
    
    def count_triples(points):
        """Count collinear triples. Fast O(N^2 log N)."""
        n = len(points)
        count = 0
        for i in range(n):
            slopes = {}
            xi, yi = points[i]
            for j in range(n):
                if i == j:
                    continue
                xj, yj = points[j]
                dx = xj - xi
                dy = yj - yi
                if dx == 0 and dy == 0:
                    continue
                if dx == 0:
                    slope = (0, 1)
                elif dy == 0:
                    slope = (1, 0)
                else:
                    g = math.gcd(abs(dx), abs(dy))
                    dx, dy = dx // g, dy // g
                    if dx < 0:
                        dx, dy = -dx, -dy
                    slope = (dx, dy)
                
                if slope in slopes:
                    slopes[slope] += 1
                else:
                    slopes[slope] = 1
            
            for k in slopes.values():
                if k >= 2:
                    count += k * (k - 1) // 2
        return count
    
    def exact_score(points):
        """Exact B&B for largest subset with no 3 collinear using bitmasks."""
        n = len(points)
        
        # Precompute line masks for each point
        # lines_from[i] is a list of bitmasks representing other points on lines through i
        lines_from = [[] for _ in range(n)]
        
        for i in range(n):
            slopes_dict = {}
            xi, yi = points[i]
            for j in range(n):
                if i == j:
                    continue
                xj, yj = points[j]
                dx = xj - xi
                dy = yj - yi
                if dx == 0 and dy == 0:
                    continue
                if dx == 0:
                    slope = (0, 1)
                elif dy == 0:
                    slope = (1, 0)
                else:
                    g = math.gcd(abs(dx), abs(dy))
                    dx, dy = dx // g, dy // g
                    if dx < 0:
                        dx, dy = -dx, -dy
                    slope = (dx, dy)
                
                if slope not in slopes_dict:
                    slopes_dict[slope] = 0
                slopes_dict[slope] |= (1 << j)
            
            lines_from[i] = list(slopes_dict.values())
        
        best = [0]
        
        # Order points by number of lines they are on (higher constraint first)
        # Actually, let's just use index order for simplicity in B&B, 
        # but sorting by degree might help pruning.
        # Let's stick to index order to avoid mapping complexity in the bitmask logic 
        # unless we permute the points themselves. Permuting points is safer.
        
        # Calculate degree (number of collinear pairs involving i)
        degrees = [0] * n
        for i in range(n):
            for mask in lines_from[i]:
                # Count bits in mask
                c = bin(mask).count('1')
                if c >= 2:
                    degrees[i] += c * (c - 1) // 2
                elif c == 1:
                    degrees[i] += 0 # No triple yet, but part of a pair
        
        # Sort points by degree descending
        order = sorted(range(n), key=lambda x: -degrees[x])
        
        # Remap points
        new_points = [points[i] for i in order]
        new_lines_from = [lines_from[i] for i in order]
        
        # Update masks in new_lines_from to reflect new indices
        # This is tricky because the masks refer to old indices.
        # It is easier to just permute the points and recompute lines_from.
        # N=20 is small, recomputing is fast.
        
        # Let's recompute lines_from for the permuted points
        permuted_lines_from = [[] for _ in range(n)]
        for i in range(n):
            slopes_dict = {}
            xi, yi = new_points[i]
            for j in range(n):
                if i == j:
                    continue
                xj, yj = new_points[j]
                dx = xj - xi
                dy = yj - yi
                if dx == 0 and dy == 0:
                    continue
                if dx == 0:
                    slope = (0, 1)
                elif dy == 0:
                    slope = (1, 0)
                else:
                    g = math.gcd(abs(dx), abs(dy))
                    dx, dy = dx // g, dy // g
                    if dx < 0:
                        dx, dy = -dx, -dy
                    slope = (dx, dy)
                
                if slope not in slopes_dict:
                    slopes_dict[slope] = 0
                slopes_dict[slope] |= (1 << j)
            
            permuted_lines_from[i] = list(slopes_dict.values())
        
        def backtrack(idx, current_mask, current_size):
            if current_size + (n - idx) <= best[0]:
                return
            if idx == n:
                if current_size > best[0]:
                    best[0] = current_size
                return
            
            # Try excluding idx
            backtrack(idx + 1, current_mask, current_size)
            
            # Try including idx
            # Check if idx can be added
            can_add = True
            for mask in permuted_lines_from[idx]:
                # Count how many points from current_mask are in this line
                # We need to check if there are >= 2 points in current_mask & mask
                intersection = current_mask & mask
                if bin(intersection).count('1') >= 2:
                    can_add = False
                    break
            
            if can_add:
                backtrack(idx + 1, current_mask | (1 << idx), current_size + 1)
        
        backtrack(0, 0, 0)
        return best[0]
    
    def generate_structured(seed_val):
        random.seed(seed_val)
        patterns = []
        
        # Pattern 1: Perturbed Grid
        # 4x5 grid
        base_x = [-40, -13, 13, 40]
        base_y = [-40, -20, 0, 20, 40]
        pts = []
        for x in base_x:
            for y in base_y:
                px = x + random.randint(-3, 3)
                py = y + random.randint(-3, 3)
                pts.append((px, py))
        # Ensure unique
        unique = list(dict.fromkeys(pts))
        while len(unique) < 20:
            unique.append((random.randint(-100, 100), random.randint(-100, 100)))
        patterns.append(unique[:20])
        
        # Pattern 2: Line Packing (5 lines of 4 points? No, max 3 per line)
        # Try to make 4 lines of 5 points? No, 5 points on a line is 4-collinear.
        # So max 3 per line.
        # Try 6 lines of 3 points? 18 points. 2 extra.
        # Let's try a "star" shape or intersecting lines.
        
        # Pattern 3: Random
        pts = []
        used = set()
        while len(pts) < 20:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            if (x, y) not in used:
                used.add((x, y))
                pts.append((x, y))
        patterns.append(pts)
        
        return patterns
    
    best_points = None
    best_score = 21
    best_triples = -1
    
    # Initial Candidates
    candidates = []
    for s in range(10):
        candidates.extend(generate_structured(s))
    
    # Filter and Evaluate Initial Candidates
    for pts in candidates:
        if time.time() - start_time > DEADLINE:
            break
        if not has_four_collinear(pts):
            triples = count_triples(pts)
            score = exact_score(pts)
            if score < best_score or (score == best_score and triples > best_triples):
                best_score = score
                best_triples = triples
                best_points = pts[:]
    
    if best_points is None:
        # Fallback
        while True:
            pts = []
            used = set()
            while len(pts) < 20:
                x = random.randint(-100, 100)
                y = random.randint(-100, 100)
                if (x, y) not in used:
                    used.add((x, y))
                    pts.append((x, y))
            if not has_four_collinear(pts):
                best_points = pts
                best_score = exact_score(pts)
                best_triples = count_triples(pts)
                break
    
    current_points = best_points[:]
    current_triples = best_triples
    current_score = best_score
    
    T = 5.0
    cooling = 0.998
    
    # Local Search
    while time.time() - start_time < DEADLINE:
        idx = random.randint(0, 19)
        old_p = current_points[idx]
        
        if random.random() < 0.7:
            dx = random.randint(-10, 10)
            dy = random.randint(-10, 10)
            x = max(-100, min(100, old_p[0] + dx))
            y = max(-100, min(100, old_p[1] + dy))
        else:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
        
        new_p = (x, y)
        if new_p == old_p or new_p in current_points:
            continue
        
        current_points[idx] = new_p
        
        if has_four_collinear(current_points):
            current_points[idx] = old_p
            continue
        
        new_triples = count_triples(current_points)
        
        if new_triples > current_triples:
            current_triples = new_triples
            # Check exact score if significant improvement or periodically
            if new_triples > best_triples or random.random() < 0.05:
                exact_s = exact_score(current_points)
                if exact_s < best_score or (exact_s == best_score and new_triples > best_triples):
                    best_score = exact_s
                    best_triples = new_triples
                    best_points = current_points[:]
        elif random.random() < math.exp((new_triples - current_triples) / T):
            current_triples = new_triples
        else:
            current_points[idx] = old_p
        
        T *= cooling
        if T < 0.1:
            T = 0.1
    
    return [[p[0], p[1]] for p in best_points]
