import time
import random
from itertools import combinations

def _collinear3(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    return (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1)

def _has_four_collinear(points):
    n = len(points)
    # Map lines to points
    line_map = {}
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            # Normalize direction
            if dx == 0:
                dy = 1
            elif dy == 0:
                dx = 1
            else:
                g = abs(dx)
                while g > 0 and dy % g == 0:
                    g = abs(g // (dy // g)) if dy != 0 else abs(dx)
                    break
                # Simple gcd
                def gcd(a, b):
                    while b:
                        a, b = b, a % b
                    return a
                g = gcd(abs(dx), abs(dy))
                dx //= g
                dy //= g
                if dx < 0:
                    dx = -dx
                    dy = -dy
            # Calculate offset (c in ax+by=c)
            # dx*y - dy*x = c
            c = dx * y1 - dy * x1
            key = (dx, dy, c)
            if key not in line_map:
                line_map[key] = set()
            line_map[key].add(i)
            line_map[key].add(j)
            
    for key, pts in line_map.items():
        if len(pts) >= 4:
            return True
    return False

def _max_3free_subset(points, deadline=None):
    n = len(points)
    # Precompute collinear masks
    # collinear_mask[i][j] is bitmask of k such that i, j, k collinear
    collinear_mask = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j: continue
            mask = 0
            for k in range(n):
                if i == k or j == k: continue
                if _collinear3(points[i], points[j], points[k]):
                    mask |= (1 << k)
            collinear_mask[i][j] = mask

    best = [0]
    
    def can_add(i, current_mask):
        # Check if adding i to current_mask creates a collinear triple
        # For any j in current_mask, if there exists k in current_mask such that i, j, k collinear
        # This is equivalent to: for any j in current_mask, (collinear_mask[i][j] & current_mask) != 0
        # Note: collinear_mask[i][j] contains k. If k is in current_mask, and j is in current_mask, then i, j, k are collinear.
        temp = current_mask
        while temp:
            lsb = temp & -temp
            j = lsb.bit_length() - 1
            temp -= lsb
            if (collinear_mask[i][j] & current_mask) != 0:
                return False
        return True

    def dfs(current_mask, candidates_mask, current_size):
        if deadline and time.time() > deadline:
            return
        if current_size > best[0]:
            best[0] = current_size
            
        if candidates_mask == 0:
            return
            
        # Pruning
        # Upper bound: current_size + popcount(candidates_mask)
        if current_size + bin(candidates_mask).count('1') <= best[0]:
            return
            
        # Heuristic: Pick a point from candidates
        # Try to pick a point that is "hard" to add or central?
        # Just pick the first one for simplicity, or random?
        # Let's pick the lowest index bit for determinism and speed
        lsb = candidates_mask & -candidates_mask
        i = lsb.bit_length() - 1
        
        # Option 1: Exclude i
        dfs(current_mask, candidates_mask & ~lsb, current_size)
        
        # Option 2: Include i
        if can_add(i, current_mask):
            new_current = current_mask | lsb
            # New candidates: remove i. 
            # Do we need to remove other points? 
            # No, because the constraint is local to the triple. 
            # However, if we add i, some points in candidates might become "un-addable" in the future 
            # if they form a triple with i and some other point added later. 
            # But the B&B handles this by checking can_add at each step.
            dfs(new_current, candidates_mask & ~lsb, current_size + 1)

    dfs(0, (1 << n) - 1, 0)
    return best[0]

def _random_point():
    return [random.randint(-100, 100), random.randint(-100, 100)]

def _construct_initial():
    # Start with a random set, ensure validity
    points = []
    seen = set()
    while len(points) < 20:
        p = _random_point()
        if tuple(p) in seen: continue
        # Check 4 collinear with existing
        points.append(p)
        if _has_four_collinear(points):
            points.pop()
            continue
        seen.add(tuple(p))
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    # Initial best
    best_points = None
    best_score = 20
    
    # Try a few structured starts
    starts = []
    
    # 1. Random
    starts.append(_construct_initial())
    
    # 2. Grid-like with noise
    pts = []
    for i in range(5):
        for j in range(4):
            x = i * 20 + random.randint(-5, 5)
            y = j * 20 + random.randint(-5, 5)
            pts.append([x, y])
    # Validate and fix if needed
    if len(set(tuple(p) for p in pts)) == 20 and not _has_four_collinear(pts):
        starts.append(pts)

    # Evaluate starts
    for pts in starts:
        if time.time() > deadline - 5: break
        score = _max_3free_subset(pts, deadline - 2)
        if score < best_score:
            best_score = score
            best_points = [p[:] for p in pts]
            
    # Simulated Annealing
    current_points = best_points if best_points else starts[0]
    current_score = best_score
    
    T = 1.0
    T_min = 0.01
    alpha = 0.995
    
    while time.time() < deadline - 5:
        # Pick a random point to move
        idx = random.randint(0, 19)
        orig = current_points[idx][:]
        
        # Random new position
        new_p = _random_point()
        current_points[idx] = new_p
        
        # Check validity
        valid = True
        if len(set(tuple(p) for p in current_points)) != 20:
            valid = False
        elif _has_four_collinear(current_points):
            valid = False
            
        if not valid:
            current_points[idx] = orig
            T *= alpha
            continue
            
        # Evaluate
        new_score = _max_3free_subset(current_points, deadline - 2)
        
        # Acceptance
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_points = [p[:] for p in current_points]
        else:
            # Accept with probability exp(-delta / T)
            delta = new_score - current_score
            if delta > 0:
                prob = random.random()
                # Approximation for small delta
                if prob < 0.1: # Simplified acceptance
                    pass # Reject
                else:
                    current_points[idx] = orig
            else:
                current_points[idx] = orig
                
        T *= alpha
        if T < T_min:
            T = T_min
            
    return best_points
