import random
import time
from itertools import combinations

def _are_collinear(p1, p2, p3):
    return (p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1])) == 0

def _has_four_collinear(points):
    n = len(points)
    for i, j, k, l in combinations(range(n), 4):
        if _are_collinear(points[i], points[j], points[k]) and _are_collinear(points[i], points[j], points[l]):
            return True
    return False

def _max_no_three_collinear(points):
    n = len(points)
    if n == 0:
        return 0
    
    # Precompute pair_conflicts: for each vertex v, pair_conflicts[v] is a bitmask
    # where bit (j<<k) or rather we need: if v is in independent set, then no pair (j,k)
    # with j,k both in independent set can be collinear with v.
    # Better: conflict_pairs[v] = bitmask of vertices u such that there exists w with v,u,w collinear.
    # Actually, let's use: for each v, forbidden_pairs[v] = set of (j,k) pairs.
    # For O(1) check: precompute for each v, a dict from j to bitmask of k's.
    
    # Simpler: conflicts[v] = list of (j, k) pairs where v,j,k are collinear.
    # For checking if we can add v to mask: for each (j,k) in conflicts[v], check if both j,k in mask.
    
    # Even faster: pair_mask[v] = for each other vertex u, if u is in mask, does that create conflict?
    # Let's use: adj[v] = dict mapping u -> bitmask of w such that v,u,w collinear.
    # Then can_add(v, mask): for each u in mask, if u in adj[v], check if adj[v][u] & mask == adj[v][u]... 
    # No, that's not right either.
    
    # Let me think differently. 
    # conflicts[v] = list of (j, k) where {v,j,k} is a collinear triple.
    # can_add(v, mask): for each (j,k) in conflicts[v], if (mask has j) and (mask has k), return False.
    # This is O(|conflicts[v]|) per check.
    
    conflicts = [[] for _ in range(n)]
    
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if _are_collinear(points[i], points[j], points[k]):
                    conflicts[i].append((j, k))
                    conflicts[j].append((i, k))
                    conflicts[k].append((i, j))
    
    # Degree for ordering
    degrees = [len(conflicts[i]) for i in range(n)]
    order = sorted(range(n), key=lambda i: -degrees[i])
    
    best_is = [0]
    
    def can_add(v, mask):
        for (j, k) in conflicts[v]:
            if (mask >> j) & 1 and (mask >> k) & 1:
                return False
        return True
    
    def bnb_is(mask, remaining, current_size):
        if current_size > best_is[0]:
            best_is[0] = current_size
            
        if remaining == 0:
            return
            
        # Upper bound: current_size + popcount(remaining)
        if current_size + bin(remaining).count('1') <= best_is[0]:
            return
            
        # Pick vertex with highest degree in remaining
        # Find best vertex from 'order' that is in remaining
        p = -1
        p_bit = 0
        for v in order:
            if (remaining >> v) & 1:
                p = v
                p_bit = 1 << v
                break
        
        if p == -1:
            return
            
        # Option 1: Add p if possible
        if can_add(p, mask):
            bnb_is(mask | p_bit, remaining ^ p_bit, current_size + 1)
            
        # Option 2: Don't add p
        bnb_is(mask, remaining ^ p_bit, current_size)
        
    bnb_is(0, (1 << n) - 1, 0)
    return best_is[0]

def _generate_initial_points():
    """
    Generate a set of 20 points with many collinear triples.
    Use a structured approach: place points on a few lines with small perturbations.
    """
    points = []
    
    # Strategy: create several lines, place 3-4 points on each, with overlaps
    # Line 1: y = 0, x = -20, -10, 0
    # Line 2: y = 10, x = -15, -5, 5
    # Line 3: y = -10, x = -10, 0, 10
    # Line 4: x = 0, y = -20, -10, 0, 10, 20 (but 0 is already used)
    
    # Let's use a base grid and deliberately create collinearities
    base_points = [
        [-20, 0], [-10, 0], [0, 0],       # y=0 line
        [-15, 10], [-5, 10], [5, 10],     # y=10 line
        [-10, -10], [0, -10], [10, -10],  # y=-10 line
        [0, -20], [0, 20],                # x=0 line (partial)
        [-20, 10], [20, -10],             # diagonal
        [-20, -10], [20, 10],             # other diagonal
        [10, 0], [-10, 10],               # more collinear potential
        [15, -5], [-15, 5],               # more
        [5, -15], [-5, 15],               # more
        [10, 10], [-10, -10],             # diagonal
        [0, 15],                          # extra
    ]
    
    # Perturb slightly to avoid 4 collinear
    for _ in range(50):
        pts = []
        for p in base_points:
            dx = random.randint(-2, 2)
            dy = random.randint(-2, 2)
            np = [p[0] + dx, p[1] + dy]
            if -100 <= np[0] <= 100 and -100 <= np[1] <= 100 and np not in pts:
                pts.append(np)
            else:
                # Find a nearby unused point
                for _ in range(10):
                    nx = p[0] + random.randint(-5, 5)
                    ny = p[1] + random.randint(-5, 5)
                    if -100 <= nx <= 100 and -100 <= ny <= 100 and [nx, ny] not in pts:
                        pts.append([nx, ny])
                        break
                else:
                    while True:
                        nx = random.randint(-100, 100)
                        ny = random.randint(-100, 100)
                        if [nx, ny] not in pts:
                            pts.append([nx, ny])
                            break
        
        if len(pts) == 20 and not _has_four_collinear(pts):
            return pts
    
    # Fallback: random
    pts = []
    for _ in range(20):
        while True:
            p = [random.randint(-100, 100), random.randint(-100, 100)]
            if p not in pts:
                break
        pts.append(p)
    
    attempts = 0
    while _has_four_collinear(pts) and attempts < 100:
        idx = random.randint(0, 19)
        pts[idx] = [random.randint(-100, 100), random.randint(-100, 100)]
        attempts += 1
    
    return pts

def _local_search(points, deadline):
    current_points = points[:]
    current_score = _max_no_three_collinear(current_points)
    best_points = current_points[:]
    best_score = current_score
    
    while time.time() < deadline:
        # Try to improve by moving a point to create a new collinear triple
        idx = random.randint(0, 19)
        old_p = current_points[idx]
        
        candidates = []
        
        # Strategy 1: Place on line through two other points
        i = random.randint(0, 19)
        if i == idx: i = (i+1) % 20
        j = random.randint(0, 19)
        if j == idx or j == i: j = (j+1) % 20
        
        pi = current_points[i]
        pj = current_points[j]
        
        dx = pj[0] - pi[0]
        dy = pj[1] - pi[1]
        
        # Try various k values to get integer points on the line
        for k in range(-8, 9):
            if k == 0 or k == 1:
                continue
            nx = pi[0] + k * dx
            ny = pi[1] + k * dy
            if -100 <= nx <= 100 and -100 <= ny <= 100:
                candidates.append([nx, ny])
        
        # Strategy 2: Random nearby point
        for _ in range(3):
            nx = old_p[0] + random.randint(-10, 10)
            ny = old_p[1] + random.randint(-10, 10)
            if -100 <= nx <= 100 and -100 <= ny <= 100:
                candidates.append([nx, ny])
        
        # Strategy 3: Fully random
        candidates.append([random.randint(-100, 100), random.randint(-100, 100)])
        
        random.shuffle(candidates)
        
        improved = False
        for new_p in candidates:
            if new_p == old_p or new_p in current_points:
                continue
                
            new_points = current_points[:]
            new_points[idx] = new_p
            
            if _has_four_collinear(new_points):
                continue
                
            new_score = _max_no_three_collinear(new_points)
            
            if new_score < current_score:
                current_score = new_score
                current_points = new_points
                if current_score < best_score:
                    best_score = current_score
                    best_points = current_points[:]
                improved = True
                # Continue looking for more improvements in this iteration
                # But break to save time if we've found a good improvement
                if current_score <= 12:  # Good enough, stop
                    break
        
        if not improved and random.random() < 0.1:
            # Occasionally try a bigger move
            idx = random.randint(0, 19)
            current_points[idx] = [random.randint(-100, 100), random.randint(-100, 100)]
            if not _has_four_collinear(current_points):
                current_score = _max_no_three_collinear(current_points)
                if current_score < best_score:
                    best_score = current_score
                    best_points = current_points[:]

    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    best_points = None
    best_score = float('inf')
    
    # Try multiple initial configurations
    for attempt in range(8):
        if time.time() > deadline - 15:
            break
            
        init_points = _generate_initial_points()
        
        if _has_four_collinear(init_points):
            continue
            
        pts, score = _local_search(init_points, deadline)
        
        if score < best_score:
            best_score = score
            best_points = pts
    
    if best_points is None:
        # Fallback
        while True:
            pts = []
            for _ in range(20):
                while True:
                    p = [random.randint(-100, 100), random.randint(-100, 100)]
                    if p not in pts:
                        break
                pts.append(p)
            if not _has_four_collinear(pts):
                return pts
                
    return best_points
