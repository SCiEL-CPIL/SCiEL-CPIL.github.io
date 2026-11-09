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
    
    # Precompute hyperedges (collinear triples)
    # hyperedges[i] is a set of frozensets({j, k}) or bitmasks
    # Let's use bitmasks for speed
    # conflicts[i] = list of bitmasks representing pairs {j, k} such that i, j, k are collinear
    conflicts = [[] for _ in range(n)]
    
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if _are_collinear(points[i], points[j], points[k]):
                    mask_jk = (1 << j) | (1 << k)
                    mask_ik = (1 << i) | (1 << k)
                    mask_ij = (1 << i) | (1 << j)
                    
                    conflicts[i].append(mask_jk)
                    conflicts[j].append(mask_ik)
                    conflicts[k].append(mask_ij)
    
    # We want to find the maximum subset with no three collinear.
    # This is equivalent to finding the minimum vertex cover in the 3-uniform hypergraph of collinear triples.
    # Score = n - min_vertex_cover_size.
    # Let's solve for min_vertex_cover_size using BnB.
    
    best_cover = [n]
    
    def can_add_to_independent_set(p, mask):
        # mask is the set of points in the independent set
        # Check if adding p violates any hyperedge
        for pair_mask in conflicts[p]:
            if (pair_mask & mask) == pair_mask:
                return False
        return True

    def bnb_cover(cover_mask, remaining_mask, current_size):
        # cover_mask: points included in the cover
        # remaining_mask: points not yet decided
        # current_size: size of cover so far
        
        if current_size >= best_cover[0]:
            return
            
        if remaining_mask == 0:
            best_cover[0] = current_size
            return
            
        # Lower bound for cover size:
        # We need to cover all hyperedges that have vertices in remaining_mask.
        # A simple lower bound: find a set of disjoint hyperedges in the remaining graph.
        # Each such hyperedge requires at least 1 vertex in the cover.
        # Greedy matching lower bound:
        # 1. Get all hyperedges fully contained in remaining_mask.
        # 2. Greedily pick hyperedges that are disjoint from each other.
        # 3. Count = number of such hyperedges.
        
        # To speed up, we can just count the number of points remaining if it's small, 
        # or use a simpler bound.
        # Actually, a better bound: 
        # Let R be remaining points.
        # If there is a hyperedge (u,v,w) in R, we must pick at least one of u,v,w.
        # Greedy set packing:
        # Collect all triples in R.
        # Sort by something? Just iterate.
        
        # Optimization: Precompute all triples in the whole graph.
        # But we need triples in R.
        
        # Let's use a simpler bound first: 
        # If we have many points left, the bound is weak.
        # However, for n=20, we can afford to be a bit careful.
        
        # Let's try a greedy matching bound.
        # 1. Find all triples (i,j,k) such that i,j,k are in remaining_mask and collinear.
        # 2. Greedily select disjoint triples.
        
        # To avoid generating triples every time, we can use the conflicts structure.
        # But conflicts are per vertex.
        
        # Let's just use the simple bound: 
        # min_cover >= ceil( number_of_uncovered_hyperedges / 3 ) ? No.
        
        # Let's stick to the independent set formulation which might be easier to prune?
        # Max Independent Set in 3-uniform hypergraph.
        # BnB on Independent Set:
        # State: mask (selected), remaining (candidates)
        # Upper bound: current_size + popcount(remaining)
        
        # Let's switch to BnB for Maximum Independent Set (MIS).
        # Score = MIS size.
        
        pass # This function is replaced below

    # Re-implementing BnB for Maximum Independent Set with better pruning
    
    best_is = [0]
    
    def bnb_is(mask, remaining, current_size):
        if current_size > best_is[0]:
            best_is[0] = current_size
            
        if remaining == 0:
            return
            
        # Upper bound
        if current_size + bin(remaining).count('1') <= best_is[0]:
            return
            
        # Pick a vertex to branch on
        # Heuristic: pick vertex with highest degree in the remaining subgraph?
        # Or just lowest bit.
        
        # Let's pick the vertex that is part of the most hyperedges in 'remaining'
        # This helps prune.
        
        # Count conflicts for each vertex in 'remaining'
        # conflicts[v] contains pairs. We need pairs where both are in 'remaining'.
        
        # This is expensive to compute every node.
        # Let's just pick the lowest set bit for now, but maybe order by index?
        
        p_bit = remaining & -remaining
        p = p_bit.bit_length() - 1
        
        # Option 1: Add p to Independent Set
        if can_add_to_independent_set(p, mask):
            bnb_is(mask | p_bit, remaining ^ p_bit, current_size + 1)
            
        # Option 2: Don't add p
        # We can also remove p from remaining, and potentially remove its neighbors?
        # No, in hypergraph, it's not that simple.
        # But if we don't add p, we just proceed.
        bnb_is(mask, remaining ^ p_bit, current_size)
        
    bnb_is(0, (1 << n) - 1, 0)
    return best_is[0]

def _generate_initial_points():
    """
    Generate a set of 20 points that is likely to have many collinear triples.
    """
    points = []
    base_x = [-20, -10, 0, 10, 20]
    base_y = [-15, -5, 5, 15]
    
    candidates = []
    for x in base_x:
        for y in base_y:
            candidates.append((x, y))
            
    pts = []
    for x, y in candidates:
        dx = random.randint(-2, 2)
        dy = random.randint(-2, 2)
        pts.append([x + dx, y + dy])
        
    valid_pts = []
    for p in pts:
        if -100 <= p[0] <= 100 and -100 <= p[1] <= 100:
            if p not in valid_pts:
                valid_pts.append(p)
            else:
                while True:
                    np = [p[0] + random.randint(-5, 5), p[1] + random.randint(-5, 5)]
                    if -100 <= np[0] <= 100 and -100 <= np[1] <= 100 and np not in valid_pts:
                        valid_pts.append(np)
                        break
                        
    while len(valid_pts) < 20:
        x = random.randint(-100, 100)
        y = random.randint(-100, 100)
        if [x, y] not in valid_pts:
            valid_pts.append([x, y])
            
    return valid_pts[:20]

def _local_search(points, deadline):
    current_points = points[:]
    current_score = _max_no_three_collinear(current_points)
    best_points = current_points[:]
    best_score = current_score
    
    # Precompute lines for faster checking?
    # Or just use the existing functions.
    
    while time.time() < deadline:
        # Strategy: Try to create a new collinear triple by moving a point.
        # Pick a random point to move.
        idx = random.randint(0, 19)
        old_p = current_points[idx]
        
        # Generate candidate new positions.
        # 1. Random position
        # 2. Position that forms a line with two other points.
        
        candidates = []
        
        # Random
        candidates.append([random.randint(-100, 100), random.randint(-100, 100)])
        
        # Line-based: pick two other points i, j, and calculate intersection or point on line?
        # We want integer points.
        # Pick i, j != idx.
        # If we move idx to a point P such that P, Pi, Pj are collinear, we create a triple.
        # P = Pi + k * (Pj - Pi).
        # We need P to be integer and within bounds and not equal to existing points.
        
        i = random.randint(0, 19)
        if i == idx: i = (i+1) % 20
        j = random.randint(0, 19)
        if j == idx or j == i: j = (j+1) % 20
        
        pi = current_points[i]
        pj = current_points[j]
        
        # Vector
        dx = pj[0] - pi[0]
        dy = pj[1] - pi[1]
        
        # Try small steps k
        for k in range(-5, 6):
            if k == 0: continue # That's pi
            if k == 1: continue # That's pj
            
            nx = pi[0] + k * dx
            ny = pi[1] + k * dy
            
            if -100 <= nx <= 100 and -100 <= ny <= 100:
                candidates.append([nx, ny])
        
        random.shuffle(candidates)
        
        moved = False
        for new_p in candidates:
            if new_p == old_p:
                continue
            if new_p in current_points:
                continue
                
            new_points = current_points[:]
            new_points[idx] = new_p
            
            if _has_four_collinear(new_points):
                continue
                
            # Evaluate
            # Optimization: Only evaluate if we think it might be better?
            # Or just evaluate.
            new_score = _max_no_three_collinear(new_points)
            
            if new_score < current_score:
                current_score = new_score
                current_points = new_points
                if current_score < best_score:
                    best_score = current_score
                    best_points = current_points[:]
                moved = True
                break # Accept first improvement? Or keep looking?
                # Let's accept first improvement to save time, or maybe best of candidates?
                # Given time constraints, first improvement is safer.
                
        if not moved:
            # If no move found, maybe try a random swap?
            pass

    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    best_points = None
    best_score = float('inf')
    
    # Try a few initial configurations
    for _ in range(5):
        if time.time() > deadline - 20:
            break
            
        init_points = _generate_initial_points()
        
        # Ensure initial points are valid (no 4 collinear)
        attempts = 0
        while _has_four_collinear(init_points) and attempts < 50 and time.time() < deadline - 20:
            idx = random.randint(0, 19)
            init_points[idx] = [random.randint(-100, 100), random.randint(-100, 100)]
            attempts += 1
            
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
