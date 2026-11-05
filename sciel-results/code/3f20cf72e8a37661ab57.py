import time
import random
import math
from functools import lru_cache

DEADLINE = 115.0

def normalize_slope(dx, dy):
    if dx == 0 and dy == 0:
        return None
    if dx == 0:
        return (0, 1)
    if dy == 0:
        return (1, 0)
    g = math.gcd(dx, dy)
    if g < 0:
        g = -g
    dx, dy = dx // g, dy // g
    if dx < 0:
        dx, dy = -dx, -dy
    return (dx, dy)

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
            slope = normalize_slope(dx, dy)
            if slope is None:
                continue
            if slope in slopes:
                slopes[slope] += 1
                if slopes[slope] >= 3:
                    return True
            else:
                slopes[slope] = 1
    return False

def count_collinear_triples(points):
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
            slope = normalize_slope(dx, dy)
            if slope is None:
                continue
            if slope in slopes:
                slopes[slope] += 1
            else:
                slopes[slope] = 1
        for k in slopes.values():
            if k >= 2:
                count += k * (k - 1) // 2
    return count

def compute_conflict_graph(points):
    """
    Returns a list of lists, where conflict[i][j] is a set of k 
    such that {i, j, k} is a collinear triple.
    Also returns the total number of triples.
    """
    n = len(points)
    # First, find all collinear triples
    triples = []
    # For each pair (i, j), find k such that i, j, k are collinear
    # To avoid duplicates, we can just iterate all triples O(N^3) which is 8000, very fast
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi = points[i]
            xj, yj = points[j]
            dx = xj - xi
            dy = yj - yi
            slope = normalize_slope(dx, dy)
            if slope is None:
                continue
            # Find k > j such that slope(i, k) == slope
            for k in range(j + 1, n):
                xk, yk = points[k]
                dxk = xk - xi
                dyk = yk - yi
                slope_k = normalize_slope(dxk, dyk)
                if slope_k == slope:
                    triples.append((i, j, k))
    
    # Build conflict graph
    # conflict[i][j] = set of k
    conflict = [[set() for _ in range(n)] for _ in range(n)]
    for i, j, k in triples:
        conflict[i][j].add(k)
        conflict[i][k].add(j)
        conflict[j][i].add(k)
        conflict[j][k].add(i)
        conflict[k][i].add(j)
        conflict[k][j].add(i)
        
    return conflict, len(triples)

def max_no_three_collinear(points):
    n = len(points)
    conflict, num_triples = compute_conflict_graph(points)
    
    # If no triples, answer is n
    if num_triples == 0:
        return n
        
    # Bitmask representation for speed
    # candidates: bitmask of points that can still be added
    # forbidden: bitmask of points that cannot be added because they form a triple with 2 already added points
    
    # We use a recursive search with bitmasks
    # State: (current_size, candidates_mask, forbidden_mask)
    # We want to maximize current_size
    
    best = [0]
    
    # Order points by degree in the conflict graph? 
    # Points involved in more triples are "more constrained".
    # Let's stick to index order for simplicity, or sort by degree.
    degrees = [0] * n
    for i in range(n):
        for j in range(n):
            degrees[i] += len(conflict[i][j])
    
    # Sort indices by degree descending? 
    # Actually, for B&B, branching on high-degree nodes first often prunes better.
    # But we need to map original indices to new indices or just use the mask as is.
    # Let's just use the mask as is, 0..n-1.
    
    def search(idx, current_mask, forbidden_mask, current_size):
        # Pruning: if current_size + popcount(candidates) <= best, return
        # Candidates are those not in current and not in forbidden
        # We pass idx to iterate in order, but bitmask is more flexible.
        # Let's use a simpler recursive structure:
        # Pick the lowest set bit in candidates.
        
        candidates_mask = ((1 << n) - 1) ^ current_mask ^ forbidden_mask
        if candidates_mask == 0:
            if current_size > best[0]:
                best[0] = current_size
            return
            
        # Upper bound: current_size + popcount(candidates_mask)
        if current_size + bin(candidates_mask).count('1') <= best[0]:
            return
            
        # Pick a point v from candidates
        # Heuristic: pick the one with highest degree in the remaining subgraph?
        # For simplicity, pick the lowest index.
        v = (candidates_mask & -candidates_mask).bit_length() - 1
        
        # Option 1: Exclude v
        new_candidates_mask = candidates_mask ^ (1 << v)
        search(idx + 1, current_mask, forbidden_mask, current_size) # Wait, search doesn't use idx in this bitmask logic properly if we just clear bits.
        # Let's restructure:
        
        # Actually, the standard B&B for Max Independent Set in Hypergraph:
        # If we exclude v, we just remove v from candidates.
        # If we include v, we must remove v from candidates, and add to forbidden
        # all k such that there exists u in current_mask where {u, v, k} is a triple.
        
        # Let's implement this carefully.
        
        # Exclude v
        candidates_mask_excl = candidates_mask ^ (1 << v)
        # We don't change current_mask or forbidden_mask
        # But we need to pass the state down.
        # The function signature should be search(current_mask, forbidden_mask)
        
        # Let's rewrite search to take masks only.
        
        pass

    def search_masks(current_mask, forbidden_mask):
        current_size = bin(current_mask).count('1')
        candidates_mask = ((1 << n) - 1) ^ current_mask ^ forbidden_mask
        
        if candidates_mask == 0:
            if current_size > best[0]:
                best[0] = current_size
            return
            
        # Pruning
        if current_size + bin(candidates_mask).count('1') <= best[0]:
            return
            
        # Pick v
        # Heuristic: pick v that maximizes the number of conflicts in the current candidate set?
        # For now, lowest index.
        v = (candidates_mask & -candidates_mask).bit_length() - 1
        
        # Option 1: Exclude v
        new_forbidden_excl = forbidden_mask
        new_current_excl = current_mask
        # Just remove v from candidates by moving it to "processed"?
        # Actually, the state is defined by current and forbidden.
        # If we exclude v, v is neither in current nor forbidden. It's just not considered anymore.
        # To handle "not considered anymore", we can have a third mask "processed".
        # Or, we can just recurse with a modified candidates mask?
        # The standard way:
        # State: (current, forbidden, candidates)
        # But candidates = All ^ current ^ forbidden.
        # If we exclude v, v becomes "processed". It is removed from candidates.
        # How to represent "processed"?
        # We can just pass a mask of available points.
        
        # Let's use: search(available_mask, current_mask, forbidden_mask)
        # Initially: available = (1<<n)-1, current=0, forbidden=0.
        # At each step, pick v from available.
        # Exclude: available' = available ^ (1<<v). current' = current. forbidden' = forbidden.
        # Include: 
        #   new_forbidden = forbidden
        #   for u in current:
        #       for k in conflict[v][u]:
        #           if k is in available:
        #               new_forbidden |= (1 << k)
        #   current' = current | (1<<v)
        #   available' = available ^ (1<<v) ^ new_forbidden_added? 
        #   Actually, if k is forbidden, it's removed from available.
        #   So available' = available ^ (1<<v)
        #   Then remove any k in new_forbidden from available'.
        
        pass

    # Let's implement the 3-mask version.
    
    def search(available, current, forbidden):
        current_size = bin(current).count('1')
        if available == 0:
            if current_size > best[0]:
                best[0] = current_size
            return
            
        # Pruning
        if current_size + bin(available).count('1') <= best[0]:
            return
            
        # Pick v from available
        # Heuristic: pick v with max degree in the induced subgraph on available?
        # Simple: lowest index.
        v = (available & -available).bit_length() - 1
        bit_v = 1 << v
        
        # Option 1: Exclude v
        search(available ^ bit_v, current, forbidden)
        
        # Option 2: Include v
        new_current = current | bit_v
        new_forbidden = forbidden
        # Check conflicts with all u in current
        # Iterate bits in current
        temp_current = current
        while temp_current:
            u_bit = temp_current & -temp_current
            u = u_bit.bit_length() - 1
            # For each k in conflict[v][u]
            for k in conflict[v][u]:
                new_forbidden |= (1 << k)
            temp_current ^= u_bit
            
        # New available: remove v and any newly forbidden points
        # Points that are in new_forbidden and were in available must be removed.
        newly_forbidden = new_forbidden & available
        new_available = (available ^ bit_v) ^ newly_forbidden
        
        search(new_available, new_current, new_forbidden)
        
    full_mask = (1 << n) - 1
    search(full_mask, 0, 0)
    return best[0]

def construct():
    start_time = time.time()
    random.seed(42)
    
    # Initial points
    points = []
    used = set()
    while len(points) < 20:
        x = random.randint(-100, 100)
        y = random.randint(-100, 100)
        if (x, y) not in used:
            used.add((x, y))
            points.append((x, y))
            
    # Repair initial points to ensure no 4 collinear
    while has_four_collinear(points):
        idx = random.randint(0, 19)
        old_p = points[idx]
        while True:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            new_p = (x, y)
            if new_p == old_p or new_p in used:
                continue
            points[idx] = new_p
            if not has_four_collinear(points):
                used.discard(old_p)
                used.add(new_p)
                break
            else:
                points[idx] = old_p
                
    current_triples = count_collinear_triples(points)
    best_score = max_no_three_collinear(points)
    best_points = points[:]
    best_triples = current_triples
    
    T = 10.0
    cooling = 0.999
    
    while time.time() - start_time < DEADLINE:
        idx = random.randint(0, 19)
        old_p = points[idx]
        
        x = random.randint(-100, 100)
        y = random.randint(-100, 100)
        new_p = (x, y)
        
        if new_p == old_p or new_p in used:
            continue
            
        points[idx] = new_p
        
        if has_four_collinear(points):
            points[idx] = old_p
            continue
            
        new_triples = count_collinear_triples(points)
        
        # Acceptance: Maximize triples
        if new_triples > current_triples or random.random() < math.exp((new_triples - current_triples) / T):
            current_triples = new_triples
            used.discard(old_p)
            used.add(new_p)
            
            # Check exact score if triples improved
            if new_triples > best_triples:
                exact_score = max_no_three_collinear(points)
                if exact_score < best_score:
                    best_score = exact_score
                    best_points = points[:]
                    best_triples = new_triples
                else:
                    # Even if score didn't drop, higher triples might be a better local landscape
                    best_triples = new_triples
        else:
            points[idx] = old_p
            
        T *= cooling
        if T < 0.01:
            T = 0.01
            
    return [[p[0], p[1]] for p in best_points]
