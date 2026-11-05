import numpy as np
import random
import time

def construct():
    # 1. Fast exact solver using bitmasks + greedy coloring B&B
    def max_no_3_collinear(points):
        n = len(points)
        # Precompute collinearity: collinear[i] is a bitmask of points j where i,j,k are collinear for some k
        # Actually: for each pair (i,j), find all k collinear with them
        # We need: for each triple (i,j,k), mark that we can't have all three
        
        # Better approach: For each point i, compute for each other point j, the set of points collinear with i and j
        # Then build a "conflict" structure
        
        # For B&B: we want to find max subset with no 3 collinear
        # This is equivalent to: in the 3-uniform hypergraph of collinear triples, find max independent set
        
        # Precompute: for each pair (i,j), the set of points k such that (i,j,k) is collinear
        # collinear_with[i][j] = set of k > j such that i,j,k collinear (to avoid duplicates)
        
        # Actually for B&B, let's precompute:
        # For each point i, and for each "direction" from i, the set of other points in that direction
        # A triple (i,j,k) is collinear iff j and k are in the same direction from i (and i is not between them necessarily, but collinear means same line)
        
        # Simpler: precompute all collinear triples as a list of (i,j,k) with i<j<k
        triples = []
        for i in range(n):
            for j in range(i+1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                dx = x2 - x1
                dy = y2 - y1
                for k in range(j+1, n):
                    x3, y3 = points[k]
                    if dx * (y3 - y1) - dy * (x3 - x1) == 0:
                        triples.append((i, j, k))
        
        if not triples:
            return n
        
        # For B&B with bitmasks:
        # state: available vertices as a bitmask
        # We need to find max independent set in 3-uniform hypergraph
        
        # Precompute for each vertex v: conflict_mask[v] = bitmask of vertices that form a triple with v and one other
        # Actually, let's think differently.
        
        # Standard approach for small n: branch and bound with greedy coloring lower bound
        # For 3-uniform hypergraph, coloring is trickier. Let's use a simpler bound.
        
        # Alternative: since n=20, we can use a different B&B strategy.
        # Pick a triple, branch on excluding one of its 3 vertices.
        
        # Precompute for fast lookup: for each vertex v, which triples contain v
        # And for each pair (v, w), which triples contain both
        
        # Let's precompute: for each vertex v, a bitmask of all vertices u such that there exists a triple containing both v and u
        # No, that's not quite right either.
        
        # Let me use a cleaner approach:
        # available: bitmask of vertices still available
        # We want to maximize |available| - (number excluded)
        # 
        # Key insight: if no triple is fully contained in available, we can take all of available.
        # 
        # For branching: find a triple (a,b,c) fully in available. Branch on excluding a, b, or c.
        
        # Precompute: for fast checking if a triple is in available, we just check (available & triple_mask) == triple_mask
        
        # To find a triple quickly, iterate through triples list
        
        best = [0]
        
        # Greedy lower bound: repeatedly pick a vertex, remove all vertices that form a triple with it and another picked vertex
        # This is complex. Let's use a simpler bound: the size of a maximal independent set found greedily
        
        def greedy_lower_bound(available):
            # Greedily build an independent set
            # Pick vertex with minimum "conflict" first
            selected = 0
            avail = available
            while avail:
                # Find vertex with fewest conflicts in avail
                best_v = -1
                min_conf = n + 1
                temp = avail
                while temp:
                    v = temp & (-temp)
                    temp &= temp - 1
                    vi = v.bit_length() - 1
                    # Count how many other vertices in avail form a triple with vi and another vertex in avail
                    # For each other vertex u in avail, check if there's a w in avail such that (vi, u, w) is collinear
                    conf = 0
                    temp2 = avail & ~v
                    while temp2:
                        u = temp2 & (-temp2)
                        temp2 &= temp2 - 1
                        ui = u.bit_length() - 1
                        # Check if there exists w in avail (w != vi, w != u) such that (min, mid, max) is collinear
                        # We need to check all triples containing both vi and ui
                        # Precompute this
                        pass
                    # This is getting complicated. Let's use a simpler bound.
                    break
                break
            # Just return 0 for now, we'll improve
            return 0
        
        # Let's use a different approach: memoization with bitmask
        # But 2^20 = 1M states, might be feasible
        
        from functools import lru_cache
        
        # Precompute: for each vertex v, for each other vertex u, whether there exists a w such that (v,u,w) is a triple
        # Actually, we need: given available set, can we determine if any triple is fully in it?
        
        # Let me precompute a 3D boolean or use the triples list directly
        
        # For efficiency, let's precompute: triple_masks = list of bitmasks for each triple
        triple_masks = []
        for (a, b, c) in triples:
            mask = (1 << a) | (1 << b) | (1 << c)
            triple_masks.append(mask)
        
        @lru_cache(maxsize=None)
        def solve(available):
            if available == 0:
                return 0
            
            # Check if any triple is fully contained in available
            found_triple = False
            for tm in triple_masks:
                if (available & tm) == tm:
                    found_triple = True
                    break
            
            if not found_triple:
                return available.bit_count()
            
            # Find a triple to branch on
            # Pick the triple with the most "impact" - one where excluding any vertex removes many constraints
            # For simplicity, pick the first one
            branch_tm = None
            for tm in triple_masks:
                if (available & tm) == tm:
                    branch_tm = tm
                    break
            
            # Branch on excluding each vertex in the triple
            best_val = 0
            temp = branch_tm
            while temp:
                v = temp & (-temp)
                temp &= temp - 1
                new_avail = available & ~v
                val = new_avail.bit_count() - (n - new_avail.bit_count())  # Not right
                # Actually: we exclude v, so we get 0 for v + solve(new_avail)
                val = solve(new_avail)
                if val > best_val:
                    best_val = val
            
            return best_val
        
        result = solve((1 << n) - 1)
        return result

    # 2. Validity Check
    def is_valid(points):
        if len(points) != 20:
            return False
        for p in points:
            if abs(p[0]) > 100 or abs(p[1]) > 100:
                return False
        if len(set(map(tuple, points))) != 20:
            return False
        
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                count = 2
                for k in range(n):
                    if k == i or k == j:
                        continue
                    x3, y3 = points[k]
                    if (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) == 0:
                        count += 1
                if count >= 4:
                    return False
        return True

    # 3. Constructive initialization: use parabola y = x^2 mod something, then perturb
    random.seed(42)
    
    # Start with points on a parabola, which has no 3 collinear
    # y = x^2 for x = -10, -9, ..., 9
    # But we need |x|,|y| <= 100
    base_points = []
    for x in range(-10, 10):
        y = x * x
        if abs(y) <= 100:
            base_points.append([x, y])
    
    # We have 11 points from x=-10..10 where x^2 <= 100, i.e., |x| <= 10
    # Actually x^2 <= 100 means |x| <= 10, so all 21 values work? No, 10^2=100, so x from -10 to 10 gives 21 points
    # But we only need 20. Let's use x = -10 to 9 (20 points)
    base_points = [[x, x*x] for x in range(-10, 10)]
    # Check: x=10 gives y=100 which is OK, but let's use -10 to 9
    # Actually range(-10, 10) gives -10,-9,...,9 which is 20 points. Good.
    # y values: 100, 81, 64, 49, 36, 25, 16, 9, 4, 1, 0, 1, 4, 9, 16, 25, 36, 49, 64, 81
    # All |y| <= 100. Good.
    
    points = [p[:] for p in base_points]
    
    # Parabola has no 3 collinear, so score should be 20 initially
    # But we need no 4 collinear. Parabola has no 3 collinear, so certainly no 4.
    # So this is valid!
    
    if not is_valid(points):
        # Fallback: perturb
        attempts = 0
        while not is_valid(points) and attempts < 1000:
            idx = random.randint(0, 19)
            new_x = points[idx][0] + random.randint(-5, 5)
            new_y = points[idx][1] + random.randint(-5, 5)
            if -100 <= new_x <= 100 and -100 <= new_y <= 100:
                collides = False
                for i, p in enumerate(points):
                    if i != idx and p[0] == new_x and p[1] == new_y:
                        collides = True
                        break
                if not collides:
                    points[idx] = [new_x, new_y]
            attempts += 1
    
    current_score = max_no_3_collinear(points)
    best_points = [p[:] for p in points]
    best_score = current_score
    
    # Local search to try to reduce score below 20
    time_limit = time.time() + 110
    
    while time.time() < time_limit:
        idx = random.randint(0, 19)
        step = random.randint(1, 15)
        new_x = points[idx][0] + random.randint(-step, step)
        new_y = points[idx][1] + random.randint(-step, step)
        
        if not (-100 <= new_x <= 100 and -100 <= new_y <= 100):
            continue
            
        collides = False
        for i, p in enumerate(points):
            if i != idx and p[0] == new_x and p[1] == new_y:
                collides = True
                break
        if collides:
            continue
            
        old_p = points[idx][:]
        points[idx] = [new_x, new_y]
        
        if is_valid(points):
            new_score = max_no_3_collinear(points)
            if new_score < best_score:
                best_score = new_score
                best_points = [p[:] for p in points]
            elif new_score == best_score:
                if random.random() < 0.3:
                    pass
                else:
                    points[idx] = old_p
            else:
                points[idx] = old_p
        else:
            points[idx] = old_p

    return best_points
