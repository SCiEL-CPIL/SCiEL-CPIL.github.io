import random
import time
import math

N = 64

# Pre-compute all grid points
ALL_POINTS = [(x, y) for x in range(N) for y in range(N)]

def squared_dist(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return dx*dx + dy*dy

def construct():
    start_time = time.time()
    # Leave a safety margin of 30 seconds before the 300s/330s kill switch
    deadline = start_time + 270.0
    
    best_set = []
    best_size = 0
    
    # We will perform multiple random restarts.
    # In each restart, we try to build a set by greedily adding points 
    # from a shuffled order, checking if the new point forms an isosceles 
    # triangle with any pair of points already in the set.
    
    # Optimization: 
    # A set S is "no-isosceles" if for any distinct a, b, c in S, 
    # no two of the distances d(a,b), d(b,c), d(c,a) are equal.
    # When adding a new point p to S, we must check for every pair (a, b) in S
    # if d(p,a) == d(p,b) or d(p,a) == d(a,b) or d(p,b) == d(a,b).
    
    # To make this faster, we can maintain a set of "forbidden" distances or positions?
    # Actually, checking pairs in S is O(k^2) per insertion, where k is current size.
    # Total O(k^3) for a full set. With N=64, grid has 4096 points.
    # Max size likely around 50-100?
    # 100^3 = 1,000,000 ops per restart. 
    # 270 seconds is plenty for thousands of restarts if optimized in Python? 
    # Maybe not. Let's optimize the check.
    
    # Alternative check:
    # For a new point p, and existing points a, b:
    # Isosceles if:
    # 1. d(p,a) == d(p,b)  => p lies on the perpendicular bisector of ab.
    # 2. d(p,a) == d(a,b)  => p lies on the circle centered at a with radius d(a,b).
    # 3. d(p,b) == d(a,b)  => p lies on the circle centered at b with radius d(a,b).
    
    # We can precompute for the current set S:
    # - A set of "bad" points? No, the set of bad points changes as S grows.
    # - We can maintain a set of "occupied" points.
    # - For each new candidate p, we check against all pairs in S.
    
    # Let's try a slightly smarter greedy:
    # Instead of checking all pairs (a,b) in S for each p, 
    # we can check if p creates an isosceles triangle with ANY a in S and ANY b in S.
    
    # Let's implement the O(k^2) check per insertion.
    # To speed it up, we can use squared distances to avoid sqrt.
    
    def is_valid_addition(p, current_set, current_sq_dists):
        # current_set is a list of points
        # current_sq_dists is a dict or list of (i, j, dist_sq) for all pairs? 
        # That's O(k^2) memory and update time.
        # Let's just iterate.
        
        n_pts = len(current_set)
        # Check 1: d(p, a) == d(p, b) for any a, b in current_set
        # This means p is equidistant from a and b.
        # Check 2: d(p, a) == d(a, b) for any a, b in current_set
        # Check 3: d(p, b) == d(a, b) for any a, b in current_set
        
        # We can combine 2 and 3: 
        # For each pair (a, b) in current_set:
        #   d_ab_sq = d(a, b)
        #   d_pa_sq = d(p, a)
        #   d_pb_sq = d(p, b)
        #   if d_pa_sq == d_pb_sq: return False
        #   if d_pa_sq == d_ab_sq: return False
        #   if d_pb_sq == d_ab_sq: return False
        
        # This is O(k^2) per insertion.
        
        for i in range(n_pts):
            a = current_set[i]
            d_pa_sq = squared_dist(p, a)
            for j in range(i + 1, n_pts):
                b = current_set[j]
                d_ab_sq = squared_dist(a, b)
                d_pb_sq = squared_dist(p, b)
                
                if d_pa_sq == d_pb_sq:
                    return False
                if d_pa_sq == d_ab_sq:
                    return False
                if d_pb_sq == d_ab_sq:
                    return False
        return True

    # To improve performance, we can pre-compute distances between all grid points?
    # 4096 * 4096 is ~16 million entries. Too much memory/time in Python.
    
    # Let's stick to the O(k^2) check but optimize the inner loop.
    # Also, we can try different initial seeds.
    
    num_restarts = 1000 # We will stop based on time
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Shuffle the points
        points = ALL_POINTS[:]
        random.shuffle(points)
        
        current_set = []
        
        for p in points:
            if time.time() > deadline:
                break
            
            # Check if adding p is valid
            # Optimization: If current_set is empty, always valid.
            if not current_set:
                current_set.append(p)
                continue
                
            # Quick check: 
            # We can skip the full O(k^2) check if we maintain some structure?
            # For now, just do the check.
            
            valid = True
            n_pts = len(current_set)
            
            # Iterate over pairs in current_set
            # To make this faster in Python, we can use local variables
            sq_dist_func = squared_dist
            
            for i in range(n_pts):
                a = current_set[i]
                d_pa_sq = sq_dist_func(p, a)
                for j in range(i + 1, n_pts):
                    b = current_set[j]
                    d_ab_sq = sq_dist_func(a, b)
                    d_pb_sq = sq_dist_func(p, b)
                    
                    if d_pa_sq == d_pb_sq or d_pa_sq == d_ab_sq or d_pb_sq == d_ab_sq:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                current_set.append(p)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
            
        # If we found a very large set, we might want to try local search?
        # But for now, random restarts with greedy is the baseline.
        
    return best_set
