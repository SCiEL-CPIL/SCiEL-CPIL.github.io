import time
import random
import math
import itertools

def _neighbors(p):
    a, b = p
    return ((a + 1, b), (a - 1, b), (a, b + 1),
            (a, b - 1), (a + 1, b - 1), (a - 1, b + 1))

def _edge_count(points):
    """Count edges in a set of points."""
    pts_set = set(points)
    cnt = 0
    for p in pts_set:
        for q in _neighbors(p):
            if q in pts_set and p < q:
                cnt += 1
    return cnt

def _valid(points):
    """Check validity: 32 distinct points, |a|,|b| <= 64, min distance >= 1."""
    if len(points) != 32:
        return False
    if len(set(points)) != 32:
        return False
    for p in points:
        if abs(p[0]) > 64 or abs(p[1]) > 64:
            return False
    # Distinct integer points on triangular lattice always have dist >= 1
    return True

def _generate_hexagon(radius):
    pts = []
    for a in range(-radius, radius + 1):
        for b in range(-radius, radius + 1):
            c = -a - b
            if max(abs(a), abs(b), abs(c)) <= radius:
                pts.append((a, b))
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    # Step 1: Brute force best 32-subset of Hex(3)
    # Hex(3) has 37 points. We remove 5.
    hex3 = _generate_hexagon(3)
    hex3_set = set(hex3)
    
    best_points = None
    best_score = -1
    
    # Iterate over combinations of 5 points to remove
    # There are C(37, 5) = 435,897 combinations
    # We can speed this up by noting that removing points on the boundary is likely better
    # but let's just iterate all to be safe if time permits.
    
    # Optimization: Pre-compute neighbor lists for hex3 points to speed up edge counting?
    # Actually, _edge_count is fast enough for 32 points.
    # 435k * (32*6/2) ~ 40M ops. Should take < 5s in PyPy, maybe 10-15s in CPython.
    # Let's try.
    
    # To save time, we can skip combinations that are obviously bad?
    # Or just run it.
    
    for rem_combo in itertools.combinations(hex3, 5):
        rem_set = set(rem_combo)
        curr_set = hex3_set - rem_set
        score = _edge_count(list(curr_set))
        if score > best_score:
            best_score = score
            best_points = list(curr_set)
            
        if time.time() > start_time + 20.0:
            # If brute force is taking too long, break and use best so far
            # But 435k is small, it should finish.
            pass

    if best_points is None:
        # Fallback if something went wrong
        best_points = hex3[:32]
        best_score = _edge_count(best_points)

    # Step 2: Simulated Annealing / Local Search
    # Pool: Hex(4) or larger
    pool = set(_generate_hexagon(5)) # Hex(5) has 91 points. Plenty of room.
    
    curr_points = set(best_points)
    curr_score = best_score
    best_global_score = best_score
    best_global_points = list(curr_points)
    
    T = 5.0 # Initial temperature
    T_min = 0.1
    cooling_rate = 0.999
    
    random.seed(42)
    
    while time.time() < deadline:
        if T < T_min:
            break
            
        # Pick a random point in current set to remove
        p_in = random.choice(list(curr_points))
        
        # Pick a random point from pool to add
        # To bias towards nearby points, we could pick from neighbors of current set
        # But random from pool is fine if pool is not too large.
        # Let's pick from neighbors of current set for locality.
        candidates = []
        for p in curr_points:
            for n in _neighbors(p):
                if n not in curr_points and n in pool:
                    candidates.append(n)
        
        if not candidates:
            T *= cooling_rate
            continue
            
        p_out = random.choice(candidates)
        
        # Perform swap
        new_set = set(curr_points)
        new_set.remove(p_in)
        new_set.add(p_out)
        
        new_score = _edge_count(list(new_set))
        
        delta = new_score - curr_score
        
        if delta > 0 or random.random() < math.exp(delta / T):
            curr_points = new_set
            curr_score = new_score
            
            if curr_score > best_global_score:
                best_global_score = curr_score
                best_global_points = list(curr_points)
                
        T *= cooling_rate

    # Final check
    if not _valid(best_global_points):
        return [[a, b] for a, b in best_global_points[:32]] # Should not happen
        
    return [[a, b] for a, b in best_global_points]
