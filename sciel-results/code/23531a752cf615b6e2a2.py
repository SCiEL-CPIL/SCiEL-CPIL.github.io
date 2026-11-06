import time
import random

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
    """Check validity: 32 distinct points, |a|,|b| <= 64."""
    if len(points) != 32:
        return False
    if len(set(points)) != 32:
        return False
    for p in points:
        if abs(p[0]) > 64 or abs(p[1]) > 64:
            return False
    return True

def _build_hexagonal_spiral(n):
    """Build a hexagonal spiral of n points centered at origin."""
    if n == 1:
        return [(0, 0)]
    points = [(0, 0)]
    radius = 1
    while len(points) < n:
        ring_points = []
        for a in range(-radius, radius + 1):
            for b in range(-radius, radius + 1):
                c = -a - b
                if max(abs(a), abs(b), abs(c)) == radius:
                    ring_points.append((a, b))
        for p in ring_points:
            if len(points) < n:
                points.append(p)
            else:
                break
        radius += 1
    return points

def _local_search(points, pool, deadline, rng):
    """Local search with swap moves and 2-swap moves."""
    best_points = list(points)
    best_score = _edge_count(best_points)
    pts_set = set(best_points)
    pool_list = list(pool)
    iterations = 0
    
    # Pre-compute neighbors for all points in the union of current and pool?
    # No, just compute on the fly. It's fast enough.
    
    while time.time() < deadline and iterations < 100000:
        iterations += 1
        
        # Try 1-swap with high probability, 2-swap with lower probability
        if rng.random() < 0.7:
            # 1-swap
            p_in = rng.choice(best_points)
            p_out = rng.choice(pool_list)
            if p_in == p_out or p_out in pts_set:
                continue
            
            edges_lost = 0
            for n in _neighbors(p_in):
                if n in pts_set:
                    edges_lost += 1
            
            edges_gained = 0
            for n in _neighbors(p_out):
                if n in pts_set and n != p_in:
                    edges_gained += 1
            
            delta = edges_gained - edges_lost
            if delta > 0 or (delta == 0 and rng.random() < 0.1):
                pts_set.remove(p_in)
                pts_set.add(p_out)
                best_points = list(pts_set)
                best_score += delta
        else:
            # 2-swap
            if len(best_points) < 2:
                continue
            p_in1, p_in2 = rng.sample(best_points, 2)
            # Pick two points from pool not in pts_set
            # To avoid O(N^2) sampling, we can just pick random indices from pool_list
            # and check validity
            p_out1 = rng.choice(pool_list)
            p_out2 = rng.choice(pool_list)
            
            if p_out1 == p_out2:
                continue
            if p_out1 in pts_set or p_out2 in pts_set:
                continue
            if p_out1 == p_in1 or p_out1 == p_in2 or p_out2 == p_in1 or p_out2 == p_in2:
                continue
                
            # Calculate delta for 2-swap
            # Remove p_in1, p_in2
            # Add p_out1, p_out2
            
            # Edges lost:
            # 1. Edges from p_in1 to others in pts_set (excluding p_in2)
            # 2. Edges from p_in2 to others in pts_set (excluding p_in1)
            # 3. Edge between p_in1 and p_in2 if it exists
            
            lost = 0
            for n in _neighbors(p_in1):
                if n in pts_set and n != p_in2:
                    lost += 1
            for n in _neighbors(p_in2):
                if n in pts_set and n != p_in1:
                    lost += 1
            if p_in2 in _neighbors(p_in1): # Check if they are adjacent
                lost += 1
                
            # Edges gained:
            # 1. Edges from p_out1 to others in pts_set (excluding p_in1, p_in2, p_out2)
            # 2. Edges from p_out2 to others in pts_set (excluding p_in1, p_in2, p_out1)
            # 3. Edge between p_out1 and p_out2 if it exists
            
            gained = 0
            # We need to check against the set AFTER removing p_in1, p_in2
            # Let's create a temporary set or just check conditions
            temp_set = pts_set
            # Actually, easier to just check if neighbor is in pts_set and not p_in1/p_in2
            
            for n in _neighbors(p_out1):
                if n in pts_set and n != p_in1 and n != p_in2 and n != p_out2:
                    gained += 1
            for n in _neighbors(p_out2):
                if n in pts_set and n != p_in1 and n != p_in2 and n != p_out1:
                    gained += 1
            if p_out2 in _neighbors(p_out1):
                gained += 1
                
            delta = gained - lost
            if delta > 0 or (delta == 0 and rng.random() < 0.05):
                pts_set.remove(p_in1)
                pts_set.remove(p_in2)
                pts_set.add(p_out1)
                pts_set.add(p_out2)
                best_points = list(pts_set)
                best_score += delta
                
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    rng = random.Random(42)
    best_points = None
    best_score = -1
    
    # Define pool
    pool = set()
    for a in range(-12, 13):
        for b in range(-12, 13):
            c = -a - b
            if max(abs(a), abs(b), abs(c)) <= 12:
                pool.add((a, b))
                
    configs = []
    configs.append(_build_hexagonal_spiral(32))
    
    # Add a few other structured configs
    base3 = []
    for a in range(-3, 4):
        for b in range(-3, 4):
            c = -a - b
            if max(abs(a), abs(b), abs(c)) <= 3:
                base3.append((a, b))
    configs.append(base3[:32])
    
    base4 = []
    for a in range(-4, 5):
        for b in range(-4, 5):
            c = -a - b
            if max(abs(a), abs(b), abs(c)) <= 4:
                base4.append((a, b))
    rng.shuffle(base4)
    configs.append(base4[:32])
    
    # Random start
    rand_pts = []
    for a in range(-6, 7):
        for b in range(-6, 7):
            c = -a - b
            if max(abs(a), abs(b), abs(c)) <= 6:
                rand_pts.append((a, b))
    rng.shuffle(rand_pts)
    configs.append(rand_pts[:32])
    
    for initial in configs:
        if time.time() >= deadline - 5.0:
            break
        pts, score = _local_search(initial, pool, deadline - 1.0, rng)
        if score > best_score:
            best_score = score
            best_points = pts
        if best_score >= 76:
            break
            
    if best_points is None:
        return [[a, b] for a in range(16) for b in range(2)]
    if not _valid(best_points):
        return [[a, b] for a in range(16) for b in range(2)]
    return [[a, b] for a, b in best_points]
