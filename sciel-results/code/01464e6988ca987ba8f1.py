"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random

def _dist2(a, b):
    """Squared distance between two lattice points (a,b) and (c,d)."""
    da = a[0] - b[0]
    db = a[1] - b[1]
    return da*da + da*db + db*db

def _edge_count(points):
    """Count pairs at distance exactly 1 (dist2 == 1)."""
    n = len(points)
    count = 0
    for i in range(n):
        for j in range(i+1, n):
            if _dist2(points[i], points[j]) == 1:
                count += 1
    return count

def _valid(points):
    """Check all pairs have distance >= 1."""
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            if _dist2(points[i], points[j]) < 1:
                return False
    return True

def _neighbors(a, b):
    """The 6 neighbors of (a,b) on the triangular lattice at distance 1."""
    return [(a+1, b), (a, b+1), (a-1, b+1), (a-1, b), (a, b-1), (a+1, b-1)]

def _harborth(n):
    return math.floor(3*n - math.sqrt(12*n - 3))

def _generate_hexagonal_seed(n, center=(0, 0)):
    """
    Generate a hexagonal (or near-hexagonal) cluster of n points on the triangular lattice.
    This is a good starting point because hexagonal clusters are known to be optimal
    for small n and provide a dense, connected structure.
    """
    cx, cy = center
    # Directions for expanding a hexagon:
    # We can generate points in rings.
    # Ring 0: (0,0)
    # Ring 1: 6 points
    # Ring 2: 12 points
    # Ring k: 6k points
    
    # Instead of rings, let's just generate a large hexagon and pick the first n.
    # Or better, generate a "hexagonal number" shape as closely as possible.
    
    # A simpler robust method:
    # Start with center, then add neighbors in a BFS-like manner prioritizing those
    # with the most existing neighbors (greedy hexagon growth).
    
    points = [(cx, cy)]
    used = set(points)
    
    # Precompute candidates in a spiral around the center
    # Actually, let's just do a greedy growth which naturally forms a hexagon.
    # To ensure it's a good seed, we can use a fixed deterministic order or
    # just use the greedy add with a fixed seed.
    
    # Let's try a deterministic hexagonal generation.
    # The points in a hexagon of radius R are those with hex_dist <= R.
    # But we want exactly n points.
    
    # Let's just use a simple greedy algorithm with a fixed random seed to generate a "compact" cluster.
    # This is fast and produces a good seed.
    
    random.seed(42)  # Deterministic for the seed generation
    for step in range(1, n):
        candidates = set()
        for (a, b) in points:
            for (na, nb) in _neighbors(a, b):
                if (na, nb) not in used:
                    candidates.add((na, nb))
        
        if not candidates:
            break
            
        # Pick the candidate with the most neighbors in the current set
        best_score = -1
        best_candidate = None
        for cand in candidates:
            score = 0
            for p in points:
                if _dist2(cand, p) == 1:
                    score += 1
            if score > best_score:
                best_score = score
                best_candidate = cand
            elif score == best_score and best_candidate is not None:
                # Tie-breaker: prefer closer to center? Or just random?
                # For a seed, deterministic is better.
                pass
        
        if best_candidate is None:
            # Fallback
            best_candidate = list(candidates)[0]
            
        points.append(best_candidate)
        used.add(best_candidate)
        
    random.seed()  # Reset random seed for the main search
    
    return points

def _local_search(points, max_iterations=1000, time_limit=None):
    """
    Perform local search to maximize edge count.
    Moves: 
    1. Swap a point with an empty neighbor.
    2. Move a point to an empty neighbor.
    3. Remove a point and add a new point elsewhere (less common, but might be needed).
    
    We will focus on moves that keep the set size constant and maintain validity.
    A move is valid if the new set has all pairwise distances >= 1.
    We accept moves that increase the edge count, or with some probability (simulated annealing)
    accept moves that decrease it.
    """
    n = len(points)
    points = [tuple(p) for p in points]
    used = set(points)
    
    current_edges = _edge_count(points)
    best_points = points[:]
    best_edges = current_edges
    
    # Precompute neighbor directions
    dirs = [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    
    # For efficiency, we can maintain the edge count incrementally,
    # but for n=32, recomputing is O(n^2) = ~1000 ops, which is fast.
    
    t_start = time.time()
    
    # Simulated Annealing
    temperature = 1.0
    cooling = 0.995
    
    for iteration in range(max_iterations):
        if time_limit is not None and time.time() > time_limit:
            break
        
        # Pick a random point to move
        idx = random.randint(0, n-1)
        old_point = points[idx]
        
        # Generate possible new positions
        # 1. Neighbors of old_point
        # 2. Neighbors of other points (to fill gaps)
        # Let's just try moving to a neighbor of the old point first.
        # If that fails, try a random empty spot nearby.
        
        candidates = set()
        for (da, db) in dirs:
            new_point = (old_point[0] + da, old_point[1] + db)
            if new_point not in used:
                candidates.add(new_point)
        
        # Also add some random candidates to allow larger jumps
        for _ in range(5):
            ra = old_point[0] + random.randint(-3, 3)
            rb = old_point[1] + random.randint(-3, 3)
            if (ra, rb) not in used:
                candidates.add((ra, rb))
        
        if not candidates:
            continue
        
        new_point = random.choice(list(candidates))
        
        # Check validity of the move
        # The new point must be at distance >= 1 from all other points (except the one being moved, which is removed)
        valid = True
        for i in range(n):
            if i == idx:
                continue
            if _dist2(new_point, points[i]) < 1:
                valid = False
                break
        
        if not valid:
            continue
        
        # Calculate new edge count
        # Remove edges incident to old_point
        # Add edges incident to new_point
        # This is faster than recomputing everything.
        
        # Let's just recompute for simplicity and correctness, it's fast enough.
        new_points = points[:]
        new_points[idx] = new_point
        new_edges = _edge_count(new_points)
        
        delta = new_edges - current_edges
        
        if delta > 0 or (temperature > 0.1 and random.random() < math.exp(delta / temperature)):
            # Accept move
            points[idx] = new_point
            used.remove(old_point)
            used.add(new_point)
            current_edges = new_edges
            
            if current_edges > best_edges:
                best_edges = current_edges
                best_points = points[:]
        
        temperature *= cooling
    
    return best_points, best_edges

def construct():
    deadline = time.time() + 140.0  # Safety margin
    n = 32
    target_edges = _harborth(n)  # 76
    
    best_points = None
    best_edges = 0
    
    # 1. Generate a good initial seed using hexagonal growth
    seed_points = _generate_hexagonal_seed(n)
    seed_edges = _edge_count(seed_points)
    
    if seed_edges > best_edges:
        best_edges = seed_edges
        best_points = seed_points[:]
    
    if best_edges >= target_edges:
        return [[a, b] for (a, b) in best_points]
    
    # 2. Run local search from this seed
    # We can run multiple local searches with different random seeds or temperatures
    # to escape local minima.
    
    for attempt in range(20):
        if time.time() > deadline - 10:  # Leave time for final checks
            break
        
        # Jitter the seed slightly to create a new starting point
        # Or just run local search with a fresh random seed
        random.seed(attempt * 12345)
        
        # Sometimes, we can try a "kick" by randomly moving a few points
        # in the seed before starting local search.
        jittered_points = seed_points[:]
        for _ in range(3):
            idx = random.randint(0, n-1)
            old_p = jittered_points[idx]
            # Try to move it to a random valid neighbor
            for _ in range(10):
                da = random.randint(-2, 2)
                db = random.randint(-2, 2)
                new_p = (old_p[0] + da, old_p[1] + db)
                if new_p in jittered_points:
                    continue
                # Check validity
                valid = True
                for i in range(n):
                    if i == idx:
                        continue
                    if _dist2(new_p, jittered_points[i]) < 1:
                        valid = False
                        break
                if valid:
                    jittered_points[idx] = new_p
                    break
        
        # Run local search
        ls_points, ls_edges = _local_search(jittered_points, max_iterations=500, time_limit=deadline - 5)
        
        if ls_edges > best_edges:
            best_edges = ls_edges
            best_points = ls_points[:]
            
            if best_edges >= target_edges:
                break
    
    # 3. If we haven't reached the target, try a more aggressive search
    # or just return the best found so far.
    
    if best_points is None:
        # Fallback
        best_points = [(a, b) for a in range(16) for b in range(2)]
    
    return [[a, b] for (a, b) in best_points]
# EVOLVE-BLOCK-END
