"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 64
import time
import random

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 280  # safety margin before 330s kill
    
    def sq_dist(p1, p2):
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return dx*dx + dy*dy

    def can_add(p, S, dist_sets):
        """
        Check if point p can be added to set S without forming an isosceles triangle.
        dist_sets is a dict: point -> set of squared distances to other points in S.
        """
        # 1. Check for isosceles triangles with p as the vertex.
        # This requires all distances from p to points in S to be distinct.
        dists_to_p = []
        for q in S:
            d = sq_dist(p, q)
            dists_to_p.append(d)
        
        if len(set(dists_to_p)) != len(dists_to_p):
            return False
            
        # 2. Check for isosceles triangles with a point q in S as the vertex.
        # This requires that for each q in S, the distance dist(p, q) is not
        # equal to the distance between q and any other point r in S.
        # dist_sets[q] contains all squared distances from q to other points in S.
        for i, q in enumerate(S):
            d = dists_to_p[i]
            if d in dist_sets[q]:
                return False
                
        return True

    def add_point(p, S, dist_sets):
        """Add point p to set S and update dist_sets."""
        S.add(p)
        # Calculate distances from p to all existing points
        dists_to_p = []
        for q in S:
            if q == p:
                continue
            d = sq_dist(p, q)
            dists_to_p.append(d)
            # Update the distance set for q
            dist_sets[q].add(d)
        
        # Store distances from p
        dist_sets[p] = set(dists_to_p)

    def greedy_run(points_order):
        """
        Greedily construct a valid set using the given order of points.
        Returns (set, dist_sets).
        """
        S = set()
        dist_sets = {}
        
        for p in points_order:
            if time.time() > deadline:
                break
            if can_add(p, S, dist_sets):
                add_point(p, S, dist_sets)
                
        return S, dist_sets

    # Generate all grid points
    all_points = [(x, y) for y in range(N) for x in range(N)]
    
    best_set = set()
    best_dist_sets = {}
    best_size = 0
    
    # 1. Try standard order (row-major)
    S, ds = greedy_run(all_points)
    if len(S) > best_size:
        best_set, best_dist_sets, best_size = S, ds, len(S)
        
    # 2. Try reversed order
    S, ds = greedy_run(all_points[::-1])
    if len(S) > best_size:
        best_set, best_dist_sets, best_size = S, ds, len(S)
        
    # 3. Try random orders
    # Use a fixed seed for reproducibility in evaluation if needed, 
    # but random is fine.
    rng = random.Random(42)
    
    # Estimate how many random runs we can afford.
    # Each run is O(N^2 * K). 
    # If K ~ 100, 4096 * 100 = 400k ops.
    # Python can do ~10M-50M simple ops/sec.
    # So ~0.01s per run? Maybe slower due to overhead.
    # Let's just loop until time is up.
    
    while time.time() < deadline - 5:
        # Shuffle a copy of all_points
        # To avoid modifying all_points, we can iterate over a shuffled range
        # or just shuffle a list
        shuffled_points = all_points.copy()
        rng.shuffle(shuffled_points)
        
        S, ds = greedy_run(shuffled_points)
        
        if len(S) > best_size:
            best_set, best_dist_sets, best_size = S, ds, len(S)
            
    # Return the best set found
    return list(best_set)
# EVOLVE-BLOCK-END
