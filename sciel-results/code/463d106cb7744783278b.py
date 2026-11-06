import itertools
import time

def construct():
    """
    Find 32 points on a triangular lattice (a + b*omega) that maximize the number of pairs
    at distance exactly 1. The optimum is 76 (Harborth's theorem).
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    # Lattice operations
    def dist_sq(p1, p2):
        dq = p1[0] - p2[0]
        dr = p1[1] - p2[1]
        return dq*dq + dr*dr + dq*dr

    def neighbors(p):
        a, b = p
        return [(a+1, b), (a-1, b), (a, b+1), (a, b-1), (a+1, b-1), (a-1, b+1)]

    def edge_count(points):
        """Count pairs at distance exactly 1."""
        pset = set(points)
        count = 0
        for p in points:
            for n in neighbors(p):
                if n in pset and p < n:
                    count += 1
        return count

    def valid(points):
        """Check all points are distinct and |a|,|b| <= 64, mutual distance >= 1."""
        pset = set(points)
        if len(pset) != len(points):
            return False
        for p in points:
            if abs(p[0]) > 64 or abs(p[1]) > 64:
                return False
            # Check mutual distance >= 1 (no two points closer than 1)
            # On triangular lattice, distance < 1 means same point (already checked distinct)
            # Actually, the minimum distance between distinct lattice points is 1.
            # So distinctness ensures distance >= 1.
        return True

    # Generate candidate region
    R = 12
    candidates = []
    for a in range(-R, R+1):
        for b in range(-R, R+1):
            if abs(a) <= 64 and abs(b) <= 64:
                candidates.append((a, b))

    # Precompute adjacency for all candidates
    cand_set = set(candidates)
    cand_adj = {}
    for p in candidates:
        cand_adj[p] = [n for n in neighbors(p) if n in cand_set]

    # Greedy construction: start with a hexagonal seed and grow
    # First, create a hexagonal cluster of 32 points
    # A hexagon of radius r has 3r(r+1)+1 points
    # r=2: 19 points, r=3: 37 points
    # We want 32 points, so use hexagon of radius 3 (37 points) and remove 5

    # Build hexagon of radius 3
    hex3 = []
    for a in range(-3, 4):
        for b in range(-3, 4):
            if max(abs(a), abs(b), abs(a+b)) <= 3:
                hex3.append((a, b))

    # Remove 5 points from the outer ring to get 32 points
    outer_ring = [p for p in hex3 if max(abs(p[0]), abs(p[1]), abs(p[0]+p[1])) == 3]

    best_points = None
    best_edges = 0

    # Try all combinations of 5 points to remove from outer ring
    for rem_combo in itertools.combinations(outer_ring, 5):
        rem_set = set(rem_combo)
        remaining = [p for p in hex3 if p not in rem_set]
        ec = edge_count(remaining)
        if ec > best_edges:
            best_edges = ec
            best_points = remaining

    # If we already have 76, return immediately
    if best_edges == 76:
        return [list(p) for p in best_points]

    # Local search to try to reach 76
    def local_search(points, max_iter=50000):
        points = list(points)
        pset = set(points)
        current_edges = edge_count(points)
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            improved = False
            # Try moving each point to a nearby empty position
            for i in range(len(points)):
                p = points[i]
                # Try all positions in a small neighborhood
                for da in range(-2, 3):
                    for db in range(-2, 3):
                        if da == 0 and db == 0:
                            continue
                        new_p = (p[0]+da, p[1]+db)
                        if abs(new_p[0]) > 64 or abs(new_p[1]) > 64:
                            continue
                        if new_p in pset:
                            continue
                        # Check if new_p is at distance >= 1 from all other points
                        # (it's a lattice point, so distinct means distance >= 1)
                        
                        # Compute new edge count
                        old_edges_with_p = sum(1 for n in neighbors(p) if n in pset)
                        new_edges_with_p = sum(1 for n in neighbors(new_p) if n in pset)
                        
                        delta = new_edges_with_p - old_edges_with_p
                        if delta > 0:
                            # Accept the move
                            pset.remove(p)
                            points[i] = new_p
                            pset.add(new_p)
                            current_edges += delta
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
        return points, current_edges

    best_points, best_edges = local_search(best_points)

    if best_edges == 76:
        return [list(p) for p in best_points]

    # Simulated annealing
    import random
    import math

    def simulated_annealing(points, max_iter=200000):
        points = list(points)
        pset = set(points)
        current_edges = edge_count(points)
        best_points = list(points)
        best_edges = current_edges
        
        T = 2.0
        T_min = 0.01
        cooling = 0.99995
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            # Pick a random point
            i = random.randint(0, len(points)-1)
            p = points[i]
            
            # Try a random nearby position
            da = random.randint(-3, 3)
            db = random.randint(-3, 3)
            if da == 0 and db == 0:
                continue
            new_p = (p[0]+da, p[1]+db)
            if abs(new_p[0]) > 64 or abs(new_p[1]) > 64:
                continue
            if new_p in pset:
                continue
            
            # Compute delta
            old_edges_with_p = sum(1 for n in neighbors(p) if n in pset)
            new_edges_with_p = sum(1 for n in neighbors(new_p) if n in pset)
            delta = new_edges_with_p - old_edges_with_p
            
            if delta >= 0 or random.random() < math.exp(delta / T):
                pset.remove(p)
                points[i] = new_p
                pset.add(new_p)
                current_edges += delta
                
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_points = list(points)
                    if best_edges == 76:
                        return best_points, best_edges
            
            T = max(T * cooling, T_min)
        
        return best_points, best_edges

    best_points, best_edges = simulated_annealing(best_points)

    if best_edges == 76:
        return [list(p) for p in best_points]

    # Final local search polish
    best_points, best_edges = local_search(best_points, max_iter=100000)

    return [list(p) for p in best_points]
