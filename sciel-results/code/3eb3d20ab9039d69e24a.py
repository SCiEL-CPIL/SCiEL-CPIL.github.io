"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

# EVOLVE-BLOCK-START
def construct():
    _t0 = time.time()
    _DEADLINE = _t0 + 140.0  # safety margin before 150s wall-clock kill
    
    N = 32
    # The six neighbors in (a,b) coordinates for the triangular lattice
    # omega = exp(i*pi/3) = (1/2, sqrt(3)/2)
    # Point = a + b*omega = (a + b/2, b*sqrt(3)/2)
    # Distance squared between (a1,b1) and (a2,b2):
    # dx = a1-a2 + (b1-b2)/2, dy = (b1-b2)*sqrt(3)/2
    # d^2 = (da + db/2)^2 + 3*db^2/4 = da^2 + da*db + db^2
    # So distance is 1 iff da^2 + da*db + db^2 = 1
    # The six unit-distance neighbors:
    NEIGHBORS = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]
    
    def dist_sq(a1, b1, a2, b2):
        da = a1 - a2
        db = b1 - b2
        return da*da + da*db + db*db
    
    def count_edges(points):
        """Count pairs at distance exactly 1."""
        edges = 0
        n = len(points)
        for i in range(n):
            a1, b1 = points[i]
            for j in range(i+1, n):
                a2, b2 = points[j]
                if dist_sq(a1, b1, a2, b2) == 1:
                    edges += 1
        return edges
    
    def is_valid(points):
        """Check all points are distinct, within bounds, and at mutual distance >= 1."""
        n = len(points)
        if n != N:
            return False
        seen = set()
        for a, b in points:
            if abs(a) > 64 or abs(b) > 64:
                return False
            if (a, b) in seen:
                return False
            seen.add((a, b))
        for i in range(n):
            a1, b1 = points[i]
            for j in range(i+1, n):
                a2, b2 = points[j]
                if dist_sq(a1, b1, a2, b2) < 1:
                    return False
        return True
    
    def hex_spiral_order(center_a, center_b, radius):
        """Generate lattice points in hexagonal spiral order around center."""
        if radius == 0:
            return [(center_a, center_b)]
        points = []
        # The hexagonal ring at distance k from center
        # For triangular lattice, ring of "radius" r has 6*r points
        # We go out to radius where we have enough points
        points.append((center_a, center_b))
        for r in range(1, radius+1):
            # 6 directions, r steps each
            for direction in range(6):
                for step in range(r):
                    # Move in direction 'direction' by 'step'
                    # Directions: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
                    dirs = [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
                    da, db = dirs[direction]
                    # Actually, we need to generate the ring properly
                    # Let me use a different approach
                    pass
            # Better: generate ring points
            # Ring r consists of 6r points
            # Starting from (center_a + r, center_b) and going around
            start_a = center_a + r
            start_b = center_b
            for i in range(6*r):
                # Move along the ring
                if i < r:
                    a = center_a + r - i
                    b = center_b + i
                elif i < 2*r:
                    a = center_a
                    b = center_b + r + (i - r)
                elif i < 3*r:
                    a = center_a - (i - 2*r)
                    b = center_b + r - (i - 2*r)
                elif i < 4*r:
                    a = center_a - r
                    b = center_b + r - (i - 3*r)
                elif i < 5*r:
                    a = center_a - r + (i - 4*r)
                    b = center_b - (i - 4*r)
                else:
                    a = center_a + (i - 5*r)
                    b = center_b - r
                points.append((a, b))
        return points
    
    def greedy_construct(start_a, start_b, seed):
        """Greedy: add points one at a time, choosing the one that maximizes new edges."""
        rng = random.Random(seed)
        points = []
        
        # Generate candidate points in spiral order from start
        candidates = hex_spiral_order(start_a, start_b, 12)
        
        # Add center first
        points.append((start_a, start_b))
        used = set([(start_a, start_b)])
        
        while len(points) < N:
            best_point = None
            best_edges = -1
            # Try candidates not yet used
            for (ca, cb) in candidates:
                if (ca, cb) in used:
                    continue
                if abs(ca) > 64 or abs(cb) > 64:
                    continue
                # Check distance >= 1 from all existing points
                valid = True
                new_edges = 0
                for (pa, pb) in points:
                    d2 = dist_sq(ca, cb, pa, pb)
                    if d2 < 1:
                        valid = False
                        break
                    if d2 == 1:
                        new_edges += 1
                if valid and new_edges > best_edges:
                    best_edges = new_edges
                    best_point = (ca, cb)
                    if best_edges >= 6:
                        break
            if best_point is None:
                # Extend spiral
                candidates.extend(hex_spiral_order(start_a, start_b, len(candidates)//6 + 1))
                # Remove duplicates
                seen = set()
                new_candidates = []
                for p in candidates:
                    if p not in seen:
                        seen.add(p)
                        new_candidates.append(p)
                candidates = new_candidates
                continue
            points.append(best_point)
            used.add(best_point)
        
        return points
    
    def local_search(points, max_iter=500, time_limit=5.0):
        """Local search: try to improve edge count by moving points."""
        points = list(points)
        current_edges = count_edges(points)
        best_points = list(points)
        best_edges = current_edges
        _ts = time.time()
        
        for iteration in range(max_iter):
            if time.time() - _ts > time_limit:
                break
            if time.time() > _DEADLINE - 5.0:
                break
            
            improved = False
            # Try moving each point to a nearby unoccupied location
            for i in range(N):
                if time.time() > _DEADLINE - 5.0:
                    break
                a, b = points[i]
                # Try all 6 neighbors and some further positions
                candidates = []
                for r in range(1, 5):
                    for da in range(-r, r+1):
                        for db in range(-r, r+1):
                            if da == 0 and db == 0:
                                continue
                            if da*da + da*db + db*db != r*r:
                                continue
                            candidates.append((a+da, b+db))
                
                for (na, nb) in candidates:
                    if abs(na) > 64 or abs(nb) > 64:
                        continue
                    if (na, nb) in points:
                        continue
                    # Check validity
                    valid = True
                    new_edges = 0
                    for j in range(N):
                        if j == i:
                            continue
                        pa, pb = points[j]
                        d2 = dist_sq(na, nb, pa, pb)
                        if d2 < 1:
                            valid = False
                            break
                        if d2 == 1:
                            new_edges += 1
                    if not valid:
                        continue
                    # Calculate total edges change
                    # Remove edges from old position, add edges from new
                    old_edges = 0
                    for j in range(N):
                        if j == i:
                            continue
                        pa, pb = points[j]
                        if dist_sq(a, b, pa, pb) == 1:
                            old_edges += 1
                    delta = new_edges - old_edges
                    if delta > 0:
                        points[i] = (na, nb)
                        current_edges += delta
                        improved = True
                        if current_edges > best_edges:
                            best_edges = current_edges
                            best_points = list(points)
                        break  # Restart scan after improvement
        
            if not improved:
                break
        
        return best_points, best_edges
    
    def simulated_annealing(points, time_limit=10.0):
        """Simulated annealing to escape local optima."""
        import math as _math
        points = list(points)
        current_edges = count_edges(points)
        best_points = list(points)
        best_edges = current_edges
        _ts = time.time()
        
        T = 5.0
        cooling = 0.9995
        
        while time.time() - _ts < time_limit:
            if time.time() > _DEADLINE - 5.0:
                break
            
            i = random.randint(0, N-1)
            a, b = points[i]
            
            # Pick a random nearby position
            r = random.randint(1, 8)
            angle = random.random() * 2 * _math.pi
            da = round(r * _math.cos(angle))
            db = round(r * _math.sin(angle))
            na, nb = a + da, b + db
            
            if abs(na) > 64 or abs(nb) > 64:
                continue
            if (na, nb) in points:
                continue
            
            # Check validity
            valid = True
            new_edges = 0
            for j in range(N):
                if j == i:
                    continue
                pa, pb = points[j]
                d2 = dist_sq(na, nb, pa, pb)
                if d2 < 1:
                    valid = False
                    break
                if d2 == 1:
                    new_edges += 1
            if not valid:
                continue
            
            old_edges = 0
            for j in range(N):
                if j == i:
                    continue
                pa, pb = points[j]
                if dist_sq(a, b, pa, pb) == 1:
                    old_edges += 1
            
            delta = new_edges - old_edges
            
            if delta >= 0 or random.random() < _math.exp(delta / T):
                points[i] = (na, nb)
                current_edges += delta
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_points = list(points)
            
            T *= cooling
            if T < 0.01:
                T = 0.01
        
        return best_points, best_edges
    
    # Main search loop
    best_overall = None
    best_overall_edges = 0
    
    # Phase 1: Multiple greedy starts with different centers
    centers = [(0, 0), (1, 0), (0, 1), (1, 1), (-1, 0), (0, -1), (2, 0), (0, 2)]
    seed_base = 42
    
    for ci, (ca, cb) in enumerate(centers):
        if time.time() > _DEADLINE - 20.0:
            break
        for attempt in range(3):
            if time.time() > _DEADLINE - 20.0:
                break
            seed = seed_base + ci * 100 + attempt
            points = greedy_construct(ca, cb, seed)
            if not is_valid(points):
                continue
            edges = count_edges(points)
            
            # Local search
            points, edges = local_search(points, max_iter=300, time_limit=3.0)
            
            if edges > best_overall_edges:
                best_overall_edges = edges
                best_overall = points
            
            if best_overall_edges >= 76:
                return [[a, b] for a, b in best_overall]
    
    # Phase 2: Simulated annealing from best result
    if best_overall is not None:
        if time.time() < _DEADLINE - 15.0:
            points, edges = simulated_annealing(best_overall, time_limit=10.0)
            if edges > best_overall_edges:
                best_overall_edges = edges
                best_overall = points
                if best_overall_edges >= 76:
                    return [[a, b] for a, b in best_overall]
        
        # Another round of local search
        if time.time() < _DEADLINE - 10.0:
            points, edges = local_search(best_overall, max_iter=500, time_limit=5.0)
            if edges > best_overall_edges:
                best_overall_edges = edges
                best_overall = points
    
    # Phase 3: More diverse starts if time permits
    if time.time() < _DEADLINE - 15.0:
        for extra_seed in range(20):
            if time.time() > _DEADLINE - 10.0:
                break
            ca = random.randint(-5, 5)
            cb = random.randint(-5, 5)
            points = greedy_construct(ca, cb, 1000 + extra_seed)
            if not is_valid(points):
                continue
            edges = count_edges(points)
            points, edges = local_search(points, max_iter=200, time_limit=2.0)
            if edges > best_overall_edges:
                best_overall_edges = edges
                best_overall = points
                if best_overall_edges >= 76:
                    return [[a, b] for a, b in best_overall]
    
    if best_overall is None:
        # Fallback: simple strip
        return [[a, b] for a in range(16) for b in range(2)]
    
    return [[a, b] for a, b in best_overall]
# EVOLVE-BLOCK-END
