"""
Combines the structural robustness of the current program with the exact 
subgraph extraction strategy from the inspiration.

Strategy:
1. Define a hexagonal region of radius 3 on the triangular lattice.
   This region contains 37 points.
2. We need to select 32 points, which is equivalent to removing 5 points.
3. Enumerate all combinations of 5 points to remove from the 37.
   For each combination, calculate the number of unit-distance edges in the 
   remaining 32 points.
4. Keep the configuration with the maximum number of edges.
5. The Harborth bound for n=32 is 76. It is expected that the optimal 
   32-point subset of the 37-point hexagon achieves this bound.
6. If for any reason the enumeration does not find 76 edges (e.g., if the 
   global optimum lies outside this hexagon, which is unlikely), fall back 
   to a heuristic local search as in the original program.
"""
# EVOLVE-BLOCK-START
import time
import math
import random
from itertools import combinations

def _count_edges(points):
    """Count the number of unit-distance pairs among given points on triangular lattice."""
    point_set = set(points)
    # All 6 neighbors in triangular lattice coordinates
    neighbors = [(1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)]
    edge_count = 0
    for a, b in points:
        for da, db in neighbors:
            if (a+da, b+db) in point_set:
                edge_count += 1
    return edge_count // 2

def _harborth_bound(n):
    """Return the Harborth bound floor(3n - sqrt(12n - 3))."""
    return int(math.floor(3*n - math.sqrt(12*n - 3)))

def _build_hexagon(R):
    """
    Build the list of points in a hexagon of radius R on the triangular lattice.
    Points are (a, b) such that |a| <= R, |b| <= R, |a+b| <= R.
    """
    points = []
    for a in range(-R, R+1):
        for b in range(-R, R+1):
            if abs(a+b) <= R:
                points.append((a, b))
    return points

def _exact_hexagon_search(R=3, n_points=32):
    """
    Find the subset of n_points from the hexagon of radius R that maximizes edges.
    Uses enumeration of removals if (total_points - n_points) is small.
    """
    hex_points = _build_hexagon(R)
    total_points = len(hex_points)
    
    if total_points < n_points:
        return None, 0
    
    n_remove = total_points - n_points
    
    # If n_remove is small, enumerate all removals
    if n_remove <= 10:
        # Map point to index for fast lookup
        point_to_idx = {p: i for i, p in enumerate(hex_points)}
        
        # Build adjacency list
        adj = [set() for _ in range(total_points)]
        neighbors = [(1,0), (0,1), (1,-1)] # Only forward directions to avoid double counting in edge list, but we need full adjacency for removal calc
        
        # Actually, for removal calc, we just need to know which edges are lost.
        # Let's precompute the full set of edges in the hexagon.
        edge_set = set()
        all_neighbors = [(1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)]
        
        for i, (a, b) in enumerate(hex_points):
            for da, db in all_neighbors:
                neighbor = (a+da, b+db)
                if neighbor in point_to_idx:
                    j = point_to_idx[neighbor]
                    if i < j:
                        edge_set.add((i, j))
        
        total_edges = len(edge_set)
        
        best_removed = None
        max_edges = -1
        
        # Enumerate all combinations of n_remove indices to remove
        # To speed up, we can prune if possible, but 435k is small.
        for combo in combinations(range(total_points), n_remove):
            removed_set = set(combo)
            
            # Count edges lost
            # An edge is lost if at least one endpoint is in removed_set
            # edges_lost = total_edges - edges_remaining
            # edges_remaining = edges where neither endpoint is in removed_set
            
            edges_remaining = 0
            for (u, v) in edge_set:
                if u not in removed_set and v not in removed_set:
                    edges_remaining += 1
            
            if edges_remaining > max_edges:
                max_edges = edges_remaining
                best_removed = removed_set
                
                # Early exit if we hit the Harborth bound
                if max_edges >= _harborth_bound(n_points):
                    break
        
        if best_removed is not None:
            # Construct the final list of points
            final_points = [p for i, p in enumerate(hex_points) if i not in best_removed]
            return final_points, max_edges
        else:
            return None, 0
    else:
        return None, 0

def _local_search(points, max_iters=5000, time_limit=None):
    """Local search: try moving a point to a nearby unused position."""
    start_time = time.time()
    n = len(points)
    point_set = set(points)
    best = points[:]
    best_edges = _count_edges(points)
    current = points[:]
    current_edges = best_edges
    temp = 1.0
    cooling = 0.999
    
    neighbors = [(1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)]
    
    for _ in range(max_iters):
        if time_limit is not None and time.time() > time_limit:
            break
        idx = random.randint(0, n-1)
        old_point = current[idx]
        
        moved = False
        for _attempt in range(20):
            da = random.randint(-5, 5)
            db = random.randint(-5, 5)
            new_point = (old_point[0] + da, old_point[1] + db)
            
            if new_point in point_set and new_point != old_point:
                continue
            
            lost_edges = 0
            for na, nb in neighbors:
                if (old_point[0]+na, old_point[1]+nb) in point_set:
                    lost_edges += 1
            
            gained_edges = 0
            for na, nb in neighbors:
                np_ = (new_point[0]+na, new_point[1]+nb)
                if np_ == old_point:
                    continue
                if np_ in point_set:
                    gained_edges += 1
            
            delta = gained_edges - lost_edges
            if delta > 0:
                point_set.remove(old_point)
                point_set.add(new_point)
                current[idx] = new_point
                current_edges += delta
                if current_edges > best_edges:
                    best_edges = current_edges
                    best = current[:]
                moved = True
                break
            elif delta == 0:
                if random.random() < 0.5:
                    point_set.remove(old_point)
                    point_set.add(new_point)
                    current[idx] = new_point
                    moved = True
                    break
            else:
                if delta < -10:
                    continue
                prob = math.exp(delta / max(temp, 0.001))
                if random.random() < prob:
                    point_set.remove(old_point)
                    point_set.add(new_point)
                    current[idx] = new_point
                    current_edges += delta
                    moved = True
                    break
        
        temp *= cooling
    
    return best, best_edges

def construct():
    """Construct 32 distinct lattice points maximizing unit-distance pairs."""
    deadline = time.time() + 115  # Safety margin for 120s limit
    
    n = 32
    target_bound = _harborth_bound(n)  # 76
    
    # Strategy 1: Exact search in Hexagon R=3 (37 points, remove 5)
    # This is very fast and likely optimal.
    best_points, best_edges = _exact_hexagon_search(R=3, n_points=n)
    
    if best_points is not None:
        # Validate
        if len(best_points) == n and all(abs(a) <= 64 and abs(b) <= 64 for a, b in best_points):
            if best_edges >= target_bound:
                return [[a, b] for a, b in best_points]
            
            # If we found a good but not optimal solution, try to improve with local search
            # But given the nature of the problem, if it's not 76, it might be hard to reach 76
            # via local search from here. However, let's try.
            if time.time() < deadline - 10:
                improved_points, improved_edges = _local_search(
                    best_points, 
                    max_iters=2000, 
                    time_limit=time.time() + 10
                )
                if improved_edges > best_edges:
                    best_edges = improved_edges
                    best_points = improved_points
                
                if best_edges >= target_bound:
                    return [[a, b] for a, b in best_points]
                
                # Return the best found so far
                return [[a, b] for a, b in best_points]
    
    # Strategy 2: Fallback to heuristic search if exact search failed or wasn't run
    # This is unlikely to be needed, but provides robustness.
    
    # Try a few rectangular strips and random configs
    candidates = []
    # 16x2 strip
    candidates.append([(a, b) for a in range(16) for b in range(2)])
    # 8x4 strip
    candidates.append([(a, b) for a in range(8) for b in range(4)])
    
    random.seed(42)
    for _ in range(5):
        pts = set()
        while len(pts) < n:
            a = random.randint(-10, 10)
            b = random.randint(-10, 10)
            pts.add((a, b))
        candidates.append(list(pts))
    
    best_points = None
    best_edges = -1
    
    for candidate in candidates:
        if time.time() > deadline - 5:
            break
        
        if len(candidate) != n:
            continue
            
        edges = _count_edges(candidate)
        if edges > best_edges:
            best_edges = edges
            best_points = candidate[:]
            
        if best_edges >= target_bound:
            break
            
        # Local search
        remaining = deadline - time.time() - 2
        if remaining > 1:
            improved_points, improved_edges = _local_search(
                candidate, 
                max_iters=1000, 
                time_limit=time.time() + max(1, remaining/2)
            )
            if improved_edges > best_edges:
                best_edges = improved_edges
                best_points = improved_points[:]
                
            if best_edges >= target_bound:
                break
    
    if best_points is None:
        # Absolute fallback
        best_points = [(a, b) for a in range(16) for b in range(2)]
    
    return [[a, b] for a, b in best_points]
# EVOLVE-BLOCK-END
