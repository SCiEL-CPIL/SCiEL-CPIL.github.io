"""
Deterministic hexagonal spiral construction for 32 points on triangular lattice.
This approach directly constructs a configuration known to achieve the Harborth bound.
"""
import time

def _count_edges(points):
    """
    Count the number of pairs of points at unit distance on the triangular lattice.
    Points are represented as (a, b) in the basis 1, omega.
    Unit neighbors are at offsets: (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1).
    """
    point_set = set(points)
    neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]
    edge_count = 0
    for a, b in point_set:
        for da, db in neighbors:
            if (a + da, b + db) in point_set:
                edge_count += 1
    return edge_count // 2

def _generate_spiral(n):
    """
    Generate n points in a hexagonal spiral starting from (0,0).
    This creates a compact hexagonal shape which is optimal for edge density.
    """
    points = []
    if n == 0:
        return points
    
    points.append((0, 0))
    
    # Directions for the spiral: East, North-East, North, West, South-West, South
    # In axial coordinates (a,b):
    # (1,0) -> a+1
    # (0,1) -> b+1
    # (-1,1) -> a-1, b+1
    # (-1,0) -> a-1
    # (0,-1) -> b-1
    # (1,-1) -> a+1, b-1
    
    directions = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    
    ring = 1
    while len(points) < n:
        for i in range(6):
            da, db = directions[i]
            # The number of steps in this direction is 'ring'
            # For ring 1: 1 step
            # For ring 2: 2 steps
            # ...
            # Actually, the standard hex spiral grows by 1 step per side per ring.
            # Let's verify:
            # Ring 1: 6 points.
            # Ring 2: 12 points.
            # Total 1 + 6 + 12 = 19.
            # We need 32.
            # 1 + 6 + 12 + 18 = 37. So we will stop in ring 3.
            
            for _ in range(ring):
                if len(points) >= n:
                    break
                # Current position is points[-1]
                a, b = points[-1]
                new_a = a + da
                new_b = b + db
                points.append((new_a, new_b))
                if len(points) >= n:
                    break
        ring += 1
        
    return points[:n]

def _local_search(points, deadline):
    """
    Perform local search to maximize edges.
    Uses single-point swaps.
    """
    current_points = list(points)
    current_edges = _count_edges(current_points)
    
    if current_edges >= 76:
        return current_points, current_edges

    # Generate a candidate pool. 
    # The spiral for 32 points is within hex radius 3 (radius 3 has 37 points).
    # Let's use radius 4 (61 points) to be safe for swaps.
    candidates = []
    max_r = 4
    for a in range(-max_r, max_r + 1):
        for b in range(-max_r, max_r + 1):
            if max(abs(a), abs(b), abs(a+b)) <= max_r:
                candidates.append((a, b))
    
    candidate_set = set(candidates)
    neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_set = set(current_points)
        
        # Try to improve by swapping one point out and one in
        # We iterate over points in current set and candidates not in set
        # To make it efficient, we can just try all pairs (remove, add)
        # N=32, Candidates=61. 32*61 = ~2000 swaps. Very fast.
        
        best_gain = 0
        best_remove = None
        best_add = None
        
        for p_remove in current_points:
            # Calculate edges lost by removing p_remove
            # And gain by adding p_add
            # Simpler: just calculate new edge count for promising swaps?
            # No, full recalc is O(N*6) = 192 ops. 2000*192 = 384k ops. Trivial.
            
            for p_add in candidates:
                if p_add in current_set:
                    continue
                
                # Create new set
                new_set = current_set.copy()
                new_set.discard(p_remove)
                new_set.add(p_add)
                
                # Count edges
                e = 0
                for a, b in new_set:
                    for da, db in neighbors:
                        if (a+da, b+db) in new_set:
                            e += 1
                e //= 2
                
                if e > current_edges + best_gain:
                    best_gain = e - current_edges
                    best_remove = p_remove
                    best_add = p_add
        
        if best_remove is not None:
            # Apply best swap
            current_points.remove(best_remove)
            current_points.append(best_add)
            current_edges += best_gain
            improved = True
            
            if current_edges >= 76:
                break

    return current_points, current_edges

def construct():
    start_time = time.time()
    deadline = start_time + 10.0  # Generous margin, should finish in <1ms
    
    # 1. Generate initial spiral configuration
    # 32 points
    points = _generate_spiral(32)
    
    # 2. Verify and optimize with local search
    # The spiral should already be optimal or very close.
    # Local search ensures we hit 76 if possible.
    points, edges = _local_search(points, deadline)
    
    # 3. Final validation
    # Check distinctness and bounds
    assert len(points) == 32
    assert len(set(points)) == 32
    for a, b in points:
        assert abs(a) <= 64 and abs(b) <= 64
        
    return [[a, b] for a, b in points]
