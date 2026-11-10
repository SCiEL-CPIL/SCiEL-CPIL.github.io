"""Improved program: Hexagonal construction for 32 points on triangular lattice."""
# EVOLVE-BLOCK-START
import time
import math
import random

def _dist2(a1, b1, a2, b2):
    """Squared distance on triangular lattice: a^2 + a*b + b^2."""
    da = a1 - a2
    db = b1 - b2
    return da*da + da*db + db*db

def _count_edges(points):
    """Count unit-distance pairs among given (a,b) points."""
    point_set = set(tuple(p) for p in points)
    neighbors = [(1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)]
    edge_count = 0
    for a, b in point_set:
        for da, db in neighbors:
            if (a+da, b+db) in point_set:
                edge_count += 1
    return edge_count // 2

def _hexagon_points(r):
    """All points with hex distance <= r from origin: max(|a|,|b|,|a+b|) <= r."""
    pts = []
    for a in range(-r, r+1):
        for b in range(-r, r+1):
            if max(abs(a), abs(b), abs(a+b)) <= r:
                pts.append((a, b))
    return pts

def _ring_points(r):
    """All points with hex distance exactly r from origin."""
    return [p for p in _hexagon_points(r) if max(abs(p[0]), abs(p[1]), abs(p[0]+p[1])) == r]

def _local_search(points, max_time=1.0):
    """Improve edge count via local search (swap/add/remove)."""
    start_time = time.time()
    current_edges = _count_edges(points)
    current_points = list(points)
    
    # Candidate pool: Points near the current set
    # Generate candidates within radius 4 from origin or near current points
    candidates = set()
    for p in current_points:
        a, b = p
        for da in range(-2, 3):
            for db in range(-2, 3):
                candidates.add((a+da, b+db))
    
    # Filter candidates: must be distinct, within bounds
    valid_candidates = []
    for c in candidates:
        if max(abs(c[0]), abs(c[1])) <= 64:
            valid_candidates.append(c)
            
    best_points = current_points
    best_edges = current_edges
    
    while time.time() - start_time < max_time:
        improved = False
        # Try swapping a point with a candidate
        random.shuffle(valid_candidates)
        for i in range(len(current_points)):
            if time.time() - start_time > max_time:
                break
            old_p = current_points[i]
            for new_p in valid_candidates[:50]: # Limit candidates per iteration
                if new_p == old_p:
                    continue
                if new_p in current_points:
                    continue
                
                # Swap
                new_points = list(current_points)
                new_points[i] = new_p
                
                # Check distinctness
                if len(set(new_points)) != 32:
                    continue
                
                # Check bounds
                if any(max(abs(a), abs(b)) > 64 for a,b in new_points):
                    continue

                new_edges = _count_edges(new_points)
                if new_edges > best_edges:
                    best_edges = new_edges
                    best_points = new_points
                    current_points = new_points
                    improved = True
                    break
            if improved:
                break
                
        if not improved:
            # Try adding a point and removing a random one? 
            # Or just break if no swap improved it.
            # To avoid getting stuck, we can try a random restart of the local search
            # or just accept the current best.
            # Given the small size, a few restarts might help.
            # But we are likely at the optimum.
            break
            
    return best_points

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    # Target: 32 points
    # Base: Hexagon of radius 2 (19 points)
    base_points = _hexagon_points(2)
    
    # Additional points needed: 32 - 19 = 13
    # Source: Ring of radius 3 (18 points)
    ring3 = _ring_points(3)
    
    # Sort ring points by angle
    def angle(p):
        a, b = p
        x = a + b/2.0
        y = b * 0.8660254037844386
        return math.atan2(y, x)
        
    sorted_ring = sorted(ring3, key=angle)
    
    best_points = None
    best_edge_count = -1
    
    # Try all 18 starting positions for a contiguous block of 13
    for i in range(18):
        if time.time() > deadline:
            break
            
        selected_ring = []
        for j in range(13):
            selected_ring.append(sorted_ring[(i + j) % 18])
        
        candidate_points = base_points + selected_ring
        
        if len(set(candidate_points)) != 32:
            continue
            
        edges = _count_edges(candidate_points)
        if edges > best_edge_count:
            best_edge_count = edges
            best_points = candidate_points
            
    if best_points is None:
        pts = _hexagon_points(4)
        best_points = pts[:32]
        best_edge_count = _count_edges(best_points)
        
    # Local search polish
    if time.time() < deadline - 5.0:
        # Run local search with remaining time
        remaining_time = deadline - 5.0 - time.time()
        improved_points = _local_search(best_points, max_time=max(0.1, remaining_time))
        improved_edges = _count_edges(improved_points)
        if improved_edges > best_edge_count:
            best_points = improved_points
            best_edge_count = improved_edges
            
    return [list(p) for p in best_points]
# EVOLVE-BLOCK-END
