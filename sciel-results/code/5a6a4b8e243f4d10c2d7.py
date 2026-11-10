"""Improved program: Hexagonal construction for 32 points on triangular lattice."""
# EVOLVE-BLOCK-START
import time

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

def construct():
    # Target: 32 points
    # Base: Hexagon of radius 2 (19 points)
    base_points = _hexagon_points(2)
    
    # Additional points needed: 32 - 19 = 13
    # Source: Ring of radius 3 (18 points)
    ring3 = _ring_points(3)
    
    # We need to select 13 contiguous points from the ring to maximize edges.
    # The ring is a cycle of 18 points. We can rotate the selection window.
    best_points = None
    best_edge_count = -1
    
    # Sort ring points to ensure they are in cyclic order for contiguous selection
    # The points in _ring_points are not necessarily in cyclic order.
    # Let's order them by angle or simply by their position in the hex grid.
    # A simple way to order them is to follow the perimeter.
    # However, for a small set, we can just try all subsets of size 13? No, 18 choose 13 is 8568, which is fine.
    # But contiguous is better. Let's order them.
    
    # To order them, we can compute the angle.
    import math
    def angle(p):
        a, b = p
        # Convert to Cartesian: x = a + b/2, y = b*sqrt(3)/2
        x = a + b/2.0
        y = b * 0.8660254037844386
        return math.atan2(y, x)
    
    sorted_ring = sorted(ring3, key=angle)
    
    # Try all 18 starting positions for a contiguous block of 13
    for i in range(18):
        selected_ring = []
        for j in range(13):
            selected_ring.append(sorted_ring[(i + j) % 18])
        
        candidate_points = base_points + selected_ring
        
        # Verify distinctness
        if len(set(candidate_points)) != 32:
            continue
            
        edges = _count_edges(candidate_points)
        if edges > best_edge_count:
            best_edge_count = edges
            best_points = candidate_points
            
    # Fallback if something went wrong (should not happen)
    if best_points is None:
        # Just take the first 32 points from a large hexagon
        pts = _hexagon_points(4)
        best_points = pts[:32]
        
    return [list(p) for p in best_points]
# EVOLVE-BLOCK-END
