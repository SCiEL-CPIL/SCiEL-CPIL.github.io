"""Improved program: hexagonal construction for 32 points on triangular lattice."""
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
            if abs(a+b) <= r:
                pts.append((a, b))
    return pts

def _hex_ring(r):
    """Points at exactly hex distance r."""
    if r == 0:
        return [(0, 0)]
    pts = []
    # Six sides of the hexagon
    for i in range(6):
        if i == 0:
            for a in range(0, r+1):
                pts.append((a, -a))
        elif i == 1:
            for b in range(-r, 0):
                pts.append((r, b))
        elif i == 2:
            for a in range(r, 0, -1):
                pts.append((a, -a + r - r))  # (a, r-a) but a goes from r to 1, b = r-a... wait
        # Let me just use the difference method
    # Simpler: get all points with hex_dist <= r, remove those with hex_dist <= r-1
    inner = set(_hexagon_points(r-1)) if r > 1 else set()
    outer = set(_hexagon_points(r))
    return [p for p in outer if p not in inner]

def _hex_dist(a, b):
    return max(abs(a), abs(b), abs(a+b))

def _build_hexagonal_32():
    """Build 32 points in a hexagonal arrangement centered at origin."""
    # Full hexagon r=2 has 19 points. Need 13 more from r=3 ring.
    # The r=3 ring has 18 points. We pick 13 that maximize edges.
    # Strategy: pick points from the ring that are adjacent to existing points.
    
    core = _hexagon_points(2)
    ring3 = _hex_ring(3)
    
    # Sort ring3 points by how many neighbors they have in core
    def neighbor_count_in(p, point_list):
        ps = set(point_list)
        a, b = p
        neighbors = [(1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)]
        return sum(1 for da, db in neighbors if (a+da, b+db) in ps)
    
    ring3_sorted = sorted(ring3, key=lambda p: neighbor_count_in(p, core), reverse=True)
    
    selected = list(core)
    for p in ring3_sorted:
        if len(selected) >= 32:
            break
        selected.append(p)
    
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 120
    
    # Build hexagonal arrangement
    points = _build_hexagonal_32()
    
    # Verify we have exactly 32 distinct points
    assert len(points) == 32, f"Expected 32 points, got {len(points)}"
    assert len(set(points)) == 32, "Points are not distinct"
    
    # Verify bounds
    for a, b in points:
        assert abs(a) <= 64 and abs(b) <= 64, f"Point ({a},{b}) out of bounds"
    
    # Count edges for verification
    edges = _count_edges(points)
    
    # Check deadline
    if time.time() > deadline - 5:
        return [[a, b] for a, b in points]
    
    return [[a, b] for a, b in points]
# EVOLVE-BLOCK-END
