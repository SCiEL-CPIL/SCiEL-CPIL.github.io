"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    """
    Return 32 distinct integer pairs [a, b] on the triangular lattice
    (point = a + b*omega, omega = exp(i*pi/3)) with |a|,|b| <= 64,
    mutual distance >= 1, maximizing the number of pairs at distance exactly 1.
    
    Uses a hexagonal cluster approach: generate a compact hexagonal region,
    then prune low-degree boundary points to reach exactly 32 points while
    maximizing edges.
    """
    start_time = time.time()
    
    # Unit distance vectors on triangular lattice in (a,b) coordinates
    # Point p = a + b*omega where omega = exp(i*pi/3) = 1/2 + i*sqrt(3)/2
    # |p1 - p2|^2 = (da)^2 + da*db + (db)^2
    # Unit distance: da^2 + da*db + db^2 = 1
    # Solutions: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    unit_vectors = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    
    def dist_sq(p1, p2):
        da = p1[0] - p2[0]
        db = p1[1] - p2[1]
        return da*da + da*db + db*db
    
    def count_edges(points):
        """Count pairs at distance exactly 1."""
        pts = set(tuple(p) for p in points)
        edges = 0
        for p in pts:
            for u in unit_vectors:
                q = (p[0] + u[0], p[1] + u[1])
                if q in pts:
                    edges += 1
        return edges // 2
    
    def degree(p, points_set):
        """Count neighbors at unit distance."""
        d = 0
        for u in unit_vectors:
            if (p[0] + u[0], p[1] + u[1]) in points_set:
                d += 1
        return d
    
    # Generate hexagonal cluster of radius r
    # A hexagon of radius r has 1 + 3r(r+1) points
    # r=1: 7, r=2: 19, r=3: 37
    # We need 32, so start with r=3 (37 points) and remove 5
    
    def hex_points(radius):
        """Generate hexagonal cluster of given radius centered at origin."""
        pts = set()
        for a in range(-radius, radius + 1):
            for b in range(-radius, radius + 1):
                # Hexagonal distance: max(|a|, |b|, |a+b|) <= radius
                # In axial coords, hex distance is (|a| + |b| + |a+b|) / 2
                c = -(a + b)
                hex_dist = max(abs(a), abs(b), abs(c))
                if hex_dist <= radius:
                    pts.add((a, b))
        return pts
    
    # Start with radius 3 hexagon (37 points)
    hex3 = hex_points(3)
    
    # We need to remove 5 points to get 32
    # Remove points with lowest degree (boundary points)
    # Sort by degree, remove lowest degree points
    
    def prune_to_n(pts, n):
        """Prune points to exactly n, removing lowest degree points first."""
        pts = list(pts)
        pts_set = set(pts)
        
        while len(pts_set) > n:
            # Compute degrees
            degrees = {}
            for p in pts_set:
                degrees[p] = degree(p, pts_set)
            
            # Find point with minimum degree (break ties by hex distance from center)
            min_deg = min(degrees.values())
            candidates = [p for p in pts_set if degrees[p] == min_deg]
            
            # Among candidates, pick the one farthest from center
            best_to_remove = max(candidates, key=lambda p: abs(p[0]) + abs(p[1]) + abs(p[0]+p[1]))
            pts_set.remove(best_to_remove)
        
        return pts_set
    
    # Try pruning from radius 3
    pruned32 = prune_to_n(hex3, 32)
    edges32 = count_edges(pruned32)
    
    # Also try other approaches: radius 2 hexagon (19 pts) + extend
    hex2 = hex_points(2)
    
    # Try different pruning strategies
    best_points = pruned32
    best_edges = edges32
    
    # Try removing different sets of 5 from hex3
    # Strategy: remove points in order of increasing degree
    def prune_by_degree_order(pts, n):
        """Remove points in order of increasing degree."""
        pts = set(pts)
        while len(pts) > n:
            degrees = {p: degree(p, pts) for p in pts}
            min_deg = min(degrees.values())
            candidates = [p for p in pts if degrees[p] == min_deg]
            # Remove one candidate (choose by symmetry breaking)
            to_remove = min(candidates, key=lambda p: (abs(p[0]) + abs(p[1]), p[0], p[1]))
            pts.remove(to_remove)
        return pts
    
    pruned32b = prune_by_degree_order(hex3, 32)
    edges32b = count_edges(pruned32b)
    if edges32b > best_edges:
        best_edges = edges32b
        best_points = pruned32b
    
    # Try a different approach: use a rhombus/parallelogram shape
    # 8x4 parallelogram has 32 points
    def parallelogram(n1, n2):
        """Generate a parallelogram of n1 x n2 points."""
        pts = set()
        for i in range(n1):
            for j in range(n2):
                pts.add((i, j))
        return pts
    
    # 8x4 = 32 points
    para32 = parallelogram(8, 4)
    edges_para = count_edges(para32)
    if edges_para > best_edges:
        best_edges = edges_para
        best_points = para32
    
    # 7x5 = 35, need to prune 3
    para35 = parallelogram(7, 5)
    pruned35_32 = prune_by_degree_order(para35, 32)
    edges_35_32 = count_edges(pruned35_32)
    if edges_35_32 > best_edges:
        best_edges = edges_35_32
        best_points = pruned35_32
    
    # Try 6x6 = 36, prune 4
    para36 = parallelogram(6, 6)
    pruned36_32 = prune_by_degree_order(para36, 32)
    edges_36_32 = count_edges(pruned36_32)
    if edges_36_32 > best_edges:
        best_edges = edges_36_32
        best_points = pruned36_32
    
    # Try hexagonal spiral: points in order of increasing hex distance
    def hex_spiral(n):
        """Generate n points in hexagonal spiral order."""
        pts = []
        # Center
        pts.append((0, 0))
        # Rings
        ring = 1
        while len(pts) < n:
            # Ring of hex distance = ring has 6*ring points
            # Start at (ring, 0) and go around
            # Directions: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
            pos = (ring, 0)
            for side in range(6):
                for step in range(ring):
                    if len(pts) < n:
                        pts.append(pos)
                    # Move in direction of this side
                    dirs = [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
                    d = dirs[side]
                    pos = (pos[0] + d[0], pos[1] + d[1])
            ring += 1
        return set(pts[:n])
    
    spiral32 = hex_spiral(32)
    edges_spiral = count_edges(spiral32)
    if edges_spiral > best_edges:
        best_edges = edges_spiral
        best_points = spiral32
    
    # Local search: try swapping points to improve edge count
    def local_search(points, max_iters=1000):
        """Try to improve edge count via local moves."""
        pts = set(points)
        current_edges = count_edges(pts)
        
        # Generate candidate positions: nearby lattice points
        for _ in range(max_iters):
            if time.time() - start_time > 140:
                break
            
            improved = False
            for p in list(pts):
                # Try moving p to a nearby empty position
                for da in range(-3, 4):
                    for db in range(-3, 4):
                        if da == 0 and db == 0:
                            continue
                        new_pos = (p[0] + da, p[1] + db)
                        if new_pos in pts:
                            continue
                        # Check bounds
                        if abs(new_pos[0]) > 64 or abs(new_pos[1]) > 64:
                            continue
                        
                        # Try the move
                        pts.remove(p)
                        pts.add(new_pos)
                        new_edges = count_edges(pts)
                        if new_edges > current_edges:
                            current_edges = new_edges
                            improved = True
                            break
                        else:
                            pts.remove(new_pos)
                            pts.add(p)
                    if improved:
                        break
                if improved:
                    break
            
            if not improved:
                break
        
        return pts, current_edges
    
    best_points_ls, best_edges_ls = local_search(best_points, max_iters=500)
    if best_edges_ls > best_edges:
        best_edges = best_edges_ls
        best_points = best_points_ls
    
    # Final check: ensure all constraints are met
    result = sorted(best_points, key=lambda p: (p[1], p[0]))
    
    # Verify constraints
    assert len(result) == 32, f"Expected 32 points, got {len(result)}"
    for p in result:
        assert abs(p[0]) <= 64 and abs(p[1]) <= 64, f"Point {p} out of bounds"
    
    # Check mutual distance >= 1
    for i in range(len(result)):
        for j in range(i+1, len(result)):
            d2 = dist_sq(result[i], result[j])
            assert d2 >= 1, f"Points {result[i]} and {result[j]} too close: d^2={d2}"
    
    # Count final edges
    final_edges = count_edges(result)
    
    return [[p[0], p[1]] for p in result]
# EVOLVE-BLOCK-END
