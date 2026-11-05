import time
import random

# EVOLVE-BLOCK-START
def construct():
    """
    Find 32 lattice points on the triangular lattice maximizing unit-distance pairs.
    Target: 76 edges (Harborth's theorem optimum for n=32).
    """
    deadline = time.time() + 140
    
    # Distance squared in axial coordinates for triangular lattice
    # omega = exp(i*pi/3), so a + b*omega
    # |a + b*omega|^2 = a^2 + b^2 + a*b
    def dist_sq(p, q):
        da = p[0] - q[0]
        db = p[1] - q[1]
        return da*da + db*db + da*db
    
    def count_edges(points):
        """Count pairs at distance exactly 1."""
        count = 0
        n = len(points)
        for i in range(n):
            pi = points[i]
            for j in range(i+1, n):
                pj = points[j]
                if dist_sq(pi, pj) == 1:
                    count += 1
        return count
    
    def is_valid(points):
        """Check all pairs have distance >= 1 (squared dist >= 1)."""
        n = len(points)
        for i in range(n):
            pi = points[i]
            for j in range(i+1, n):
                pj = points[j]
                if dist_sq(pi, pj) < 1:
                    return False
        return True
    
    # Generate a hexagonal cluster of radius R (axial coords: |a|<=R, |b|<=R, |a+b|<=R)
    # For R=3, this gives 37 points. We need 32, so we'll trim.
    # For R=4, this gives 61 points.
    
    def hex_points(R):
        pts = []
        for a in range(-R, R+1):
            for b in range(-R, R+1):
                if abs(a+b) <= R:
                    pts.append([a, b])
        return pts
    
    # Start with a good initial configuration:
    # Use a hexagonal cluster and randomly remove points to get 32,
    # then do local search.
    
    best_points = None
    best_edges = -1
    
    # Try multiple random initializations from hex clusters
    rng = random.Random(42)
    
    # Strategy 1: Start with hex radius 3 (37 pts), remove 5 randomly
    # Strategy 2: Start with hex radius 4 (61 pts), remove 29 randomly
    # Strategy 3: Random points in a bounded region
    
    for attempt in range(50):
        if time.time() > deadline - 5:
            break
        
        if attempt < 20:
            # Hex radius 3, remove 5
            pts = hex_points(3)
            rng.shuffle(pts)
            pts = pts[:32]
        elif attempt < 40:
            # Hex radius 4, remove 29
            pts = hex_points(4)
            rng.shuffle(pts)
            pts = pts[:32]
        else:
            # Random points in a larger region
            pts = []
            while len(pts) < 32:
                a = rng.randint(-10, 10)
                b = rng.randint(-10, 10)
                if [a, b] not in pts:
                    pts.append([a, b])
        
        # Local search: single point moves
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            edges_before = count_edges(pts)
            
            for i in range(32):
                if time.time() > deadline - 5:
                    break
                orig = pts[i][:]
                # Try moving point i to nearby positions
                for da in range(-3, 4):
                    for db in range(-3, 4):
                        if da == 0 and db == 0:
                            continue
                        new_pt = [orig[0]+da, orig[1]+db]
                        pts[i] = new_pt
                        
                        if not is_valid(pts):
                            pts[i] = orig
                            continue
                        
                        edges_after = count_edges(pts)
                        if edges_after > edges_before:
                            edges_before = edges_after
                            improved = True
                        else:
                            pts[i] = orig
        
        edges = count_edges(pts)
        if edges > best_edges:
            best_edges = edges
            best_points = [p[:] for p in pts]
        
        if best_edges >= 76:
            break
    
    # Additional refinement: try swap moves on the best solution
    if best_points is not None and best_edges < 76:
        pts = best_points[:]
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            edges_before = count_edges(pts)
            
            for i in range(32):
                if time.time() > deadline - 3:
                    break
                for j in range(i+1, 32):
                    if time.time() > deadline - 3:
                        break
                    orig_i = pts[i][:]
                    orig_j = pts[j][:]
                    pts[i], pts[j] = orig_j, orig_i
                    
                    if not is_valid(pts):
                        pts[i], pts[j] = orig_i, orig_j
                        continue
                    
                    edges_after = count_edges(pts)
                    if edges_after > edges_before:
                        edges_before = edges_after
                        improved = True
                    else:
                        pts[i], pts[j] = orig_i, orig_j
            
            # Also try single moves
            for i in range(32):
                if time.time() > deadline - 3:
                    break
                orig = pts[i][:]
                for da in range(-4, 5):
                    for db in range(-4, 5):
                        if da == 0 and db == 0:
                            continue
                        new_pt = [orig[0]+da, orig[1]+db]
                        pts[i] = new_pt
                        
                        if not is_valid(pts):
                            pts[i] = orig
                            continue
                        
                        edges_after = count_edges(pts)
                        if edges_after > edges_before:
                            edges_before = edges_after
                            improved = True
                        else:
                            pts[i] = orig
        
        edges = count_edges(pts)
        if edges > best_edges:
            best_edges = edges
            best_points = [p[:] for p in pts]
    
    # Verify bounds
    result = []
    for p in best_points:
        if abs(p[0]) <= 64 and abs(p[1]) <= 64:
            result.append(p)
    
    # If we lost points due to bounds, add random valid ones
    while len(result) < 32:
        a = rng.randint(-64, 64)
        b = rng.randint(-64, 64)
        if [a, b] not in result:
            result.append([a, b])
    
    # Ensure exactly 32 distinct points
    seen = set()
    final = []
    for p in result:
        key = (p[0], p[1])
        if key not in seen:
            seen.add(key)
            final.append(p)
    
    # Pad if needed
    while len(final) < 32:
        a = rng.randint(-64, 64)
        b = rng.randint(-64, 64)
        if (a, b) not in seen:
            seen.add((a, b))
            final.append([a, b])
    
    return final[:32]
# EVOLVE-BLOCK-END
