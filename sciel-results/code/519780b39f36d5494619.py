"""Triangular lattice packing: maximize edges for 32 points using hexagonal seed + local search."""
import time
import random

def _neighbors(p):
    a, b = p
    return ((a + 1, b), (a - 1, b), (a, b + 1),
            (a, b - 1), (a + 1, b - 1), (a - 1, b + 1))

def _dist_sq(p, q):
    """Squared distance between two lattice points a+b*omega where omega = exp(i*pi/3)."""
    da = p[0] - q[0]
    db = p[1] - q[1]
    # |a + b*omega|^2 = a^2 + ab + b^2
    return da * da + da * db + db * db

def _edge_count(Sset):
    """Count pairs at distance exactly 1."""
    cnt = 0
    for p in Sset:
        for q in _neighbors(p):
            if q in Sset and p < q:
                cnt += 1
    return cnt

def _valid(Sset):
    """Check validity: 32 distinct points, |a|,|b| <= 64, min distance >= 1."""
    if len(Sset) != 32:
        return False
    if len(set(tuple(p) for p in Sset)) != 32:
        return False
    for p in Sset:
        if abs(p[0]) > 64 or abs(p[1]) > 64:
            return False
    # Check mutual distance at least 1
    pts = list(Sset)
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            if _dist_sq(pts[i], pts[j]) < 1:
                return False
    return True

def _make_hexagonal_seed(n=32):
    """Create a hexagonal (triangular spiral) arrangement of n points."""
    points = [(0, 0)]
    # Triangular spiral: expand rings
    # Ring k has 6k points for k >= 1
    # We want to fill rings until we have n points
    ring = 1
    while len(points) < n:
        # Ring k: 6k points
        ring_points = []
        # The 6 sides of the hexagon at ring k
        # Side 1: (k, -k) to (2k, -k) -- actually let me think more carefully
        # For triangular lattice with basis vectors e1=(1,0), e2=(0,1) in (a,b) coords
        # Ring k consists of points at "hexagonal distance" k from origin
        # Hexagonal distance: max(|a|, |b|, |a+b|) ... no, that's for cube coords
        
        # Let me use a different approach: generate all points at distance k
        # and add them in spiral order
        
        # Actually, let me just use a known hexagonal packing pattern
        # For a hexagon of side length s (number of points on each side),
        # total points = 3s(s-1) + 1
        
        # For s=1: 1 point (just center)
        # For s=2: 7 points
        # For s=3: 19 points
        # For s=4: 37 points
        
        # So for 32 points, we need s=4 hexagon but only 32 of the 37 points
        pass
    
    # Alternative: use a rectangular patch of the triangular lattice
    # A 6x6 hexagonal patch gives 36 points, trim to 32
    
    # Let me use a simpler approach: hexagonal numbers
    # Hexagonal number H_k = k(3k-1)/2
    # H_1 = 1, H_2 = 7, H_3 = 19, H_4 = 37
    # So we need 32 points, which is between H_3=19 and H_4=37
    
    # Let me construct a hexagonal shape of side 4 (37 points) and remove 5
    # Or construct side 3 (19 points) and add 13 more
    
    # Actually, let me just use a direct construction:
    # Place points in a hexagonal pattern
    
    points = []
    # Center
    points.append((0, 0))
    
    # Ring 1: 6 points
    ring1 = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    points.extend(ring1)
    
    # Ring 2: 12 points
    ring2 = []
    # (2, -1), (2, 0), (2, 1)
    # (1, 2), (0, 2), (-1, 2)
    # (-2, 1), (-2, 0), (-2, -1)
    # (-1, -2), (0, -2), (1, -2)
    ring2 = [(2, -1), (2, 0), (2, 1),
             (1, 2), (0, 2), (-1, 2),
             (-2, 1), (-2, 0), (-2, -1),
             (-1, -2), (0, -2), (1, -2)]
    points.extend(ring2)
    
    # Ring 3: 18 points
    ring3 = []
    # Top-right edge: (3, -2), (3, -1), (3, 0), (3, 1), (3, 2)
    # Wait, let me be more systematic.
    
    # Ring k has 6k points. For k=3, that's 18 points.
    # The 6 sides:
    # Side 1 (top-right): from (3, -2) to (3, 2) going through... 
    # Let me think in terms of the 6 directions:
    # d1 = (1, 0), d2 = (0, 1), d3 = (-1, 1), d4 = (-1, 0), d5 = (0, -1), d6 = (1, -1)
    
    # Ring 3:
    # Start at (3, -2) [which is 3*d1 - 2*d6... hmm]
    
    # Let me just enumerate:
    # Corner points of ring 3:
    # (3, -2), (3, 0) no wait...
    
    # Hexagon corners at ring 3:
    # (3, -3+3) = (3, 0)? No.
    
    # Let me use: corner i is at position k * direction_i where directions are:
    # Actually for a hexagon in triangular lattice coords:
    # Corners are: (k, -k), (2k, -k), (2k, k), (k, 2k), (-k, k), (-2k, k)
    # Wait that doesn't look right either.
    
    # Let me just hardcode ring 3:
    # Going around the hexagon:
    # (3, -2), (3, -1), (3, 0), (3, 1), (3, 2) -- no, that's 5 points on one side
    
    # I think the cleanest approach is to just generate all points with hex distance <= 3
    # and take the first 32.
    
    # Hex distance in (a,b) coords for triangular lattice:
    # The 6 neighbors of (0,0) are at: (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)
    # These correspond to directions. The hex distance of (a,b) is:
    # d = max(|a|, |b|, |a+b|) ... let me verify:
    # (1,0): max(1,0,1) = 1 ✓
    # (0,1): max(0,1,1) = 1 ✓
    # (1,-1): max(1,1,0) = 1 ✓
    # (1,1): max(1,1,2) = 2 ✓ (should be distance 2)
    # (2,0): max(2,0,2) = 2 ✓
    
    # Great, so hex distance = max(|a|, |b|, |a+b|)
    
    all_points = []
    for a in range(-3, 4):
        for b in range(-3, 4):
            d = max(abs(a), abs(b), abs(a + b))
            if d <= 3:
                all_points.append((a, b))
    
    # Sort by ring, then by angle for spiral order
    def ring_sort_key(p):
        a, b = p
        d = max(abs(a), abs(b), abs(a + b))
        # Angle for ordering within ring
        import math
        # Convert to Cartesian: x = a + b/2, y = b*sqrt(3)/2
        x = a + b * 0.5
        y = b * (3 ** 0.5) / 2
        angle = math.atan2(y, x)
        return (d, angle)
    
    all_points.sort(key=ring_sort_key)
    
    # Take first 32
    points = all_points[:32]
    
    return [list(p) for p in points]

def _local_search(points, max_iter=50000, timeout=None):
    """Local search: try moving points to adjacent positions to increase edge count."""
    start = time.time()
    Sset = set(tuple(p) for p in points)
    best_edges = _edge_count(Sset)
    
    for iteration in range(max_iter):
        if timeout is not None and time.time() - start > timeout:
            break
        
        # Pick a random point
        idx = random.randint(0, len(points) - 1)
        p = tuple(points[idx])
        
        # Try all 6 neighbors
        best_gain = 0
        best_new_pos = None
        
        for np in _neighbors(p):
            if np in Sset:
                continue
            if abs(np[0]) > 64 or abs(np[1]) > 64:
                continue
            
            # Check if moving p to np maintains min distance >= 1
            valid = True
            for q in Sset:
                if q == p:
                    continue
                if _dist_sq(np, q) < 1:
                    valid = False
                    break
            if not valid:
                continue
            
            # Calculate edge count change
            # Remove p, add np
            new_Sset = Sset - {p} | {np}
            gain = _edge_count(new_Sset) - best_edges
            
            if gain > best_gain:
                best_gain = gain
                best_new_pos = np
        
        if best_new_pos is not None:
            points[idx] = list(best_new_pos)
            Sset = set(tuple(p) for p in points)
            best_edges += best_gain
        else:
            # No improvement found; try a random valid move (for simulated annealing-like behavior)
            # For now, just skip
            pass
    
    return points, best_edges

def construct():
    deadline = 140.0  # Safety margin before 150s limit
    start_time = time.time()
    
    # Start with hexagonal seed
    points = _make_hexagonal_seed(32)
    Sset = set(tuple(p) for p in points)
    current_edges = _edge_count(Sset)
    
    best_points = [list(p) for p in points]
    best_edges = current_edges
    
    # Local search with multiple restarts
    remaining = deadline - (time.time() - start_time)
    
    while time.time() - start_time < deadline - 5.0:  # 5s safety margin
        # Run local search
        pts, edges = _local_search(points, max_iter=20000, 
                                    timeout=remaining - (time.time() - start_time) - 2.0)
        
        if edges > best_edges:
            best_edges = edges
            best_points = [list(p) for p in pts]
            points = [list(p) for p in pts]
        else:
            # Perturb and try again
            # Randomly swap two points or move one point
            if random.random() < 0.5:
                # Move a random point to a random nearby position
                idx = random.randint(0, len(points) - 1)
                p = tuple(points[idx])
                candidates = []
                for da in range(-3, 4):
                    for db in range(-3, 4):
                        np = (p[0] + da, p[1] + db)
                        if np != p and abs(np[0]) <= 64 and abs(np[1]) <= 64:
                            candidates.append(np)
                if candidates:
                    np = random.choice(candidates)
                    if np not in Sset:
                        # Check validity
                        valid = True
                        for q in Sset:
                            if q == p:
                                continue
                            if _dist_sq(np, q) < 1:
                                valid = False
                                break
                        if valid:
                            points[idx] = list(np)
                            Sset = set(tuple(x) for x in points)
            else:
                # Swap two points
                i, j = random.sample(range(32), 2)
                if tuple(points[i]) != tuple(points[j]):
                    points[i], points[j] = points[j], points[i]
                    Sset = set(tuple(x) for x in points)
        
        # Update best
        current_edges = _edge_count(Sset)
        if current_edges > best_edges:
            best_edges = current_edges
            best_points = [list(p) for p in points]
    
    # Verify final result
    result = best_points
    if not _valid(result):
        # Fallback to seed
        result = [[a, b] for a in range(16) for b in range(2)]
    
    return result
