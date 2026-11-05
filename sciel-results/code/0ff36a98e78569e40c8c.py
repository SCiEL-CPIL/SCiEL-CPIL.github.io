import time
import math
from itertools import combinations

def construct():
    """
    Find 32 points on triangular lattice maximizing number of unit-distance edges.
    Uses hexagonal spiral (Harborth's construction) as starting point, then local search.
    """
    deadline = time.time() + 140
    
    def dist_sq(p, q):
        da = p[0] - q[0]
        db = p[1] - q[1]
        return da*da + db*db + da*db
    
    def count_edges(points):
        """Count pairs at distance exactly 1."""
        count = 0
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                if dist_sq(points[i], points[j]) == 1:
                    count += 1
        return count
    
    def is_valid(points):
        """Check all points are distinct and at distance >= 1."""
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                if dist_sq(points[i], points[j]) < 1:
                    return False
        return True
    
    def hex_spiral_points(r):
        """Generate hexagonal spiral of radius r (centered at origin)."""
        points = [(0, 0)]
        for radius in range(1, r + 1):
            # 6 directions
            dirs = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
            # Start at the first point on this ring
            # The ring starts at (radius, 0) and goes around
            # Actually, let's use the standard hex ring generation
            # For radius r, the ring has 6*r points
            # Starting point: (r, 0)
            # Then move in direction (0,1) for r steps, then (-1,1) for r steps, etc.
            
            # Let's use a cleaner approach: enumerate all points with hex distance = r
            for a in range(-r, r + 1):
                for b in range(-r, r + 1):
                    c = -a - b
                    if max(abs(a), abs(b), abs(c)) == r:
                        points.append((a, b))
        return points
    
    # Generate hex spiral of radius 3 (37 points)
    R = 3
    all_hex = hex_spiral_points(R)
    
    # We need 32 points, so remove 5
    # Start with all 37, then try removing 5 to maximize edges
    # First, let's try the natural spiral order: remove outermost points
    
    # Sort by hex distance from center, then by angle
    def hex_dist(p):
        a, b = p
        c = -a - b
        return max(abs(a), abs(b), abs(c))
    
    def hex_angle(p):
        a, b = p
        if a == 0 and b == 0:
            return 0
        import math
        # Convert to cartesian for angle
        x = a + b / 2.0
        y = b * math.sqrt(3) / 2.0
        return math.atan2(y, x)
    
    # Sort: by distance, then angle
    sorted_hex = sorted(all_hex, key=lambda p: (hex_dist(p), hex_angle(p)))
    
    # Take first 32 in this order (innermost first)
    initial_points = sorted_hex[:32]
    
    # Verify validity
    if not is_valid(initial_points):
        # Fallback: just use a strip
        initial_points = [(a, b) for a in range(16) for b in range(2)]
    
    best_points = list(initial_points)
    best_edges = count_edges(best_points)
    
    # Local search: try swapping one point with another
    # Candidate pool: all hex points within radius 4
    R2 = 4
    candidate_pool = hex_spiral_points(R2)
    
    # Also add some extra points
    for a in range(-R2, R2 + 1):
        for b in range(-R2, R2 + 1):
            if (a, b) not in candidate_pool:
                candidate_pool.append((a, b))
    
    # Deduplicate
    candidate_pool = list(set(candidate_pool))
    
    # Local search iterations
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() < deadline - 5 and no_improve_count < max_no_improve:
        improved = False
        n = len(best_points)
        
        for i in range(n):
            if time.time() > deadline - 5:
                break
            for new_pt in candidate_pool:
                if new_pt in best_points:
                    continue
                
                # Check if replacing best_points[i] with new_pt is valid
                trial = list(best_points)
                trial[i] = new_pt
                
                if not is_valid(trial):
                    continue
                
                edges = count_edges(trial)
                if edges > best_edges:
                    best_points = trial
                    best_edges = edges
                    improved = True
                    no_improve_count = 0
                    break
            if improved:
                break
        
        if not improved:
            no_improve_count += 1
    
    # Convert to list of lists
    result = [[p[0], p[1]] for p in best_points]
    
    # Final validity check
    if not is_valid(result):
        # Fallback to strip
        result = [[a, b] for a in range(16) for b in range(2)]
    
    return result
