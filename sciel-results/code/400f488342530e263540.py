"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    N = 30
    MIN_COORD = -30
    MAX_COORD = 30

    def calculate_max_multiplicity(pts):
        counts = {}
        for i, j, k in combinations(range(N), 3):
            x1, y1 = pts[i]
            x2, y2 = pts[j]
            x3, y3 = pts[k]
            area = abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
            if area > 0:
                counts[area] = counts.get(area, 0) + 1
        if not counts:
            return 0
        return max(counts.values())

    def generate_grid(rows, cols):
        """Generate a grid of size rows x cols, centered at (0,0) if possible."""
        points = []
        # Center the grid
        # If rows/cols are even, center is between points. If odd, on a point.
        # We just generate relative coords and then shift.
        raw_points = [(x, y) for x in range(cols) for y in range(rows)]
        
        # Calculate bounding box to center it
        min_x = min(p[0] for p in raw_points)
        max_x = max(p[0] for p in raw_points)
        min_y = min(p[1] for p in raw_points)
        max_y = max(p[1] for p in raw_points)
        
        # Center coordinates
        cx = (min_x + max_x) / 2.0
        cy = (min_y + max_y) / 2.0
        
        centered_points = []
        for x, y in raw_points:
            # Round to integer to maintain lattice structure? 
            # If we shift by 0.5, we lose integer coords.
            # We must keep integer coords.
            # So we shift by integers only.
            # Let's just place them starting from a specific offset.
            pass
            
        # Simpler: just generate x in [0, cols-1], y in [0, rows-1]
        # Then shift so that the center is near (0,0).
        # Shift x by -(cols-1)//2, y by -(rows-1)//2
        shift_x = -(cols - 1) // 2
        shift_y = -(rows - 1) // 2
        
        final_points = []
        for x in range(cols):
            for y in range(rows):
                px = x + shift_x
                py = y + shift_y
                final_points.append((px, py))
                
        # Check bounds
        for p in final_points:
            if p[0] < MIN_COORD or p[0] > MAX_COORD or p[1] < MIN_COORD or p[1] > MAX_COORD:
                # This grid is too large, return None
                return None
                
        return final_points

    candidates = []
    
    # Factors of 30
    factors = [1, 2, 3, 5, 6, 10, 15, 30]
    for r in factors:
        c = 30 // r
        # Try both orientations
        pts1 = generate_grid(r, c)
        if pts1:
            candidates.append(pts1)
        if r != c:
            pts2 = generate_grid(c, r)
            if pts2:
                candidates.append(pts2)

    # Also try some shifted variants? 
    # The centering logic above is deterministic.
    # Maybe a grid aligned to the corner is better? Unlikely for symmetry, but let's add one.
    # Grid 5x6 starting at (0,0)
    pts_corner = [(x, y) for x in range(6) for y in range(5)]
    candidates.append(pts_corner)
    
    # Grid 6x5 starting at (0,0)
    pts_corner2 = [(x, y) for x in range(5) for y in range(6)]
    candidates.append(pts_corner2)

    best_score = -1
    best_points = None
    
    for pts in candidates:
        if len(pts) != N:
            continue
        # Check distinctness
        if len(set(pts)) != N:
            continue
        score = calculate_max_multiplicity(pts)
        if score > best_score:
            best_score = score
            best_points = pts

    if best_points is None:
        # Fallback to the original seed
        return [(x, y) for x in range(10) for y in range(3)]
        
    return best_points
# EVOLVE-BLOCK-END
