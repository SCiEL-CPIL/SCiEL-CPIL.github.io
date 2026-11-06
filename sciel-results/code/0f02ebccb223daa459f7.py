import itertools
from collections import Counter

def _max_multiplicity(points):
    """Compute the largest multiplicity of a single positive doubled area."""
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                det = abs(x1*(y2-y3) - x2*(y1-y3) + x3*(y1-y2))
                if det > 0:
                    areas.append(det)
    if not areas:
        return 0
    counter = Counter(areas)
    return max(counter.values())

def _generate_rect(W, H):
    """Generate a W x H grid centered at (0,0)."""
    points = []
    # Center the grid
    # x range: 0 to W-1. Center: (W-1)/2
    # y range: 0 to H-1. Center: (H-1)/2
    # We want integer coordinates.
    # If W is odd, center is integer. If even, center is half-integer.
    # Shift to make coordinates integers and centered.
    # Simple shift: x - W//2, y - H//2
    for x in range(W):
        for y in range(H):
            points.append((x - W//2, y - H//2))
    return points

def _generate_rect_minus(W, H, remove_frac):
    """Generate a W x H grid and remove points from the boundary/corners."""
    points = _generate_rect(W, H)
    # Sort points by "centrality" or just remove corners?
    # For now, let's just use the full rect for simplicity in this thought process,
    # but in the code I will try specific removals.
    return points

def construct():
    candidates = []
    
    # Candidate 1: 5x6 Grid
    p1 = _generate_rect(5, 6)
    candidates.append(p1)
    
    # Candidate 2: 3x10 Grid
    p2 = _generate_rect(3, 10)
    candidates.append(p2)
    
    # Candidate 3: 6x6 Grid minus 6 points (4 corners + 2 mid-edge?)
    # Let's just try 6x6 and 4x8 to see if the search finds a better one.
    # Actually, 6x6 has 36 points. We need 30.
    # Let's create a 6x6 grid and remove the 6 points with the lowest "local density" 
    # which are the corners.
    p3_full = _generate_rect(6, 6)
    # Identify corners: (-3,-3), (-3,3), (3,-3), (3,3)
    # And maybe (-3, -2), (-2, -3) etc.
    # Let's just remove the 4 corners and 2 random boundary points?
    # Better: Remove the 6 points with the smallest x+y? No.
    # Let's stick to rectangles for the search to keep it simple and robust.
    # If 5x6 is not the best, 3x10 or 4x8 (subset) might be.
    # But 4x8 subset requires specific removal.
    # Let's add a 4x8 grid and remove 2 corners.
    p4_full = _generate_rect(4, 8)
    # Corners of 4x8: (-2,-4), (-2,3), (1,-4), (1,3)
    # Remove 2 corners.
    # Which 2? Any 2.
    # Let's remove (-2,-4) and (1,3).
    corners_48 = [(-2,-4), (1,3)]
    p4 = [p for p in p4_full if p not in corners_48]
    candidates.append(p4)
    
    # Candidate 4: 2x15 Grid
    p5 = _generate_rect(2, 15)
    candidates.append(p5)
    
    best_points = None
    best_score = -1
    
    for pts in candidates:
        if len(pts) != 30:
            continue
        # Check bounds
        if not all(-30 <= x <= 30 and -30 <= y <= 30 for x, y in pts):
            continue
        # Check distinct
        if len(set(pts)) != 30:
            continue
        score = _max_multiplicity(pts)
        if score > best_score:
            best_score = score
            best_points = pts
            
    return best_points
