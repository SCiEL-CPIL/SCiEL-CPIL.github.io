"""Number-theoretic construction: lattice points in a parallelogram to maximize area multiplicity."""
import itertools
from collections import Counter

def _evaluate_grid(w, h, n_points=30):
    """Generate a w x h grid, trim to n_points if necessary (from corners), and return max multiplicity."""
    # Generate full grid
    points = [(x, y) for x in range(w) for y in range(h)]
    if len(points) > n_points:
        # Remove points from corners first to preserve internal structure
        # Sort by distance from center? Or just remove corners (0,0), (w-1,0) etc
        # For simplicity, we'll just take the first n_points in a spiral or sorted order?
        # Better: Remove points with highest x+y or something?
        # Actually, for small trims, removing corners is best.
        corners = [(0,0), (w-1,0), (0,h-1), (w-1,h-1)]
        # This is getting complex to generalize "optimal trim".
        # Given 5x6 is exact, we don't need to trim.
        # We will just evaluate exact grids.
        return -1, []
    
    if len(points) < n_points:
        return -1, []
        
    # Compute areas
    areas = []
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            for k in range(j+1, len(points)):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                det = abs(x1*(y2-y3) - x2*(y1-y3) + x3*(y1-y2))
                if det > 0:
                    areas.append(det)
    
    if not areas:
        return 0, []
        
    counter = Counter(areas)
    max_mult = max(counter.values())
    return max_mult, points

def construct():
    # Candidate grids (w, h) such that w*h == 30
    candidates = [(5, 6), (6, 5), (3, 10), (10, 3), (2, 15), (15, 2)]
    # Also consider slightly larger grids trimmed? 
    # Let's stick to exact fits for now as they are most regular.
    
    best_mult = 0
    best_points = []
    
    for w, h in candidates:
        mult, pts = _evaluate_grid(w, h, 30)
        if mult > best_mult:
            best_mult = mult
            best_points = pts
            
    # Center the points
    if best_points:
        xs = [p[0] for p in best_points]
        ys = [p[1] for p in best_points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        offset_x = -(min_x + max_x) // 2
        offset_y = -(min_y + max_y) // 2
        centered_points = [(x + offset_x, y + offset_y) for x, y in best_points]
        
        # Verify bounds
        if all(-30 <= x <= 30 and -30 <= y <= 30 for x, y in centered_points):
            return centered_points
        else:
            # Fallback to original 5x6 centered
            points = [(x - 2, y - 2) for x in range(5) for y in range(6)]
            return points
    else:
        # Fallback
        points = [(x - 2, y - 2) for x in range(5) for y in range(6)]
        return points
