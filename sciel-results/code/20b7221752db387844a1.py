"""Heilbronn triangles: 11 points in equilateral triangle, maximize min triangle area."""
import numpy as np
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def _min_triangle_area(points):
    """Minimum area among all triangles formed by triples of points."""
    n = len(points)
    min_area = float('inf')
    # Vectorized calculation for speed
    # We can't easily vectorize the triple loop in pure numpy without huge memory usage for N=11 (11 choose 3 = 165 triangles), so a loop is fine.
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                # Area = 0.5 * | (x2-x1)(y3-y1) - (x3-x1)(y2-y1) |
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
                if area < min_area:
                    min_area = area
    return min_area

def _penalty_outside(points):
    """
    Calculate penalty for points outside the equilateral triangle (0,0), (1,0), (0.5, sqrt(3)/2).
    Returns a large positive value if any point is outside.
    """
    sqrt3 = np.sqrt(3.0)
    h = sqrt3 / 2.0
    penalty = 0.0
    for p in points:
        x, y = p
        # Check boundaries
        # Bottom: y >= 0
        if y < 0:
            penalty += (0 - y) * 100
        # Left: y <= sqrt(3) * x  => sqrt(3)*x - y >= 0
        # Line from (0,0) to (0.5, h): slope sqrt(3). Eq: y = sqrt(3)x => sqrt(3)x - y = 0
        # Inside is to the right/below? 
        # At (0.5, 0), sqrt(3)*0.5 - 0 > 0. So inside is sqrt(3)x - y >= 0.
        val_left = np.sqrt(3) * x - y
        if val_left < 0:
            penalty += (-val_left) * 100
        # Right: y <= -sqrt(3) * (x - 1) => sqrt(3)(1-x) - y >= 0
        val_right = np.sqrt(3) * (1 - x) - y
        if val_right < 0:
            penalty += (-val_right) * 100
    return penalty

def construct():
    """
    Construct 11 points in the equilateral triangle (0,0), (1,0), (0.5, sqrt(3)/2)
    maximizing the minimum triangle area.
    Uses symmetry about x=0.5 and Nelder-Mead optimization.
    """
    sqrt3 = np.sqrt(3.0)
    h = sqrt3 / 2.0
    
    # We will parameterize points using symmetry.
    # N=11.
    # One point on the axis of symmetry (x=0.5) or pairs.
    # Let's assume pairs + 1 center point.
    # 11 = 5 pairs + 1 center? Or 4 pairs + 3 centers?
    # Let's try a configuration with 5 pairs and 1 center point.
    # Variables: 5 * (x_offset, y) + 1 * (y_center) = 11 variables.
    # x_coords for pair i: 0.5 - x_i, 0.5 + x_i
    # y_coords for pair i: y_i
    # Center point: (0.5, y_c)
    
    # Initial guess: Hexagonal packing style
    # Rows of points.
    # Let's try to place them in a grid that fits.
    
    # Topology:
    # Bottom row: 4 points? If symmetric, 2 pairs.
    # Middle row: 3 points? 1 pair + 1 center.
    # Top row: 4 points? 2 pairs.
    # Total 11.
    
    # Let's define the variables:
    # v[0..3]: x_offsets for bottom pair 1, bottom pair 2, middle pair, top pair 1, top pair 2?
    # This is getting complex. Let's just optimize all 22 coordinates but initialize them symmetrically and use a symmetry-enforcing projection or just rely on the optimizer staying near the symmetric manifold if initialized well.
    # Actually, Nelder-Mead can handle 22 vars.
    
    # Better: Use a specific symmetric parameterization.
    # Let's assume the optimal solution is symmetric.
    # Points:
    # P0: (0.5, y0) - Center of bottom? Or is bottom a pair?
    # Let's look at known N=11 solutions.
    # Often points are on the boundary.
    
    # Let's use a flexible parameterization:
    # 11 points, but we enforce symmetry by only optimizing half and mirroring.
    # If N is odd, one point is on the axis.
    # Let's assume 1 point on axis, 5 pairs.
    # Variables: 
    #   y_axis: y-coord of the axis point
    #   For each of 5 pairs: (dx, y) where points are (0.5-dx, y) and (0.5+dx, y)
    # Total variables: 1 + 5*2 = 11.
    
    def params_to_points(params):
        y_axis = params[0]
        pts = [(0.5, y_axis)]
        for i in range(5):
            dx = params[1 + 2*i]
            y = params[2 + 2*i]
            pts.append((0.5 - dx, y))
            pts.append((0.5 + dx, y))
        return np.array(pts)

    def objective(params):
        pts = params_to_points(params)
        min_area = _min_triangle_area(pts)
        pen = _penalty_outside(pts)
        # We want to maximize min_area, so minimize -min_area
        return -min_area + pen

    # Initial guesses
    # We need to provide 11 parameters.
    # Let's try a few heuristic starts.
    
    best_score = -float('inf')
    best_params = None
    
    # Heuristic 1: Evenly spaced rows
    # y levels: 0.1, 0.3, 0.5, 0.7, 0.85 (5 levels)
    # Widths at these y levels:
    # w(y) = 1 - 2*y/sqrt(3) ? No.
    # Triangle width at height y:
    # At y=0, width=1. At y=h, width=0.
    # Width(y) = 1 - y/h = 1 - 2y/sqrt(3).
    
    # Let's assign points to rows.
    # Row 0 (y=0.1): 2 pairs (4 pts) + maybe axis?
    # If we have 1 axis point, it can be in any row.
    # Let's put axis point in the middle row.
    
    # Start 1:
    # y_axis = 0.4
    # Pairs:
    # Row 1 (y=0.1): 2 pairs. dx approx width/4.
    # Row 2 (y=0.3): 1 pair.
    # Row 3 (y=0.5): 1 pair.
    # Row 4 (y=0.7): 1 pair.
    # Wait, 1 axis + 5 pairs = 11 pts.
    # If axis is at y=0.4, and we have 5 pairs, we can distribute them.
    
    y_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    
    # Let's just use random symmetric starts
    np.random.seed(42)
    
    candidates = []
    
    # Candidate 1: Grid-like
    # Axis at y=0.5
    # Pairs at y=0.1, 0.2, 0.3, 0.7, 0.8
    p1 = [0.5]
    for y in [0.1, 0.2, 0.3, 0.7, 0.8]:
        # Max dx at y is (1 - 2y/sqrt(3)) / 2
        max_dx = (1 - 2*y/sqrt3) / 2 * 0.8 # 80% of max width
        dx = max_dx * 0.5 # Half of that? Just a guess
        p1.extend([dx, y])
    candidates.append(np.array(p1))
    
    # Candidate 2: More points at bottom
    # Axis at y=0.2
    # Pairs at y=0.1, 0.2 (same level as axis? no, separate), 0.3, 0.5, 0.7
    p2 = [0.2]
    for y in [0.05, 0.15, 0.3, 0.5, 0.7]:
        max_dx = (1 - 2*y/sqrt3) / 2 * 0.8
        dx = max_dx * 0.5
        p2.extend([dx, y])
    candidates.append(np.array(p2))
    
    # Candidate 3: Random symmetric
    for _ in range(10):
        p = [np.random.uniform(0, h)] # Axis y
        for _ in range(5):
            y = np.random.uniform(0, h)
            max_dx = max(0.01, (1 - 2*y/sqrt3) / 2 * 0.9)
            dx = np.random.uniform(0.01, max_dx)
            p.extend([dx, y])
        candidates.append(np.array(p))

    for cand in candidates:
        res = minimize(objective, cand, method='Nelder-Mead', 
                       options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6})
        
        if res.success or res.fun < best_score:
            pts = params_to_points(res.x)
            # Check validity
            if _penalty_outside(pts) < 1e-6:
                score = _min_triangle_area(pts)
                if score > best_score:
                    best_score = score
                    best_params = res.x
    
    if best_params is None:
        # Fallback to first candidate
        best_params = candidates[0]
        
    return params_to_points(best_params)
# EVOLVE-BLOCK-END
