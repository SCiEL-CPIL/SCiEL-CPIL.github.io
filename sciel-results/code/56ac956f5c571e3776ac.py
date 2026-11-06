"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from itertools import combinations
from collections import Counter

def _compute_best_area_multiplicity(points):
    """Compute the largest multiplicity of a single positive doubled area."""
    n = len(points)
    pts = np.array(points, dtype=np.int64)
    areas = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                det = abs(int(pts[i][0])*(pts[j][1]-pts[k][1]) + 
                         int(pts[j][0])*(pts[k][1]-pts[i][1]) + 
                         int(pts[k][0])*(pts[i][1]-pts[j][1]))
                if det > 0:
                    areas.append(det)
    if not areas:
        return 0
    counter = Counter(areas)
    return counter.most_common(1)[0][1]

def _generate_grid(w, h, offset_x=0, offset_y=0):
    """Create a w x h grid of points."""
    points = []
    for x in range(w):
        for y in range(h):
            points.append((x + offset_x, y + offset_y))
    return points

def _generate_sheared_grid(w, h, offset_x=0, offset_y=0):
    """Create a sheared grid (parallelogram)."""
    points = []
    for x in range(w):
        for y in range(h):
            # Shear: x' = x + y, y' = y
            px = x + y
            py = y
            points.append((px + offset_x, py + offset_y))
    return points

def _try_configurations():
    """Try various configurations and return the best one."""
    candidates = []
    
    # Rectangular grids
    grid_dims = [(1,30), (2,15), (3,10), (5,6), (6,5), (10,3), (15,2), (30,1)]
    for w, h in grid_dims:
        ox = -(w - 1) // 2
        oy = -(h - 1) // 2
        pts = _generate_grid(w, h, ox, oy)
        if all(-30 <= p[0] <= 30 and -30 <= p[1] <= 30 for p in pts) and len(set(pts)) == 30:
            candidates.append(pts)
            
    # Sheared grids
    for w, h in grid_dims:
        pts = _generate_sheared_grid(w, h)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ox = -(min(xs) + max(xs)) // 2
        oy = -(min(ys) + max(ys)) // 2
        pts = [(p[0]+ox, p[1]+oy) for p in pts]
        if all(-30 <= p[0] <= 30 and -30 <= p[1] <= 30 for p in pts) and len(set(pts)) == 30:
            candidates.append(pts)

    best_pts = None
    best_mult = -1
    for pts in candidates:
        if len(pts) != 30:
            continue
        mult = _compute_best_area_multiplicity(pts)
        if mult > best_mult:
            best_mult = mult
            best_pts = pts
    
    if best_pts is None:
        best_pts = _generate_grid(6, 5)
        
    return best_pts

def construct():
    return _try_configurations()
