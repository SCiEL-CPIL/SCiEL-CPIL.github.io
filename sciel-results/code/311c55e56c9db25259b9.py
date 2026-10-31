import numpy as np
from scipy.optimize import dual_annealing
from itertools import combinations

N = 11
tri_indices = list(combinations(range(N), 3))
tri_indices_arr = np.array(tri_indices)

def _calculate_min_area(points):
    p1 = points[tri_indices_arr[:, 0]]
    p2 = points[tri_indices_arr[:, 1]]
    p3 = points[tri_indices_arr[:, 2]]
    x1, y1 = p1[:, 0], p1[:, 1]
    x2, y2 = p2[:, 0], p2[:, 1]
    x3, y3 = p3[:, 0], p3[:, 1]
    areas = np.abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)) * 0.5
    return np.min(areas)

def _objective(flat_points):
    pts = flat_points.reshape(N, 2)
    x = pts[:, 0]
    y = pts[:, 1]
    sqrt3 = np.sqrt(3)
    
    # Penalty for outside
    pen1 = np.maximum(0, -y)
    pen2 = np.maximum(0, y - sqrt3 * x)
    pen3 = np.maximum(0, y - sqrt3 * (1 - x))
    penalty = np.sum(pen1**2 + pen2**2 + pen3**2)
    
    min_area = _calculate_min_area(pts)
    return -min_area + 1000.0 * penalty

def _project_to_triangle(pts):
    """Project points to the triangle (0,0)-(1,0)-(0.5, sqrt(3)/2)."""
    proj = pts.copy()
    sqrt3 = np.sqrt(3)
    h = sqrt3 / 2.0
    
    for i in range(N):
        px, py = proj[i]
        # Clip to bounding box first
        px = np.clip(px, 0, 1)
        py = np.clip(py, 0, h)
        
        # Check edge 1: y = 0
        if py < 0:
            py = 0
        
        # Check edge 2: y = sqrt(3) * x  => x = y / sqrt(3)
        # If point is above this line (left side), it's outside.
        # Actually, the triangle is to the right of this line?
        # Vertex (0,0) and (0.5, h). Line passes through (0,0) and (0.5, sqrt(3)/2).
        # Slope is sqrt(3).
        # Points inside have y <= sqrt(3)*x?
        # Let's check (0.5, 0). y=0, sqrt(3)*0.5 = 0.866. 0 <= 0.866. Yes.
        # So inside is y <= sqrt(3)*x.
        if py > sqrt3 * px:
            # Project onto line y = sqrt(3)*x
            # Closest point on line ax+by+c=0?
            # Line: sqrt(3)x - y = 0.
            # Distance formula projection.
            # Or simply, if we are above the line, we can reduce y or increase x?
            # Actually, the constraint is y <= sqrt(3)x.
            # If violated, we are in the region x < y/sqrt(3).
            # We want to move to the line.
            # Vector from (px,py) to line.
            # Normal vector to line sqrt(3)x - y = 0 is (sqrt(3), -1).
            # Signed distance d = (sqrt(3)*px - py) / sqrt(3+1) = (sqrt(3)*px - py)/2.
            # Since py > sqrt(3)px, d is negative.
            # We want to move by -d * normal_unit?
            # Let's just use the geometric property:
            # The triangle is convex. The closest point on the boundary is on the edge.
            # We can solve for the projection.
            # Alternatively, just clamp:
            # If py > sqrt(3)px, then x is too small.
            # Set x = py / sqrt(3)? No, that moves horizontally.
            # Orthogonal projection is better.
            # Let's use a simple iterative clip or just set to the edge if close.
            # For robustness in a black box optimizer, a simple clip to the edge might suffice.
            # If py > sqrt(3)px, set px = py / sqrt(3) if py/sqrt(3) <= 0.5?
            # Actually, the edge is from (0,0) to (0.5, h).
            # Parametric: t * (0.5, h).
            # Let's just use the fact that we can scale the vector from origin?
            # No, let's use a generic projection helper.
            
            # Project (px, py) onto segment from (0,0) to (0.5, h)
            # v = (0.5, h)
            # u = (px, py)
            # proj = (u.dot(v) / v.dot(v)) * v
            v = np.array([0.5, h])
            u = np.array([px, py])
            t = np.dot(u, v) / np.dot(v, v)
            t = np.clip(t, 0, 1)
            proj_pt = t * v
            px, py = proj_pt[0], proj_pt[1]
            
        # Check edge 3: y = sqrt(3) * (1-x)
        # Line: sqrt(3)x + y - sqrt(3) = 0.
        # Inside: y <= sqrt(3)(1-x) => sqrt(3)x + y <= sqrt(3).
        if sqrt3 * px + py > sqrt3:
            # Project onto segment from (1,0) to (0.5, h)
            # v = (-0.5, h) from (1,0)
            # Or just project (px,py) onto segment (1,0)-(0.5,h)
            p1 = np.array([1.0, 0.0])
            p2 = np.array([0.5, h])
            v = p2 - p1
            u = np.array([px, py]) - p1
            t = np.dot(u, v) / np.dot(v, v)
            t = np.clip(t, 0, 1)
            proj_pt = p1 + t * v
            px, py = proj_pt[0], proj_pt[1]
            
        proj[i] = [px, py]
        
    return proj

def construct():
    best_score = -np.inf
    best_final = None
    
    # Run dual_annealing multiple times
    for seed in range(5):
        # Random start inside triangle
        pts = np.zeros((N, 2))
        for i in range(N):
            u = np.random.rand()
            v = np.random.rand() * (1 - u)
            pts[i, 0] = u + 0.5 * v
            pts[i, 1] = v * (np.sqrt(3) / 2)
            
        x0 = pts.flatten()
        bounds = [(0.0, 1.0), (0.0, np.sqrt(3)/2)] * N
        
        try:
            result = dual_annealing(
                _objective, 
                bounds, 
                x0=x0,
                maxiter=50,
                restarts=5,
                seed=seed,
                initial_temp=100.0,
                schedule=(1.0, 2.0),
                maxfun=10000
            )
        except Exception:
            # Fallback if dual_annealing fails
            continue
            
        final_pts = _project_to_triangle(result.x.reshape(N, 2))
        score = _calculate_min_area(final_pts)
        
        if score > best_score:
            best_score = score
            best_final = final_pts
            
    if best_final is None:
        # Fallback to random
        return np.random.rand(N, 2) * 0.5 + 0.25 # Roughly inside
        
    return best_final
