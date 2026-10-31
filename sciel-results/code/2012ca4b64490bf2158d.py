"""Improved program for Heilbronn triangles task."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay, Voronoi

N = 11
S3 = np.sqrt(3)
H = S3 / 2.0
AREA_EQ = S3 / 4.0  # area of equilateral triangle with side 1

# Vertices of the equilateral triangle
_VERTS = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, H]])


def _inside_triangle(p, eps=1e-6):
    """Check if point p (x,y) is inside or on the boundary of the equilateral triangle."""
    # Barycentric coordinates
    # p = a*A + b*B + c*C where a+b+c=1, a,b,c >= 0
    x, y = p
    # Using the standard barycentric formula for equilateral triangle
    c = y / H  # coefficient for vertex C (top)
    a = 1.0 - x - c  # coefficient for vertex A (bottom-left)
    b = c - (x - 0.5)  # wait, let me recompute
    # Actually: for triangle A=(0,0), B=(1,0), C=(0.5,H)
    # barycentric: 
    #   c = y / H
    #   b = (x - 0.5*c) ... hmm let me use the standard formula
    # Using: a = 1 - b - c
    # Solve: x = b*1 + c*0.5, y = c*H
    # So c = y/H, b = x - 0.5*c = x - 0.5*y/H
    # a = 1 - b - c
    c_b = y / H
    b_b = x - 0.5 * c_b
    a_b = 1.0 - a_b if False else 1.0 - b_b - c_b
    return a_b >= -eps and b_b >= -eps and c_b >= -eps


def _min_triangle_area(points):
    """Compute the minimum triangle area formed by any 3 of the points."""
    n = len(points)
    min_area = float('inf')
    # Vectorized: compute all pairwise cross products
    # For points p_i, p_j, p_k, area = 0.5 * |cross(p_j - p_i, p_k - p_i)|
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            v1 = points[j] - points[i]
            # Vectorize over k > j
            rest = points[j + 1:]
            v2 = rest - points[i]
            areas = 0.5 * np.abs(v1[0] * v2[:, 1] - v1[1] * v2[:, 0])
            local_min = np.min(areas)
            if local_min < min_area:
                min_area = local_min
    return min_area


def _soft_min_triangle_area(points, beta=100.0):
    """
    Smooth surrogate for the min triangle area using log-sum-exp.
    We want to maximize the min area, so we minimize -log(sum(exp(-area/beta)))
    which approximates -min(area) for large beta.
    
    Actually, to maximize min area, we can minimize:
      f = -log(sum_i exp(-area_i / beta)) * beta
    For large beta, this approaches -min(area_i), so minimizing it maximizes min(area_i).
    
    But we need to be careful with numerical stability.
    """
    n = len(points)
    areas = []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            v1 = points[j] - points[i]
            rest = points[j + 1:]
            v2 = rest - points[i]
            a = 0.5 * np.abs(v1[0] * v2[:, 1] - v1[1] * v2[:, 0])
            areas.append(a)
    areas = np.concatenate(areas)
    
    # Soft min: -beta * log(sum(exp(-area/beta)))
    # For numerical stability, subtract max
    neg_areas = -areas / beta
    max_neg = np.max(neg_areas)
    log_sum = max_neg + np.log(np.sum(np.exp(neg_areas - max_neg)))
    soft_min = -beta * log_sum
    return soft_min


def _objective_flat(x, beta=50.0):
    """Objective: minimize the negative soft-min triangle area. x is flattened 2D coords."""
    points = x.reshape(-1, 2)
    # Penalty for points outside triangle
    penalty = 0.0
    for p in points:
        if not _inside_triangle(p, eps=0.0):
            # Compute distance to boundary roughly
            c_b = p[1] / H
            b_b = p[0] - 0.5 * c_b
            a_b = 1.0 - b_b - c_b
            # Penalty proportional to how far outside
            penalty += max(0, -a_b) ** 2 + max(0, -b_b) ** 2 + max(0, -c_b) ** 2
    if penalty > 1e-4:
        return penalty * 100.0 - soft_min_triangle_area(points, beta)
    return -_soft_min_triangle_area(points, beta)


def _construct_initial(seed=42):
    """Generate an initial set of points using a grid-like pattern."""
    rng = np.random.RandomState(seed)
    # Start with a structured pattern: corners + center + edge midpoints + interior
    # For 11 points in an equilateral triangle
    # Let's use a hexagonal-like packing adapted to triangle
    points = []
    # Three corners
    points.append([0.05, 0.02])
    points.append([0.95, 0.02])
    points.append([0.5, H - 0.03])
    # Edge midpoints
    points.append([0.5, 0.02])
    points.append([0.25, H * 0.5])
    points.append([0.75, H * 0.5])
    # Interior points - use a smaller grid
    # Center
    points.append([0.5, H / 3.0])
    # A few more interior
    points.append([0.35, H / 3.0])
    points.append([0.65, H / 3.0])
    points.append([0.5, 2 * H / 3.0])
    # One more
    points.append([0.4, H / 2.0])
    points.append([0.6, H / 2.0])
    
    # Ensure we have exactly N points
    while len(points) < N:
        points.append([0.5 + rng.randn() * 0.1, H / 3.0 + rng.randn() * 0.1])
    points = points[:N]
    
    # Clamp to be inside triangle
    for i in range(N):
        p = points[i]
        c_b = p[1] / H
        b_b = p[0] - 0.5 * c_b
        a_b = 1.0 - b_b - c_b
        # If outside, project back
        if a_b < 0.01:
            # Project to edge BC or adjust
            p[1] = max(0.01, p[1])
            p[0] = min(0.99, max(0.01, p[0]))
        if b_b < 0.01:
            p[0] = max(0.01, p[0])
        if c_b < 0.01:
            p[1] = max(0.01, p[1])
        points[i] = p
    
    return np.array(points)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Start with a good initial guess
    best_points = _construct_initial(42)
    best_score = _min_triangle_area(best_points)
    
    # Try multiple random initializations and optimize
    rng = np.random.RandomState(123)
    n_restarts = 8
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        if restart == 0:
            points = best_points.copy()
        else:
            # Random initialization with some structure
            pts = []
            # Add corners with jitter
            pts.append([0.05 + rng.randn() * 0.02, 0.02 + rng.randn() * 0.01])
            pts.append([0.95 + rng.randn() * 0.02, 0.02 + rng.randn() * 0.01])
            pts.append([0.5 + rng.randn() * 0.02, H - 0.03 + rng.randn() * 0.01])
            # Add random interior points
            for _ in range(N - 3):
                u = rng.random()
                v = rng.random()
                while u + v > 0.9:
                    u = rng.random()
                    v = rng.random()
                x = u + 0.5 * v
                y = H * v
                pts.append([x, y])
            points = np.array(pts)
        
        # Optimize using L-BFGS-B with bounds
        x0 = points.flatten()
        
        # Define bounds for each coordinate
        # x in [0, 1], y in [0, H], but also must satisfy triangle constraint
        # We'll use simple bounds and rely on penalty for triangle constraint
        bounds = [(0.0, 1.0), (0.0, H)] * N
        
        # Use a smooth objective
        def obj(x):
            return _objective_flat(x, beta=50.0)
        
        result = minimize(obj, x0, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
        
        opt_points = result.x.reshape(-1, 2)
        
        # Project points back inside triangle
        for i in range(N):
            p = opt_points[i]
            c_b = p[1] / H
            b_b = p[0] - 0.5 * c_b
            a_b = 1.0 - b_b - c_b
            if a_b < 0.005:
                # Move to edge
                if c_b < 0.005:
                    # Near vertex A
                    p[0] = 0.005
                    p[1] = 0.005
                elif b_b < 0.005:
                    # Near edge AB
                    p[1] = 0.005
                else:
                    # Near edge BC
                    # Project onto line from B to C
                    # Parametric: B + t*(C-B)
                    # B=(1,0), C=(0.5,H)
                    # Closest point on segment
                    bc = _VERTS[2] - _VERTS[1]
                    bp = p - _VERTS[1]
                    t = np.dot(bp, bc) / np.dot(bc, bc)
                    t = np.clip(t, 0.0, 1.0)
                    p = _VERTS[1] + t * bc
                    p += 0.005 * np.array([-0.5, H]) / np.linalg.norm([-0.5, H])  # small inward
                    p = np.clip(p, [0.005, 0.005], [0.995, H - 0.005])
            elif b_b < 0.005:
                # Near edge AC
                ac = _VERTS[2] - _VERTS[0]
                ap = p - _VERTS[0]
                t = np.dot(ap, ac) / np.dot(ac, ac)
                t = np.clip(t, 0.0, 1.0)
                p = _VERTS[0] + t * ac
                p += 0.005 * np.array([0.5, -H]) / np.linalg.norm([0.5, -H])
                p = np.clip(p, [0.005, 0.005], [0.995, H - 0.005])
            elif c_b < 0.005:
                # Near edge AB
                p[1] = 0.005
                p[0] = np.clip(p[0], 0.005, 0.995)
            opt_points[i] = p
        
        score = _min_triangle_area(opt_points)
        if score > best_score:
            best_score = score
            best_points = opt_points.copy()
        
        # If we have time, do a local polish
        if time.time() < deadline - 5.0:
            # Try a finer optimization with higher beta
            def obj_fine(x):
                return _objective_flat(x, beta=200.0)
            
            result2 = minimize(obj_fine, opt_points.flatten(), method='L-BFGS-B', 
                              bounds=bounds, options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10})
            opt_points2 = result2.x.reshape(-1, 2)
            
            # Project again
            for i in range(N):
                p = opt_points2[i]
                c_b = p[1] / H
                b_b = p[0] - 0.5 * c_b
                a_b = 1.0 - b_b - c_b
                if a_b < 0.005:
                    if c_b < 0.005:
                        p[0] = 0.005
                        p[1] = 0.005
                    elif b_b < 0.005:
                        p[1] = 0.005
                    else:
                        bc = _VERTS[2] - _VERTS[1]
                        bp = p - _VERTS[1]
                        t = np.dot(bp, bc) / np.dot(bc, bc)
                        t = np.clip(t, 0.0, 1.0)
                        p = _VERTS[1] + t * bc
                        p += 0.005 * np.array([-0.5, H]) / np.linalg.norm([-0.5, H])
                        p = np.clip(p, [0.005, 0.005], [0.995, H - 0.005])
                elif b_b < 0.005:
                    ac = _VERTS[2] - _VERTS[0]
                    ap = p - _VERTS[0]
                    t = np.dot(ap, ac) / np.dot(ac, ac)
                    t = np.clip(t, 0.0, 1.0)
                    p = _VERTS[0] + t * ac
                    p += 0.005 * np.array([0.5, -H]) / np.linalg.norm([0.5, -H])
                    p = np.clip(p, [0.005, 0.005], [0.995, H - 0.005])
                elif c_b < 0.005:
                    p[1] = 0.005
                    p[0] = np.clip(p[0], 0.005, 0.995)
                opt_points2[i] = p
            
            score2 = _min_triangle_area(opt_points2)
            if score2 > best_score:
                best_score = score2
                best_points = opt_points2.copy()
    
    return best_points
