"""Heilbronn triangles: place 11 points to maximize the smallest triangle area."""
N = 11
import numpy as np
from scipy.optimize import minimize
import itertools
import time

V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])

def point_in_triangle(p, a=V0, b=V1, c=V2, eps=1e-9):
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    d1 = sign(p, a, b)
    d2 = sign(p, b, c)
    d3 = sign(p, c, a)
    has_neg = (d1 < -eps) or (d2 < -eps) or (d3 < -eps)
    has_pos = (d1 > eps) or (d2 > eps) or (d3 > eps)
    return not (has_neg and has_pos)

def min_triangle_area(points):
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = 0.5 * abs((points[j][0] - points[i][0]) * (points[k][1] - points[i][1]) -
                                  (points[k][0] - points[i][0]) * (points[j][1] - points[i][1]))
                if area < min_area:
                    min_area = area
    return min_area

def negative_min_area_flat(x):
    points = x.reshape(-1, 2)
    penalty = 0.0
    for p in points:
        if not point_in_triangle(p):
            penalty += 100.0
    return -min_triangle_area(points) + penalty

def generate_grid_initialization(rows, cols, jitter=0.0, seed=42):
    """
    Generate a grid of points in barycentric coordinates and convert to Cartesian.
    The triangle is defined by V0, V1, V2.
    Barycentric coords (a, b, c) with a+b+c=1.
    We can parameterize by (u, v) where point = u*V1 + v*V2.
    u in [0, 1], v in [0, 1-u].
    """
    rng = np.random.RandomState(seed)
    pts = []
    for i in range(rows):
        for j in range(cols):
            # Map i, j to barycentric-like grid
            # u is roughly j/cols, v is roughly i/rows * (1-u)
            # To make it more uniform, we use a triangular grid
            pass
    
    # Better approach: Use a hexagonal packing or simple grid in barycentric space
    # Let's use a simple grid in (u,v) space where u+v <= 1
    # Number of points is rows * cols. We need exactly 11.
    # This function is a placeholder. I will implement a specific 11-point generator.
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_points = None
    best_score = -float('inf')
    
    # Pre-compute a set of good initial configurations
    # 1. Grid-like in barycentric coords
    # 2. Random restarts
    
    def bary_to_cart(u, v):
        # p = u*V1 + v*V2
        return u * V1 + v * V2

    def generate_structured_init(seed):
        rng = np.random.RandomState(seed)
        # Create a grid of points in the (u,v) plane where u>=0, v>=0, u+v<=1
        # We want 11 points.
        # A 4x3 grid has 12 points, so we can drop one or adjust.
        # Let's try a 3x4 grid (12 points) and drop the center or a corner?
        # Or just use 11 random points jittered from a grid.
        
        # Grid points in barycentric (u, v)
        # u values: 0.1, 0.3, 0.5, 0.7
        # v values depend on u.
        
        # Alternative: Use a known good configuration for 11 points if available?
        # No, we must compute.
        
        # Let's use a simple jittered grid in the triangle.
        # Divide the triangle into smaller triangles.
        # For 11 points, a 3x3 grid (9) + 2 extra?
        
        # Let's try a grid of 4x3 = 12 points, then remove the one closest to the centroid?
        # Or just generate 11 points on a grid.
        
        # Grid in barycentric coordinates (a, b, c)
        # a + b + c = 1
        # Let's iterate over i, j in a grid that covers the simplex.
        # i from 0 to 3, j from 0 to 2?
        # a = i/4, b = j/3, c = 1 - a - b
        # Need a,b,c >= 0.
        
        candidates = []
        # 4x3 grid
        for i in range(4):
            for j in range(3):
                a = i / 3.0 # 0, 1/3, 2/3, 1
                b = j / 2.0 # 0, 1/2, 1
                c = 1 - a - b
                if a >= -1e-9 and b >= -1e-9 and c >= -1e-9:
                    # Convert to (u, v) where p = b*V1 + c*V2 ? No.
                    # p = a*V0 + b*V1 + c*V2
                    # Since V0=0, p = b*V1 + c*V2
                    # So u = b, v = c
                    u = b
                    v = c
                    p = bary_to_cart(u, v)
                    candidates.append(p)
        
        # We have 12 candidates (4*3 grid, all valid? 
        # i=0: a=0. b=0, 0.5, 1. c=1, 0.5, 0. All valid.
        # i=1: a=1/3. b=0, 0.5, 1. c=2/3, 1/6, -1/3. Last one invalid.
        # i=2: a=2/3. b=0, 0.5, 1. c=1/3, -1/6, -2/3. Only first valid.
        # i=3: a=1. b=0, 0.5, 1. c=0, -0.5, -1. Only first valid.
        # Total valid: 3 + 2 + 1 + 1 = 7 points. Not enough.
        
        # Let's use a finer grid or different mapping.
        # For 11 points, we can use a grid of size that yields ~11 points.
        # Or just use random points with a seed.
        
        # Let's stick to random restarts but with Nelder-Mead.
        # The key improvement is the optimizer.
        return None

    # Define a few specific initializations
    inits = []
    
    # Init 1: Random
    rng = np.random.RandomState(42)
    for _ in range(20):
        pts = []
        for _ in range(N):
            u = rng.random()
            v = rng.random() * (1 - u)
            p = bary_to_cart(u, v)
            pts.append(p)
        inits.append(np.array(pts))
        
    # Init 2: Grid-based
    # Try to place points at vertices and midpoints and center
    # Vertices: V0, V1, V2 (3)
    # Midpoints: (V0+V1)/2, (V1+V2)/2, (V2+V0)/2 (3)
    # Center: (V0+V1+V2)/3 (1)
    # That's 7 points. Need 4 more.
    # Place them near the midpoints of the segments from vertices to center?
    
    v0 = V0; v1 = V1; v2 = V2
    m01 = (v0+v1)/2; m12 = (v1+v2)/2; m20 = (v2+v0)/2
    c = (v0+v1+v2)/3
    
    base_pts = [v0, v1, v2, m01, m12, m20, c]
    
    # Add 4 more points
    # Maybe at 1/4 and 3/4 along the medians?
    # Median from V0 to m12: p = V0 + t*(m12-V0)
    # t=0.25, 0.75
    # Median from V1 to m20: p = V1 + t*(m20-V1)
    # t=0.25, 0.75
    
    extra1 = v0 + 0.25*(m12-v0)
    extra2 = v0 + 0.75*(m12-v0)
    extra3 = v1 + 0.25*(m20-v1)
    extra4 = v1 + 0.75*(m20-v1)
    
    grid_pts = np.array(base_pts + [extra1, extra2, extra3, extra4])
    inits.append(grid_pts)
    
    # Init 3: Jittered Grid
    rng2 = np.random.RandomState(123)
    jittered = grid_pts + rng2.normal(0, 0.05, grid_pts.shape)
    inits.append(jittered)
    
    # Run Nelder-Mead on each init
    for i, init_pts in enumerate(inits):
        if time.time() > deadline - 10.0:
            break
            
        x0 = init_pts.flatten()
        
        try:
            result = minimize(
                negative_min_area_flat,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
        except Exception:
            continue
            
        if time.time() > deadline - 5.0:
            break
            
        optimized_points = result.x.reshape(-1, 2)
        
        # Project points outside triangle back inside (simple clamping/projection)
        # For Nelder-Mead, points might drift out.
        for k in range(len(optimized_points)):
            p = optimized_points[k]
            if not point_in_triangle(p):
                # Simple projection: find closest point in triangle
                # This is a bit expensive to do correctly. 
                # Instead, we can just reject this solution if it's invalid?
                # Or use the penalty result which should have driven it inside.
                # If penalty is high, score will be low.
                pass
        
        valid = all(point_in_triangle(p) for p in optimized_points)
        if valid:
            score = min_triangle_area(optimized_points)
            if score > best_score:
                best_score = score
                best_points = optimized_points.copy()
                
    if best_points is None:
        # Fallback
        best_points = inits[0]
        
    return best_points
