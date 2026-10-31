import numpy as np
from scipy.optimize import minimize
import time

N = 11

# Vertices of the equilateral triangle
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, np.sqrt(3) / 2.0])
V = np.array([A, B, C])

def is_inside_triangle(points):
    """Check if points are inside the triangle defined by V."""
    # Using barycentric coordinates or cross products
    # For a point P, it is inside if it is on the same side of each edge as the opposite vertex.
    # Let's use a simpler method: sum of areas of sub-triangles equals area of main triangle.
    # Area of triangle with vertices P, A, B, C...
    # Actually, just check barycentric coords.
    # P = u*A + v*B + w*C, u+v+w=1, u,v,w >= 0
    # We can solve for u,v,w for each point.
    
    # Matrix form:
    # [x]   [Ax Bx Cx] [u]
    # [y] = [Ay By Cy] [v]
    # [1]   [1  1  1 ] [w]
    
    # Let's just use the cross product method for robustness.
    # Edge AB, vertex C. Edge BC, vertex A. Edge CA, vertex B.
    
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
        
    # Vectorized version for multiple points
    # d1 = sign(P, A, B)
    # d2 = sign(P, B, C)
    # d3 = sign(P, C, A)
    # Inside if all same sign (assuming CCW order A,B,C)
    
    # A=(0,0), B=(1,0), C=(0.5, h). CCW.
    # d1 = (P[0]-0)*(0-0) - (1-0)*(P[1]-0) = -P[1]
    # d2 = (P[0]-1)*(h-0) - (0.5-1)*(P[1]-0) = h(P[0]-1) + 0.5 P[1]
    # d3 = (P[0]-0.5)*(0-h) - (0-0.5)*(P[1]-h) = -h(P[0]-0.5) + 0.5(P[1]-h)
    
    # Let's just compute areas and check sum.
    # Area(PAB) + Area(PBC) + Area(PAC) == Area(ABC)
    # And all sub-areas > 0.
    
    pass # Will implement in objective

def calc_areas(points):
    """Calculate areas of all triangles formed by N points."""
    n = len(points)
    indices = np.indices((n, n, n)).reshape(3, -1).T
    # Filter i < j < k
    mask = (indices[:, 0] < indices[:, 1]) & (indices[:, 1] < indices[:, 2])
    idx = indices[mask]
    
    p1 = points[idx[:, 0]]
    p2 = points[idx[:, 1]]
    p3 = points[idx[:, 2]]
    
    # Area = 0.5 * | (p2-p1) x (p3-p1) |
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return areas

def penalty_function(points):
    """Calculate penalty for points outside the triangle."""
    # Area of main triangle
    area_main = 0.5 * np.abs((B[0]-A[0])*(C[1]-A[1]) - (C[0]-A[0])*(B[1]-A[1]))
    
    # Areas of sub-triangles
    # PAB, PBC, PAC
    pab = 0.5 * np.abs((B[0]-A[0])*(points[:,1]-A[1]) - (B[1]-A[1])*(points[:,0]-A[0]))
    pbc = 0.5 * np.abs((C[0]-B[0])*(points[:,1]-B[1]) - (C[1]-B[1])*(points[:,0]-B[0]))
    pac = 0.5 * np.abs((A[0]-C[0])*(points[:,1]-C[1]) - (A[1]-C[1])*(points[:,0]-C[0]))
    
    # If point is inside, sum of areas == area_main
    # If outside, sum > area_main
    # Penalty is sum of (sum_areas - area_main)
    sum_areas = pab + pbc + pac
    violation = np.maximum(0, sum_areas - area_main)
    return np.sum(violation ** 2)

def objective(x, beta=100.0):
    points = x.reshape(N, 2)
    
    # Penalty for being outside
    pen = penalty_function(points)
    if pen > 1e-6:
        # If significantly outside, return large value
        # We can also just rely on the optimizer staying close if we start inside
        pass
        
    areas = calc_areas(points)
    
    # Soft min
    # To avoid overflow, subtract max
    max_a = np.max(areas)
    log_sum_exp = np.log(np.sum(np.exp(-beta * (areas - max_a))))
    soft_min = max_a - (1.0 / beta) * log_sum_exp
    
    return -soft_min + 100.0 * pen

def construct():
    start_time = time.time()
    timeout = 110.0 # Safety margin
    
    best_val = -np.inf
    best_points = None
    
    # Initial guesses
    # 1. Hexagonal grid
    # 2. Random
    
    # Generate a hexagonal grid pattern
    # Side length of triangle is 1.
    # Hex grid spacing h
    # Let's try a few densities
    
    def get_hex_grid(n_rows, scale):
        pts = []
        h = np.sqrt(3) / 2 * scale
        for i in range(n_rows):
            y = i * h
            # x offset for odd rows
            offset = 0 if i % 2 == 0 else scale / 2
            # x range
            # y is from 0 to sqrt(3)/2
            # x bounds at height y:
            # Left edge: x = y / sqrt(3)
            # Right edge: x = 1 - y / sqrt(3)
            x_min = y / np.sqrt(3)
            x_max = 1 - y / np.sqrt(3)
            x = offset
            while x <= x_max:
                if x >= x_min:
                    pts.append([x, y])
                x += scale
        return np.array(pts)
        
    # We need exactly 11 points.
    # We can generate a larger grid and pick the best 11? Or just perturb a specific 11-point config.
    # A known good config for 11 points in a triangle often involves 3 in corners, some on edges, some interior.
    
    # Let's use a random restart strategy with Nelder-Mead.
    # Nelder-Mead is robust to non-smoothness (penalty).
    
    rng = np.random.default_rng(42)
    
    # Try a few different initializations
    inits = []
    
    # Init 1: Random points inside
    for _ in range(5):
        u = rng.random(N)
        v = rng.random(N)
        mask = u + v < 1
        # Rejection sampling for uniform in triangle
        pts = np.zeros((N, 2))
        for i in range(N):
            while True:
                u, v = rng.random(2)
                if u + v <= 1:
                    break
            pts[i] = u * A + v * B + (1 - u - v) * C
        inits.append(pts.flatten())
        
    # Init 2: Perturbed Grid
    # Create a 3x3 grid in barycentric coords
    # (0,0), (0,1/2), (0,1), (1/2,0)...
    # Let's just use a few structured starts
    
    # Start optimization
    for init in inits:
        if time.time() - start_time > timeout:
            break
            
        res = minimize(objective, init, method='Nelder-Mead', 
                       options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
        
        if res.success or res.fun < best_val:
            pts = res.x.reshape(N, 2)
            # Verify inside
            if penalty_function(pts) < 1e-5:
                val = -res.fun # This is roughly the soft-min area
                # Calculate true min area for scoring
                areas = calc_areas(pts)
                true_min = np.min(areas)
                if true_min > best_val:
                    best_val = true_min
                    best_points = pts

    # If best_points is None, return a fallback
    if best_points is None:
        # Return random points
        pts = np.zeros((N, 2))
        for i in range(N):
            while True:
                u, v = rng.random(2)
                if u + v <= 1:
                    break
            pts[i] = u * A + v * B + (1 - u - v) * C
        best_points = pts

    return best_points

# EVOLVE-BLOCK-END
