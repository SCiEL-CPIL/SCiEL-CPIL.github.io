import numpy as np
import time
from scipy.optimize import minimize

N = 11
S3H = np.sqrt(3) / 2
TRI_AREA = S3H / 2.0

# Vertices of the equilateral triangle
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, S3H])

def bary_to_cart(bary):
    """
    Convert barycentric coordinates (N, 3) to Cartesian (N, 2).
    bary[i] = [u, v, w] such that P = u*V0 + v*V1 + w*V2
    Since V0=(0,0), P = v*V1 + w*V2
    """
    # v = bary[:, 1], w = bary[:, 2]
    x = bary[:, 1] * V1[0] + bary[:, 2] * V2[0]
    y = bary[:, 1] * V1[1] + bary[:, 2] * V2[1]
    return np.column_stack((x, y))

def cart_to_bary(cart):
    """
    Convert Cartesian (N, 2) to Barycentric (N, 3).
    Solve P = v*V1 + w*V2
    x = v + 0.5*w
    y = 0      + S3H*w  => w = y / S3H
    v = x - 0.5*w
    u = 1 - v - w
    """
    x = cart[:, 0]
    y = cart[:, 1]
    w = y / S3H
    v = x - 0.5 * w
    u = 1.0 - v - w
    return np.column_stack((u, v, w))

def compute_min_area_cartesian(points):
    """
    Compute the minimum area of any triangle formed by three points.
    Returns the area normalized by the area of the equilateral triangle.
    """
    n = len(points)
    min_area = float('inf')
    # Vectorized calculation for all triplets is faster but memory heavy for large N.
    # For N=11, C(11,3) = 165 triplets. We can loop or vectorize.
    # Let's vectorize for speed.
    indices = np.array(list(itertools.combinations(range(n), 3)))
    if len(indices) == 0:
        return 0.0
    
    p1 = points[indices[:, 0]]
    p2 = points[indices[:, 1]]
    p3 = points[indices[:, 2]]
    
    # Cross product magnitude / 2 gives area
    # Area = 0.5 * | (p2-p1) x (p3-p1) |
    dx1 = p2[:, 0] - p1[:, 0]
    dy1 = p2[:, 1] - p1[:, 1]
    dx2 = p3[:, 0] - p1[:, 0]
    dy2 = p3[:, 1] - p1[:, 1]
    
    cross = np.abs(dx1 * dy2 - dy1 * dx2)
    areas = 0.5 * cross
    
    min_area = np.min(areas)
    # Normalize by triangle area
    return min_area / TRI_AREA

def objective(bary_coords):
    """
    Objective function for scipy.optimize.minimize.
    We want to maximize the min area, so we minimize the negative of it.
    Also, we must ensure points are strictly inside (or on boundary).
    Barycentric coords must be >= 0 and sum to 1.
    Since we are using Nelder-Mead which doesn't handle constraints well,
    we will penalize any coordinate < 0.
    """
    # Check for negative barycentric coordinates
    if np.any(bary_coords < -1e-9):
        # Strong penalty to push back inside
        penalty = np.sum(bary_coords[bary_coords < 0] ** 2) * 1e6
        return penalty
    
    # Clamp to 0 for calculation
    bary_clamped = np.clip(bary_coords, 0, None)
    
    # Convert to cartesian
    points = bary_to_cart(bary_clamped)
    
    # Compute min area
    min_area = compute_min_area_cartesian(points)
    
    return -min_area

def structured_init_bary():
    """
    Generate a structured initial configuration in barycentric coordinates.
    We want points well-distributed.
    A triangular grid of 11 points?
    Rows: 1, 2, 3, 4, 5? No, 1+2+3+4 = 10. 1+2+3+4+5 = 15.
    Maybe 1, 2, 3, 5? Or just a grid in barycentric space.
    
    Let's try a grid in barycentric coordinates (u, v, w) where u+v+w=1.
    We can sample a grid on the 2-simplex.
    
    Alternatively, place points at specific symmetric locations.
    Center, midpoints of edges, vertices?
    Vertices: (1,0,0), (0,1,0), (0,0,1)
    Midpoints: (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)
    Center: (1/3, 1/3, 1/3)
    That's 7 points.
    Need 4 more.
    Points at 1/4 and 3/4 along edges?
    (0.75, 0.25, 0), (0.25, 0.75, 0), (0, 0.75, 0.25), (0.25, 0, 0.75), (0.75, 0, 0.25), (0, 0.25, 0.75)
    That's 6 more. Total 13.
    
    Let's just use a simple random init but with symmetry, or a grid.
    Grid on simplex:
    Let u = i/(n-1), v = j/(n-1), w = 1-u-v.
    We need 11 points.
    Maybe a 4x4 grid? 16 points. Pick 11?
    
    Let's try a specific known good configuration or a dense grid.
    Actually, Nelder-Mead is sensitive to init.
    Let's try a grid of points in Cartesian space, then convert to barycentric.
    
    Cartesian grid points inside triangle:
    We can generate points on a triangular lattice.
    """
    points = []
    # Add vertices
    points.append([0.0, 0.0])
    points.append([1.0, 0.0])
    points.append([0.5, S3H])
    
    # Add midpoints
    points.append([0.5, 0.0])
    points.append([0.25, S3H/2])
    points.append([0.75, S3H/2])
    
    # Add center
    points.append([0.5, S3H/3])
    
    # Add 4 more points to make 11
    # Maybe closer to vertices or edges
    points.append([0.25, 0.0])
    points.append([0.75, 0.0])
    points.append([0.5, S3H/6])
    points.append([0.25, S3H/6])
    
    cart = np.array(points)
    bary = cart_to_bary(cart)
    return bary

def random_init_bary():
    """
    Generate random barycentric coordinates.
    """
    # Generate random numbers and normalize
    r = np.random.rand(N, 3)
    r /= r.sum(axis=1, keepdims=True)
    return r

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    best_bary = None
    best_score = -float('inf')
    
    # Try multiple initializations
    inits = []
    
    # 1. Structured
    inits.append(structured_init_bary())
    
    # 2. Random
    np.random.seed(42)
    for _ in range(5):
        inits.append(random_init_bary())
    
    for bary_init in inits:
        if time.time() > deadline:
            break
            
        # Ensure init is valid (non-negative)
        bary_init = np.clip(bary_init, 1e-6, None)
        # Re-normalize to sum to 1
        bary_init = bary_init / bary_init.sum(axis=1, keepdims=True)
        
        # Run Nelder-Mead
        # x0 is the initial barycentric coordinates
        res = minimize(objective, bary_init, method='Nelder-Mead', 
                       options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
        
        if time.time() > deadline:
            break
            
        # Check result
        bary_opt = res.x
        # Clip and normalize again just in case
        bary_opt = np.clip(bary_opt, 0, None)
        bary_opt = bary_opt / bary_opt.sum(axis=1, keepdims=True)
        
        # Calculate score
        cart_opt = bary_to_cart(bary_opt)
        score = compute_min_area_cartesian(cart_opt)
        
        if score > best_score:
            best_score = score
            best_bary = bary_opt.copy()
            
    if best_bary is None:
        # Fallback
        best_bary = structured_init_bary()
        
    cart_final = bary_to_cart(best_bary)
    return cart_final

# Import itertools for combinations
import itertools

if __name__ == "__main__":
    pts = construct()
    print("Points:", pts)
    print("Min area (normalized):", compute_min_area_cartesian(pts))
