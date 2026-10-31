"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11
import numpy as np
from scipy.optimize import differential_evolution, minimize
import time
import itertools
import sys

# EVOLVE-BLOCK-START

# Vertices of the equilateral triangle
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, np.sqrt(3) / 2])
H = np.sqrt(3) / 2  # height

def barycentric_to_cartesian(bary):
    """Convert barycentric coordinates to Cartesian coordinates."""
    # bary is (N, 3) with rows summing to 1
    pts = bary[:, 0:1] * A + bary[:, 1:2] * B + bary[:, 2:3] * C
    return pts

def cartesian_to_barycentric(pts):
    """Convert Cartesian coordinates to barycentric coordinates."""
    # Solve for barycentric coordinates
    # [x, y] = a*A + b*B + c*C where a+b+c=1
    # A=[0,0], B=[1,0], C=[0.5, H]
    # x = b + 0.5*c
    # y = H*c
    # c = y/H
    # b = x - 0.5*c
    # a = 1 - b - c
    c = pts[:, 1] / H
    b = pts[:, 0] - 0.5 * c
    a = 1.0 - b - c
    return np.column_stack((a, b, c))

def min_triangle_area_from_bary(bary):
    """
    Compute the minimum area of any triangle formed by 3 points,
    given barycentric coordinates.
    Area of triangle formed by points i, j, k in barycentric coords:
    The area is proportional to |det of the 3x3 matrix with rows bary_i, bary_j, bary_k|.
    Specifically, area = (area of reference triangle) * |det|.
    The area of the reference triangle is sqrt(3)/4.
    """
    n = len(bary)
    min_area = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                M = np.array([bary[i], bary[j], bary[k]])
                det = np.linalg.det(M)
                area = (np.sqrt(3) / 4) * abs(det)
                if area < min_area:
                    min_area = area
    return min_area

def objective(params):
    """
    Objective function to minimize (we want to maximize min area).
    params: flattened array of length N*3 (barycentric coordinates)
    Returns: negative of min area (so we minimize -min_area)
    """
    bary = params.reshape(N, 3)
    # Check if all barycentric coordinates are valid (>= 0 and sum to 1)
    # We use a penalty approach or handle in bounds
    if np.any(bary < -1e-9):
        return 1e10
    # Normalize to ensure they sum to 1 (in case of numerical drift)
    bary = bary / bary.sum(axis=1, keepdims=True)
    min_area = min_triangle_area_from_bary(bary)
    return -min_area

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s safety margin
    
    # Initial guess: random points inside the triangle
    # Use barycentric coordinates for representation
    # Random barycentric: generate 3 random numbers, normalize
    def random_bary_point():
        r = np.random.random(3)
        return r / r.sum()
    
    # Differential evolution over barycentric coordinates
    # Bounds: each barycentric coord in [0, 1]
    # But we need to ensure they sum to 1. 
    # Standard DE doesn't handle equality constraints well.
    # Instead, we can use 2 coordinates per point (since a+b+c=1, c=1-a-b).
    # Constraints: a>=0, b>=0, a+b<=1.
    
    def objective_2d(params):
        """
        params: length N*2. For each point, (a, b). c = 1-a-b.
        """
        ab = params.reshape(N, 2)
        a = ab[:, 0]
        b = ab[:, 1]
        c = 1.0 - a - b
        
        # Check constraints
        if np.any(a < -1e-9) or np.any(b < -1e-9) or np.any(c < -1e-9):
            return 1e10
        
        bary = np.column_stack((a, b, c))
        min_area = min_triangle_area_from_bary(bary)
        return -min_area

    # Bounds for a, b: [0, 1]
    # But a+b<=1 is a constraint. DE handles bounds but not linear constraints directly.
    # We can use a penalty in the objective if a+b > 1.
    
    def objective_penalized(params):
        ab = params.reshape(N, 2)
        a = ab[:, 0]
        b = ab[:, 1]
        c = 1.0 - a - b
        
        penalty = 0.0
        if np.any(a < 0):
            penalty += np.sum(-a[a < 0]) * 100
        if np.any(b < 0):
            penalty += np.sum(-b[b < 0]) * 100
        if np.any(c < 0):
            penalty += np.sum(-c[c < 0]) * 100
            
        if penalty > 0:
            return penalty
            
        bary = np.column_stack((a, b, c))
        min_area = min_triangle_area_from_bary(bary)
        return -min_area

    # Use differential evolution
    # Number of parameters: N*2 = 22
    bounds = [(0, 1)] * (N * 2)
    
    best_result = None
    best_score = -np.inf
    
    # Run differential evolution
    try:
        # Use a seed for reproducibility
        np.random.seed(42)
        
        # DE with maxiter and popsize tuned for time
        # 22 dimensions. Popsize 10-20 is reasonable.
        # Maxiter 50-100.
        # Tolerance for convergence.
        
        res = differential_evolution(
            objective_penalized, 
            bounds, 
            maxiter=200, 
            popsize=15, 
            tol=1e-6, 
            seed=42,
            polish=True
        )
        
        if time.time() > deadline:
            raise TimeoutError("DE took too long")
            
        if res.success or res.fun < 0:
            best_params = res.x
            best_score = -res.fun
            best_result = best_params
            
    except Exception as e:
        print(f"DE failed: {e}", file=sys.stderr)
        
    if best_result is None:
        # Fallback: random search
        best_params = None
        for _ in range(1000):
            if time.time() > deadline:
                break
            params = np.random.random((N, 2))
            a = params[:, 0]
            b = params[:, 1]
            c = 1.0 - a - b
            if np.any(a < 0) or np.any(b < 0) or np.any(c < 0):
                continue
            score = -objective_penalized(params.flatten())
            if score > best_score:
                best_score = score
                best_params = params.flatten()
                best_result = params
        
    if best_result is None:
        # Last resort: return a simple grid or random points
        # Just return 11 random points
        pts = []
        for _ in range(N):
            u, v = np.random.random(2)
            if u + v > 1:
                u, v = 1 - u, 1 - v
            x = u + 0.5 * v
            y = H * v
            pts.append([x, y])
        return np.array(pts)
    
    # Convert best barycentric params to Cartesian
    ab = best_result.reshape(N, 2)
    a = ab[:, 0]
    b = ab[:, 1]
    c = 1.0 - a - b
    
    # Ensure validity
    a = np.clip(a, 0, None)
    b = np.clip(b, 0, None)
    c = np.clip(c, 0, None)
    # Re-normalize if clipping changed sums
    sums = a + b + c
    a = a / sums
    b = b / sums
    c = c / sums
    
    bary = np.column_stack((a, b, c))
    pts = barycentric_to_cartesian(bary)
    
    return pts

# EVOLVE-BLOCK-END
