"""Heilbronn triangles: maximize the minimum triangle area using soft-min and L-BFGS-B."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

N = 11

# Equilateral triangle vertices
V = np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [0.5, np.sqrt(3) / 2.0]
])

def bary_to_cart(bary):
    """Convert barycentric coordinates (n, 3) to Cartesian (n, 2)."""
    return bary @ V

def triangle_area(p1, p2, p3):
    """Area of triangle formed by three 2D points."""
    return 0.5 * abs(np.cross(p2 - p1, p3 - p1))

def all_triangle_areas(points):
    """Compute areas of all C(11,3)=165 triangles."""
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                areas.append(triangle_area(points[i], points[j], points[k]))
    return np.array(areas)

def soft_min(areas, tau):
    """Soft minimum using log-sum-exp: -tau * log(sum(exp(-areas/tau)))."""
    return -tau * np.log(np.sum(np.exp(-areas / tau)))

def objective(params, tau):
    """Objective: negative soft-min of triangle areas (to minimize)."""
    # params is barycentric coords flattened: (3*N,)
    bary = params.reshape(N, 3)
    # Normalize to sum to 1 (handle numerical drift)
    bary = bary / bary.sum(axis=1, keepdims=True)
    # Clip to non-negative
    bary = np.maximum(bary, 0)
    bary = bary / bary.sum(axis=1, keepdims=True)
    
    points = bary_to_cart(bary)
    areas = all_triangle_areas(points)
    return -soft_min(areas, tau)

def construct():
    # Start with a reasonable initial configuration
    # Place points in a grid-like pattern within the triangle
    rng = np.random.RandomState(42)
    
    # Initial barycentric coordinates: start with points spread out
    bary_init = np.zeros((N, 3))
    # Place first 3 at vertices, rest in interior
    bary_init[0] = [1, 0, 0]
    bary_init[1] = [0, 1, 0]
    bary_init[2] = [0, 0, 1]
    # Interior points: random barycentric with sum=1
    for i in range(3, N):
        r = rng.dirichlet([1, 1, 1])
        bary_init[i] = r
    
    params_init = bary_init.flatten()
    
    # Optimize with decreasing temperature
    best_params = params_init
    best_val = -objective(params_init, 0.01)
    
    for tau in [0.05, 0.03, 0.02, 0.01, 0.005]:
        result = minimize(
            objective,
            best_params,
            args=(tau,),
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if result.success and -result.fun > best_val:
            best_params = result.x
            best_val = -result.fun
    
    # Final evaluation with hard min
    bary = best_params.reshape(N, 3)
    bary = bary / bary.sum(axis=1, keepdims=True)
    bary = np.maximum(bary, 0)
    bary = bary / bary.sum(axis=1, keepdims=True)
    
    points = bary_to_cart(bary)
    
    # Ensure points are inside the triangle (small margin)
    # Re-normalize to stay strictly inside
    bary = np.maximum(bary, 0.001)
    bary = bary / bary.sum(axis=1, keepdims=True)
    points = bary_to_cart(bary)
    
    return points

# EVOLVE-BLOCK-START
import random
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

N = 11

# Equilateral triangle vertices
V = np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [0.5, np.sqrt(3) / 2.0]
])

def bary_to_cart(bary):
    """Convert barycentric coordinates (n, 3) to Cartesian (n, 2)."""
    return bary @ V

def triangle_area(p1, p2, p3):
    """Area of triangle formed by three 2D points."""
    return 0.5 * abs(np.cross(p2 - p1, p3 - p1))

def all_triangle_areas(points):
    """Compute areas of all C(11,3)=165 triangles."""
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                areas.append(triangle_area(points[i], points[j], points[k]))
    return np.array(areas)

def soft_min(areas, tau):
    """Soft minimum using log-sum-exp: -tau * log(sum(exp(-areas/tau)))."""
    return -tau * np.log(np.sum(np.exp(-areas / tau)))

def objective(params, tau):
    """Objective: negative soft-min of triangle areas (to minimize)."""
    bary = params.reshape(N, 3)
    bary = bary / bary.sum(axis=1, keepdims=True)
    bary = np.maximum(bary, 0)
    bary = bary / bary.sum(axis=1, keepdims=True)
    
    points = bary_to_cart(bary)
    areas = all_triangle_areas(points)
    return -soft_min(areas, tau)

def construct():
    rng = np.random.RandomState(42)
    
    bary_init = np.zeros((N, 3))
    bary_init[0] = [1, 0, 0]
    bary_init[1] = [0, 1, 0]
    bary_init[2] = [0, 0, 1]
    for i in range(3, N):
        r = rng.dirichlet([1, 1, 1])
        bary_init[i] = r
    
    params_init = bary_init.flatten()
    
    best_params = params_init
    best_val = -objective(params_init, 0.01)
    
    for tau in [0.05, 0.03, 0.02, 0.01, 0.005]:
        result = minimize(
            objective,
            best_params,
            args=(tau,),
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if result.success and -result.fun > best_val:
            best_params = result.x
            best_val = -result.fun
    
    bary = best_params.reshape(N, 3)
    bary = bary / bary.sum(axis=1, keepdims=True)
    bary = np.maximum(bary, 0)
    bary = bary / bary.sum(axis=1, keepdims=True)
    
    points = bary_to_cart(bary)
    
    bary = np.maximum(bary, 0.001)
    bary = bary / bary.sum(axis=1, keepdims=True)
    points = bary_to_cart(bary)
    
    return points
# EVOLVE-BLOCK-END
