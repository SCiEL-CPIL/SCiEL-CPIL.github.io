import numpy as np
from scipy.optimize import differential_evolution

N = 11

# Triangle vertices
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, np.sqrt(3)/2])

# Pre-compute all triangle indices
tri_indices = []
for i in range(N-2):
    for j in range(i+1, N-1):
        for k in range(j+1, N):
            tri_indices.append((i, j, k))
tri_indices = np.array(tri_indices)

def min_area_vectorized(pts):
    """
    Compute minimum triangle area for N points, vectorized.
    pts: (N, 2) array
    """
    pi = pts[tri_indices[:, 0]]
    pj = pts[tri_indices[:, 1]]
    pk = pts[tri_indices[:, 2]]
    
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def objective_de(x):
    """
    Objective function for Differential Evolution.
    x: flat array of length 2*N (barycentric coords u, v for each point)
    """
    # Reshape to (N, 2)
    uv = x.reshape(N, 2)
    u = uv[:, 0]
    v = uv[:, 1]
    
    # Check if points are inside the triangle
    # Conditions: u >= 0, v >= 0, u + v <= 1
    # Since DE bounds are [0, 1] for both u and v, u and v are always >= 0.
    # We just need to check u + v <= 1.
    if np.any(u + v > 1.0 + 1e-9):
        # Return a large penalty to discourage this region
        return 1e6
    
    # Calculate third barycentric coord
    w = 1.0 - u - v
    
    # Convert to Cartesian coordinates
    # P = u*A + v*B + w*C
    # A is (0,0), so term u*A is 0.
    # P = v*B + w*C
    pts = v[:, np.newaxis] * B + w[:, np.newaxis] * C
    
    area = min_area_vectorized(pts)
    return -area

def construct():
    """
    Main function: find 11 points in the unit triangle maximizing minimum triangle area.
    Uses Differential Evolution.
    """
    # Bounds for barycentric coordinates u and v
    # u in [0, 1], v in [0, 1]
    bounds = [(0.0, 1.0)] * (2 * N)
    
    # Run Differential Evolution
    # maxiter controls the number of iterations of the main loop
    # tol is the stopping tolerance
    # popsize is the number of members of the population
    result = differential_evolution(
        objective_de,
        bounds,
        maxiter=1000,
        tol=1e-6,
        popsize=20,
        seed=42,
        polish=True # Use Nelder-Mead to polish the best solution found by DE
    )
    
    # Extract best solution
    x_best = result.x
    uv = x_best.reshape(N, 2)
    u = uv[:, 0]
    v = uv[:, 1]
    w = 1.0 - u - v
    
    # Convert to Cartesian
    pts = v[:, np.newaxis] * B + w[:, np.newaxis] * C
    
    return pts
