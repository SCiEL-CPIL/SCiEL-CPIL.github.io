"""Packing 50 equal circles in a unit square: maximize minimum distance."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import itertools

def _initial_guess(n=50):
    """Create a reasonable initial guess using a grid with some perturbation."""
    m = int(np.ceil(np.sqrt(n)))
    # Place points in a grid pattern within [-0.5, 0.5]^2
    xs = np.linspace(-0.45, 0.45, m)
    ys = np.linspace(-0.45, 0.45, m)
    points = []
    for i in range(m):
        for j in range(m):
            if len(points) < n:
                points.append([xs[i], ys[j]])
    points = np.array(points[:n])
    # Add small random perturbation to break symmetry
    np.random.seed(42)
    points += np.random.normal(0, 0.01, size=points.shape)
    # Clip to stay within bounds
    points = np.clip(points, -0.48, 0.48)
    return points

def _objective_and_constraints(points):
    """
    Compute the minimum distance and check constraints.
    Returns (min_dist, constraints_satisfied)
    """
    n = len(points)
    # Wall constraint: distance to each wall
    wall_dists = []
    for p in points:
        wall_dists.append(min(0.5 - abs(p[0]), 0.5 - abs(p[1])))
    
    # Pairwise distances
    min_pair_dist = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(points[i] - points[j])
            if d < min_pair_dist:
                min_pair_dist = d
    
    min_dist = min(min(wall_dists), min_pair_dist)
    return min_dist

def _negative_min_dist(points_flat):
    """Objective function: minimize negative of minimum distance."""
    n = N
    points = points_flat.reshape(n, 2)
    
    # Wall constraint: 0.5 - |x| and 0.5 - |y| for each point
    # We want all these to be >= 0
    wall_margins = np.minimum(
        0.5 - np.abs(points[:, 0]),
        0.5 - np.abs(points[:, 1])
    )
    
    # Pairwise distances
    min_pair = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(points[i] - points[j])
            if d < min_pair:
                min_pair = d
    
    min_dist = min(min(wall_margins), min_pair)
    return -min_dist

def _wall_constraints(points_flat):
    """Constraints: each point must be inside the square."""
    n = N
    points = points_flat.reshape(n, 2)
    constraints = []
    for i in range(n):
        # x >= -0.5, x <= 0.5, y >= -0.5, y <= 0.5
        constraints.append(0.5 + points[i, 0])  # x >= -0.5
        constraints.append(0.5 - points[i, 0])  # x <= 0.5
        constraints.append(0.5 + points[i, 1])  # y >= -0.5
        constraints.append(0.5 - points[i, 1])  # y <= 0.5
    return constraints

def construct():
    """
    Find the packing of 50 equal circles in a unit square [-0.5, 0.5]^2
    that maximizes the minimum distance between circle centers and to walls.
    """
    # Start with a grid-based initial guess
    x0 = _initial_guess(N)
    
    # Define bounds for each coordinate
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    # Define constraints as a list of dictionaries
    constraints = []
    n = N
    # Wall constraints
    for i in range(n):
        # x_i >= -0.5
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, i=i: x[2*i] + 0.5
        })
        # x_i <= 0.5
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, i=i: 0.5 - x[2*i]
        })
        # y_i >= -0.5
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, i=i: x[2*i+1] + 0.5
        })
        # y_i <= 0.5
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, i=i: 0.5 - x[2*i+1]
        })
    
    # Pairwise distance constraints: ||p_i - p_j|| >= 0 (they should be positive, but we just need them valid)
    # Actually, for SLSQP we need to ensure circles don't overlap, but since we're maximizing min dist,
    # the constraint is implicitly handled. Let's add non-overlap as a soft constraint via the objective.
    # For hard constraints, we just need the points to be within bounds.
    
    # Run optimization
    result = minimize(
        _negative_min_dist,
        x0.flatten(),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 2000, 'ftol': 1e-12}
    )
    
    if result.success:
        points = result.x.reshape(N, 2)
    else:
        # Fall back to initial guess if optimization fails
        points = x0
    
    # Ensure points are strictly inside the square (add small margin for numerical safety)
    points = np.clip(points, -0.4999, 0.4999)
    
    return points
# EVOLVE-BLOCK-END
