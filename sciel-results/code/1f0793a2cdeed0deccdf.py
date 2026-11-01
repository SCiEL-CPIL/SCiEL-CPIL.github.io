"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14
D = 3

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist, squareform

def _score(points):
    """Compute the squared ratio of max to min pairwise distance."""
    # Ensure we have at least 2 points
    if len(points) < 2:
        return np.inf
    d = pdist(points)
    d_min = d.min()
    d_max = d.max()
    if d_min < 1e-12:
        return np.inf
    return (d_max / d_min) ** 2

def _make_stella(params):
    """
    Construct 14 points from two regular tetrahedra (stella octangula) plus 6 more.
    
    Actually, let's think about this differently.
    
    A better approach: Start with a known good configuration for 14 points in 3D.
    
    Known optimal or near-optimal configurations for packing/sphere packing:
    - For small n, we can use known optimal packings.
    - For n=14, one approach is to start with an icosahedron (12 vertices) plus 2 more points.
    
    Actually, let me use a more robust approach:
    1. Start with multiple random initial configurations
    2. Use differential evolution to optimize the positions
    3. Return the best found configuration
    
    To make this more efficient, I'll use a parameterization that respects some symmetry.
    
    Alternative: Use a grid-based approach with optimization.
    
    Let me use differential evolution directly on the 42 coordinates (14 points x 3 dims),
    with constraints to keep points distinct.
    """
    # This function will be replaced by the main optimization
    pass

def _objective(params):
    """Objective function for optimization: minimize squared max/min distance ratio."""
    points = params.reshape((-1, 3))
    return _score(points)

def construct():
    start_time = time.time()
    deadline = 120.0  # seconds, with safety margin from 150s limit
    
    # Strategy: Use differential evolution to find 14 points in 3D
    # that minimize (max_dist / min_dist)^2
    
    # Initial best: use a simple configuration
    # Start with points on a sphere or in a grid
    best_score = np.inf
    best_points = None
    
    # Create initial grid configuration
    side = int(np.ceil(N ** (1 / D)))
    grid = list(np.meshgrid(range(side), range(side), range(side), indexing='ij'))
    grid_points = np.array([[x, y, z] for x, y, z in zip(*grid)]).reshape((-1, 3))[:N]
    best_points = grid_points.astype(float)
    best_score = _score(best_points)
    
    # Try a few random initial configurations
    np.random.seed(42)
    for trial in range(5):
        if time.time() - start_time > deadline - 30:
            break
        # Random points in a cube
        pts = np.random.uniform(0, 1, size=(N, 3))
        s = _score(pts)
        if s < best_score:
            best_score = s
            best_points = pts.copy()
    
    # Use differential evolution for global optimization
    # Bounds: points in [0, 10]^3 (scale doesn't matter for the ratio)
    bounds = [(0, 10)] * (N * 3)
    
    # Run differential evolution with time limit
    try:
        def de_objective(params):
            points = params.reshape((-1, 3))
            # Add penalty for very small distances (to avoid division by zero)
            d = pdist(points)
            d_min = d.min()
            if d_min < 1e-8:
                return 1e10
            d_max = d.max()
            return (d_max / d_min) ** 2
        
        # Use differential evolution
        result = differential_evolution(
            de_objective,
            bounds=bounds,
            maxiter=200,
            popsize=15,
            tol=1e-6,
            seed=42,
            polish=False,  # We'll do local refinement ourselves
            init='sobol',
            updating='deferred'
        )
        
        if result.fun < best_score:
            best_score = result.fun
            best_points = result.x.reshape((-1, 3)).copy()
            
    except Exception as e:
        pass
    
    # Local refinement with L-BFGS-B
    if time.time() - start_time < deadline - 10:
        try:
            # Multiple local refinements from different starting points
            for trial in range(10):
                if time.time() - start_time > deadline - 5:
                    break
                
                if trial == 0:
                    start_pts = best_points.copy()
                else:
                    # Perturb the best solution
                    start_pts = best_points + np.random.normal(0, 0.1, size=(N, 3))
                
                def local_obj(params):
                    points = params.reshape((-1, 3))
                    d = pdist(points)
                    d_min = d.min()
                    if d_min < 1e-10:
                        return 1e10
                    d_max = d.max()
                    return (d_max / d_min) ** 2
                
                res = minimize(
                    local_obj,
                    start_pts.reshape(-1),
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-10}
                )
                
                if res.fun < best_score:
                    best_score = res.fun
                    best_points = res.x.reshape((-1, 3)).copy()
                    
        except Exception as e:
            pass
    
    # Ensure all points are distinct
    d = pdist(best_points)
    if d.min() < 1e-10:
        # If points are too close, add small perturbation
        for i in range(len(best_points)):
            for j in range(i + 1, len(best_points)):
                if np.linalg.norm(best_points[i] - best_points[j]) < 1e-8:
                    best_points[j] += np.random.uniform(-0.01, 0.01, size=3)
    
    return best_points
# EVOLVE-BLOCK-END
