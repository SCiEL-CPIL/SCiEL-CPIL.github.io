"""Improved packing of 50 equal circles in a unit square via multi-start SLSQP."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import random

def _radius_score(centres):
    """Compute the maximum common radius for given centres in [-0.5, 0.5]^2."""
    # Distance to boundaries
    dist_bound = np.min(np.stack([
        0.5 - centres[:, 0],
        0.5 + centres[:, 0],
        0.5 - centres[:, 1],
        0.5 + centres[:, 1]
    ], axis=1), axis=1)
    
    # Pairwise distances
    from scipy.spatial.distance import pdist
    if len(centres) < 2:
        min_pair = np.inf
    else:
        min_pair = pdist(centres, metric='euclidean').min()
    
    r = min(min_pair / 2.0, dist_bound.min())
    return r

def _slsqp_objective_and_constraints(x, N):
    """
    x: array of length 2*N + 1. First 2*N are coordinates, last is r.
    Minimize -r.
    """
    coords = x[:2*N].reshape(N, 2)
    r = x[2*N]
    
    # Objective: minimize -r
    obj = -r
    
    # Constraints
    # 1. Boundary constraints:
    # 0.5 - x_i - r >= 0
    # 0.5 + x_i - r >= 0
    # 0.5 - y_i - r >= 0
    # 0.5 + y_i - r >= 0
    
    cons = []
    
    # Boundary constraints
    for i in range(N):
        xi = coords[i, 0]
        yi = coords[i, 1]
        # Right wall
        cons.append({'type': 'ineq', 'fun': lambda xi=xi, r=r: 0.5 - xi - r})
        # Left wall
        cons.append({'type': 'ineq', 'fun': lambda xi=xi, r=r: 0.5 + xi - r})
        # Top wall
        cons.append({'type': 'ineq', 'fun': lambda yi=yi, r=r: 0.5 - yi - r})
        # Bottom wall
        cons.append({'type': 'ineq', 'fun': lambda yi=yi, r=r: 0.5 + yi - r})
        
    # Pairwise distance constraints: ||Ci - Cj|| - 2*r >= 0
    # To avoid N^2 constraints which might be slow, we can try to be smart, 
    # but for N=50, 1225 constraints might be okay for SLSQP if optimized.
    # However, defining 1225 lambda functions in a loop is slow in Python.
    # We can define a single constraint function that returns an array? 
    # SLSQP expects a list of dicts or a single dict with 'fun' returning an array for 'ineq'.
    
    # Let's use a single vector constraint for pairwise distances.
    # We need to construct the vector of distances - 2r.
    
    def pairwise_dist_constraint(x):
        coords = x[:2*N].reshape(N, 2)
        r = x[2*N]
        # Calculate all pairwise distances
        # Using broadcasting for efficiency
        # diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        # dists = np.linalg.norm(diff, axis=-1)
        # This creates a NxN matrix. We only need i < j.
        # Let's use pdist for speed inside the constraint? No, pdist is not differentiable easily for gradients if needed, 
        # but SLSQP approximates gradients.
        # For N=50, a simple loop or vectorized calculation is fine.
        
        # Vectorized calculation of upper triangle distances
        i_idx, j_idx = np.triu_indices(N, k=1)
        diffs = coords[i_idx] - coords[j_idx]
        dists = np.linalg.norm(diffs, axis=1)
        return dists - 2.0 * r

    # Add pairwise constraint as a single vector inequality
    cons.append({'type': 'ineq', 'fun': pairwise_dist_constraint})
    
    # Bounds for coordinates: [-0.5, 0.5]
    # Bounds for r: [0, 0.5]
    bounds = [(-0.5, 0.5)] * (2 * N)
    bounds.append((0.0, 0.5))
    
    return obj, cons, bounds

def _optimize_slsqp(centres, r_init, max_iter=1000):
    """
    Run SLSQP optimization starting from given centres and r_init.
    """
    x0 = np.concatenate([centres.flatten(), [r_init]])
    
    # We need to pass constraints correctly. 
    # The lambda closures in _slsqp_objective_and_constraints are created every time? 
    # It's better to define the constraint functions once or use a closure that captures N.
    
    # Let's refactor to avoid recreating lambdas in a loop if possible, 
    # but SLSQP calls the functions many times.
    
    # A more efficient way: define the constraint functions outside or use a class.
    # Given the complexity, let's just call minimize with a helper that returns the correct structure.
    
    def objective(x):
        return -x[2*N] # Maximize r by minimizing -r

    def cons_func(x):
        coords = x[:2*N].reshape(N, 2)
        r = x[2*N]
        
        # Boundary constraints
        # 0.5 - x - r >= 0
        # 0.5 + x - r >= 0
        # 0.5 - y - r >= 0
        # 0.5 + y - r >= 0
        c1 = 0.5 - coords[:, 0] - r
        c2 = 0.5 + coords[:, 0] - r
        c3 = 0.5 - coords[:, 1] - r
        c4 = 0.5 + coords[:, 1] - r
        
        # Pairwise
        i_idx, j_idx = np.triu_indices(N, k=1)
        diffs = coords[i_idx] - coords[j_idx]
        dists = np.linalg.norm(diffs, axis=1)
        c5 = dists - 2.0 * r
        
        return np.concatenate([c1, c2, c3, c4, c5])

    constraints = {'type': 'ineq', 'fun': cons_func}
    
    bounds = [(-0.5, 0.5)] * (2 * N)
    bounds.append((0.0, 0.5))
    
    try:
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': max_iter, 'ftol': 1e-9})
        if res.success or res.fun < -r_init:
            best_coords = res.x[:2*N].reshape(N, 2)
            best_r = res.x[2*N]
            # Verify the result
            actual_r = _radius_score(best_coords)
            return best_coords, actual_r
        else:
            return centres, _radius_score(centres)
    except Exception:
        return centres, _radius_score(centres)

def _construct_grid():
    """Create a grid-based initial configuration."""
    # 7x7 grid = 49 points. Add one more.
    # Or 8x7 = 56 points, take 50.
    # Let's use a slightly perturbed grid to avoid symmetry issues.
    
    # 7x7 grid
    m = 7
    ticks_x = np.linspace(-0.5 + 1/(2*m), 0.5 - 1/(2*m), m)
    ticks_y = np.linspace(-0.5 + 1/(2*m), 0.5 - 1/(2*m), m)
    grid = np.array([(x, y) for y in ticks_y for x in ticks_x])
    
    # We need 50. Add one point in the center of a cell?
    # Or use 8x7 and drop 6.
    # Let's use 8x7 grid, take first 50.
    m_x = 8
    m_y = 7
    ticks_x2 = np.linspace(-0.5 + 1/(2*m_x), 0.5 - 1/(2*m_x), m_x)
    ticks_y2 = np.linspace(-0.5 + 1/(2*m_y), 0.5 - 1/(2*m_y), m_y)
    grid2 = np.array([(x, y) for y in ticks_y2 for x in ticks_x2])
    
    return grid2[:N]

def construct():
    """Construct 50 circle centres in [-0.5, 0.5]^2 maximizing the common radius."""
    # Start with a grid configuration
    centres = _construct_grid()
    
    # Initial radius estimate
    r_init = _radius_score(centres)
    
    best_centres = centres.copy()
    best_score = r_init
    
    # Multi-start SLSQP
    # We'll try a few different starting points
    starts = []
    
    # 1. The grid itself
    starts.append(centres.copy())
    
    # 2. Perturbed grids
    np.random.seed(42)
    for i in range(5):
        noise = np.random.normal(0, 0.02, size=(N, 2))
        perturbed = centres + noise
        perturbed = np.clip(perturbed, -0.5, 0.5)
        starts.append(perturbed)
        
    # 3. Random points (might be bad, but worth a shot)
    # Actually, random points usually have very small min distance, leading to small r.
    # SLSQP might struggle to escape. 
    # Let's stick to grid-based starts which are already decent.
    
    for start in starts:
        # Estimate r for this start
        r_s = _radius_score(start)
        if r_s > best_score:
            best_score = r_s
            best_centres = start.copy()
            
        # Run optimization
        opt_centres, opt_score = _optimize_slsqp(start, r_s, max_iter=500)
        
        if opt_score > best_score:
            best_score = opt_score
            best_centres = opt_centres
    
    return best_centres
# EVOLVE-BLOCK-END
