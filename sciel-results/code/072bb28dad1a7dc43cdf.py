import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
from itertools import combinations

N = 11

# Vertices of the equilateral triangle
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2.0])
TRI_AREA = np.sqrt(3) / 4.0

# Precompute all triples
_TRIPLES = list(combinations(range(N), 3))
_N_TRIPLES = len(_TRIPLES)

def _min_area(points):
    """Compute minimum triangle area for N points."""
    min_a = np.inf
    for i, j, k in _TRIPLES:
        pi, pj, pk = points[i], points[j], points[k]
        cross = (pj[0]-pi[0])*(pk[1]-pi[1]) - (pj[1]-pi[1])*(pk[0]-pi[0])
        a = 0.5 * abs(cross)
        if a < min_a:
            min_a = a
    return min_a

def _bary_to_cart(bary):
    """Convert barycentric coords (N,3) to cartesian (N,2)."""
    # bary is (N, 3). V0, V1, V2 are (2,).
    # We want (N, 2).
    # bary[:, 0] is (N,). Multiply by V0 (2,) -> (N, 2).
    return bary[:, 0:1] * V0 + bary[:, 1:2] * V1 + bary[:, 2:3] * V2

def _cart_to_bary(cart):
    """Convert cartesian (N,2) to barycentric coords (N,3)."""
    bary = np.zeros((N, 3))
    for i in range(N):
        x, y = cart[i]
        # l2 = y / (sqrt(3)/2)
        l2 = 2.0 * y / np.sqrt(3)
        # l1 = x - 0.5*l2
        l1 = x - 0.5 * l2
        # l0 = 1 - l1 - l2
        l0 = 1.0 - l1 - l2
        bary[i] = [l0, l1, l2]
    return bary

def _process_bary(params):
    """
    Takes a flat array of 3*N barycentric coordinates.
    Normalizes each triplet to sum to 1 and ensures non-negativity.
    Returns (N, 2) Cartesian points.
    """
    bary = params.reshape(N, 3)
    # Clip to non-negative
    bary = np.maximum(bary, 0)
    # Normalize rows to sum to 1
    sums = bary.sum(axis=1, keepdims=True)
    # Avoid division by zero
    sums[sums == 0] = 1.0
    bary = bary / sums
    return _bary_to_cart(bary)

def _objective_de(params):
    """
    Objective function for Differential Evolution.
    Returns negative min area (to minimize).
    """
    points = _process_bary(params)
    area = _min_area(points)
    return -area

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # Leave 5s margin
    
    def time_left():
        return deadline - time.time()
    
    # Bounds for barycentric coordinates: [0, 1] for each of the 3*N variables
    bounds = [(0.0, 1.0)] * (3 * N)
    
    # Differential Evolution
    # popsize=15, maxiter=50 as suggested by knowledge
    # tol=1e-6
    # strategy='best1bin'
    
    try:
        result = differential_evolution(
            _objective_de, 
            bounds, 
            seed=42, 
            maxiter=50, 
            popsize=15,
            tol=1e-6,
            polish=True,
            callback=None,
            init='sobol' # Use Sobol for better coverage if available, else 'latinhypercube'
        )
        
        best_params = result.x
        best_score = -result.fun
        
    except Exception as e:
        print(f"DE failed: {e}")
        # Fallback to random
        best_params = np.random.rand(3 * N)
        best_score = _objective_de(best_params)

    # Polish with Nelder-Mead on Cartesian coordinates for fine-tuning
    if time_left() > 10.0:
        points = _process_bary(best_params)
        
        # Use Nelder-Mead on Cartesian coordinates
        # We need to map Cartesian back to a parameter vector for NM?
        # NM works on the input vector. Let's use Cartesian (2*N) as input.
        # But we need to ensure points stay in triangle.
        # We can use a penalty function or project.
        # Simpler: Use NM on Barycentric params again?
        # The gradient issue was the problem. NM doesn't need gradients.
        
        def _obj_nm_bary(params):
            pts = _process_bary(params)
            return -_min_area(pts)
            
        try:
            res_nm = minimize(
                _obj_nm_bary, 
                best_params, 
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            
            if -res_nm.fun > best_score:
                best_score = -res_nm.fun
                best_params = res_nm.x
        except Exception as e:
            print(f"NM failed: {e}")

    final_points = _process_bary(best_params)
    
    # Final safety check: ensure points are strictly inside or on boundary
    # _process_bary already normalizes and clips, so they should be valid.
    
    return np.array(final_points)
