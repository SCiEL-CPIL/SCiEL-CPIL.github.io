import numpy as np
from scipy.optimize import basinhopping, minimize
from itertools import combinations
import time

N = 11
TIME_LIMIT = 115.0
H = np.sqrt(3) / 2

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, H])

# Pre-compute triangle index combinations
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)  # (165, 3)

def convert_bary_to_cart(x):
    """
    Convert barycentric representation to Cartesian points.
    x shape: (2*N,)
    For each point i: x[2*i] = u_i (lambda_1), x[2*i+1] = v_i (lambda_2)
    lambda_0 = 1 - u_i - v_i
    """
    pts = np.zeros((N, 2))
    for i in range(N):
        u = x[2*i]
        v = x[2*i+1]
        l0 = 1.0 - u - v
        pts[i] = l0 * V0 + u * V1 + v * V2
    return pts

def compute_min_area_cart(pts):
    """Compute minimum triangle area for points in Cartesian coordinates."""
    pi = pts[tri_idx[:, 0]]  # (165, 2)
    pj = pts[tri_idx[:, 1]]  # (165, 2)
    pk = pts[tri_idx[:, 2]]  # (165, 2)
    
    v1 = pj - pi  # (165, 2)
    v2 = pk - pi  # (165, 2)
    
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def objective(x):
    """
    Objective function for optimization.
    x shape: (2*N,)
    """
    pts = convert_bary_to_cart(x)
    min_area = compute_min_area_cart(pts)
    return -min_area

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    # Define constraints for SLSQP
    constraints = []
    for i in range(N):
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i+1]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2*i] - x[2*i+1]})
    
    # Initial guess
    rng = np.random.default_rng(42)
    x0 = np.zeros(2*N)
    for i in range(N):
        u = rng.uniform(0, 1)
        v = rng.uniform(0, 1-u)
        x0[2*i] = u
        x0[2*i+1] = v
    
    # Define local minimizer wrapper for SLSQP
    def local_minimizer(x):
        res = minimize(
            objective, 
            x, 
            method='SLSQP', 
            constraints=constraints,
            options={'maxiter': 100, 'ftol': 1e-12}
        )
        return res.x, res.fun, res.success

    best_x = x0
    best_f = objective(x0)
    current_x = x0
    
    # Run Basin Hopping in chunks to respect time limit
    while time.time() < deadline - 10.0:
        niter_chunk = 5 # Small chunks to check time frequently
        
        try:
            res = basinhopping(
                objective, 
                x0=current_x, 
                minimizer=local_minimizer,
                niter=niter_chunk,
                stepsize=0.5,
                seed=int(time.time()) # Random seed for diversity
            )
            
            # Update best
            if res.fun < best_f:
                best_f = res.fun
                best_x = res.x
            
            # Use the result of this run as the start for the next
            current_x = res.x
            
        except Exception as e:
            break

    # Final polish
    try:
        res_final = minimize(
            objective, 
            best_x, 
            method='SLSQP', 
            constraints=constraints,
            options={'maxiter': 200, 'ftol': 1e-14}
        )
        if res_final.fun < best_f:
            best_x = res_final.x
    except Exception:
        pass

    best_pts = convert_bary_to_cart(best_x)
    return best_pts

# EVOLVE-BLOCK-END
