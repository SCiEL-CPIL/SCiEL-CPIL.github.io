"""Heilbronn triangles: maximize min triangle area for 11 points in unit-area equilateral triangle.
Strategy: Barycentric coordinates + Trust-Region Reflective optimization.
"""
import numpy as np
import random
import time
from scipy.optimize import minimize

N = 11

# Equilateral triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2.0])
AREA = np.sqrt(3) / 4.0  # area of the equilateral triangle

# Barycentric to Cartesian conversion
def bary_to_cart(bary):
    """Convert barycentric coordinates (n, 3) to Cartesian (n, 2)."""
    # bary[:, 0] * V0 + bary[:, 1] * V1 + bary[:, 2] * V2
    # V0 is (0,0), so term drops
    x = bary[:, 1] * V1[0] + bary[:, 2] * V2[0]
    y = bary[:, 1] * V1[1] + bary[:, 2] * V2[1]
    return np.column_stack((x, y))

def cart_to_bary(cart):
    """Convert Cartesian (n, 2) to barycentric (n, 3)."""
    # Solve for barycentric coords u, v, w such that p = u*V0 + v*V1 + w*V2
    # and u+v+w=1
    # x = v + 0.5w
    # y = sqrt(3)/2 * w
    # w = 2*y / sqrt(3)
    # v = x - 0.5w
    # u = 1 - v - w
    w = 2 * cart[:, 1] / np.sqrt(3)
    v = cart[:, 0] - 0.5 * w
    u = 1 - v - w
    return np.column_stack((u, v, w))

def _min_triangle_area_from_bary(bary):
    """Compute the minimum area of any triangle formed by 3 of the points, normalized by triangle area.
    Input: barycentric coordinates (n, 3)
    """
    # Area of triangle with barycentric coords b_i, b_j, b_k is:
    # Area = 0.5 * |det([b_i - b_j, b_i - b_k])| * Area_of_Unit_Triangle
    # Actually, the determinant of the matrix with rows b_i, b_j, b_k gives the area ratio.
    # Let M be the matrix with rows b_i, b_j, b_k.
    # Area(i,j,k) / Area(Triangle) = |det(M)|
    # We want to maximize the minimum of these.
    # So we minimize the negative of the minimum, or maximize the minimum.
    # For scipy minimize, we minimize -min_area.
    
    n = bary.shape[0]
    min_det = float('inf')
    
    # Precompute differences? No, just loop.
    # To speed up, we can vectorize the determinant calculation for all triples?
    # There are C(11,3) = 165 triples.
    # We can compute all determinants at once.
    
    # Indices for triples
    # This is a fixed set of indices for N=11
    # i < j < k
    # We can precompute the indices once globally or inside.
    
    # For efficiency, let's just loop, as N is small.
    for i in range(n):
        bi = bary[i]
        for j in range(i + 1, n):
            bj = bary[j]
            for k in range(j + 1, n):
                bk = bary[k]
                # det of [[bi], [bj], [bk]]
                # | bi0 bi1 bi2 |
                # | bj0 bj1 bj2 |
                # | bk0 bk1 bk2 |
                det = (bi[0]*(bj[1]*bk[2] - bj[2]*bk[1]) - 
                       bi[1]*(bj[0]*bk[2] - bj[2]*bk[0]) + 
                       bi[2]*(bj[0]*bk[1] - bj[1]*bk[0]))
                abs_det = abs(det)
                if abs_det < min_det:
                    min_det = abs_det
    return min_det

def _objective(params):
    """Objective function for scipy.optimize.minimize.
    params: flat array of barycentric coordinates (n*3,)
    Returns: -min_triangle_area (to be minimized)
    """
    bary = params.reshape(N, 3)
    # Ensure valid barycentric coordinates (sum to 1, non-negative)
    # The constraints handle this, but for safety in objective evaluation:
    # If invalid, return a large penalty
    if np.any(bary < -1e-9) or np.any(np.abs(bary.sum(axis=1) - 1.0) > 1e-6):
        return 1000.0
    
    # Clip small negatives for numerical stability in area calc
    bary_clipped = np.clip(bary, 0, None)
    # Renormalize just in case
    bary_clipped = bary_clipped / bary_clipped.sum(axis=1, keepdims=True)
    
    min_area = _min_triangle_area_from_bary(bary_clipped)
    return -min_area

def _generate_structured_initial_bary():
    """Generate a structured initial configuration in barycentric coordinates."""
    points = []
    # Vertices
    points.append([1.0, 0.0, 0.0])
    points.append([0.0, 1.0, 0.0])
    points.append([0.0, 0.0, 1.0])
    
    # Midpoints of edges
    points.append([0.5, 0.5, 0.0])
    points.append([0.5, 0.0, 0.5])
    points.append([0.0, 0.5, 0.5])
    
    # Centroid
    points.append([1/3, 1/3, 1/3])
    
    # Additional points
    # 1/3 and 2/3 along edges
    points.append([1/3, 2/3, 0.0])
    points.append([2/3, 1/3, 0.0])
    
    # Two more interior points
    # Symmetric placement
    points.append([0.4, 0.3, 0.3])
    points.append([0.3, 0.4, 0.3])
    
    # Ensure exactly N points
    points = points[:N]
    return np.array(points).flatten()

def _generate_random_bary(rng):
    """Generate random barycentric coordinates."""
    bary = []
    for _ in range(N):
        # Random point in simplex
        u = rng.random()
        v = rng.random()
        if u + v > 1.0:
            u = 1.0 - u
            v = 1.0 - v
        bary.append([1.0 - u - v, u, v])
    return np.array(bary).flatten()

def _optimize_barycentric(initial_bary, rng, time_limit):
    """Run trust-constr optimization from an initial barycentric configuration."""
    start_time = time.time()
    
    # Constraints:
    # 1. Sum of barycentric coords for each point is 1
    # 2. Each barycentric coord is non-negative
    
    # Linear constraints for sum to 1
    # For each point i, sum(bary[i, :]) = 1
    # This is a linear equality constraint.
    # Inequality constraints: bary[i, j] >= 0
    
    # scipy trust-constr supports LinearConstraint and NonlinearConstraint.
    # We can define bounds for each variable to be [0, 1].
    # And linear equality constraints for the sums.
    
    # Bounds: (0, 1) for all 3*N variables
    bounds = [(0.0, 1.0)] * (N * 3)
    
    # Linear Equality Constraints:
    # A @ x = b
    # For point 0: x[0] + x[1] + x[2] = 1
    # For point 1: x[3] + x[4] + x[5] = 1
    # ...
    A_eq = np.zeros((N, N * 3))
    for i in range(N):
        A_eq[i, 3*i:3*i+3] = 1.0
    b_eq = np.ones(N)
    
    # LinearConstraint(A_eq, lb=b_eq, ub=b_eq)
    
    from scipy.optimize import LinearConstraint
    
    cons = [
        LinearConstraint(A_eq, lb=b_eq, ub=b_eq)
    ]
    
    # Use trust-constr
    # Method: 'trust-constr'
    # We don't have an analytic gradient, so it will use numerical approximation.
    # This might be slow, but N=33 variables is small.
    
    try:
        res = minimize(
            _objective,
            initial_bary,
            method='trust-constr',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 200, 'gtol': 1e-8, 'xtol': 1e-10}
        )
        
        if time.time() > start_time + time_limit:
            return None, 0.0
            
        # Check if result is valid
        if res.success or res.fun < 0: # fun is -min_area, so negative means positive area
            final_bary = res.x.reshape(N, 3)
            # Ensure constraints are met strictly for output
            final_bary = np.clip(final_bary, 0, None)
            final_bary = final_bary / final_bary.sum(axis=1, keepdims=True)
            final_bary_flat = final_bary.flatten()
            
            min_area = _min_triangle_area_from_bary(final_bary)
            return final_bary_flat, min_area
        else:
            return None, 0.0
            
    except Exception:
        return None, 0.0

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s
    
    best_params = None
    best_min = 0.0
    
    rng = random.Random(42)
    
    # Strategy 1: Structured initial
    init_struct = _generate_structured_initial_bary()
    params, min_val = _optimize_barycentric(init_struct, rng, 30.0)
    if params is not None and min_val > best_min:
        best_min = min_val
        best_params = params
    
    # Strategy 2: Random restarts
    # We have ~110s left. Each optimization takes ~20-30s?
    # Let's do a few more restarts with shorter time limits.
    remaining_time = deadline - time.time()
    
    if remaining_time > 10:
        # Try 3 random restarts
        for i in range(3):
            if time.time() > deadline - 10:
                break
            init_rand = _generate_random_bary(rng)
            # Use the best found so far as a seed for perturbation?
            # Or purely random?
            # Let's perturb the best solution.
            if best_params is not None:
                # Add small noise
                noise = np.random.normal(0, 0.05, size=best_params.shape)
                perturbed = best_params + noise
                # Project back to simplex approximately? 
                # The optimizer will handle constraints.
                # Just use perturbed as start.
                init_rand = perturbed
            
            params, min_val = _optimize_barycentric(init_rand, rng, 20.0)
            if params is not None and min_val > best_min:
                best_min = min_val
                best_params = params
                
    # Strategy 3: Final polish if time permits
    if best_params is not None and time.time() < deadline - 5:
        params, min_val = _optimize_barycentric(best_params, rng, 10.0)
        if params is not None and min_val > best_min:
            best_min = min_val
            best_params = params
            
    # Convert best barycentric to Cartesian
    if best_params is None:
        # Fallback to structured
        best_params = _generate_structured_initial_bary()
        
    best_bary = best_params.reshape(N, 3)
    # Ensure valid
    best_bary = np.clip(best_bary, 0, None)
    best_bary = best_bary / best_bary.sum(axis=1, keepdims=True)
    
    points = bary_to_cart(best_bary)
    
    return points
# EVOLVE-BLOCK-END
