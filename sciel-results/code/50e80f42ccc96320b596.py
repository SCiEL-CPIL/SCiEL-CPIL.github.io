"""Heilbronn triangles: maximize min triangle area for 11 points in unit-area equilateral triangle.
Strategy: Smooth Surrogate Optimization (Log-Sum-Exp) + Nelder-Mead Polish.
1. Transform coordinates to barycentric coordinates to enforce convex constraints (u>=0, v>=0, u+v<=1).
2. Define objective as negative log-sum-exp of squared triangle areas (smooth approximation of -min area).
3. Optimize using L-BFGS-B with bounds.
4. Polish the result using Nelder-Mead on the true min-area objective to handle non-differentiability.
5. Multi-start to escape local minima.
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

def _params_to_points(params):
    """Convert barycentric parameters (2*N) to Cartesian points."""
    # params shape (2*N): [u0, v0, u1, v1, ...]
    points = np.zeros((N, 2))
    for i in range(N):
        u = params[2*i]
        v = params[2*i+1]
        # Point = (1-u-v)*V0 + u*V1 + v*V2
        points[i] = u * V1 + v * V2
    return points

def _points_to_params(points):
    """Convert Cartesian points to barycentric parameters."""
    # Inverse of above.
    # P = u*V1 + v*V2
    # x = u + 0.5v
    # y = (sqrt(3)/2) v
    # v = 2y / sqrt(3)
    # u = x - 0.5v
    params = np.zeros(2*N)
    for i in range(N):
        x, y = points[i]
        v = 2 * y / np.sqrt(3)
        u = x - 0.5 * v
        params[2*i] = u
        params[2*i+1] = v
    return params

def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by 3 of the points, normalized by AREA."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        pi = points[i]
        for j in range(i + 1, n):
            pj = points[j]
            for k in range(j + 1, n):
                pk = points[k]
                a = 0.5 * abs((pj[0] - pi[0]) * (pk[1] - pi[1]) - (pk[0] - pi[0]) * (pj[1] - pi[1]))
                if a < min_area:
                    min_area = a
    return min_area / AREA

def _log_sum_exp_surrogate(params, temperature=1.0):
    """
    Objective function for L-BFGS-B.
    Minimizes -log(sum(exp(-area_i^2 / temperature)))
    which approximates -min(area_i^2) for small temperature.
    We maximize min area, so we minimize negative log-sum-exp.
    """
    points = _params_to_points(params)
    
    # Calculate all triangle areas squared
    n = len(points)
    areas_sq = []
    for i in range(n):
        pi = points[i]
        for j in range(i + 1, n):
            pj = points[j]
            for k in range(j + 1, n):
                pk = points[k]
                cross = (pj[0] - pi[0]) * (pk[1] - pi[1]) - (pk[0] - pi[0]) * (pj[1] - pi[1])
                area_sq = 0.25 * cross * cross # (0.5 * cross)^2
                areas_sq.append(area_sq)
    
    areas_sq = np.array(areas_sq)
    
    # Scale for numerical stability
    # We want to approximate min(areas_sq).
    # log(sum(exp(-a_i/T))) ~ -min(a_i)/T + log(count)
    # So -log(sum(exp(-a_i/T))) ~ min(a_i)/T - log(count)
    # Minimizing this is equivalent to minimizing min(a_i).
    
    # Clip to prevent overflow/underflow issues in exp
    # If T is small, -a_i/T is large negative.
    # We can subtract max to stabilize, but since we want min, we look at the tail.
    # Actually, standard log-sum-exp stability:
    # L = log(sum(exp(x_i))) = max(x) + log(sum(exp(x_i - max(x))))
    # Here x_i = -a_i / T.
    # max(x_i) corresponds to min(a_i).
    
    x = -areas_sq / temperature
    max_x = np.max(x)
    
    # If max_x is very negative (all areas huge), exp(x-max_x) is fine.
    # If max_x is close to 0, fine.
    
    lse = max_x + np.log(np.sum(np.exp(x - max_x)))
    
    # We want to minimize -min(area).
    # The surrogate is -lse.
    # Note: lse includes log(N_triangles) which is constant.
    return -lse

def _true_objective(params):
    """Objective for Nelder-Mead: -min_triangle_area."""
    points = _params_to_points(params)
    return -_min_triangle_area(points)

def _generate_random_params(rng):
    """Generate random valid barycentric coordinates."""
    params = np.zeros(2*N)
    for i in range(N):
        u = rng.random()
        v = rng.random()
        if u + v > 1.0:
            u = 1.0 - v
        # Add small randomization to avoid exact boundaries if possible, 
        # but boundaries are allowed.
        params[2*i] = u
        params[2*i+1] = v
    return params

def _generate_structured_params():
    """Generate structured initial configuration in barycentric coords."""
    pts = [
        V0 + np.array([0.05, 0.05]),
        V1 + np.array([-0.05, 0.05]),
        V2 + np.array([0.0, -0.05]),
        (V0 + V1) / 2.0,
        (V1 + V2) / 2.0,
        (V2 + V0) / 2.0,
        (V0 + V1 + V2) / 3.0,
        (V0 + 2*V1 + V2) / 4.0,
        (2*V0 + V1 + V2) / 4.0,
        (V0 + V1 + 2*V2) / 4.0,
        (2*V0 + 2*V1 + V2) / 5.0
    ]
    return _points_to_params(np.array(pts))

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    best_params = None
    best_score = 0.0
    
    rng = random.Random(42)
    
    # Bounds for barycentric coords: u >= 0, v >= 0, u+v <= 1
    # This is a triangular region in the (u,v) plane.
    # L-BFGS-B handles box constraints, not linear ones directly.
    # However, u in [0,1], v in [0,1] is a box.
    # We need to enforce u+v <= 1.
    # We can use a penalty or project. 
    # Better: Use a transformation or just rely on the fact that 
    # if we start inside and step size is small, we stay inside? 
    # No, L-BFGS-B might step outside.
    # Let's use a penalty function for the constraint u+v > 1.
    
    def _penalized_surrogate(params, temperature=0.01, penalty_weight=1e6):
        points = _params_to_points(params)
        # Check constraints
        penalty = 0.0
        for i in range(N):
            u = params[2*i]
            v = params[2*i+1]
            if u < 0: penalty += penalty_weight * u * u
            if v < 0: penalty += penalty_weight * v * v
            if u + v > 1: penalty += penalty_weight * (u + v - 1) * (u + v - 1)
        
        base = _log_sum_exp_surrogate(params, temperature)
        return base + penalty

    # Initializations
    initial_params_list = [_generate_structured_params()]
    for _ in range(5):
        initial_params_list.append(_generate_random_params(rng))
    
    bounds = [(0.0, 1.0)] * (2*N)
    
    for idx, init_p in enumerate(initial_params_list):
        if time.time() > deadline:
            break
            
        # Phase 1: L-BFGS-B on smooth surrogate
        # Use adaptive temperature? Start with larger T for global view, then smaller.
        # Given time constraints, a single run with moderate T might be best.
        # T=0.01 is quite sharp. Let's try T=0.05 first, then polish.
        
        try:
            res = minimize(
                _penalized_surrogate, 
                init_p, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            current_params = res.x
            current_score = _min_triangle_area(_params_to_points(current_params))
            
            if current_score > best_score:
                best_score = current_score
                best_params = current_params.copy()
                
        except Exception:
            continue
            
        # Phase 2: Nelder-Mead Polish on true objective
        # NM is derivative-free and handles non-smoothness well.
        # It is local, so it needs a good start (provided by L-BFGS-B).
        
        if time.time() > deadline - 10:
            break
            
        try:
            res_nm = minimize(
                _true_objective, 
                current_params, 
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            
            polished_params = res_nm.x
            polished_score = _min_triangle_area(_params_to_points(polished_params))
            
            if polished_score > best_score:
                best_score = polished_score
                best_params = polished_params.copy()
                
        except Exception:
            pass

    # Fallback
    if best_params is None:
        best_params = _generate_random_params(rng)
        
    final_points = _params_to_points(best_params)
    
    # Final safety check: ensure points are strictly inside (or on boundary)
    # The optimization should have kept them there, but let's clip just in case.
    # Actually, _params_to_points with u,v in [0,1] and u+v<=1 guarantees inside.
    # If penalty failed, points might be outside.
    # Let's verify and clip if necessary.
    for i in range(N):
        x, y = final_points[i]
        if x < 0: final_points[i, 0] = 0
        if x > 1: final_points[i, 0] = 1
        if y < 0: final_points[i, 1] = 0
        max_y = np.sqrt(3) * x if x <= 0.5 else np.sqrt(3) * (1.0 - x)
        if y > max_y: final_points[i, 1] = max_y

    return final_points
