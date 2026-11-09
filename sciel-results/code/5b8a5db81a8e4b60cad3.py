"""Heilbronn triangles: place 11 points in unit equilateral triangle to maximize min triangle area."""
import numpy as np
import time
import itertools
from scipy.optimize import differential_evolution, minimize

N = 11
H = np.sqrt(3) / 2.0
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, H])

# Precompute all C(11,3) = 165 triangle index triplets
_idx_i, _idx_j, _idx_k = [], [], []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _idx_i.append(i)
            _idx_j.append(j)
            _idx_k.append(k)
_idx_i = np.array(_idx_i)
_idx_j = np.array(_idx_j)
_idx_k = np.array(_idx_k)


def _min_area(points):
    """Compute minimum triangle area among all C(N,3) triples. points: (N,2)."""
    p1 = points[_idx_i]
    p2 = points[_idx_j]
    p3 = points[_idx_k]
    areas = 0.5 * np.abs(np.cross(p2 - p1, p3 - p1))
    return np.min(areas)

def _bary_to_cart(u, v):
    """Convert barycentric coordinates (u, v, 1-u-v) to Cartesian (x, y).
    Assumes u, v are such that u+v <= 1.
    A=(0,0), B=(1,0), C=(0.5, H)
    P = u*A + v*B + (1-u-v)*C
    x = u*0 + v*1 + (1-u-v)*0.5 = v + 0.5 - 0.5u - 0.5v = 0.5 + 0.5v - 0.5u
    y = u*0 + v*0 + (1-u-v)*H = H(1-u-v)
    """
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = H * w
    return np.array([x, y])

def _params_to_points(params):
    """
    params: 11 floats.
    params[0] = u0 for the symmetric point (u0, u0, 1-2u0).
    params[1:6] = u1..u5
    params[6:11] = v1..v5
    
    Returns 11 points.
    """
    points = np.zeros((N, 2))
    
    # Point 0: Symmetric on axis
    u0 = params[0]
    # Ensure u0 is in [0, 0.5] so that w = 1-2u0 >= 0
    u0 = np.clip(u0, 0.0, 0.5)
    points[0] = _bary_to_cart(u0, u0)
    
    # Pairs
    for i in range(5):
        ui = params[1 + i]
        vi = params[6 + i]
        
        # Clip to simplex if necessary (u+v <= 1)
        s = ui + vi
        if s > 1.0:
            ui /= s
            vi /= s
        # Also ensure non-negative
        ui = np.clip(ui, 0.0, 1.0)
        vi = np.clip(vi, 0.0, 1.0)
        # Re-check sum after clipping? Clipping might break sum<=1 if one was negative? 
        # But DE bounds are [0,1], so ui,vi >= 0.
        # If ui+vi > 1, we scaled.
        
        # Point i+1
        points[i + 1] = _bary_to_cart(ui, vi)
        # Point 11-(i+1) = 10-i
        points[10 - i] = _bary_to_cart(vi, ui)
        
    return points

def _objective(params):
    points = _params_to_points(params)
    return -_min_area(points)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    # Bounds for 11 parameters
    # u0 in [0, 0.5]
    # ui, vi in [0, 1]
    bounds = [(0.0, 0.5)] + [(0.0, 1.0)] * 10

    best_params = None
    best_value = -np.inf

    # Phase 1: Differential Evolution
    # Reduced dimensionality (11 vs 22) should help convergence
    n_restarts = 3
    for r in range(n_restarts):
        if time.time() > deadline - 20.0:
            break
        try:
            result = differential_evolution(
                _objective,
                bounds,
                seed=42 + r,
                maxiter=300,
                popsize=15,
                tol=1e-6,
                polish=False,
                init='latinhypercube',
                mutation=(0.5, 1.5),
                recombination=0.7,
            )
            val = -result.fun
            if val > best_value:
                best_value = val
                best_params = result.x.copy()
        except Exception:
            continue

    # Phase 2: Nelder-Mead refinement
    if best_params is not None:
        if time.time() < deadline - 5.0:
            try:
                result2 = minimize(
                    _objective,
                    best_params,
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True},
                )
                val2 = -result2.fun
                if val2 > best_value:
                    best_value = val2
                    best_params = result2.x.copy()
            except Exception:
                pass

    if best_params is None:
        # Fallback
        rng = np.random.RandomState(42)
        params = rng.rand(11)
        params[0] *= 0.5
        best_params = params

    points = _params_to_points(best_params)
    return points
