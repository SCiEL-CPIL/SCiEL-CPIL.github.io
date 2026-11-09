"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from scipy.optimize import differential_evolution, minimize

H = np.sqrt(3) / 2
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, H])

# Precompute all C(11,3) triangle index triplets
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
    # Area = 0.5 * |cross(p2-p1, p3-p1)|
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - \
            (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = 0.5 * np.abs(cross)
    return areas.min()


def _is_inside(point):
    """Check if a point is inside or on the boundary of the equilateral triangle."""
    x, y = point
    # Inside triangle with vertices (0,0), (1,0), (0.5, sqrt(3)/2)
    if y < -1e-10:
        return False
    # Left edge: y <= sqrt(3) * x  =>  sqrt(3)*x - y >= 0
    if np.sqrt(3) * x - y < -1e-10:
        return False
    # Right edge: y <= sqrt(3) * (1 - x)  =>  sqrt(3)*(1-x) - y >= 0
    if np.sqrt(3) * (1 - x) - y < -1e-10:
        return False
    # Top edge: y <= sqrt(3)/2
    if y - H < -1e-10:
        return False
    return True


def _params_to_points(params):
    """Convert 2N parameters to N points, ensuring they stay inside the triangle.
    
    We use barycentric coordinates: each point is represented by (u, v) where
    u >= 0, v >= 0, u + v <= 1. The point is u*B + v*C + (1-u-v)*A = (u + 0.5*v, H*v).
    """
    points = np.zeros((N, 2))
    for i in range(N):
        u = params[2 * i]
        v = params[2 * i + 1]
        # Clip to valid range
        u = max(0.0, min(1.0, u))
        v = max(0.0, min(1.0 - u, v))
        points[i, 0] = u + 0.5 * v
        points[i, 1] = H * v
    return points


def _objective(params):
    """Objective function for optimization: minimize the negative of the min area."""
    points = _params_to_points(params)
    return -_min_area(points)


def _penalized_objective(params):
    """Objective with penalty for points outside the triangle (shouldn't happen with clipping)."""
    points = _params_to_points(params)
    # Check if any point is outside (with tolerance)
    for p in points:
        if not _is_inside(p):
            return 1e10  # Large penalty
    return -_min_area(points)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Initial random points
    rng = np.random.RandomState(42)
    
    # Try multiple random restarts with DE
    best_params = None
    best_score = -np.inf
    
    # First, get a decent starting point with random search
    for _ in range(50):
        params = np.random.rand(N * 2)
        # Ensure u + v <= 1
        for i in range(N):
            u = params[2 * i]
            v = params[2 * i + 1]
            if u + v > 1:
                scale = 1.0 / (u + v)
                params[2 * i] = u * scale
                params[2 * i + 1] = v * scale
        points = _params_to_points(params)
        score = _min_area(points)
        if score > best_score:
            best_score = score
            best_params = params.copy()
    
    # Differential Evolution for global optimization
    bounds = [(0, 1)] * (N * 2)
    
    def de_objective(params):
        """DE objective with constraint handling."""
        points = _params_to_points(params)
        # Check constraints
        for i in range(N):
            u = params[2 * i]
            v = params[2 * i + 1]
            if u < -1e-9 or v < -1e-9 or u + v > 1 + 1e-9:
                return 1e10
        return -_min_area(points)
    
    try:
        result = differential_evolution(
            de_objective,
            bounds,
            seed=42,
            maxiter=100,
            popsize=20,
            tol=1e-6,
            mutation=(0.5, 1.0),
            recombination=0.8,
            polish=False,
            init='sobol'
        )
        
        if result.fun < 1e9:  # Valid solution
            score = -result.fun
            if score > best_score:
                best_score = score
                best_params = result.x.copy()
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 20:
        # Not enough time for local refinement
        points = _params_to_points(best_params)
        return points
    
    # Local refinement with Nelder-Mead
    def local_objective(params):
        points = _params_to_points(params)
        return -_min_area(points)
    
    try:
        result_local = minimize(
            local_objective,
            best_params,
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10}
        )
        
        if result_local.fun < 1e9:
            score = -result_local.fun
            if score > best_score:
                best_score = score
                best_params = result_local.x.copy()
    except Exception:
        pass
    
    # Final check and return
    elapsed = time.time() - start_time
    if elapsed > deadline - 5:
        # Return best found
        points = _params_to_points(best_params)
        return points
    
    # One more round of local refinement if time permits
    if time.time() - start_time < deadline - 10:
        try:
            result_local2 = minimize(
                local_objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            
            if result_local2.fun < 1e9:
                score = -result_local2.fun
                if score > best_score:
                    best_score = score
                    best_params = result_local2.x.copy()
        except Exception:
            pass
    
    points = _params_to_points(best_params)
    return points

# EVOLVE-BLOCK-END
