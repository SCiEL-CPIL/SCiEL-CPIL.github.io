"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from scipy.optimize import minimize

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
_n_triangles = len(_idx_i)


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


def _is_inside(p):
    """Check if point p=(x,y) is inside the equilateral triangle."""
    x, y = p
    if y < -1e-9 or x < -1e-9 or x > 1 + 1e-9:
        return False
    if y > np.sqrt(3) * x + 1e-9:
        return False
    if y > np.sqrt(3) * (1 - x) + 1e-9:
        return False
    return True


def _clamp_to_triangle(points):
    """Clamp points to be inside the triangle. Simple projection or rejection."""
    # For simplicity and speed in SA, we just check and nudge.
    # A more robust method would project onto the triangle boundaries.
    # Here we use a simple heuristic: if outside, move towards centroid.
    centroid = np.array([0.5, H / 3])
    for i in range(N):
        if not _is_inside(points[i]):
            # Move towards centroid by 50%
            points[i] = 0.5 * points[i] + 0.5 * centroid
    return points


def _objective_exact(points):
    """Minimize negative exact min-area."""
    for i in range(N):
        if not _is_inside(points[i]):
            return 1e6
    return -_min_area(points)


def construct():
    t_start = time.time()
    deadline = t_start + 140.0

    rng = np.random.RandomState(42)

    # Initial configuration: Vertices + Centroid + Random points
    # This is a reasonable starting point for packing.
    initial_points = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, H],
        [0.5, H / 3],
        [0.25, H / 6],
        [0.75, H / 6],
        [0.5, H / 2],
        [0.1, 0.1],
        [0.9, 0.1],
        [0.5, 0.05],
        [0.5, 0.4]
    ])
    # Ensure they are inside
    initial_points = _clamp_to_triangle(initial_points.copy())
    
    best_points = initial_points.copy()
    best_score = _objective_exact(best_points)
    current_points = initial_points.copy()
    current_score = best_score

    # Simulated Annealing Parameters
    T = 0.01  # Initial temperature
    T_min = 1e-6
    alpha = 0.995  # Cooling rate
    max_iter = 50000  # Max iterations for SA

    # SA Loop
    for i in range(max_iter):
        if time.time() > deadline - 20.0:
            break
        
        # Perturb one random point
        idx = rng.randint(N)
        # Gaussian noise, scale decreases with temperature
        scale = 0.1 * (T + 0.01)
        delta = rng.normal(0, scale, 2)
        
        new_points = current_points.copy()
        new_points[idx] += delta
        
        # Clamp to triangle
        new_points = _clamp_to_triangle(new_points)
        
        new_score = _objective_exact(new_points)
        
        # Acceptance criterion
        if new_score > current_score:
            current_score = new_score
            current_points = new_points
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
        else:
            # Accept worse solution with probability exp((new - current) / T)
            # Note: objective is negative area, so "better" is larger (less negative)
            # So new_score < current_score is worse.
            # Delta E = current_score - new_score (positive if worse)
            delta_e = current_score - new_score
            if delta_e < 0:
                accept = True
            else:
                accept_prob = np.exp(-delta_e / T)
                accept = rng.random() < accept_prob
            
            if accept:
                current_score = new_score
                current_points = new_points
        
        # Cool down
        T *= alpha
        if T < T_min:
            # Reheat occasionally to escape local minima
            T = 0.01
            # Optional: Random restart or large kick
            # Here we just reheat

    # Final Polish with Nelder-Mead
    if time.time() < deadline - 5.0:
        try:
            # Flatten points to 1D array for minimize
            x0 = best_points.flatten()
            
            def _obj_flat(x):
                pts = x.reshape(N, 2)
                pts = _clamp_to_triangle(pts)
                return _objective_exact(pts)
            
            res = minimize(
                _obj_flat,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            if res.fun < best_score:
                best_points = _clamp_to_triangle(res.x.reshape(N, 2))
                best_score = res.fun
        except Exception:
            pass

    return best_points

# EVOLVE-BLOCK-END
