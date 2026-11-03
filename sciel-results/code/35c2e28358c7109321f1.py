"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = 145.0  # seconds; evaluator kills at 150s, safety margin
_TRI_AREA = np.sqrt(3) / 4.0  # area of equilateral triangle with side 1


def _bary_to_cart(bary):
    """Convert barycentric (a, b) with c=1-a-b to Cartesian.
    Triangle vertices: V0=(0,0), V1=(1,0), V2=(0.5, sqrt(3)/2)
    Point = a*V0 + b*V1 + c*V2 where c = 1-a-b
    """
    a = bary[:, 0]
    b = bary[:, 1]
    c = 1.0 - a - b
    x = b * 1.0 + c * 0.5
    y = c * (np.sqrt(3) / 2)
    return np.column_stack([x, y])


def _cart_to_bary(points):
    """Convert Cartesian to barycentric (a, b) where c = 1-a-b."""
    x = points[:, 0]
    y = points[:, 1]
    # c = y / (sqrt(3)/2) = 2y/sqrt(3)
    c = 2.0 * y / np.sqrt(3)
    # b = x - c*0.5 = x - y/sqrt(3)
    b = x - c * 0.5
    a = 1.0 - b - c
    return np.column_stack([a, b])


def _clamp_bary(bary):
    """Clamp barycentric coordinates to valid region: a>=0, b>=0, a+b<=1."""
    a = np.clip(bary[:, 0], 0, 1)
    b = np.clip(bary[:, 1], 0, 1)
    # Ensure a + b <= 1
    s = a + b
    mask = s > 1.0
    if np.any(mask):
        scale = 1.0 / s[mask]
        a[mask] *= scale
        b[mask] *= scale
    return np.column_stack([a, b])


def _min_triangle_area(points):
    """Compute minimum triangle area among all triples of points."""
    n = points.shape[0]
    min_area = float('inf')
    for i in range(n):
        pi = points[i]
        for j in range(i + 1, n):
            pj = points[j]
            d = pj - pi
            for k in range(j + 1, n):
                pk = points[k]
                cross = d[0] * (pk[1] - pi[1]) - d[1] * (pk[0] - pi[0])
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
    return min_area


def _min_triangle_area_fast(points):
    """Vectorized minimum triangle area computation."""
    n = points.shape[0]
    min_area = float('inf')
    for i in range(n):
        pi = points[i]
        for j in range(i + 1, n):
            pj = points[j]
            d = pj - pi
            if j + 1 < n:
                rest = points[j + 1:]
                cross = d[0] * (rest[:, 1] - pi[1]) - d[1] * (rest[:, 0] - pi[0])
                areas = 0.5 * np.abs(cross)
                local_min = np.min(areas)
                if local_min < min_area:
                    min_area = local_min
    return min_area


def _objective_bary(bary_flat):
    """Objective: minimize negative min triangle area. bary_flat: 2N array."""
    bary = bary_flat.reshape(N, 2)
    bary = _clamp_bary(bary)
    points = _bary_to_cart(bary)
    area = _min_triangle_area_fast(points)
    return -area


def _make_initial_bary(rng):
    """Generate a random valid barycentric configuration."""
    bary = np.zeros((N, 2))
    for i in range(N):
        # Random point in triangle using sqrt method
        u = rng.random()
        v = rng.random()
        if u + v > 1.0:
            u = 1.0 - u
            v = 1.0 - v
        bary[i, 0] = u
        bary[i, 1] = v
    return bary


def _sa_optimize(bary_init, start_time, deadline, rng):
    """Simulated Annealing in barycentric space."""
    bary = _clamp_bary(bary_init.copy())
    current_bary = bary.copy()
    current_points = _bary_to_cart(current_bary)
    current_area = _min_triangle_area_fast(current_points)
    
    best_bary = current_bary.copy()
    best_area = current_area
    
    T = 0.05  # initial temperature
    T_min = 1e-8
    cooling = 0.995
    step_size = 0.1
    
    n_iter = 0
    while time.time() < deadline:
        n_iter += 1
        
        # Propose move: perturb one or two points
        n_points = 1 if rng.random() < 0.7 else 2
        idxs = rng.choice(N, size=n_points, replace=False)
        
        new_bary = current_bary.copy()
        for idx in idxs:
            delta = rng.normal(0, step_size, size=2)
            new_bary[idx] = current_bary[idx] + delta
        new_bary = _clamp_bary(new_bary)
        
        new_points = _bary_to_cart(new_bary)
        new_area = _min_triangle_area_fast(new_points)
        
        delta_area = new_area - current_area
        
        if delta_area >= 0 or rng.random() < np.exp(delta_area / T):
            current_bary = new_bary
            current_points = new_points
            current_area = new_area
            
            if current_area > best_area:
                best_area = current_area
                best_bary = current_bary.copy()
        
        T *= cooling
        if T < T_min:
            T = T_min
        
        # Adapt step size based on acceptance rate (simple heuristic)
        if n_iter % 1000 == 0:
            step_size *= 0.95
    
    return best_bary, best_area


def _local_refine(bary, start_time, deadline):
    """Local refinement using Nelder-Mead on barycentric coordinates."""
    bary_flat = bary.flatten()
    
    # Use Nelder-Mead with bounds handled by clamping in objective
    def obj(x):
        return _objective_bary(x)
    
    try:
        result = minimize(
            obj,
            bary_flat,
            method='Nelder-Mead',
            options={
                'maxiter': 2000,
                'xatol': 1e-10,
                'fatol': 1e-12,
                'adaptive': True,
            },
        )
        if result.fun < -best_area_global[0]:
            pass  # tracked externally
        return result.x.reshape(N, 2), -result.fun
    except Exception:
        return bary, _min_triangle_area_fast(_bary_to_cart(bary))


def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    rng = np.random.default_rng(42)
    
    best_bary = None
    best_area = -float('inf')
    
    # Multi-restart SA with local refinement
    n_restarts = 8
    per_restart_time = (_DEADLINE - 10.0) / n_restarts  # 10s safety
    
    for r in range(n_restarts):
        elapsed = time.time() - start_time
        if elapsed > _DEADLINE - 5.0:
            break
        
        # Generate initial configuration
        bary_init = _make_initial_bary(rng)
        
        # Run SA
        restart_deadline = min(start_time + (r + 1) * per_restart_time, deadline - 5.0)
        sa_bary, sa_area = _sa_optimize(bary_init, start_time, restart_deadline, rng)
        
        if sa_area > best_area:
            best_area = sa_area
            best_bary = sa_bary.copy()
        
        # Local refinement on best SA result
        elapsed = time.time() - start_time
        if elapsed < deadline - 15.0:
            refine_deadline = min(start_time + (r + 1) * per_restart_time, deadline - 10.0)
            refined_bary, refined_area = _local_refine(sa_bary, start_time, refine_deadline)
            
            if refined_area > best_area:
                best_area = refined_area
                best_bary = refined_bary.copy()
        
        # Also try refining from a slightly perturbed best
        if best_bary is not None and elapsed < deadline - 10.0:
            perturbed = best_bary + rng.normal(0, 0.05, size=(N, 2))
            perturbed = _clamp_bary(perturbed)
            refined2_bary, refined2_area = _local_refine(perturbed, start_time, deadline - 5.0)
            if refined2_area > best_area:
                best_area = refined2_area
                best_bary = refined2_bary.copy()
    
    # If no valid result, fall back to a grid-like placement
    if best_bary is None:
        best_bary = np.zeros((N, 2))
        for i in range(N):
            row = i // 4
            col = i % 4
            a = 0.15 + 0.7 * col / 3.0
            b = 0.15 + 0.5 * row / 2.0
            if a + b > 0.85:
                b = 0.85 - a
            best_bary[i, 0] = a
            best_bary[i, 1] = b
        best_bary = _clamp_bary(best_bary)
    
    # Convert to Cartesian
    points = _bary_to_cart(best_bary)
    
    return points
# EVOLVE-BLOCK-END
