import numpy as np
from scipy.optimize import differential_evolution, minimize
from itertools import combinations
import time

N = 11
TIME_LIMIT = 115.0

# Pre-compute triangle indices
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)  # (165, 3)

# Triangle vertices in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
H = np.sqrt(3) / 2

def bary_to_cart(bary):
    """Convert barycentric coords (u, v) to Cartesian (x, y)."""
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = w * H
    return np.column_stack([x, y])

def calc_areas_bary(bary):
    """Calculate areas of all triangles formed by N points in barycentric coords."""
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    ui = u[tri_idx[:, 0]]
    vi = v[tri_idx[:, 0]]
    wi = w[tri_idx[:, 0]]
    
    uj = u[tri_idx[:, 1]]
    vj = v[tri_idx[:, 1]]
    wj = w[tri_idx[:, 1]]
    
    uk = u[tri_idx[:, 2]]
    vk = v[tri_idx[:, 2]]
    wk = w[tri_idx[:, 2]]
    
    det = ui * (vj * wk - vk * wj) - vi * (uj * wk - uk * wj) + wi * (uj * vk - uk * vj)
    
    return np.abs(det)

def clamp_bary(bary, eps=1e-10):
    """Clamp barycentric coordinates to the valid triangle region."""
    u = bary[:, 0].copy()
    v = bary[:, 1].copy()
    
    # Ensure u >= 0, v >= 0, u + v <= 1
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    
    sums = u + v
    invalid = sums > (1 - eps)
    if np.any(invalid):
        scale = (1 - eps) / np.maximum(sums, eps)
        u[invalid] *= scale[invalid]
        v[invalid] *= scale[invalid]
        
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    
    return np.column_stack([u, v])

def hard_min_value(bary):
    """Compute the hard minimum area value."""
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return np.min(areas)

def de_objective(x_flat):
    """
    Objective function for Differential Evolution.
    Minimizes -min_area.
    Returns a large penalty if points are outside the triangle.
    """
    bary = x_flat.reshape(N, 2)
    
    # Check validity
    u = bary[:, 0]
    v = bary[:, 1]
    
    # If any u < 0, v < 0, or u+v > 1, return penalty
    if np.any(u < 0) or np.any(v < 0) or np.any(u + v > 1):
        return 10.0  # Large value since we minimize
    
    areas = calc_areas_bary(bary)
    min_area = np.min(areas)
    return -min_area

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    # Bounds for DE: u in [0, 1], v in [0, 1]
    bounds = [(0.0, 1.0)] * (N * 2)
    
    # Run Differential Evolution
    # popsize=15, maxiter=1000, tol=1e-10, seed=42
    # strategy='best1bin' is standard
    try:
        de_result = differential_evolution(
            de_objective,
            bounds,
            maxiter=150,  # Limit iterations to stay within time
            popsize=20,
            tol=1e-10,
            seed=42,
            polish=False,  # We will do our own polish
            init='sobol',  # Sobol sequence for better initial spread
            maxfev=5000
        )
        
        best_bary = de_result.x.reshape(N, 2)
        best_bary = clamp_bary(best_bary)
        
    except Exception as e:
        # Fallback to random if DE fails
        rng = np.random.default_rng(42)
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        best_bary = np.column_stack([u, v])

    # Refinement with Nelder-Mead
    # We use Nelder-Mead to polish the solution found by DE.
    # Nelder-Mead is derivative-free and works well on non-smooth functions (min is non-smooth).
    
    def nm_objective(x_flat):
        bary = x_flat.reshape(N, 2)
        bary = clamp_bary(bary)
        areas = calc_areas_bary(bary)
        return -np.min(areas)

    current_bary = best_bary.copy()
    
    # Check time
    if time.time() < deadline - 10:
        try:
            nm_result = minimize(
                nm_objective,
                current_bary.flatten(),
                method='Nelder-Mead',
                options={
                    'maxiter': 2000,
                    'xatol': 1e-12,
                    'fatol': 1e-14,
                    'adaptive': True
                }
            )
            current_bary = clamp_bary(nm_result.x.reshape(N, 2))
        except Exception:
            pass

    # Final Polish with SLSQP or L-BFGS-B on a smoothed objective?
    # The min function is non-differentiable.
    # We can try to maximize the smallest area by optimizing the 165 areas directly?
    # No, that's complex.
    
    # Let's try one more round of Nelder-Mead with tighter tolerance if time permits
    if time.time() < deadline - 5:
        try:
            nm_result2 = minimize(
                nm_objective,
                current_bary.flatten(),
                method='Nelder-Mead',
                options={
                    'maxiter': 1000,
                    'xatol': 1e-14,
                    'fatol': 1e-16,
                    'adaptive': True
                }
            )
            candidate = clamp_bary(nm_result2.x.reshape(N, 2))
            # Verify improvement
            if -nm_result2.fun > hard_min_value(current_bary):
                current_bary = candidate
        except Exception:
            pass

    # Convert to cartesian
    cart = bary_to_cart(current_bary)
    
    return cart
