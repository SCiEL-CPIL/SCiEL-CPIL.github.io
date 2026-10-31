import numpy as np
from scipy.optimize import differential_evolution, minimize
from itertools import combinations
import time

N = 11
TIME_LIMIT = 115.0  # Safety margin for 120s limit

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

def clamp_bary(bary, eps=1e-9):
    """Clamp barycentric coordinates to the strictly valid triangle region."""
    u = bary[:, 0].copy()
    v = bary[:, 1].copy()
    
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    
    sums = u + v
    invalid = sums > (1 - eps)
    if np.any(invalid):
        scale = (1 - eps) / np.where(sums > 0, sums, 1.0)
        u[invalid] *= scale[invalid]
        v[invalid] *= scale[invalid]
        
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    
    return np.column_stack([u, v])

def perimeter_to_bary(t):
    """
    Convert a scalar t in [0, 1) to barycentric coordinates on the triangle boundary.
    The perimeter is divided into 3 equal segments (each of length 1).
    t=0 is at V0, t=1/3 is at V1, t=2/3 is at V2, t=1 is back at V0.
    """
    t = np.mod(t, 1.0)
    bary = np.zeros((len(t), 2))
    
    # Segment 0: V0 -> V1, t in [0, 1/3)
    mask0 = (t >= 0) & (t < 1.0/3)
    if np.any(mask0):
        s = t[mask0] * 3  # parameter along edge, 0 to 1
        bary[mask0, 0] = 0  # w = 0 (on edge V0-V1)
        bary[mask0, 1] = s  # v = s, u = 1-s... wait, let's be careful
        
    # Let's use a cleaner formulation:
    # Edge V0-V1: bary = (1-s, s, 0) where s = t*3
    # Edge V1-V2: bary = (1-s, 0, s) where s = (t-1/3)*3
    # Edge V2-V0: bary = (s, 0, 1-s) where s = (t-2/3)*3
    
    # Actually, barycentric (u, v) means point = u*V0 + v*V1 + (1-u-v)*V2
    # So:
    # V0 = (0,0), V1 = (1,0), V2 = (0.5, sqrt(3)/2)
    
    # Edge V0->V1: point = (1-s)*V0 + s*V1 + 0*V2 => bary = (0, s)
    # Edge V1->V2: point = 0*V0 + (1-s)*V1 + s*V2 => bary = (0, 1-s)
    # Edge V2->V0: point = s*V0 + 0*V1 + (1-s)*V2 => bary = (s, 0)
    
    bary = np.zeros((len(t), 2))
    
    # Segment 0: V0 -> V1, t in [0, 1/3)
    mask0 = (t >= 0) & (t < 1.0/3)
    if np.any(mask0):
        s = t[mask0] * 3
        bary[mask0, 0] = 0
        bary[mask0, 1] = s
    
    # Segment 1: V1 -> V2, t in [1/3, 2/3)
    mask1 = (t >= 1.0/3) & (t < 2.0/3)
    if np.any(mask1):
        s = (t[mask1] - 1.0/3) * 3
        bary[mask1, 0] = 0
        bary[mask1, 1] = 1 - s
    
    # Segment 2: V2 -> V0, t in [2/3, 1)
    mask2 = (t >= 2.0/3) & (t < 1.0)
    if np.any(mask2):
        s = (t[mask2] - 2.0/3) * 3
        bary[mask2, 0] = s
        bary[mask2, 1] = 0
    
    return bary

def bary_to_perimeter(bary):
    """
    Convert barycentric coordinates to a perimeter parameter t in [0, 1).
    This is the inverse of perimeter_to_bary for points on the boundary.
    For interior points, we project to the nearest boundary point.
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    t = np.zeros(len(bary))
    
    # Determine which edge is closest (smallest barycentric coordinate)
    # If w is smallest, point is near edge V0-V1
    # If u is smallest, point is near edge V1-V2
    # If v is smallest, point is near edge V2-V0
    
    min_idx = np.argmin(np.column_stack([u, v, w]), axis=1)
    
    for i in range(len(bary)):
        if min_idx[i] == 2:  # w is smallest, near V0-V1 edge
            # bary = (0, s) where s = v
            s = v[i]
            t[i] = s / 3.0
        elif min_idx[i] == 0:  # u is smallest, near V1-V2 edge
            # bary = (0, 1-s) where s = 1-v... wait
            # On V1-V2: bary = (0, 1-s) => v = 1-s => s = 1-v
            s = 1 - v[i]
            t[i] = (1.0/3) + s / 3.0
        else:  # v is smallest, near V2-V0 edge
            # bary = (s, 0) where s = u
            s = u[i]
            t[i] = (2.0/3) + s / 3.0
    
    return np.mod(t, 1.0)

def perimeter_obj(t_flat):
    """
    Objective function for perimeter parameterization.
    t_flat is an array of N values in [0, 1).
    We sort them to ensure consistent ordering, then compute min triangle area.
    """
    t = np.mod(t_flat, 1.0)
    t_sorted = np.sort(t)
    bary = perimeter_to_bary(t_sorted)
    areas = calc_areas_bary(bary)
    return -np.min(areas)

def generate_regular_init():
    """Generate a regular distribution on the perimeter."""
    t = np.linspace(0, 1, N, endpoint=False)
    return t

def generate_random_init(seed=None):
    """Generate random perimeter parameters."""
    rng = np.random.default_rng(seed)
    t = rng.random(N)
    return t

def refine_local_perimeter(t_init, time_limit=5.0):
    """
    Refine a perimeter configuration using local optimization.
    """
    start_time = time.time()
    bounds = [(0, 1)] * N
    
    t_current = t_init.copy()
    
    # Stage 1: Differential Evolution for global exploration in 1D
    try:
        result_de = differential_evolution(
            perimeter_obj,
            bounds,
            seed=42,
            maxiter=50,
            popsize=15,
            tol=1e-8,
            polish=False
        )
        t_current = result_de.x
    except Exception:
        pass
    
    if time.time() - start_time > time_limit * 0.5:
        return t_current
    
    # Stage 2: Nelder-Mead polish
    try:
        result_nm = minimize(
            perimeter_obj,
            t_current,
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
        )
        t_current = result_nm.x
    except Exception:
        pass
    
    if time.time() - start_time > time_limit * 0.8:
        return t_current
    
    # Stage 3: SLSQP final polish
    try:
        result_slsqp = minimize(
            perimeter_obj,
            t_current,
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-14}
        )
        t_current = result_slsqp.x
    except Exception:
        pass
    
    return t_current

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_t = None
    best_val = -np.inf
    
    def update_best(t):
        nonlocal best_t, best_val
        t = np.mod(t, 1.0)
        val = -perimeter_obj(t)
        if val > best_val:
            best_val = val
            best_t = t.copy()
            return True
        return False

    # 1. Start with regular distribution
    regular_init = generate_regular_init()
    refined_regular = refine_local_perimeter(regular_init, time_limit=10.0)
    update_best(refined_regular)
    
    # 2. Multi-start with random initializations
    rng = np.random.default_rng(123)
    restart_count = 0
    while time.time() < deadline - 15:
        restart_count += 1
        if best_t is not None and restart_count % 3 == 0:
            # Perturbation of best solution
            noise = rng.normal(0, 0.02, N)
            init = np.mod(best_t + noise, 1.0)
        else:
            # Fresh random start
            init = generate_random_init(seed=5000 + restart_count)
        
        refined = refine_local_perimeter(init, time_limit=4.0)
        update_best(refined)
    
    # 3. Final high-precision polish
    if best_t is not None and time.time() < deadline - 5:
        try:
            bounds = [(0, 1)] * N
            result_final = minimize(
                perimeter_obj,
                best_t,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-15}
            )
            update_best(result_final.x)
        except Exception:
            pass
        
        if best_t is not None and time.time() < deadline - 2:
            try:
                result_nm = minimize(
                    perimeter_obj,
                    best_t,
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
                )
                update_best(result_nm.x)
            except Exception:
                pass
    
    # Convert best perimeter params to barycentric, then to cartesian
    t_best = np.mod(best_t, 1.0)
    t_sorted = np.sort(t_best)
    bary = perimeter_to_bary(t_sorted)
    bary = clamp_bary(bary)
    cart = bary_to_cart(bary)
    
    return cart
