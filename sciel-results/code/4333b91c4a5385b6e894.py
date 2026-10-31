import numpy as np
from scipy.optimize import differential_evolution, minimize
import time
from itertools import combinations

N = 11
TIME_LIMIT = 115.0

# Triangle vertices
P0 = np.array([0.0, 0.0])
P1 = np.array([1.0, 0.0])
P2 = np.array([0.5, np.sqrt(3) / 2])
H = np.sqrt(3) / 2

# Pre-compute triangle indices for vectorization
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices) # shape (165, 3)

def perimeter_to_cartesian(t):
    """
    Map a scalar parameter t in [0, 3) to Cartesian coordinates on the perimeter.
    Side lengths are 1.
    t in [0, 1): P0 -> P1
    t in [1, 2): P1 -> P2
    t in [2, 3): P2 -> P0
    """
    x = np.zeros_like(t)
    y = np.zeros_like(t)
    
    # Segment 1: P0(0,0) to P1(1,0)
    mask1 = (t >= 0) & (t < 1)
    x[mask1] = t[mask1]
    y[mask1] = 0.0
    
    # Segment 2: P1(1,0) to P2(0.5, H)
    mask2 = (t >= 1) & (t < 2)
    t2 = t[mask2] - 1.0
    x[mask2] = 1.0 + t2 * (0.5 - 1.0)
    y[mask2] = 0.0 + t2 * (H - 0.0)
    
    # Segment 3: P2(0.5, H) to P0(0,0)
    mask3 = (t >= 2) & (t < 3)
    t3 = t[mask3] - 2.0
    x[mask3] = 0.5 + t3 * (0.0 - 0.5)
    y[mask3] = H + t3 * (0.0 - H)
    
    # Handle t >= 3 by wrapping, though bounds should prevent this
    mask4 = (t >= 3)
    if np.any(mask4):
        t_wrapped = t[mask4] - 3.0
        # Recursively call or just map again
        x[mask4], y[mask4] = perimeter_to_cartesian(t_wrapped)
        
    return np.column_stack([x, y])

def calc_min_area(points):
    """Calculate the minimum area of triangles formed by N points."""
    # points is (N, 2)
    pi = points[tri_idx[:, 0]]
    pj = points[tri_idx[:, 1]]
    pk = points[tri_idx[:, 2]]
    
    v1 = pj - pi
    v2 = pk - pi
    
    # 2D cross product
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def objective_perimeter(t_params):
    """
    Objective function for perimeter optimization.
    t_params is an array of 11 values in [0, 3).
    """
    # Ensure t is in [0, 3)
    t = np.mod(t_params, 3.0)
    points = perimeter_to_cartesian(t)
    area = calc_min_area(points)
    return -area

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_points = None
    best_score = -np.inf
    
    def update_best(points):
        nonlocal best_points, best_score
        score = calc_min_area(points)
        if score > best_score:
            best_score = score
            best_points = points.copy()
            return True
        return False

    # Strategy: Optimize 11 parameters t_i in [0, 3)
    bounds = [(0, 3)] * N
    
    # 1. Global Search with Differential Evolution on Perimeter Parameters
    try:
        de_result = differential_evolution(
            objective_perimeter,
            bounds,
            seed=42,
            maxiter=100,
            popsize=20,
            tol=1e-6,
            polish=False,
            init='latinhypercube'
        )
        
        de_params = de_result.x
        de_points = perimeter_to_cartesian(np.mod(de_params, 3.0))
        update_best(de_points)
        
        # Refine the DE result using local optimization on parameters
        if time.time() < deadline - 20:
            try:
                res_local = minimize(
                    objective_perimeter,
                    de_params,
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                local_params = res_local.x
                local_points = perimeter_to_cartesian(np.mod(local_params, 3.0))
                update_best(local_points)
            except Exception:
                pass
                
    except Exception:
        pass

    # 2. Multi-start with structured initializations on perimeter
    # Evenly spaced
    init_even = np.linspace(0, 3, N, endpoint=False)
    update_best(perimeter_to_cartesian(init_even))
    
    # Slightly perturbed even spacing
    rng = np.random.default_rng(123)
    for i in range(10):
        if time.time() > deadline - 15:
            break
        noise = rng.uniform(-0.1, 0.1, N)
        init_perturbed = np.mod(init_even + noise, 3.0)
        try:
            res_init = minimize(
                objective_perimeter,
                init_perturbed,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True}
            )
            init_points = perimeter_to_cartesian(np.mod(res_init.x, 3.0))
            update_best(init_points)
        except Exception:
            pass

    # 3. Final Polish on the best points found
    if best_points is not None and time.time() < deadline - 2:
        # We need to map best_points back to perimeter parameters for polishing?
        # It's easier to just do a final high-precision Nelder-Mead on the parameters
        # that produced best_score. However, we didn't store the params.
        # Let's just run one more DE iteration with smaller bounds around the best solution?
        # Or simply, since best_points are on the perimeter, we can try to recover params?
        # Actually, just running a final high-precision Nelder-Mead on a random set of params
        # close to the previous best is risky if we don't have the params.
        
        # Let's just do a final polish on the parameters from the last successful local opt
        # If we have best_points, we can assume they are on the perimeter.
        # To be safe, let's just return the best found so far.
        pass

    if best_points is None:
        # Fallback: return evenly spaced points
        best_points = perimeter_to_cartesian(np.linspace(0, 3, N, endpoint=False))
        
    return best_points

# EVOLVE-BLOCK-END
