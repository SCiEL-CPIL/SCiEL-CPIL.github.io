import numpy as np
from scipy.optimize import basin_hopping, minimize
from itertools import combinations
import time

N = 11
TIME_LIMIT = 115.0
H = np.sqrt(3) / 2

# Triangle vertices in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, H])

# Pre-compute triangle index combinations
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)  # (165, 3)

def barycentric_to_cartesian(uv):
    """
    Convert barycentric coords (u, v) to Cartesian.
    w = 1 - u - v
    P = u*V0 + v*V1 + w*V2
    """
    u = uv[:, 0]
    v = uv[:, 1]
    w = 1.0 - u - v
    
    # P = u*(0,0) + v*(1,0) + w*(0.5, H)
    # x = v*1 + w*0.5
    # y = w*H
    x = v + 0.5 * w
    y = H * w
    return np.column_stack((x, y))

def project_to_simplex(uv):
    """
    Project (u, v) such that u>=0, v>=0, u+v<=1.
    Returns (u, v).
    """
    u = uv[:, 0].copy()
    v = uv[:, 1].copy()
    
    # Clamp negative values
    u[u < 0] = 0
    v[v < 0] = 0
    
    # Check if u+v > 1
    s = u + v
    mask = s > 1.0
    if np.any(mask):
        u[mask] /= s[mask]
        v[mask] /= s[mask]
        
    return np.column_stack((u, v))

def compute_min_area_barycentric(uv):
    """
    Compute minimum triangle area using barycentric coordinates.
    Area(i,j,k) = Area(ABC) * |det(B_i, B_j, B_k)|
    where B_i = [u_i, v_i, w_i]
    """
    u = uv[:, 0]
    v = uv[:, 1]
    w = 1.0 - u - v
    
    B = np.column_stack((u, v, w))  # (N, 3)
    
    # Extract points for each triangle
    bi = B[tri_idx[:, 0]]  # (165, 3)
    bj = B[tri_idx[:, 1]]  # (165, 3)
    bk = B[tri_idx[:, 2]]  # (165, 3)
    
    # Determinant of 3x3 matrix formed by rows bi, bj, bk
    # det = bi_x * (bj_y * bk_z - bj_z * bk_y) - bi_y * (bj_x * bk_z - bj_z * bk_x) + bi_z * (bj_x * bk_y - bj_y * bk_x)
    
    det = (bi[:, 0] * (bj[:, 1] * bk[:, 2] - bj[:, 2] * bk[:, 1]) -
           bi[:, 1] * (bj[:, 0] * bk[:, 2] - bj[:, 2] * bk[:, 0]) +
           bi[:, 2] * (bj[:, 0] * bk[:, 1] - bj[:, 1] * bk[:, 0]))
           
    areas = np.abs(det) * 0.5 * H # Area of ABC is 0.5 * base * height = 0.5 * 1 * H
    
    return np.min(areas)

def objective_bary(x):
    """
    Objective function for Basin Hopping.
    x is a vector of length 2*N (u_0, v_0, u_1, v_1, ...)
    """
    uv = x.reshape(N, 2)
    uv = project_to_simplex(uv)
    min_area = compute_min_area_barycentric(uv)
    return -min_area

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    # Initial guess: Random points in barycentric space
    rng = np.random.default_rng(42)
    
    # Generate random points in simplex
    # Method: Generate 3 random numbers, normalize
    # Or just uniform in [0,1] and project
    x0 = rng.uniform(0, 1, (N, 2))
    x0 = project_to_simplex(x0).flatten()
    
    best_x = x0.copy()
    best_val = -objective_bary(best_x)
    
    # Use Basin Hopping
    # minimizer_kwargs for Nelder-Mead
    minimizer_kwargs = {
        'method': 'Nelder-Mead',
        'options': {
            'xatol': 1e-10,
            'fatol': 1e-10,
            'maxiter': 100,
            'disp': False
        }
    }
    
    # Basin Hopping parameters
    # niter: number of Monte Carlo steps
    # T: temperature
    # stepsize: initial step size
    
    try:
        bh_result = basin_hopping(
            objective_bary,
            x0,
            niter=100,
            T=1.0,
            stepsize=0.5,
            minimizer_kwargs=minimizer_kwargs,
            seed=42
        )
        
        if bh_result.fun < best_val:
            best_val = bh_result.fun
            best_x = bh_result.x
    except Exception as e:
        pass
    
    # Multi-start refinement
    remaining_time = deadline - time.time()
    if remaining_time > 10:
        num_restarts = 5
        for i in range(num_restarts):
            if time.time() > deadline - 5:
                break
            
            # Perturb best solution
            sigma = 0.1 + 0.05 * i
            x_start = best_x + rng.normal(0, sigma, best_x.shape)
            x_start = project_to_simplex(x_start.reshape(N, 2)).flatten()
            
            # Quick local optimization
            try:
                res = minimize(
                    objective_bary,
                    x_start,
                    method='Nelder-Mead',
                    options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 200, 'disp': False}
                )
                val = res.fun
                if val < best_val:
                    best_val = val
                    best_x = res.x
            except Exception:
                pass
            
            # Try Basin Hopping from this start if time permits
            if time.time() < deadline - 20:
                try:
                    bh_res = basin_hopping(
                        objective_bary,
                        x_start,
                        niter=20,
                        T=0.5,
                        stepsize=0.2,
                        minimizer_kwargs=minimizer_kwargs,
                        seed=42 + i
                    )
                    if bh_res.fun < best_val:
                        best_val = bh_res.fun
                        best_x = bh_res.x
                except Exception:
                    pass
    
    # Final polish
    if time.time() < deadline - 5:
        try:
            res = minimize(
                objective_bary,
                best_x,
                method='Nelder-Mead',
                options={'xatol': 1e-12, 'fatol': 1e-12, 'maxiter': 1000, 'disp': False}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass
    
    # Convert best barycentric solution to Cartesian
    uv_best = project_to_simplex(best_x.reshape(N, 2))
    pts_cart = barycentric_to_cartesian(uv_best)
    
    return pts_cart

# EVOLVE-BLOCK-END
