import numpy as np
from scipy.optimize import minimize
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

def generate_hexagonal_init():
    """
    Generate a hexagonal lattice-like initial configuration.
    """
    best_bary = None
    best_val = -np.inf
    
    for scale in [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]:
        pts = []
        max_k = int(1.0 / scale) + 1
        for k in range(max_k + 1):
            w_val = k * scale
            if w_val > 1.0:
                break
            num_pts_in_row = int((1 - w_val) / scale) + 1
            for j in range(num_pts_in_row):
                u_val = j * scale
                v_val = (1 - w_val) - u_val
                if u_val >= -1e-9 and v_val >= -1e-9 and u_val + v_val <= 1 + 1e-9:
                    pts.append([u_val, v_val])
        
        if len(pts) >= N:
            # Try different subsets
            rng = np.random.default_rng(42)
            for _ in range(5):
                if len(pts) == N:
                    bary = np.array(pts)
                else:
                    idx = rng.choice(len(pts), N, replace=False)
                    bary = np.array(pts)[idx]
                val = -np.min(calc_areas_bary(bary))
                if val > best_val:
                    best_val = val
                    best_bary = bary.copy()
                
    if best_bary is None:
        rng = np.random.default_rng(42)
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        best_bary = np.column_stack([u, v])
        
    return best_bary

def generate_random_init(seed=None):
    """Generate random valid barycentric coordinates."""
    rng = np.random.default_rng(seed)
    u = rng.random(N)
    v = rng.random(N) * (1 - u)
    return np.column_stack([u, v])

def nlp_objective(x):
    """
    Objective function for the NLP formulation.
    x = [t, u_1, v_1, ..., u_N, v_N]
    We maximize t, so we minimize -t.
    """
    return -x[0]

def nlp_constraints(x):
    """
    Constraints for the NLP formulation.
    1. Barycentric coordinates must be non-negative and sum to 1.
       u_i >= 0, v_i >= 0, u_i + v_i <= 1
    2. Area of every triangle must be >= t.
       area_ij_k - t >= 0
    """
    t = x[0]
    bary = x[1:].reshape(N, 2)
    bary = clamp_bary(bary) # Ensure validity for calculation, though constraints handle it
    
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    # 1. Boundary constraints: u >= 0, v >= 0, u+v <= 1
    # SLSQP inequality constraints are of the form: c(x) >= 0
    cons = []
    
    # u_i >= 0
    cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[1 + 2*i]})
    # v_i >= 0
    cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2 + 2*i]})
    # u_i + v_i <= 1  =>  1 - u_i - v_i >= 0
    cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[1 + 2*i] - x[2 + 2*i]})
    
    # 2. Area constraints: area >= t
    # Calculate areas for current bary
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
    areas = np.abs(det)
    
    # We need area_ij_k - t >= 0 for all triangles
    # This is a vector-valued constraint. SLSQP supports this.
    cons.append({'type': 'ineq', 'fun': lambda x: np.abs(calc_areas_bary(x[1:].reshape(N, 2))) - x[0]})
    
    return cons

def solve_nlp(bary_init, time_limit=5.0):
    """
    Solve the NLP problem starting from bary_init.
    """
    start_time = time.time()
    
    # Initial t estimate: min area of initial configuration
    min_area = np.min(calc_areas_bary(bary_init))
    
    # x = [t, u1, v1, ..., uN, vN]
    x0 = np.zeros(1 + N * 2)
    x0[0] = min_area
    x0[1:] = bary_init.flatten()
    
    # Bounds: t >= 0, u,v in [0, 1]
    bounds = [(0, None)] + [(0, 1)] * (N * 2)
    
    # SLSQP is good for problems with many constraints and smooth objectives
    # We use a small tolerance to ensure high precision
    try:
        res = minimize(
            nlp_objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=nlp_constraints,
            options={'maxiter': 500, 'ftol': 1e-12}
        )
        
        if res.success or res.fun < -min_area:
            best_t = res.x[0]
            best_bary = clamp_bary(res.x[1:].reshape(N, 2))
            return best_bary, best_t
        else:
            return bary_init, min_area
            
    except Exception as e:
        # If SLSQP fails, try a softer approach or return original
        return bary_init, min_area

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_bary = None
    best_val = -np.inf
    
    def update_best(bary):
        nonlocal best_bary, best_val
        bary = clamp_bary(bary)
        val = np.min(calc_areas_bary(bary))
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
            return True
        return False

    # 1. Try Hexagonal Init
    hex_init = generate_hexagonal_init()
    refined_hex, val_hex = solve_nlp(hex_init, time_limit=10.0)
    update_best(refined_hex)
    
    # 2. Multi-start with NLP
    rng = np.random.default_rng(123)
    restart_count = 0
    while time.time() < deadline - 15:
        restart_count += 1
        # Mix of random and perturbed best
        if best_bary is not None and restart_count % 2 == 0:
            # Perturb best solution
            noise = rng.normal(0, 0.05, best_bary.shape)
            init = clamp_bary(best_bary + noise)
        else:
            init = generate_random_init(seed=3000 + restart_count)
        
        refined, val = solve_nlp(init, time_limit=5.0)
        update_best(refined)
        
        # Check time
        if time.time() > deadline - 15:
            break

    # 3. Final high-precision polish on the best solution
    if best_bary is not None and time.time() < deadline - 5:
        # Use a very tight tolerance for the final polish
        # We can also try to increase maxiter
        final_bary, final_val = solve_nlp(best_bary, time_limit=10.0)
        update_best(final_bary)
        
        # One more pass if time permits
        if time.time() < deadline - 2:
            final_bary2, final_val2 = solve_nlp(best_bary, time_limit=5.0)
            update_best(final_bary2)

    # Convert to cartesian
    cart = bary_to_cart(best_bary)
    
    return cart

# EVOLVE-BLOCK-END
