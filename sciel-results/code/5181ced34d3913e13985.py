import numpy as np
from scipy.optimize import minimize
import time
import math
from itertools import combinations

N = 11
TIME_LIMIT = 145.0

# Triangle vertices in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
H = np.sqrt(3) / 2

# Pre-compute triangle indices
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)  # (165, 3)

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

def soft_min_obj(x, beta):
    """
    Soft-min objective: -log-sum-exp(-beta * areas) / beta
    This approximates -min(areas) as beta -> infinity.
    """
    bary = x[:N*2].reshape(N, 2)
    areas = calc_areas_bary(bary)
    
    # Soft min of areas
    max_a = np.max(areas)
    # log(sum(exp(-beta * (areas - max_a)))) / beta - max_a
    # This is the soft min value
    log_sum = np.log(np.sum(np.exp(-beta * (areas - max_a))))
    soft_min_val = max_a - log_sum / beta
    
    return -soft_min_val

def soft_min_grad(x, beta):
    """
    Gradient of soft-min objective.
    d/dx_i [-soft_min] = -sum_j (exp(-beta*(a_j - max_a)) / sum_k exp(-beta*(a_k - max_a))) * (d a_j / dx_i) * beta
    Actually, gradient of soft_min = sum_j w_j * grad(a_j) where w_j = exp(-beta*(a_j-max_a)) / sum_k exp(-beta*(a_k-max_a))
    So gradient of -soft_min = -sum_j w_j * grad(a_j)
    
    But computing grad(a_j) for all j and all x is expensive. Let's use numerical or just rely on the smoothness.
    Actually, for SLSQP, providing a gradient is helpful but not strictly required if we use a good method.
    However, the gradient of the soft-min is a convex combination of the gradients of the areas.
    Let's just return zeros for the barycentric coords and let the optimizer handle it? No, that's wrong.
    
    Let's compute the gradient numerically if needed, or just use the objective only.
    For SLSQP, it can compute numerical gradients if jac is not provided, but it's slower.
    Let's try to provide the gradient.
    
    The gradient of area_i with respect to bary coords is:
    For point p, q, r, area = 0.5 * |det(p-q, r-q)| (in Cartesian) or in barycentric:
    The determinant formula: det = u_i(v_j w_k - v_k w_j) - v_i(u_j w_k - u_k w_j) + w_i(u_j v_k - u_k v_j)
    This is linear in each point's coordinates.
    
    Let's just use numerical gradient for simplicity and robustness, or provide a sparse gradient.
    Given the complexity, let's just let SLSQP compute the numerical gradient for the soft-min objective.
    It might be slow, but the objective is smooth.
    """
    return None

def epigraph_obj(x):
    """
    Objective function for SLSQP: minimize -t subject to t <= Area_i.
    x = [bary_0_u, bary_0_v, ..., bary_10_u, bary_10_v, t]
    """
    bary = x[:N*2].reshape(N, 2)
    t = x[N*2]
    areas = calc_areas_bary(bary)
    return -t

def epigraph_grad(x):
    """
    Gradient of the objective function.
    Only t has a gradient (-1). Barycentric coords have 0 gradient in the objective.
    """
    grad = np.zeros_like(x)
    grad[N*2] = -1.0
    return grad

def epigraph_constraints(x):
    """
    Constraints: t <= Area_i for all i.
    Formulated as Area_i - t >= 0.
    """
    bary = x[:N*2].reshape(N, 2)
    t = x[N*2]
    areas = calc_areas_bary(bary)
    return areas - t

def generate_hexagonal_init(seed=0):
    """
    Generate initial configuration based on a hexagonal lattice pattern.
    """
    rng = np.random.default_rng(seed)
    
    # Hexagonal lattice points inside the triangle
    # This is a common good starting point for such problems.
    # Let's define a few key points and then add random perturbations.
    
    # Center
    pts = [[1/3, 1/3]]
    
    # Points near vertices (slightly inward)
    pts.append([0.1, 0.1])
    pts.append([0.1, 0.9])
    pts.append([0.9, 0.1])
    
    # Points near edge midpoints
    pts.append([0.0, 0.5])
    pts.append([0.5, 0.0])
    pts.append([0.5, 0.5])
    
    # Additional points to reach 11
    pts.append([0.25, 0.25])
    pts.append([0.25, 0.5])
    pts.append([0.5, 0.25])
    pts.append([0.33, 0.33]) # Another center-ish point
    
    bary = np.array(pts)
    
    # Add small random perturbation
    noise = rng.normal(0, 0.02, bary.shape)
    bary += noise
    bary = clamp_bary(bary)
    
    return bary

def generate_random_init(seed):
    rng = np.random.default_rng(seed)
    u = rng.random(N)
    v = rng.random(N) * (1 - u)
    return np.column_stack([u, v])

def generate_perturbed_init(base_bary, seed, sigma=0.1):
    """Generate a perturbed version of a base configuration."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, sigma, base_bary.shape)
    bary = base_bary + noise
    bary = clamp_bary(bary)
    return bary

def refine_local(bary, method='SLSQP', max_iter=1000, ftol=1e-12):
    """
    Refine a local configuration using SLSQP with epigraph formulation.
    """
    # Flatten: [u0, v0, ..., u10, v10, t]
    # Initial t: small value
    t0 = np.min(calc_areas_bary(bary)) * 0.9
    x0 = np.concatenate([bary.flatten(), [t0]])
    
    # Bounds for SLSQP
    bounds = []
    for _ in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
    bounds.append((0.0, 1.0)) # t
    
    # Constraints
    def con_fun(x):
        bary_x = x[:N*2].reshape(N, 2)
        t = x[N*2]
        areas = calc_areas_bary(bary_x)
        cons = areas - t
        
        # Barycentric constraints
        bary_cons = np.zeros(3*N)
        bary_cons[0::3] = bary_x[:, 0]       # u
        bary_cons[1::3] = bary_x[:, 1]       # v
        bary_cons[2::3] = 1.0 - bary_x[:, 0] - bary_x[:, 1] # 1-u-v
        
        return np.concatenate([cons, bary_cons])
    
    constraints = {'type': 'ineq', 'fun': con_fun}
    
    try:
        res = minimize(
            epigraph_obj,
            x0,
            method=method,
            jac=epigraph_grad,
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': max_iter, 'ftol': ftol}
        )
        
        if res.success or (res.fun < 0 and not np.isnan(res.fun)):
            res_bary = res.x[:N*2].reshape(N, 2)
            res_bary = clamp_bary(res_bary)
            return res_bary
        else:
            return bary
    except Exception:
        return bary

def refine_soft_min(bary, beta_schedule=[20, 50, 200]):
    """
    Refine using graduated soft-min approach.
    """
    current_bary = bary.copy()
    
    for beta in beta_schedule:
        # Use SLSQP with soft-min objective
        x0 = current_bary.flatten()
        
        bounds = []
        for _ in range(N):
            bounds.append((0.0, 1.0))
            bounds.append((0.0, 1.0))
        
        def con_fun(x):
            bary_x = x[:N*2].reshape(N, 2)
            bary_cons = np.zeros(3*N)
            bary_cons[0::3] = bary_x[:, 0]       # u
            bary_cons[1::3] = bary_x[:, 1]       # v
            bary_cons[2::3] = 1.0 - bary_x[:, 0] - bary_x[:, 1] # 1-u-v
            return bary_cons
        
        constraints = {'type': 'ineq', 'fun': con_fun}
        
        try:
            res = minimize(
                lambda x: soft_min_obj(x, beta),
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-10}
            )
            
            if res.success or not np.isnan(res.fun):
                current_bary = clamp_bary(res.x[:N*2].reshape(N, 2))
        except Exception:
            pass
    
    return current_bary

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_bary = None
    best_val = -np.inf
    
    def update_best(bary):
        nonlocal best_bary, best_val
        if bary is None: return
        bary = clamp_bary(bary)
        areas = calc_areas_bary(bary)
        val = np.min(areas)
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
            return True
        return False

    # Generate multiple initial candidates
    inits = []
    
    # 1. Hexagonal based
    inits.append(generate_hexagonal_init(0))
    
    # 2. Random starts
    for i in range(5):
        inits.append(generate_random_init(100 + i))
    
    # 3. Perturbed hexagonal
    base_hex = generate_hexagonal_init(0)
    for i in range(3):
        inits.append(generate_perturbed_init(base_hex, 200 + i, sigma=0.05))
    
    # Run refinement on each init
    for i, init_bary in enumerate(inits):
        if time.time() > deadline - 20:
            break
        
        # First try soft-min refinement
        try:
            refined_bary = refine_soft_min(init_bary)
            update_best(refined_bary)
        except Exception:
            pass
        
        # Then try epigraph refinement
        if time.time() > deadline - 10:
            break
        
        try:
            refined_bary2 = refine_local(refined_bary if 'refined_bary' in locals() else init_bary)
            update_best(refined_bary2)
        except Exception:
            pass
    
    # Final high-precision polish
    if best_bary is not None and time.time() < deadline - 5:
        try:
            polished = refine_local(best_bary, max_iter=2000, ftol=1e-14)
            update_best(polished)
        except Exception:
            pass
    
    # Fallback
    if best_bary is None:
        best_bary = generate_random_init(0)
        
    cart = bary_to_cart(best_bary)
    return cart
