import numpy as np
from itertools import combinations
import time
from scipy.optimize import minimize

N = 11
TIME_LIMIT = 145.0  # Safety margin for 150s limit

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

def cart_to_bary(cart):
    """Convert Cartesian (x, y) to barycentric coords (u, v)."""
    x = cart[:, 0]
    y = cart[:, 1]
    u = 1.0 - x * (2.0 / 3.0) - y * (2.0 / (3.0 * H))
    v = 1.0 - u - y / H
    return np.column_stack([u, v])

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

def get_min_area(bary):
    """Get the minimum area for a given barycentric configuration."""
    areas = calc_areas_bary(bary)
    return np.min(areas)

def epigraph_obj(x, t):
    """
    Epigraph formulation objective: minimize t subject to all triangle areas >= t.
    We use a smooth approximation: sum of soft violations + t.
    """
    bary = x.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    
    # Soft violation: for each area a, if a < t, add penalty
    # Using a smooth hinge: max(0, t - a) approximated by log(1 + exp(beta*(t-a)))
    beta = 50.0
    violations = t - areas
    # Soft plus: log(1 + exp(beta * violations)) / beta
    # For stability:
    max_v = np.max(violations)
    if max_v > 0:
        soft_plus = (max_v + (1.0 / beta) * np.log(np.sum(np.exp(beta * (violations - max_v))))) / beta
    else:
        soft_plus = 0.0
    
    return t + 10.0 * soft_plus

def epigraph_grad(x, t):
    """
    Gradient of the epigraph objective.
    This is complex due to the soft plus. We'll use numerical gradient or 
    rely on scipy's minimize with method that handles this.
    Actually, let's just use the objective and let scipy handle it, 
    or compute gradient numerically if needed.
    For now, we'll use the objective function directly with L-BFGS-B 
    which can use numerical gradients if not provided, but it's better to provide one.
    """
    # We will compute the gradient numerically inside the minimizer if not provided analytically.
    # However, providing an analytic gradient is much faster.
    # Let's derive it.
    # f = t + 10 * sum_i (1/beta) * log(1 + exp(beta*(t - a_i)))
    # df/dt = 1 + 10 * sum_i (1/beta) * exp(beta*(t-a_i)) / (1 + exp(beta*(t-a_i))) * beta
    #        = 1 + 10 * sum_i sigmoid(beta*(t-a_i))
    # df/dx_j = 10 * sum_i (1/beta) * exp(beta*(t-a_i)) / (1 + exp(beta*(t-a_i))) * beta * (-da_i/dx_j)
    #          = -10 * sum_i sigmoid(beta*(t-a_i)) * da_i/dx_j
    
    bary = x.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    beta = 50.0
    violations = t - areas
    # sigmoid(beta * violations)
    # For stability:
    max_v = np.max(violations)
    if max_v > 0:
        sigmoids = np.exp(beta * (violations - max_v)) / (1.0 + np.exp(beta * (violations - max_v)))
    else:
        sigmoids = np.exp(beta * violations) / (1.0 + np.exp(beta * violations))
    
    # dt
    dt = 1.0 + 10.0 * np.sum(sigmoids)
    
    # dx
    # We need gradients of areas w.r.t. barycentric coords.
    # This is a bit involved. Let's approximate the gradient of the objective numerically 
    # for the x part, or derive it.
    # Given time constraints, let's use a hybrid: analytic for t, numerical for x? 
    # No, scipy's L-BFGS-B can take a function that returns (f, g).
    # Let's just use the function and let scipy approximate the gradient for x, 
    # but that's slow.
    
    # Alternative: Use a simpler smooth objective that is easier to differentiate.
    # Let's stick to the soft-min approach but with a better local search strategy.
    # The epigraph approach is good but the gradient is tricky.
    
    # Let's return None for gradient and let scipy use finite differences, 
    # but that might be too slow.
    # Instead, let's use a different strategy: 
    # Maximize the minimum area directly using a smooth approximation (soft-min) 
    # and then use a local search that focuses on the active constraints.
    
    # Actually, let's implement a simple gradient for the soft-min objective.
    # soft_min = -log-sum-exp(-beta * areas)
    # d(soft_min)/dx = beta * sum_i (exp(-beta*a_i) / sum_j exp(-beta*a_j)) * da_i/dx
    # This is a weighted average of the gradients of the areas.
    
    # Let's compute the gradients of the areas.
    # Area(i,j,k) = 0.5 * |det|. 
    # Let's assume the sign of the determinant is consistent for each triangle 
    # in the local neighborhood (which is true for non-degenerate triangles).
    # We can compute the signed area and its gradient.
    
    # Signed Area(i,j,k) = 0.5 * (u_i(v_j w_k - v_k w_j) - v_i(u_j w_k - u_k w_j) + w_i(u_j v_k - u_k v_j))
    # Let's compute the gradient of the signed area w.r.t. u_m, v_m for m in {i,j,k}.
    
    # This is getting complex. Let's use a simpler approach: 
    # Use the soft-min objective and let scipy minimize it. 
    # We already have soft_min_obj. Let's use that.
    
    # For the epigraph approach, let's just use the objective and let scipy 
    # handle the gradient numerically. It might be slower but more robust.
    
    # However, to be safe and fast, let's just use the soft-min objective 
    # with a high beta and L-BFGS-B.
    
    # Let's return the objective value and let scipy compute the gradient.
    # But we need to return a tuple (f, g) if we want to provide the gradient.
    # If we don't provide the gradient, we should just return f.
    
    # Let's change the approach: use the soft-min objective directly.
    
    # For now, let's just return the objective value.
    # We will use a different function for the minimization.
    
    return None  # Placeholder, we won't use this gradient function.

def soft_min_obj(x, beta=100.0):
    """
    Smooth surrogate for the hard min via log-sum-exp.
    """
    bary = x.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    a_max = np.max(areas)
    val = a_max - (1.0 / beta) * np.log(np.sum(np.exp(-beta * (areas - a_max))))
    return -val

def refine_local(bary):
    """
    Local refinement using L-BFGS-B on the soft-min objective.
    """
    current_bary = clamp_bary(bary)
    current_min = get_min_area(current_bary)
    
    # Use a high beta for precision in the local search
    beta = 500.0
    
    def local_obj(x):
        return soft_min_obj(x, beta)
    
    try:
        res = minimize(local_obj, current_bary.flatten(), method='L-BFGS-B', 
                       options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
        if res.success:
            new_bary = clamp_bary(res.x.reshape(N, 2))
            new_min = get_min_area(new_bary)
            if new_min > current_min:
                return new_bary
    except Exception:
        pass
        
    return current_bary

def generate_hexagonal_init():
    """Generate a hexagonal lattice-like initialization."""
    pts = []
    # Use a grid in barycentric space
    grid_size = 10
    us = np.linspace(0.1, 0.8, grid_size)
    vs = np.linspace(0.1, 0.8, grid_size)
    
    candidates = []
    for u in us:
        for v in vs:
            if u + v <= 0.9:
                candidates.append([u, v])
                
    if len(candidates) < N:
        return None
        
    # We want to select 11 points that are well spread.
    # Just picking a subset of the grid.
    # To be deterministic but diverse, we can use a fixed seed for selection.
    rng = np.random.default_rng(42)
    idx = rng.choice(len(candidates), N, replace=False)
    return np.array(candidates)[idx]

def generate_random_init(rng):
    """Generate a random initialization inside the triangle."""
    u = rng.random(N)
    v = rng.random(N) * (1 - u)
    return np.column_stack([u, v])

def generate_perturbed_init(bary, rng, sigma=0.05):
    """Generate a perturbed version of an existing configuration."""
    pert = rng.normal(0, sigma, size=(N, 2))
    return clamp_bary(bary + pert)

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_bary = None
    best_val = -np.inf
    
    def update_best(bary):
        nonlocal best_bary, best_val
        if bary is None: return
        bary = clamp_bary(bary)
        val = get_min_area(bary)
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
            return True
        return False

    # 1. Initialization
    start_points = []
    
    hex_init = generate_hexagonal_init()
    if hex_init is not None:
        start_points.append(hex_init)
        
    rng_master = np.random.default_rng(123)
    for _ in range(5):
        start_points.append(generate_random_init(rng_master))
        
    # 2. Local Refinement on Initial Points
    for sp in start_points:
        if time.time() > deadline - 10:
            break
        refined_bary = refine_local(sp)
        update_best(refined_bary)
        
    # 3. Iterative Perturbation and Refinement (Hill Climbing)
    # This is the key structural change: instead of global search, we do 
    # multiple rounds of local search with perturbations, starting from 
    # the best point found so far.
    
    current_best = best_bary.copy() if best_bary is not None else start_points[0]
    
    for iteration in range(50):  # Max iterations
        if time.time() > deadline - 5:
            break
            
        # Try several perturbations of the current best
        for _ in range(10):
            if time.time() > deadline - 2:
                break
                
            # Decrease perturbation size over time
            sigma = 0.1 / (1.0 + 0.1 * iteration)
            perturbed = generate_perturbed_init(current_best, rng_master, sigma)
            refined = refine_local(perturbed)
            
            if update_best(refined):
                current_best = best_bary.copy()
                break  # If we found a better point, restart perturbation loop
        
    # 4. Final Fallback
    if best_bary is None:
        rng = np.random.default_rng(0)
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        best_bary = np.column_stack([u, v])
        
    cart = bary_to_cart(best_bary)
    
    return cart

# EVOLVE-BLOCK-END
