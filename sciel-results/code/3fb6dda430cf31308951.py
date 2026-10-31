import numpy as np
from scipy.optimize import differential_evolution, minimize
from itertools import combinations
import time

N = 11
TIME_LIMIT = 145.0

tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)

V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
H = np.sqrt(3) / 2

def bary_to_cart(bary):
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = w * H
    return np.column_stack([x, y])

def calc_areas_bary(bary):
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

def hard_min_obj(bary_flat):
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return -np.min(areas)

def get_active_triangles(bary, threshold=1.05):
    """Get triangles with area close to the minimum."""
    areas = calc_areas_bary(bary)
    min_area = np.min(areas)
    active = np.where(areas <= threshold * min_area)[0]
    return active, min_area, areas

def coordinated_ascent_step(bary, step_size=0.02, time_budget=1.0):
    """
    Perform coordinated ascent: move points to increase the minimum area.
    Focus on active constraints (triangles near minimum).
    """
    start_time = time.time()
    bary = bary.copy()
    areas = calc_areas_bary(bary)
    min_area = np.min(areas)
    
    # Get active triangles
    active_mask = areas <= 1.1 * min_area
    active_tris = tri_idx[active_mask]
    
    if len(active_tris) == 0:
        return bary
    
    # For each point, compute gradient direction to increase minimum area
    # Use subgradient: average of gradients from active triangles containing that point
    
    for i in range(N):
        if time.time() - start_time > time_budget:
            break
            
        # Find which active triangles contain point i
        mask_i = (active_tris[:, 0] == i) | (active_tris[:, 1] == i) | (active_tris[:, 2] == i)
        if not np.any(mask_i):
            continue
        
        active_for_i = active_tris[mask_i]
        
        # Compute subgradient for point i
        # The area of triangle (i,j,k) in barycentric coords is |det|
        # We want to increase this area
        
        grad_u = 0.0
        grad_v = 0.0
        
        for tri in active_for_i:
            i_idx, j_idx, k_idx = tri
            
            ui = bary[i_idx, 0]
            vi = bary[i_idx, 1]
            wi = 1 - ui - vi
            
            uj = bary[j_idx, 0]
            vj = bary[j_idx, 1]
            wj = 1 - uj - vj
            
            uk = bary[k_idx, 0]
            vk = bary[k_idx, 1]
            wk = 1 - uk - vk
            
            # Area = |ui*(vj*wk - vk*wj) - vi*(uj*wk - uk*wj) + wi*(uj*vk - uk*vj)|
            # = |ui*(vj*wk - vk*wj) - vi*(uj*wk - uk*wj) + (1-ui-vi)*(uj*vk - uk*vj)|
            
            # Derivative w.r.t. ui (treating vi as constant, so wi = 1-ui-vi)
            # d/dui [ui*(vj*wk - vk*wj) - vi*(uj*wk - uk*wj) + (1-ui-vi)*(uj*vk - uk*vj)]
            # = (vj*wk - vk*wj) - (uj*vk - uk*vj)
            d_ui = (vj * wk - vk * wj) - (uj * vk - uk * vj)
            
            # Derivative w.r.t. vi
            # d/dvi [-vi*(uj*wk - uk*wj) + (1-ui-vi)*(uj*vk - uk*vj)]
            # = -(uj*wk - uk*wj) - (uj*vk - uk*vj)
            d_vi = -(uj * wk - uk * wj) - (uj * vk - uk * vj)
            
            # The sign depends on whether the determinant is positive or negative
            det_val = ui * (vj * wk - vk * wj) - vi * (uj * wk - uk * wj) + wi * (uj * vk - uk * vj)
            sign = np.sign(det_val) if det_val != 0 else 1.0
            
            # We want to INCREASE the absolute area, so move in direction that increases |det|
            # If det > 0, increase det; if det < 0, decrease det
            grad_u += sign * d_ui
            grad_v += sign * d_vi
        
        # Normalize gradient
        grad_norm = np.sqrt(grad_u**2 + grad_v**2)
        if grad_norm > 1e-12:
            grad_u /= grad_norm
            grad_v /= grad_norm
            
            # Move in gradient direction
            bary[i, 0] += step_size * grad_u
            bary[i, 1] += step_size * grad_v
    
    bary = clamp_bary(bary)
    return bary

def refine_coordinated(bary_init, time_budget=5.0):
    """
    Refine using coordinated ascent with decreasing step size.
    """
    start_time = time.time()
    bary = bary_init.copy()
    best_bary = bary.copy()
    best_val = -hard_min_obj(bary.flatten())
    
    step_size = 0.03
    patience = 0
    
    while time.time() - start_time < time_budget:
        # Perform several coordinated ascent steps
        for _ in range(50):
            if time.time() - start_time > time_budget:
                break
            bary = coordinated_ascent_step(bary, step_size=step_size, time_budget=0.05)
        
        current_val = -hard_min_obj(bary.flatten())
        
        if current_val > best_val:
            best_val = current_val
            best_bary = bary.copy()
            patience = 0
        else:
            patience += 1
        
        if patience > 5:
            step_size *= 0.7
            patience = 0
        
        if step_size < 1e-5:
            break
    
    return best_bary

def generate_heilbronn_init():
    """
    Generate initial configuration based on known Heilbronn triangle results.
    For n=11, use a configuration that approximates the optimal arrangement.
    """
    # Start with a hexagonal-like pattern
    pts = []
    
    # Corners and near-corners
    pts.append([0.01, 0.01])  # Near V0
    pts.append([0.99, 0.01])  # Near V1
    pts.append([0.5, 0.86])   # Near V2
    
    # Interior points in a hexagonal arrangement
    # Center
    pts.append([0.5, 0.33])
    
    # Six points around center in hexagonal pattern
    r = 0.18
    angles = np.linspace(0, 2*np.pi, 7)[:-1]
    for angle in angles:
        u = 0.5 + r * np.cos(angle)
        v = 0.33 + r * np.sin(angle) * 0.5  # Scale v to account for triangle geometry
        if u >= 0 and v >= 0 and u + v <= 1:
            pts.append([u, v])
    
    # Add more points if needed
    while len(pts) < N:
        # Add points at strategic locations
        u = 0.25
        v = 0.25
        pts.append([u, v])
    
    pts = np.array(pts[:N])
    pts = clamp_bary(pts)
    return pts

def generate_random_init(rng):
    """Generate random initial configuration."""
    u = rng.random(N)
    v = rng.random(N) * (1 - u)
    return clamp_bary(np.column_stack([u, v]))

def generate_perturbed_init(bary, rng, scale=0.05):
    """Generate perturbed version of existing configuration."""
    noise = rng.normal(0, scale, bary.shape)
    return clamp_bary(bary + noise)

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_bary = None
    best_val = -np.inf
    
    def update_best(bary):
        nonlocal best_bary, best_val
        bary = clamp_bary(bary)
        val = -hard_min_obj(bary.flatten())
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
            return True
        return False
    
    rng = np.random.default_rng(42)
    
    # 1. Start with Heilbronn-inspired initialization
    init_heilbronn = generate_heilbronn_init()
    refined = refine_coordinated(init_heilbronn, time_budget=min(10.0, deadline - start_time - 30))
    update_best(refined)
    
    # 2. Try several random initializations
    n_random = 5
    for i in range(n_random):
        if time.time() > deadline - 40:
            break
        init_rand = generate_random_init(rng)
        refined = refine_coordinated(init_rand, time_budget=min(5.0, deadline - start_time - 35))
        update_best(refined)
    
    # 3. Global search with differential evolution in reduced parameter space
    if time.time() < deadline - 50:
        try:
            # Use a parameterization: center + scale + rotation + per-point radial offsets
            # This is a 2 + 1 + 1 + N = 5 + N = 16 dimensional space
            
            def param_to_bary(params):
                center_u = params[0]
                center_v = params[1]
                scale = params[2]
                rotation = params[3]
                
                bary = np.zeros((N, 2))
                bary[0] = [center_u, center_v]
                
                # Distribute remaining points in a circle
                for i in range(1, N):
                    angle = rotation + 2 * np.pi * i / (N - 1)
                    r = scale * (1 + 0.1 * np.sin(3 * angle))  # Slight modulation
                    bary[i, 0] = center_u + r * np.cos(angle)
                    bary[i, 1] = center_v + r * np.sin(angle) * 0.5  # Scale for triangle
                    
                return clamp_bary(bary)
            
            def param_obj(params):
                bary = param_to_bary(params)
                return -np.min(calc_areas_bary(bary))
            
            bounds = [
                (0.3, 0.7),  # center_u
                (0.1, 0.4),  # center_v
                (0.1, 0.4),  # scale
                (0, 2*np.pi)  # rotation
            ]
            
            de_result = differential_evolution(
                param_obj,
                bounds,
                seed=42,
                maxiter=50,
                popsize=15,
                tol=1e-4,
                polish=False
            )
            
            bary_de = param_to_bary(de_result.x)
            refined_de = refine_coordinated(bary_de, time_budget=min(8.0, deadline - start_time - 20))
            update_best(refined_de)
        except Exception:
            pass
    
    # 4. Perturb and refine best solution
    if best_bary is not None and time.time() < deadline - 20:
        for i in range(3):
            if time.time() > deadline - 15:
                break
            perturbed = generate_perturbed_init(best_bary, rng, scale=0.03)
            refined = refine_coordinated(perturbed, time_budget=min(5.0, deadline - start_time - 10))
            update_best(refined)
    
    # 5. Final high-precision polish with SLSQP
    if best_bary is not None and time.time() < deadline - 5:
        try:
            bounds = [(0, 1)] * (N * 2)
            result_slsqp = minimize(
                hard_min_obj,
                best_bary.flatten(),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-14}
            )
            update_best(clamp_bary(result_slsqp.x.reshape(N, 2)))
        except Exception:
            pass
    
    # 6. One more coordinated ascent pass
    if best_bary is not None and time.time() < deadline - 2:
        try:
            final_refined = refine_coordinated(best_bary, time_budget=min(3.0, deadline - start_time - 1))
            update_best(final_refined)
        except Exception:
            pass
    
    cart = bary_to_cart(best_bary)
    return cart
