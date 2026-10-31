import numpy as np
from scipy.optimize import minimize
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

def clamp_bary(bary, eps=1e-9):
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

def cart_to_bary(cart):
    """Convert Cartesian to barycentric coordinates."""
    x = cart[:, 0]
    y = cart[:, 1]
    u = 1.0 - 2.0 * x - 2.0 * y / H
    v = 2.0 * y / H
    return np.column_stack([u, v])

def hexagonal_pattern_11():
    """
    Generate 11 points in a hexagonal lattice pattern.
    Center + 6 in first ring + 4 in second ring (partial).
    Returns points in a local coordinate system centered at origin.
    """
    pts = []
    # Center
    pts.append([0.0, 0.0])
    # First ring: 6 points at distance 1
    for i in range(6):
        angle = i * np.pi / 3
        pts.append([np.cos(angle), np.sin(angle)])
    # Second ring: pick 4 points to make 11 total
    # Second ring has 12 points at distance 2, pick every 3rd
    for i in range(0, 12, 3):
        angle = i * np.pi / 6
        pts.append([2.0 * np.cos(angle), 2.0 * np.sin(angle)])
        if len(pts) == 11:
            break
    return np.array(pts[:11])

def apply_affine(params, base_pts):
    """
    Apply affine transformation: x' = A @ x + t
    params = [a11, a12, a21, a22, tx, ty]
    """
    a11, a12, a21, a22, tx, ty = params
    A = np.array([[a11, a12], [a21, a22]])
    t = np.array([tx, ty])
    return base_pts @ A.T + t

def affine_obj(params, base_pts):
    """
    Objective for affine transform optimization.
    Returns negative min triangle area (to minimize).
    Also penalizes points outside the triangle.
    """
    cart = apply_affine(params, base_pts)
    bary = cart_to_bary(cart)
    
    # Check if all points are inside the triangle
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    # Penalty for points outside
    outside_penalty = np.sum(np.maximum(0, -u + 1e-6) ** 2 + np.maximum(0, -v + 1e-6) ** 2 + np.maximum(0, -w + 1e-6) ** 2)
    
    if outside_penalty > 1e-4:
        return outside_penalty + 1.0
    
    bary_clamped = clamp_bary(bary)
    areas = calc_areas_bary(bary_clamped)
    return -np.min(areas) + 100.0 * outside_penalty

def optimize_affine(base_pts, init_params, time_limit=5.0):
    """Optimize affine transformation parameters."""
    start_time = time.time()
    
    # Bounds: reasonable range for affine parameters
    bounds = [(-1.5, 1.5)] * 6
    
    best_x = init_params.copy()
    best_val = affine_obj(init_params, base_pts)
    
    # Try multiple methods
    methods = [
        ('L-BFGS-B', {'maxiter': 500, 'ftol': 1e-15}),
        ('Nelder-Mead', {'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-16, 'adaptive': True}),
        ('SLSQP', {'maxiter': 800, 'ftol': 1e-15}),
    ]
    
    for method, opts in methods:
        if time.time() - start_time > time_limit:
            break
        try:
            result = minimize(
                affine_obj,
                best_x,
                args=(base_pts,),
                method=method,
                bounds=bounds if method != 'Nelder-Mead' else None,
                options=opts
            )
            val = result.fun
            if val < best_val:
                best_val = val
                best_x = result.x.copy()
        except Exception:
            pass
    
    return best_x, best_val

def generate_hex_init_with_affine():
    """
    Generate initial configurations using hexagonal pattern + affine optimization.
    """
    base_pts = hexagonal_pattern_11()
    
    best_bary = None
    best_val = -np.inf
    
    # Try different initial affine parameters
    rng = np.random.default_rng(42)
    
    # Symmetric initial guess: scale to fit in triangle
    # The triangle has area sqrt(3)/4, height H
    # A symmetric placement would center the hexagon
    for trial in range(20):
        if trial == 0:
            # Centered, scaled
            s = 0.3
            params = np.array([s, 0, 0, s, 0.5, H/3])
        elif trial == 1:
            # Slightly rotated
            angle = np.pi / 6
            s = 0.3
            c, sn = np.cos(angle), np.sin(angle)
            params = np.array([s*c, s*sn, -s*sn, s*c, 0.5, H/3])
        else:
            # Random perturbation around symmetric
            s = 0.2 + 0.3 * rng.random()
            angle = rng.uniform(-np.pi/6, np.pi/6)
            c, sn = np.cos(angle), np.sin(angle)
            tx = 0.5 + rng.uniform(-0.1, 0.1)
            ty = H/3 + rng.uniform(-0.1, 0.1)
            params = np.array([s*c, s*sn, -s*sn, s*c, tx, ty])
        
        # Optimize
        opt_params, val = optimize_affine(base_pts, params, time_limit=3.0)
        
        if val < 0:  # Valid configuration
            cart = apply_affine(opt_params, base_pts)
            bary = cart_to_bary(cart)
            bary = clamp_bary(bary)
            actual_val = -hard_min_obj(bary.flatten())
            
            if actual_val > best_val:
                best_val = actual_val
                best_bary = bary.copy()
    
    return best_bary, best_val

def refine_local(bary_init, time_limit=5.0):
    start_time = time.time()
    bounds = [(0, 1)] * (N * 2)
    bary_current = bary_init.copy()
    
    try:
        result1 = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        if result1.success or result1.fun < -hard_min_obj(bary_current.flatten()):
            bary_current = clamp_bary(result1.x.reshape(N, 2))
    except Exception:
        pass
    
    if time.time() - start_time > time_limit * 0.4:
        return bary_current
    
    try:
        result2 = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-16, 'adaptive': True}
        )
        bary_current = clamp_bary(result2.x.reshape(N, 2))
    except Exception:
        pass
    
    if time.time() - start_time > time_limit * 0.8:
        return bary_current
    
    try:
        result3 = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 600, 'ftol': 1e-14}
        )
        bary_current = clamp_bary(result3.x.reshape(N, 2))
    except Exception:
        pass
    
    return bary_current

def generate_hexagonal_init():
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
            rng = np.random.default_rng(42)
            for _ in range(5):
                if len(pts) == N:
                    bary = np.array(pts)
                else:
                    idx = rng.choice(len(pts), N, replace=False)
                    bary = np.array(pts)[idx]
                val = -hard_min_obj(bary.flatten())
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
    rng = np.random.default_rng(seed)
    u = rng.random(N)
    v = rng.random(N) * (1 - u)
    return np.column_stack([u, v])

def generate_perturbed_init(base_bary, sigma=0.05, seed=None):
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, sigma, base_bary.shape)
    return clamp_bary(base_bary + noise)

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

    # 1. Affine hexagonal optimization (structural approach)
    hex_affine_bary, hex_affine_val = generate_hex_init_with_affine()
    if hex_affine_bary is not None:
        update_best(hex_affine_bary)
        # Refine the affine result with full local optimization
        if time.time() < deadline - 30:
            refined = refine_local(hex_affine_bary, time_limit=5.0)
            update_best(refined)

    # 2. Traditional hexagonal init + local refinement
    if time.time() < deadline - 30:
        hex_init = generate_hexagonal_init()
        refined_hex = refine_local(hex_init, time_limit=5.0)
        update_best(refined_hex)

    # 3. Multi-start with random and perturbed restarts
    rng = np.random.default_rng(123)
    restart_count = 0
    while time.time() < deadline - 15:
        restart_count += 1
        if best_bary is not None and restart_count % 3 == 0:
            init = generate_perturbed_init(best_bary, sigma=0.03, seed=2000 + restart_count)
        elif best_bary is not None and restart_count % 3 == 1:
            init = generate_perturbed_init(best_bary, sigma=0.08, seed=2500 + restart_count)
        else:
            init = generate_random_init(seed=3000 + restart_count)
        
        refined = refine_local(init, time_limit=3.0)
        update_best(refined)

    # 4. Final polish
    if best_bary is not None and time.time() < deadline - 5:
        try:
            bounds = [(0, 1)] * (N * 2)
            result_slsqp = minimize(
                hard_min_obj,
                best_bary.flatten(),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-15}
            )
            update_best(clamp_bary(result_slsqp.x.reshape(N, 2)))
        except Exception:
            pass
        
        if best_bary is not None and time.time() < deadline - 2:
            try:
                result_nm = minimize(
                    hard_min_obj,
                    best_bary.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 3000, 'xatol': 1e-14, 'fatol': 1e-18, 'adaptive': True}
                )
                update_best(clamp_bary(result_nm.x.reshape(N, 2)))
            except Exception:
                pass

    cart = bary_to_cart(best_bary)
    return cart
