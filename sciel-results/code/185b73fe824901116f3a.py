import numpy as np
from scipy.optimize import minimize, differential_evolution
from itertools import combinations
import time

N = 11
TIME_LIMIT = 145.0

# Pre-compute triangle indices
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)

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

def soft_min_obj(bary_flat, beta=50.0):
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    
    if np.any(areas < 1e-15):
        return 1e10
    
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    
    if max_scaled > 700:
        return -np.min(areas)
        
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min = -log_sum / beta
    return -soft_min

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

def generate_symmetric_init():
    """Generate a symmetric configuration: 3 vertices + 3 edge midpoints + 3 near-centers + 2 more"""
    # Equilateral triangle vertices in barycentric: (1,0), (0,1), (0,0)
    # Edge midpoints: (0.5, 0.5), (0.5, 0), (0, 0.5)
    # Center: (1/3, 1/3)
    # We need 11 points. Let's use a structured approach.
    
    # Start with vertices and midpoints (6 points)
    base = np.array([
        [1.0, 0.0], [0.0, 1.0], [0.0, 0.0],  # vertices
        [0.5, 0.5], [0.5, 0.0], [0.0, 0.5],  # edge midpoints
    ])
    
    # Add center
    center = np.array([[1/3, 1/3]])
    
    # Add 4 more points: perturb the center slightly in different directions
    # or use a smaller hexagonal pattern around center
    extra = []
    r = 0.15
    angles = [0, 60, 120, 180]
    for a in angles:
        rad = np.radians(a)
        u = 1/3 + r * np.cos(rad)
        v = 1/3 + r * np.sin(rad)
        if u >= 0 and v >= 0 and u + v <= 1:
            extra.append([u, v])
    
    if len(extra) < 4:
        # Fallback: random points near center
        rng = np.random.default_rng(42)
        while len(extra) < 4:
            u = rng.uniform(0.2, 0.8)
            v = rng.uniform(0.2, 0.8)
            if u + v <= 0.9:
                extra.append([u, v])
    
    all_pts = np.vstack([base, center, np.array(extra)])
    return clamp_bary(all_pts[:N])

def generate_lattice_init(spacing=0.25):
    """Generate points on a triangular lattice clipped to the triangle"""
    pts = []
    n_max = int(1.0 / spacing) + 1
    for i in range(n_max + 1):
        for j in range(n_max + 1):
            u = i * spacing
            v = j * spacing
            if u >= 0 and v >= 0 and u + v <= 1 + 1e-9:
                pts.append([u, v])
    
    if len(pts) < N:
        # Fallback to random
        return generate_random_init(42)
    
    # Select N points from the lattice to maximize min area
    best_val = -np.inf
    best_bary = None
    rng = np.random.default_rng(42)
    
    for _ in range(20):
        idx = rng.choice(len(pts), N, replace=False)
        bary = np.array(pts)[idx]
        val = -hard_min_obj(bary.flatten())
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
    
    return best_bary if best_bary is not None else generate_random_init(42)

def refine_local(bary_init, time_limit=5.0):
    start_time = time.time()
    bounds = [(0, 1)] * (N * 2)
    
    bary_current = bary_init.copy()
    
    # Stage 1: Graduated soft-min
    for i in range(4):
        beta = 10.0 * (150.0 / 10.0) ** (i / 3.0)
        try:
            result = minimize(
                soft_min_obj,
                bary_current.flatten(),
                args=(beta,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if result.success or result.fun < -hard_min_obj(bary_current.flatten()):
                bary_current = clamp_bary(result.x.reshape(N, 2))
        except Exception:
            pass
        
        if time.time() - start_time > time_limit * 0.4:
            break
    
    if time.time() - start_time > time_limit * 0.4:
        return bary_current
        
    # Stage 2: Nelder-Mead
    try:
        result2 = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 1500, 'xatol': 1e-13, 'fatol': 1e-17, 'adaptive': True}
        )
        bary_current = clamp_bary(result2.x.reshape(N, 2))
    except Exception:
        pass
        
    if time.time() - start_time > time_limit * 0.7:
        return bary_current
    
    # Stage 3: SLSQP
    try:
        result3 = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-15}
        )
        bary_current = clamp_bary(result3.x.reshape(N, 2))
    except Exception:
        pass
        
    return bary_current

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

    # 1. Multiple structured initializations
    inits = [
        generate_hexagonal_init(),
        generate_symmetric_init(),
        generate_lattice_init(0.25),
        generate_lattice_init(0.2),
        generate_random_init(42),
        generate_random_init(123),
    ]
    
    for init in inits:
        refined = refine_local(init, time_limit=8.0)
        update_best(refined)
        if time.time() > deadline - 30:
            break

    # 2. Global Search with Differential Evolution
    if time.time() < deadline - 40:
        try:
            def de_obj(x):
                bary = x.reshape(N, 2)
                bary = clamp_bary(bary)
                areas = calc_areas_bary(bary)
                return -np.min(areas)

            bounds = [(0, 1)] * (N * 2)
            result_de = differential_evolution(
                de_obj,
                bounds,
                seed=42,
                maxiter=30,
                popsize=12,
                tol=1e-7,
                polish=False
            )

            de_bary = clamp_bary(result_de.x.reshape(N, 2))
            refined_de = refine_local(de_bary, time_limit=10.0)
            update_best(refined_de)
        except Exception:
            pass

    # 3. Multi-start
    rng = np.random.default_rng(123)
    restart_count = 0
    while time.time() < deadline - 15:
        restart_count += 1
        if best_bary is not None and restart_count % 2 == 0:
            sigma = 0.03 * (1.0 / (1.0 + 0.1 * restart_count))
            init = generate_perturbed_init(best_bary, sigma=sigma, seed=2000 + restart_count)
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
                options={'maxiter': 1000, 'ftol': 1e-16}
            )
            update_best(clamp_bary(result_slsqp.x.reshape(N, 2)))
        except Exception:
            pass
        
    if best_bary is not None and time.time() < deadline - 3:
        try:
            result_nm = minimize(
                hard_min_obj,
                best_bary.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-15, 'fatol': 1e-19, 'adaptive': True}
            )
            update_best(clamp_bary(result_nm.x.reshape(N, 2)))
        except Exception:
            pass
        
    # Final SLSQP
    if best_bary is not None and time.time() < deadline - 1:
        try:
            bounds = [(0, 1)] * (N * 2)
            result_final = minimize(
                hard_min_obj,
                best_bary.flatten(),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-17}
            )
            update_best(clamp_bary(result_final.x.reshape(N, 2)))
        except Exception:
            pass

    cart = bary_to_cart(best_bary)
    return cart
