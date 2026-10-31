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

def min_area(bary):
    """Compute the minimum triangle area for a configuration."""
    areas = calc_areas_bary(bary)
    return np.min(areas)

def generate_hexagonal_init():
    """Generate a hexagonal lattice-like initial configuration."""
    best_bary = None
    best_val = -np.inf
    
    for scale in [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
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
            for _ in range(10):
                if len(pts) == N:
                    bary = np.array(pts)
                else:
                    idx = rng.choice(len(pts), N, replace=False)
                    bary = np.array(pts)[idx]
                val = min_area(bary)
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

def refine_sa(bary_init, time_limit=5.0, rng=None):
    """
    Refine using Simulated Annealing.
    This is a stochastic global optimization method suitable for non-convex landscapes.
    """
    if rng is None:
        rng = np.random.default_rng()
        
    start_time = time.time()
    current = bary_init.copy()
    current_val = min_area(current)
    best = current.copy()
    best_val = current_val
    
    # Initial temperature and cooling schedule
    # The "energy" is -min_area. We want to minimize energy.
    # Step size should be small enough to explore locally but large enough to jump.
    T = 0.01
    T_min = 1e-6
    alpha = 0.9999  # Cooling rate
    
    # Pre-calculate step size based on initial T
    # A good heuristic: step size proportional to sqrt(T) or just T
    # Let's use a fixed step size that decays slightly or stays constant with rejection.
    # Actually, adaptive step size is better.
    step_size = 0.05
    
    while time.time() - start_time < time_limit:
        if T < T_min:
            break
            
        # Generate a neighbor
        # Perturb each coordinate randomly
        # Use a Gaussian perturbation
        delta = rng.normal(0, step_size, current.shape)
        neighbor = current + delta
        neighbor = clamp_bary(neighbor)
        
        neighbor_val = min_area(neighbor)
        delta_e = current_val - neighbor_val # If negative, neighbor is better (higher area)
        
        if neighbor_val > current_val:
            current = neighbor
            current_val = neighbor_val
            if current_val > best_val:
                best_val = current_val
                best = current.copy()
        else:
            # Accept with probability exp(-delta_e / T)
            # delta_e is positive here (neighbor is worse)
            prob = np.exp(-delta_e / T)
            if rng.random() < prob:
                current = neighbor
                current_val = neighbor_val
                
        T *= alpha
        # Optionally adjust step size
        # step_size *= 0.999
        
    return best, best_val

def refine_local(bary_init, time_limit=5.0):
    """
    Refine a configuration using a combination of SA and local polishing.
    """
    start_time = time.time()
    
    bary_current = bary_init.copy()
    current_val = min_area(bary_current)
    
    # Stage 1: Simulated Annealing (Global search)
    # Allocate 70% of time to SA
    sa_time = time_limit * 0.7
    try:
        bary_sa, val_sa = refine_sa(bary_current, time_limit=sa_time)
        if val_sa > current_val:
            bary_current = bary_sa
            current_val = val_sa
    except Exception:
        pass
    
    if time.time() - start_time > time_limit * 0.8:
        return bary_current, current_val
        
    # Stage 2: SLSQP on min-area directly (Local polish)
    # Allocate remaining time to SLSQP
    try:
        bounds = [(0, 1)] * (N * 2)
        # Use a tight tolerance for fast convergence
        result2 = minimize(
            lambda x: -min_area(x.reshape(N, 2)),
            bary_current.flatten(),
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-12}
        )
        candidate = clamp_bary(result2.x.reshape(N, 2))
        val = min_area(candidate)
        if val > current_val:
            bary_current = candidate
            current_val = val
    except Exception:
        pass
        
    # Stage 3: Nelder-Mead (Final polish)
    if time.time() - start_time < time_limit:
        try:
            result3 = minimize(
                lambda x: -min_area(x.reshape(N, 2)),
                bary_current.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            candidate = clamp_bary(result3.x.reshape(N, 2))
            val = min_area(candidate)
            if val > current_val:
                bary_current = candidate
                current_val = val
        except Exception:
            pass

    return bary_current, current_val

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_bary = None
    best_val = -np.inf
    
    def update_best(bary):
        nonlocal best_bary, best_val
        bary = clamp_bary(bary)
        val = min_area(bary)
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
            return True
        return False

    # 1. Generate diverse initializations
    inits = []
    
    # Hexagonal
    hex_init = generate_hexagonal_init()
    inits.append(hex_init)
    
    # Random
    for i in range(5):
        inits.append(generate_random_init(seed=1000 + i))
        
    # Perturbed hex
    rng = np.random.default_rng(42)
    for i in range(5):
        noise = rng.normal(0, 0.05, hex_init.shape)
        inits.append(clamp_bary(hex_init + noise))
        
    # 2. Refine each initialization
    # We have 11 initializations.
    # Total time ~145s.
    # Let's give each refinement a budget.
    # If we have 11 inits, 145/11 ~ 13s each.
    # But SA can be effective with less time if step size is right.
    # Let's try 10s per refinement.
    
    for i, init in enumerate(inits):
        if time.time() > deadline - 15:
            break
            
        # Estimate time budget for this refinement
        # Leave 15s for final polish
        time_left = deadline - time.time()
        # Don't spend more than 15s on a single init if we have many left,
        # but don't starve it either.
        # Simple heuristic: 10s per init, capped by remaining time.
        alloc_time = min(10.0, time_left / (len(inits) - i))
        
        # Create a dedicated RNG for reproducibility within this run
        rng_init = np.random.default_rng(12345 + i)
        
        refined, val = refine_local(init, time_limit=alloc_time)
        update_best(refined)
        
    # 3. If time remains, do a final high-precision polish on the best found
    if best_bary is not None and time.time() < deadline - 5:
        try:
            # One more SLSQP pass with tighter tolerance
            bounds = [(0, 1)] * (N * 2)
            result_final = minimize(
                lambda x: -min_area(x.reshape(N, 2)),
                best_bary.flatten(),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-14}
            )
            update_best(clamp_bary(result_final.x.reshape(N, 2)))
        except Exception:
            pass
        
        # One more Nelder-Mead pass
        if time.time() < deadline - 2:
            try:
                result_nm = minimize(
                    lambda x: -min_area(x.reshape(N, 2)),
                    best_bary.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
                )
                update_best(clamp_bary(result_nm.x.reshape(N, 2)))
            except Exception:
                pass

    # Convert to cartesian
    cart = bary_to_cart(best_bary)
    
    return cart

# EVOLVE-BLOCK-END
