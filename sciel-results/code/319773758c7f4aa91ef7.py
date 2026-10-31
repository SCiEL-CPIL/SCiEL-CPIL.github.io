import numpy as np
from scipy.optimize import minimize
import time
from itertools import combinations

N = 11
TIME_LIMIT = 115.0

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
    """Convert Cartesian (x, y) to barycentric (u, v)."""
    x = cart[:, 0]
    y = cart[:, 1]
    S = 1 - y / H
    u = 0.5 + 0.5 * S - x
    v = S - u
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

def calc_min_area(bary):
    """Calculate minimum area efficiently."""
    return np.min(calc_areas_bary(bary))

def clamp_bary(bary, eps=1e-12):
    """Clamp barycentric coordinates to the strictly valid triangle region."""
    u = bary[:, 0].copy()
    v = bary[:, 1].copy()
    
    # Simple projection to barycentric simplex
    # u, v >= 0, u+v <= 1
    u = np.maximum(u, 0)
    v = np.maximum(v, 0)
    s = u + v
    invalid = s > 1.0
    if np.any(invalid):
        scale = 1.0 / s[invalid]
        u[invalid] *= scale
        v[invalid] *= scale
        
    # Add small epsilon to avoid boundary issues if needed, 
    # but for max area, boundary is usually not optimal unless N is small.
    # We can allow boundary but ensure valid.
    return np.column_stack([u, v])

def objective(bary_flat):
    """Objective function: negative of min area."""
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return -np.min(areas)

def generate_initial_configs():
    """Generate diverse high-quality initial configurations."""
    configs = []
    
    # 1. Symmetric Hexagonal/Lattice-like
    # Center at (1/3, 1/3)
    center = np.array([1/3, 1/3])
    # Try different scales
    for scale in [0.1, 0.15, 0.2, 0.25]:
        pts = [center]
        # 6 neighbors
        for k in range(6):
            theta = np.pi/3 * k
            dx = scale * np.cos(theta)
            dy = scale * np.sin(theta)
            pts.append(center + np.array([dx, dy]))
        # 6 more neighbors (2nd ring)
        for k in range(6):
            if len(pts) >= N: break
            theta = np.pi/3 * k + np.pi/6
            dx = 2 * scale * np.cos(theta)
            dy = 2 * scale * np.sin(theta)
            pts.append(center + np.array([dx, dy]))
        
        if len(pts) >= N:
            cart = np.array(pts[:N])
            bary = cart_to_bary(cart)
            bary = clamp_bary(bary)
            # Check validity
            if np.all(bary[:, 0] >= 0) and np.all(bary[:, 1] >= 0) and np.all(bary[:, 0] + bary[:, 1] <= 1):
                configs.append(bary)

    # 2. Random with symmetry breaking
    rng = np.random.default_rng(42)
    for _ in range(20):
        # Random points
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        bary = np.column_stack([u, v])
        configs.append(bary)
        
    # 3. Perturbed Grid
    for _ in range(10):
        # 3x4 grid approx
        u = np.linspace(0.1, 0.8, 4)
        v = np.linspace(0.1, 0.8, 3)
        grid_u, grid_v = np.meshgrid(u, v)
        # Take 11 points
        idx = np.random.choice(grid_u.size, N, replace=False)
        bary = np.column_stack([grid_u.flat[idx], grid_v.flat[idx]])
        # Perturb
        bary += rng.normal(0, 0.05, bary.shape)
        bary = clamp_bary(bary)
        configs.append(bary)
        
    return configs

def refine_local(bary_init, time_limit=2.0):
    """
    High-precision local refinement.
    Uses a combination of SLSQP and a custom active-set Nelder-Mead.
    """
    start_time = time.time()
    bounds = [(0, 1)] * (N * 2)
    
    bary_current = bary_init.copy()
    curr_val = calc_min_area(bary_current)
    
    # Step 1: SLSQP to get close to a local optimum
    try:
        res = minimize(
            objective, 
            bary_current.flatten(), 
            method='SLSQP', 
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-15}
        )
        bary_sl = clamp_bary(res.x.reshape(N, 2))
        val_sl = calc_min_area(bary_sl)
        if val_sl > curr_val:
            bary_current = bary_sl
            curr_val = val_sl
    except Exception:
        pass
        
    # Step 2: Active Set Nelder-Mead
    # Identify critical triangles
    areas = calc_areas_bary(bary_current)
    min_area = np.min(areas)
    # Points involved in min-area triangles
    critical_idx = np.where(areas <= min_area * 1.01)[0]
    critical_triangles = tri_idx[critical_idx]
    involved_points = np.unique(critical_triangles.flatten())
    
    # We can try to optimize only the involved points while keeping others fixed?
    # No, moving others might improve the min area by moving the min away.
    # Instead, use Nelder-Mead on the full set but with tight tolerances.
    
    if time.time() - start_time < time_limit * 0.7:
        try:
            res_nm = minimize(
                objective,
                bary_current.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-16, 'adaptive': True}
            )
            bary_nm = clamp_bary(res_nm.x.reshape(N, 2))
            val_nm = calc_min_area(bary_nm)
            if val_nm > curr_val:
                bary_current = bary_nm
                curr_val = val_nm
        except Exception:
            pass
            
    # Step 3: Fine-grained random search around current best
    if time.time() - start_time < time_limit:
        rng = np.random.default_rng(123)
        step = 0.01
        for _ in range(50):
            if time.time() - start_time > time_limit:
                break
            improved = False
            for _ in range(10):
                pert = rng.normal(0, step, (N, 2))
                new_bary = clamp_bary(bary_current + pert)
                new_val = calc_min_area(new_bary)
                if new_val > curr_val:
                    bary_current = new_bary
                    curr_val = new_val
                    improved = True
                    break
            if not improved:
                step *= 0.8
            if step < 1e-8:
                break
                
    return bary_current

def simulated_annealing(bary_init, time_limit=5.0):
    """
    Simulated Annealing for global search.
    """
    start_time = time.time()
    current = bary_init.copy()
    current_val = calc_min_area(current)
    
    best = current.copy()
    best_val = current_val
    
    T = 0.01 # Initial temperature
    T_min = 1e-6
    cooling = 0.99
    
    rng = np.random.default_rng(456)
    
    while time.time() - start_time < time_limit:
        # Generate neighbor
        step = 0.05
        pert = rng.normal(0, step, (N, 2))
        new_bary = clamp_bary(current + pert)
        new_val = calc_min_area(new_bary)
        
        delta = new_val - current_val
        
        if delta > 0:
            # Improvement
            current = new_bary
            current_val = new_val
            if current_val > best_val:
                best = current.copy()
                best_val = current_val
        else:
            # Accept with probability exp(delta / T)
            # Note: delta is negative
            if rng.random() < np.exp(delta / T):
                current = new_bary
                current_val = new_val
                
        T *= cooling
        if T < T_min:
            break
            
    return best

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_bary = None
    best_val = -np.inf
    
    def update_best(bary):
        nonlocal best_bary, best_val
        bary = clamp_bary(bary)
        val = calc_min_area(bary)
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
            return True
        return False

    # 1. Generate Initials
    inits = generate_initial_configs()
    
    # Sort by initial quality
    inits_with_vals = [(init, calc_min_area(init)) for init in inits]
    inits_with_vals.sort(key=lambda x: -x[1])
    inits = [x[0] for x in inits_with_vals]
    
    # 2. Refine Top Initials Locally
    for i, init in enumerate(inits[:10]):
        if time.time() > deadline - 30:
            break
        # Allocate more time to better starts
        time_budget = 5.0 if i < 3 else 3.0
        refined = refine_local(init, time_limit=time_budget)
        update_best(refined)
        
    # 3. Simulated Annealing from Best Found
    if best_bary is not None and time.time() < deadline - 10:
        sa_time = min(20.0, deadline - time.time() - 10)
        sa_result = simulated_annealing(best_bary, time_limit=sa_time)
        if update_best(sa_result):
             # Refine SA result
             if time.time() < deadline - 5:
                refined_sa = refine_local(sa_result, time_limit=5.0)
                update_best(refined_sa)
                
    # 4. Final Polish
    if best_bary is not None and time.time() < deadline - 2:
        refined_final = refine_local(best_bary, time_limit=min(4.0, deadline - time.time() - 1))
        update_best(refined_final)
        
    # Convert to cartesian
    cart = bary_to_cart(best_bary)
    return cart
