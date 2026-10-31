import numpy as np
from scipy.optimize import minimize, differential_evolution
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

def hard_min_obj(bary_flat):
    """Objective using hard min."""
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return -np.min(areas)

def generate_lattice_inits():
    """
    Generate initial configurations based on a triangular lattice.
    This aligns with the 'grid of critical points' directive.
    """
    inits = []
    rng = np.random.default_rng(42)
    
    # Generate a fine triangular lattice in barycentric coordinates
    # We sample points on a grid and select subsets or perturb them.
    # A simple approach: place points on a hexagonal lattice centered in the triangle.
    
    # Center of triangle in Cartesian
    cx, cy = 0.5, np.sqrt(3)/6
    
    # Try different lattice spacings and offsets
    for spacing in [0.15, 0.2, 0.25, 0.3]:
        for offset in [0.0, 0.05, 0.1]:
            bary = np.zeros((N, 2))
            
            # Start with a regular hexagonal pattern
            # 1 center
            bary[0] = [1/3, 1/3]
            
            # 6 neighbors at distance spacing
            for i in range(6):
                angle = i * np.pi / 3
                x = cx + spacing * np.cos(angle)
                y = cy + spacing * np.sin(angle)
                pt = np.array([[x, y]])
                bary_pt = cart_to_bary(pt)[0]
                bary[1+i] = bary_pt
            
            # 4 more points further out? 
            # For N=11, we have 1 + 6 = 7. We need 4 more.
            # Place them at distance 2*spacing in the same directions? Or random perturbations.
            # Let's just use random perturbations of a known good config or random lattice points.
            
            # Instead, let's just generate random points on the lattice
            # Sample N points from a dense lattice
            lattice_points = []
            for i in range(-5, 6):
                for j in range(-5, 6):
                    # Triangular lattice basis vectors
                    x = i * spacing + j * spacing * 0.5
                    y = j * spacing * np.sqrt(3)/2
                    # Check if inside triangle (approx)
                    # Convert to barycentric to check
                    pt = np.array([[x + cx, y + cy]]) # Shift to center
                    # This is getting complicated. Let's stick to the symmetric ring idea but call it lattice-inspired.
                    pass
            
            # Fallback: Use the symmetric ring idea but with more diversity
            # We already have generate_symmetric_inits. Let's just make a simpler lattice-based init.
            # Place 11 points on a grid.
            # Grid 3x3x3? No, 11 points.
            # Let's just use random points for now in this function and rely on the other inits.
            # Actually, let's implement a proper lattice sampling.
            
            # Sample N points uniformly from a triangular lattice
            # Define lattice in Cartesian
            a = spacing
            h = a * np.sqrt(3) / 2
            pts = []
            for i in range(-3, 4):
                for j in range(-3, 4):
                    x = i * a + j * a * 0.5
                    y = j * h
                    # Check if inside triangle (0,0), (1,0), (0.5, sqrt(3)/2)
                    # Barycentric coords for (x,y) relative to triangle
                    # u = 0.5 + 0.5*(1-y/H) - x
                    # v = (1-y/H) - u
                    # w = 1 - u - v
                    S = 1 - y / H
                    u = 0.5 + 0.5 * S - x
                    v = S - u
                    w = 1 - u - v
                    if u > 0.01 and v > 0.01 and w > 0.01:
                        pts.append((u, v))
                        
            if len(pts) >= N:
                # Select N points randomly from the lattice
                indices = rng.choice(len(pts), N, replace=False)
                for k in range(N):
                    bary[k] = pts[indices[k]]
                bary = clamp_bary(bary)
                inits.append(bary)
                
    return inits

def simulated_annealing(bary_init, time_limit=5.0):
    """
    Refine a configuration using Simulated Annealing.
    """
    start_time = time.time()
    rng = np.random.default_rng(12345)
    
    bary_current = bary_init.copy()
    curr_val = calc_min_area(bary_current)
    
    T = 0.01
    T_min = 1e-8
    alpha = 0.99
    
    while time.time() - start_time < time_limit:
        # Generate a neighbor
        step_size = 0.05 * (T / 0.01) # Scale step with temperature
        perturbation = rng.normal(0, step_size, (N, 2))
        new_bary = clamp_bary(bary_current + perturbation)
        new_val = calc_min_area(new_bary)
        
        delta = new_val - curr_val
        
        # Accept if better, or with probability exp(delta/T)
        if delta > 0 or np.random.random() < np.exp(delta / T):
            bary_current = new_bary
            curr_val = new_val
            
        T *= alpha
        if T < T_min:
            T = T_min
            
    return bary_current

def refine_local(bary_init, time_limit=5.0):
    """
    Refine a configuration using Simulated Annealing and then Nelder-Mead.
    """
    start_time = time.time()
    
    # Stage 1: Simulated Annealing
    if time.time() - start_time < time_limit * 0.7:
        bary_sa = simulated_annealing(bary_init, time_limit=time_limit * 0.7)
    else:
        bary_sa = bary_init.copy()
        
    # Stage 2: Nelder-Mead polish
    if time.time() - start_time < time_limit:
        try:
            result_nm = minimize(
                hard_min_obj,
                bary_sa.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-15, 'adaptive': True}
            )
            new_bary = clamp_bary(result_nm.x.reshape(N, 2))
            new_val = calc_min_area(new_bary)
            curr_val = calc_min_area(bary_sa)
            if new_val > curr_val:
                bary_sa = new_bary
        except Exception:
            pass
            
    return bary_sa

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

    # Generate diverse initial points
    rng = np.random.default_rng(123)
    inits = []
    
    # 1. Lattice-based starts
    lattice_inits = generate_lattice_inits()
    inits.extend(lattice_inits)
    
    # 2. Random starts
    for i in range(10):
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        inits.append(np.column_stack([u, v]))
        
    # Deduplicate inits
    unique_inits = []
    for init in inits:
        init_clamped = clamp_bary(init)
        is_unique = True
        for existing in unique_inits:
            if np.allclose(init_clamped, existing, atol=0.015):
                is_unique = False
                break
        if is_unique:
            unique_inits.append(init_clamped)
    
    inits = unique_inits
    
    # Sort inits by initial quality
    inits_with_vals = [(init, calc_min_area(init)) for init in inits]
    inits_with_vals.sort(key=lambda x: -x[1])
    inits = [x[0] for x in inits_with_vals]
    
    # Refine each initial point
    for i, init in enumerate(inits):
        if time.time() > deadline - 12:
            break
            
        remaining_inits = len(inits) - i
        if i < 3:
            base_time = 6.0
        elif i < 8:
            base_time = 4.0
        else:
            base_time = 2.5
            
        time_per_init = min(base_time, (deadline - time.time() - 12) / max(1, remaining_inits))
        if time_per_init < 0.5:
            time_per_init = 0.5
            
        refined = refine_local(init, time_limit=time_per_init)
        update_best(refined)
        
    # Global search with DE
    if best_bary is not None and time.time() < deadline - 25:
        try:
            def de_objective(x):
                bary = x.reshape(N, 2)
                bary = clamp_bary(bary)
                return -np.min(calc_areas_bary(bary))
            
            bounds_de = [(0, 1)] * (N * 2)
            result_de = differential_evolution(
                de_objective,
                bounds_de,
                seed=42,
                maxiter=100,
                tol=1e-12,
                popsize=25,
                mutation=(0.5, 1.0),
                recombination=0.7,
                polish=False,
                update_interval=1,
                callback=None,
                init='sobol'
            )
            if result_de.success or result_de.fun < hard_min_obj(best_bary.flatten()):
                bary_de = clamp_bary(result_de.x.reshape(N, 2))
                if update_best(bary_de):
                    if time.time() < deadline - 8:
                        refined_de = refine_local(bary_de, time_limit=min(8.0, deadline - time.time() - 4))
                        update_best(refined_de)
        except Exception:
            pass
        
    # Final polish
    if best_bary is not None and time.time() < deadline - 8:
        try:
            result_nm = minimize(
                hard_min_obj,
                best_bary.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 20000, 'xatol': 1e-16, 'fatol': 1e-21, 'adaptive': True}
            )
            if result_nm.fun < hard_min_obj(best_bary.flatten()):
                update_best(clamp_bary(result_nm.x.reshape(N, 2)))
        except Exception:
            pass
            
    cart = bary_to_cart(best_bary)
    return cart

# EVOLVE-BLOCK-END
