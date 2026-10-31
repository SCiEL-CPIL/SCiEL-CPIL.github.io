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

def soft_min_obj(bary_flat, beta=50.0):
    """
    Smooth surrogate for the hard min via log-sum-exp.
    Minimizing this pushes ALL areas up, with emphasis on the smallest.
    """
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

def equalize_min_obj(bary_flat, n_crit=5):
    """
    Objective that tries to equalize the smallest n_crit triangle areas.
    At the optimum, multiple triangles should have exactly the minimum area.
    We minimize the variance of the smallest areas while maximizing their mean.
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    
    # Get the smallest n_crit areas
    sorted_areas = np.sort(areas)
    crit_areas = sorted_areas[:n_crit]
    
    min_a = crit_areas[0]
    mean_a = np.mean(crit_areas)
    std_a = np.std(crit_areas)
    
    # Penalize variance strongly, maximize mean
    return -mean_a + 10.0 * std_a

def generate_symmetric_inits():
    """
    Generate initial configurations with 3-fold rotational symmetry.
    We place 1 point at the center and 10 points in 2 rings of 5? No, 11 is odd.
    Structure: 1 center + 3 orbits of 3 points = 1 + 9 = 10. Need 11.
    Maybe: 1 center + 1 orbit of 3 + 1 orbit of 6? Or just break symmetry slightly.
    
    Let's try:
    1. Center point.
    2. 3 points at distance r1, angles 0, 120, 240.
    3. 3 points at distance r2, angles 60, 180, 300.
    4. 3 points at distance r3, angles 0, 120, 240 (closer to vertices).
    5. 1 point? This sums to 1+3+3+3+1 = 11.
    
    Actually, simpler:
    Start with a regular hexagonal lattice clipped to triangle, or just random symmetric perturbations of a known good pattern.
    """
    inits = []
    rng = np.random.default_rng(42)
    
    # Pattern 1: Center + 2 rings of 3 (rotated) + 1 extra?
    # Let's use a set of base parameters and perturb them.
    
    # Base configuration 1: 
    # Points at (1/3, 1/3)
    # 3 points at dist 0.15 from center
    # 3 points at dist 0.30 from center
    # 3 points at dist 0.45 from center
    # 1 point? This is 10 points. 
    # Let's add a 4th ring of 1? No, symmetry breaks.
    
    # Let's try a different decomposition:
    # 1 center
    # 3 points near midpoints of sides
    # 3 points near vertices
    # 3 points in between
    # 1 point?
    
    # Instead of forcing strict symmetry which might be hard to optimize with standard optimizers on full 22D space,
    # we will generate high-quality random starts and rely on the optimizer.
    # But we will bias them towards symmetry.
    
    for trial in range(20):
        bary = np.zeros((N, 2))
        # Center
        bary[0] = [1/3, 1/3]
        
        # Ring 1: 3 points
        r1 = 0.1 + rng.uniform(0, 0.1)
        for i in range(3):
            angle = 2 * np.pi * i / 3 + rng.uniform(-0.1, 0.1)
            # Map polar to barycentric approx
            # Center is (1/3, 1/3). 
            # x = 1/3 + r cos(theta), y = 1/3 + r sin(theta) * (2/sqrt(3)) ? No, aspect ratio.
            # Cartesian: cx=0.5, cy=sqrt(3)/6.
            cx = 0.5
            cy = np.sqrt(3)/6
            x = cx + r1 * np.cos(angle)
            y = cy + r1 * np.sin(angle) * (2/np.sqrt(3)) # Scale y to match triangle geometry?
            # Actually, just use Cartesian and convert back.
            cart_pt = np.array([[x, y]])
            bary_pt = cart_to_bary(cart_pt)[0]
            bary[1+i] = bary_pt
            
        # Ring 2: 3 points
        r2 = 0.25 + rng.uniform(0, 0.1)
        angle_offset = np.pi/3 + rng.uniform(-0.1, 0.1)
        for i in range(3):
            angle = 2 * np.pi * i / 3 + angle_offset
            cx = 0.5
            cy = np.sqrt(3)/6
            x = cx + r2 * np.cos(angle)
            y = cy + r2 * np.sin(angle) * (2/np.sqrt(3))
            cart_pt = np.array([[x, y]])
            bary_pt = cart_to_bary(cart_pt)[0]
            bary[4+i] = bary_pt
            
        # Ring 3: 3 points
        r3 = 0.35 + rng.uniform(0, 0.1)
        angle_offset2 = 0 + rng.uniform(-0.1, 0.1)
        for i in range(3):
            angle = 2 * np.pi * i / 3 + angle_offset2
            cx = 0.5
            cy = np.sqrt(3)/6
            x = cx + r3 * np.cos(angle)
            y = cy + r3 * np.sin(angle) * (2/np.sqrt(3))
            cart_pt = np.array([[x, y]])
            bary_pt = cart_to_bary(cart_pt)[0]
            bary[7+i] = bary_pt
            
        # Point 11: Perturbed center or random?
        # Let's put it at a random position near center to break symmetry slightly if needed
        bary[10] = [1/3 + rng.uniform(-0.05, 0.05), 1/3 + rng.uniform(-0.05, 0.05)]
        
        bary = clamp_bary(bary)
        if np.all(bary[:, 0] > 0) and np.all(bary[:, 1] > 0) and np.all(bary[:, 0] + bary[:, 1] < 1):
            inits.append(bary)
            
    # Add some fully random starts for diversity
    for i in range(10):
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        inits.append(np.column_stack([u, v]))
        
    return inits

def refine_local(bary_init, time_limit=5.0):
    """
    Refine a configuration using multiple optimization stages.
    """
    start_time = time.time()
    bounds = [(0, 1)] * (N * 2)
    
    bary_current = bary_init.copy()
    curr_val = calc_min_area(bary_current)
    
    # Stage 1: Smooth soft-min minimization with increasing beta
    for beta in [20.0, 100.0, 500.0]:
        if time.time() - start_time > time_limit * 0.3:
            break
        try:
            result1 = minimize(
                soft_min_obj,
                bary_current.flatten(),
                args=(beta,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1500, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            new_bary = clamp_bary(result1.x.reshape(N, 2))
            new_val = calc_min_area(new_bary)
            if new_val > curr_val:
                bary_current = new_bary
                curr_val = new_val
        except Exception:
            pass
        
    # Stage 2: Equalize the smallest triangle areas
    if time.time() - start_time < time_limit * 0.5:
        try:
            # Try different numbers of critical triangles to equalize
            for n_crit in [3, 5, 7]:
                if time.time() - start_time > time_limit * 0.5:
                    break
                result2 = minimize(
                    equalize_min_obj,
                    bary_current.flatten(),
                    args=(n_crit,),
                    method='SLSQP',
                    bounds=bounds,
                    options={'maxiter': 2000, 'ftol': 1e-14}
                )
                new_bary = clamp_bary(result2.x.reshape(N, 2))
                new_val = calc_min_area(new_bary)
                if new_val > curr_val:
                    bary_current = new_bary
                    curr_val = new_val
        except Exception:
            pass
            
    # Stage 3: Hard min refinement with SLSQP
    if time.time() - start_time < time_limit * 0.7:
        try:
            result3 = minimize(
                hard_min_obj,
                bary_current.flatten(),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 3000, 'ftol': 1e-16}
            )
            new_bary = clamp_bary(result3.x.reshape(N, 2))
            new_val = calc_min_area(new_bary)
            if new_val > curr_val:
                bary_current = new_bary
                curr_val = new_val
        except Exception:
            pass
            
    # Stage 4: Nelder-Mead for final polish
    if time.time() - start_time < time_limit * 0.9:
        try:
            result4 = minimize(
                hard_min_obj,
                bary_current.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-18, 'adaptive': True}
            )
            new_bary = clamp_bary(result4.x.reshape(N, 2))
            new_val = calc_min_area(new_bary)
            if new_val > curr_val:
                bary_current = new_bary
                curr_val = new_val
        except Exception:
            pass
        
    # Stage 5: Adaptive local search with decreasing step size
    if time.time() - start_time < time_limit:
        rng = np.random.default_rng(int(time.time() * 1000) % 100000)
        step_size = 0.03
        for iteration in range(80):
            if time.time() - start_time > time_limit:
                break
            # Try multiple perturbations with current step size
            improved = False
            for _ in range(8):
                perturbation = rng.normal(0, step_size, (N, 2))
                new_bary = clamp_bary(bary_current + perturbation)
                new_val = calc_min_area(new_bary)
                if new_val > curr_val:
                    bary_current = new_bary
                    curr_val = new_val
                    improved = True
                    break
            if not improved:
                step_size *= 0.75
            if step_size < 1e-7:
                break
                
    return bary_current

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
    
    # 1. Symmetric starts (reduced count)
    sym_inits = generate_symmetric_inits()
    inits.extend(sym_inits[:10])  # Take only top 10 symmetric inits
    
    # 2. Random starts (reduced count)
    for i in range(5):
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        inits.append(np.column_stack([u, v]))
        
    # Deduplicate inits (approximately)
    unique_inits = []
    for init in inits:
        init_clamped = clamp_bary(init)
        is_unique = True
        for existing in unique_inits:
            if np.allclose(init_clamped, existing, atol=0.02):
                is_unique = False
                break
        if is_unique:
            unique_inits.append(init_clamped)
    
    inits = unique_inits
    
    # Sort inits by initial quality (best first)
    inits_with_vals = [(init, calc_min_area(init)) for init in inits]
    inits_with_vals.sort(key=lambda x: -x[1])
    inits = [x[0] for x in inits_with_vals]
    
    # Refine each initial point with time budget
    for i, init in enumerate(inits):
        if time.time() > deadline - 15:
            break
            
        # Allocate time based on rank and remaining time
        remaining_inits = len(inits) - i
        # Give more time to top-ranked inits
        if i < 2:
            base_time = 8.0
        elif i < 5:
            base_time = 5.0
        else:
            base_time = 3.0
            
        time_per_init = min(base_time, (deadline - time.time() - 15) / max(1, remaining_inits))
        if time_per_init < 0.5:
            time_per_init = 0.5
            
        refined = refine_local(init, time_limit=time_per_init)
        update_best(refined)
        
    # If we have time, do a global search with differential evolution
    if best_bary is not None and time.time() < deadline - 30:
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
                maxiter=80,
                tol=1e-10,
                popsize=20,
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
                    # Refine the DE result
                    if time.time() < deadline - 10:
                        refined_de = refine_local(bary_de, time_limit=min(10.0, deadline - time.time() - 5))
                        update_best(refined_de)
        except Exception:
            pass
        
    # Final high-precision polish on the best found
    if best_bary is not None and time.time() < deadline - 10:
        try:
            bounds = [(0, 1)] * (N * 2)
            # First, try to equalize the minimum areas
            result_eq = minimize(
                equalize_min_obj,
                best_bary.flatten(),
                args=(5,),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 3000, 'ftol': 1e-15}
            )
            new_bary = clamp_bary(result_eq.x.reshape(N, 2))
            if calc_min_area(new_bary) > best_val:
                best_bary = new_bary
                best_val = calc_min_area(best_bary)
        except Exception:
            pass
        
        if best_bary is not None and time.time() < deadline - 6:
            try:
                result_slsqp = minimize(
                    hard_min_obj,
                    best_bary.flatten(),
                    method='SLSQP',
                    bounds=bounds,
                    options={'maxiter': 5000, 'ftol': 1e-17}
                )
                if result_slsqp.fun < hard_min_obj(best_bary.flatten()):
                    update_best(clamp_bary(result_slsqp.x.reshape(N, 2)))
            except Exception:
                pass
        
        if best_bary is not None and time.time() < deadline - 3:
            try:
                result_nm = minimize(
                    hard_min_obj,
                    best_bary.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 10000, 'xatol': 1e-15, 'fatol': 1e-20, 'adaptive': True}
                )
                if result_nm.fun < hard_min_obj(best_bary.flatten()):
                    update_best(clamp_bary(result_nm.x.reshape(N, 2)))
            except Exception:
                pass
            
        # Final adaptive perturbation search
        if best_bary is not None and time.time() < deadline - 1:
            rng_final = np.random.default_rng(999)
            step = 0.02
            for _ in range(40):
                if time.time() > deadline - 1:
                    break
                improved = False
                for _ in range(6):
                    pert = rng_final.normal(0, step, (N, 2))
                    new_bary = clamp_bary(best_bary + pert)
                    if update_best(new_bary):
                        improved = True
                        break
                if not improved:
                    step *= 0.85
                if step < 1e-8:
                    break

    # Convert to cartesian
    cart = bary_to_cart(best_bary)
    
    return cart

# EVOLVE-BLOCK-END
