import numpy as np
from scipy.optimize import differential_evolution, minimize
from itertools import combinations
import time

N = 11
TIME_LIMIT = 148.0  # Safety margin for 150s limit

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
    
    ui = u[tri_idx[:, 0]]
    vi = v[tri_idx[:, 0]]
    
    uj = u[tri_idx[:, 1]]
    vj = v[tri_idx[:, 1]]
    
    uk = u[tri_idx[:, 2]]
    vk = v[tri_idx[:, 2]]
    
    cross = (uj - ui) * (vk - vi) - (vj - vi) * (uk - ui)
    
    return np.abs(cross)

def clamp_bary(bary, eps=1e-10):
    """Clamp barycentric coordinates to the valid triangle region."""
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

def get_active_triangles(bary, threshold=1e-6):
    """Get indices of triangles with area near the minimum."""
    areas = calc_areas_bary(bary)
    min_area = np.min(areas)
    active = np.where(areas <= min_area + threshold)[0]
    return active, min_area

def epigraph_obj(x):
    """
    Epigraph formulation: maximize t subject to areas >= t.
    x = [bary_flat (22), t]
    Minimize -t subject to areas_i >= t for all i.
    """
    bary_flat = x[:N*2]
    t = x[N*2]
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    
    # Penalize violations of areas >= t
    violations = np.maximum(0, t - areas)
    penalty = np.sum(violations**2)
    
    return -t + 100.0 * penalty

def generate_symmetric_init():
    """
    Generate a symmetric configuration exploiting 3-fold symmetry.
    """
    bary = np.zeros((N, 2))
    
    # Near vertex 0: barycentric (1,0,0) -> (u=1,v=0)
    bary[0] = [0.95, 0.02]
    # Near vertex 1: barycentric (0,1,0) -> (u=0,v=1)
    bary[1] = [0.02, 0.95]
    # Near vertex 2: barycentric (0,0,1) -> (u=0,v=0)
    bary[2] = [0.02, 0.02]
    
    # Near edge midpoints (offset inward)
    bary[3] = [0.48, 0.03]  # edge 0-1
    bary[4] = [0.03, 0.48]  # edge 1-2
    bary[5] = [0.48, 0.48]  # edge 2-0
    
    # Center
    bary[6] = [1/3, 1/3]
    
    # Additional points: form a small hexagon around center
    r = 0.15
    for i in range(3):
        angle = 2 * np.pi * i / 3 + np.pi / 6
        du = r * np.cos(angle)
        dv = r * np.sin(angle)
        bary[7 + i] = [1/3 + du, 1/3 + dv]
    
    # One more point
    bary[10] = [0.5, 0.25]
    
    return clamp_bary(bary)

def generate_radial_init():
    """
    Generate points using radial/angular parameterization from centroid.
    """
    bary = np.zeros((N, 2))
    cx, cy = 1/3, 1/3
    
    # Ring 1: 3 points at radius r1, angles 0, 120, 240
    r1 = 0.35
    for i in range(3):
        angle = 2 * np.pi * i / 3
        bary[i] = [cx + r1 * np.cos(angle), cy + r1 * np.sin(angle)]
    
    # Ring 2: 3 points at radius r2, angles 60, 180, 300
    r2 = 0.25
    for i in range(3):
        angle = 2 * np.pi * i / 3 + np.pi / 3
        bary[3 + i] = [cx + r2 * np.cos(angle), cy + r2 * np.sin(angle)]
    
    # Ring 3: 3 points at radius r3, angles 0, 120, 240 (closer to center)
    r3 = 0.12
    for i in range(3):
        angle = 2 * np.pi * i / 3
        bary[6 + i] = [cx + r3 * np.cos(angle), cy + r3 * np.sin(angle)]
    
    # Center point
    bary[9] = [cx, cy]
    
    # Additional point offset
    bary[10] = [0.5, 0.25]
    
    return clamp_bary(bary)

def generate_boundary_init():
    """
    Generate points with some on/near the boundary of the triangle.
    """
    bary = np.zeros((N, 2))
    
    # 3 near vertices
    bary[0] = [0.97, 0.015]
    bary[1] = [0.015, 0.97]
    bary[2] = [0.015, 0.015]
    
    # 3 near edge midpoints
    bary[3] = [0.49, 0.01]
    bary[4] = [0.01, 0.49]
    bary[5] = [0.49, 0.49]
    
    # 3 inner ring
    cx, cy = 1/3, 1/3
    r = 0.2
    for i in range(3):
        angle = 2 * np.pi * i / 3 + np.pi / 6
        bary[6 + i] = [cx + r * np.cos(angle), cy + r * np.sin(angle)]
    
    # Center
    bary[9] = [cx, cy]
    
    # Additional
    bary[10] = [0.45, 0.35]
    
    return clamp_bary(bary)

def generate_hexagonal_init():
    """
    Generate a hexagonal lattice-like initial configuration.
    """
    best_bary = None
    best_val = -np.inf
    
    # Try a range of scales to find a good starting point
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
    """Generate random valid barycentric coordinates."""
    rng = np.random.default_rng(seed)
    u = rng.random(N)
    v = rng.random(N) * (1 - u)
    return np.column_stack([u, v])

def generate_perturbed_init(base_bary, sigma=0.05, seed=None):
    """Generate a perturbed version of a base configuration."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, sigma, base_bary.shape)
    return clamp_bary(base_bary + noise)

def refine_local(bary_init, time_limit=5.0):
    """
    Refine a configuration using multiple optimization stages including epigraph.
    """
    start_time = time.time()
    bounds = [(0, 1)] * (N * 2)
    
    bary_current = bary_init.copy()
    current_val = -hard_min_obj(bary_current.flatten())
    
    # Stage 1: Graduated soft-min minimization (beta 20 -> 50 -> 200)
    for beta in [20.0, 50.0, 200.0]:
        if time.time() - start_time > time_limit * 0.25:
            break
        try:
            result1 = minimize(
                soft_min_obj,
                bary_current.flatten(),
                args=(beta,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 150, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            new_val = -hard_min_obj(clamp_bary(result1.x.reshape(N, 2)).flatten())
            if new_val > current_val:
                bary_current = clamp_bary(result1.x.reshape(N, 2))
                current_val = new_val
        except Exception:
            pass
    
    if time.time() - start_time > time_limit * 0.35:
        return bary_current
    
    # Stage 2: Active-constraint coordinate refinement
    if time.time() - start_time < time_limit * 0.50:
        try:
            areas = calc_areas_bary(bary_current)
            min_area = np.min(areas)
            active = np.where(areas <= min_area + 1e-5)[0]
            active_pts = set()
            for tri in active:
                active_pts.add(tri_idx[tri, 0])
                active_pts.add(tri_idx[tri, 1])
                active_pts.add(tri_idx[tri, 2])
            
            for pt_idx in sorted(active_pts):
                if time.time() - start_time > time_limit * 0.50:
                    break
                for step in [0.01, 0.02, 0.03, 0.005]:
                    for du in [-step, 0, step]:
                        for dv in [-step, 0, step]:
                            if du == 0 and dv == 0:
                                continue
                            candidate = bary_current.copy()
                            candidate[pt_idx, 0] += du
                            candidate[pt_idx, 1] += dv
                            candidate = clamp_bary(candidate)
                            val = -hard_min_obj(candidate.flatten())
                            if val > current_val:
                                bary_current = candidate
                                current_val = val
        except Exception:
            pass
    
    if time.time() - start_time > time_limit * 0.55:
        return bary_current
    
    # Stage 3: Epigraph formulation with SLSQP
    if time.time() - start_time < time_limit * 0.75:
        try:
            areas_init = calc_areas_bary(bary_current)
            t_init = np.min(areas_init) * 0.999
            x_init = np.concatenate([bary_current.flatten(), [t_init]])
            
            bounds_epigraph = bounds + [(0, 1)]
            
            result_ep = minimize(
                epigraph_obj,
                x_init,
                method='SLSQP',
                bounds=bounds_epigraph,
                options={'maxiter': 400, 'ftol': 1e-14}
            )
            
            if result_ep.success or result_ep.fun < epigraph_obj(x_init):
                new_bary = clamp_bary(result_ep.x[:N*2].reshape(N, 2))
                new_val = -hard_min_obj(new_bary.flatten())
                if new_val > current_val:
                    bary_current = new_bary
                    current_val = new_val
        except Exception:
            pass
    
    if time.time() - start_time > time_limit * 0.80:
        return bary_current
    
    # Stage 4: Hard min refinement with Nelder-Mead
    try:
        result2 = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 1500, 'xatol': 1e-12, 'fatol': 1e-16, 'adaptive': True}
        )
        new_bary = clamp_bary(result2.x.reshape(N, 2))
        new_val = -hard_min_obj(new_bary.flatten())
        if new_val > current_val:
            bary_current = new_bary
            current_val = new_val
    except Exception:
        pass
        
    if time.time() - start_time > time_limit * 0.90:
        return bary_current
    
    # Stage 5: SLSQP polish
    try:
        result3 = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 400, 'ftol': 1e-15}
        )
        new_bary = clamp_bary(result3.x.reshape(N, 2))
        new_val = -hard_min_obj(new_bary.flatten())
        if new_val > current_val:
            bary_current = new_bary
            current_val = new_val
    except Exception:
        pass
        
    return bary_current

def coordinated_ascent(bary_init, max_iter=50, time_limit=3.0):
    """
    Coordinated ascent targeting active constraints.
    """
    start_time = time.time()
    bary = bary_init.copy()
    current_val = -hard_min_obj(bary.flatten())
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        areas = calc_areas_bary(bary)
        min_area = np.min(areas)
        active = np.where(areas <= min_area + 1e-6)[0]
        
        improved = False
        for pt_idx in range(N):
            point_in_active = False
            for tri in active:
                if pt_idx in tri_idx[tri]:
                    point_in_active = True
                    break
            
            if not point_in_active:
                continue
            
            best_pert = None
            best_pert_val = current_val
            
            for trial in range(20):
                rng = np.random.default_rng(it * 100 + pt_idx * 10 + trial)
                direction = rng.normal(0, 1, (N, 2))
                for step in [0.005, 0.01, 0.02]:
                    candidate = bary + step * direction
                    candidate = clamp_bary(candidate)
                    val = -hard_min_obj(candidate.flatten())
                    if val > best_pert_val:
                        best_pert_val = val
                        best_pert = candidate.copy()
            
            if best_pert is not None and best_pert_val > current_val + 1e-10:
                bary = best_pert
                current_val = best_pert_val
                improved = True
        
        if not improved:
            break
    
    return bary, current_val

def high_precision_polish(bary_init, time_limit=10.0):
    """
    High-precision polish pass with very tight tolerances and multiple methods.
    """
    start_time = time.time()
    bary_current = bary_init.copy()
    current_val = -hard_min_obj(bary_current.flatten())
    bounds = [(0, 1)] * (N * 2)
    
    # Very tight soft-min with high beta
    for beta in [200.0, 500.0, 1000.0]:
        if time.time() - start_time > time_limit * 0.4:
            break
        try:
            result = minimize(
                soft_min_obj,
                bary_current.flatten(),
                args=(beta,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-16, 'gtol': 1e-14}
            )
            new_bary = clamp_bary(result.x.reshape(N, 2))
            new_val = -hard_min_obj(new_bary.flatten())
            if new_val > current_val:
                bary_current = new_bary
                current_val = new_val
        except Exception:
            pass
    
    if time.time() - start_time > time_limit * 0.5:
        return bary_current, current_val
    
    # Epigraph with very tight tolerance
    try:
        areas_init = calc_areas_bary(bary_current)
        t_init = np.min(areas_init) * 0.9999
        x_init = np.concatenate([bary_current.flatten(), [t_init]])
        
        bounds_epigraph = bounds + [(0, 1)]
        
        result_ep = minimize(
            epigraph_obj,
            x_init,
            method='SLSQP',
            bounds=bounds_epigraph,
            options={'maxiter': 800, 'ftol': 1e-16}
        )
        
        if result_ep.success or result_ep.fun < epigraph_obj(x_init):
            new_bary = clamp_bary(result_ep.x[:N*2].reshape(N, 2))
            new_val = -hard_min_obj(new_bary.flatten())
            if new_val > current_val:
                bary_current = new_bary
                current_val = new_val
    except Exception:
        pass
    
    if time.time() - start_time > time_limit * 0.7:
        return bary_current, current_val
    
    # SLSQP with extreme precision
    try:
        result_slsqp = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 800, 'ftol': 1e-16}
        )
        new_bary = clamp_bary(result_slsqp.x.reshape(N, 2))
        new_val = -hard_min_obj(new_bary.flatten())
        if new_val > current_val:
            bary_current = new_bary
            current_val = new_val
    except Exception:
        pass
    
    if time.time() - start_time > time_limit * 0.85:
        return bary_current, current_val
    
    # Nelder-Mead with extreme precision
    try:
        result_nm = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-18, 'adaptive': True}
        )
        new_bary = clamp_bary(result_nm.x.reshape(N, 2))
        new_val = -hard_min_obj(new_bary.flatten())
        if new_val > current_val:
            bary_current = new_bary
            current_val = new_val
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
        val = -hard_min_obj(bary.flatten())
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
            return True
        return False

    # 1. Try multiple structured inits
    inits = [
        generate_hexagonal_init(),
        generate_symmetric_init(),
        generate_radial_init(),
        generate_boundary_init()
    ]
    
    for init in inits:
        if time.time() > deadline - 50:
            break
        refined = refine_local(init, time_limit=8.0)
        update_best(refined)
        # Also try coordinated ascent on the refined result
        if best_bary is not None and time.time() < deadline - 40:
            ca_bary, ca_val = coordinated_ascent(best_bary, max_iter=40, time_limit=3.0)
            if ca_val > best_val:
                best_bary = ca_bary
                best_val = ca_val
    
    # 2. Global Search with Differential Evolution (increased population and iterations)
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
                maxiter=50,
                popsize=18,
                tol=1e-8,
                polish=False
            )
            
            de_bary = clamp_bary(result_de.x.reshape(N, 2))
            refined_de = refine_local(de_bary, time_limit=8.0)
            update_best(refined_de)
            
            # Coordinated ascent on DE result
            if best_bary is not None and time.time() < deadline - 30:
                ca_bary, ca_val = coordinated_ascent(best_bary, max_iter=40, time_limit=3.0)
                if ca_val > best_val:
                    best_bary = ca_bary
                    best_val = ca_val
        except Exception:
            pass
    
    # 3. Multi-start: random + perturbed restarts with local refinement
    restart_count = 0
    while time.time() < deadline - 25:
        restart_count += 1
        if best_bary is not None and restart_count % 3 == 0:
            # Perturbation of best solution
            init = generate_perturbed_init(best_bary, sigma=0.03, seed=2000 + restart_count)
        elif best_bary is not None and restart_count % 3 == 1:
            # Radial-style perturbation
            init = generate_perturbed_init(best_bary, sigma=0.06, seed=3000 + restart_count)
        else:
            # Random restart
            init = generate_random_init(seed=4000 + restart_count)
        
        remaining = deadline - 25 - time.time()
        if remaining < 5:
            break
            
        t_limit = min(5.0, remaining)
        refined = refine_local(init, time_limit=t_limit)
        update_best(refined)
        
        # Quick coordinated ascent
        if best_bary is not None and time.time() < deadline - 15:
            ca_bary, ca_val = coordinated_ascent(best_bary, max_iter=30, time_limit=2.0)
            if ca_val > best_val:
                best_bary = ca_bary
                best_val = ca_val

    # Final high-precision polish on the best candidate
    if best_bary is not None and time.time() < deadline - 12:
        # Progressive soft-min polish with very high beta
        for beta in [300.0, 500.0, 800.0, 1200.0]:
            if time.time() > deadline - 8:
                break
            try:
                bounds = [(0, 1)] * (N * 2)
                result_soft = minimize(
                    soft_min_obj,
                    best_bary.flatten(),
                    args=(beta,),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 300, 'ftol': 1e-16, 'gtol': 1e-14}
                )
                new_bary = clamp_bary(result_soft.x.reshape(N, 2))
                new_val = -hard_min_obj(new_bary.flatten())
                if new_val > best_val:
                    best_bary = new_bary
                    best_val = new_val
            except Exception:
                pass
        
        # High-precision epigraph pass
        if best_bary is not None and time.time() < deadline - 8:
            try:
                areas_final = calc_areas_bary(best_bary)
                t_init = np.min(areas_final) * 0.99995
                x_init = np.concatenate([best_bary.flatten(), [t_init]])
                
                bounds_epigraph = [(0, 1)] * (N * 2) + [(0, 1)]
                
                result_ep = minimize(
                    epigraph_obj,
                    x_init,
                    method='SLSQP',
                    bounds=bounds_epigraph,
                    options={'maxiter': 1000, 'ftol': 1e-16}
                )
                
                if result_ep.success or result_ep.fun < epigraph_obj(x_init):
                    new_bary = clamp_bary(result_ep.x[:N*2].reshape(N, 2))
                    new_val = -hard_min_obj(new_bary.flatten())
                    if new_val > best_val:
                        best_bary = new_bary
                        best_val = new_val
            except Exception:
                pass
        
        # SLSQP high-precision
        if best_bary is not None and time.time() < deadline - 6:
            try:
                bounds = [(0, 1)] * (N * 2)
                result_slsqp = minimize(
                    hard_min_obj,
                    best_bary.flatten(),
                    method='SLSQP',
                    bounds=bounds,
                    options={'maxiter': 1000, 'ftol': 1e-16}
                )
                new_bary = clamp_bary(result_slsqp.x.reshape(N, 2))
                new_val = -hard_min_obj(new_bary.flatten())
                if new_val > best_val:
                    best_bary = new_bary
                    best_val = new_val
            except Exception:
                pass
        
        # Nelder-Mead final
        if best_bary is not None and time.time() < deadline - 4:
            try:
                result_nm = minimize(
                    hard_min_obj,
                    best_bary.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-15, 'fatol': 1e-18, 'adaptive': True}
                )
                new_bary = clamp_bary(result_nm.x.reshape(N, 2))
                new_val = -hard_min_obj(new_bary.flatten())
                if new_val > best_val:
                    best_bary = new_bary
                    best_val = new_val
            except Exception:
                pass
        
        # Final coordinated ascent
        if best_bary is not None and time.time() < deadline - 2:
            ca_bary, ca_val = coordinated_ascent(best_bary, max_iter=60, time_limit=2.0)
            if ca_val > best_val:
                best_bary = ca_bary
                best_val = ca_val

    # Convert to cartesian
    cart = bary_to_cart(best_bary)
    
    return cart
