import numpy as np
from scipy.optimize import differential_evolution, minimize, linprog
from itertools import combinations
import time

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

def get_active_triangles(bary, tolerance=1e-8):
    """Find indices of triangles with area close to the minimum (active constraints)."""
    areas = calc_areas_bary(bary)
    min_area = np.min(areas)
    active = np.where(areas <= min_area + tolerance)[0]
    return active, min_area

def active_set_coordinate_ascent(bary_init, time_limit=10.0, max_sweeps=50):
    """
    Structured coordinate ascent targeting active (minimum-area) triangles.
    For each sweep: identify active triangles, then optimize each point 
    involved in those triangles using a constrained 2D optimizer.
    """
    start_time = time.time()
    bary = bary_init.copy()
    best_val = -hard_min_obj(bary.flatten())
    
    # Precompute which triangles include each point
    point_tri_list = [[] for _ in range(N)]
    for t_idx, tri in enumerate(tri_idx):
        for p in tri:
            point_tri_list[p].append(t_idx)
    
    for sweep in range(max_sweeps):
        if time.time() - start_time > time_limit:
            break
            
        # Identify active triangles
        active, min_area = get_active_triangles(bary, tolerance=min_area * 0.1 + 1e-10)
        
        # Determine which points are in active triangles
        active_points = set()
        for t in active:
            for p in tri_idx[t]:
                active_points.add(p)
        
        # If no active points found, optimize all
        if not active_points:
            active_points = set(range(N))
        
        # Optimize active points (shuffled order for diversity)
        pts_to_opt = list(active_points)
        np.random.shuffle(pts_to_opt)
        
        for i in pts_to_opt:
            if time.time() - start_time > time_limit:
                break
                
            orig_pos = bary[i].copy()
            
            # Objective: maximize min area considering triangles involving point i
            def obj_active(x):
                bary[i] = x
                bary_clamped = clamp_bary(bary)
                areas = calc_areas_bary(bary_clamped)
                # Focus on triangles involving point i for speed
                mask_i = np.array([i in tri for tri in tri_idx])
                if np.any(mask_i):
                    min_with_i = np.min(areas[mask_i])
                else:
                    min_with_i = np.min(areas)
                # Also track global min
                global_min = np.min(areas)
                return -min(global_min, min_with_i)
            
            try:
                result = minimize(
                    obj_active,
                    orig_pos,
                    method='Nelder-Mead',
                    options={'maxiter': 150, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
                )
                new_pos = result.x
                bary[i] = new_pos
                bary = clamp_bary(bary)
                
                new_val = -hard_min_obj(bary.flatten())
                if new_val > best_val:
                    best_val = new_val
            except Exception:
                bary[i] = orig_pos
        
        # Every 5 sweeps, do a full pass over all points
        if (sweep + 1) % 5 == 0:
            for i in range(N):
                if time.time() - start_time > time_limit:
                    break
                orig_pos = bary[i].copy()
                
                def obj_full(x):
                    bary[i] = x
                    bary_clamped = clamp_bary(bary)
                    areas = calc_areas_bary(bary_clamped)
                    return -np.min(areas)
                
                try:
                    result = minimize(
                        obj_full,
                        orig_pos,
                        method='Nelder-Mead',
                        options={'maxiter': 100, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
                    )
                    bary[i] = result.x
                    bary = clamp_bary(bary)
                    new_val = -hard_min_obj(bary.flatten())
                    if new_val > best_val:
                        best_val = new_val
                except Exception:
                    bary[i] = orig_pos
    
    return bary

def epigraph_obj(params, beta=30.0):
    """
    Epigraph formulation: minimize t subject to all areas >= t.
    Uses log-sum-exp penalty for smoothness.
    """
    bary = params[:N*2].reshape(N, 2)
    t = params[N*2]
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    
    # Penalty for areas below t
    violations = np.maximum(0, t - areas)
    penalty = np.sum(violations ** 2)
    
    # Smooth approximation: -t + penalty
    # Use soft version: t - (1/beta) * log(sum(exp(-beta*(areas - t))))
    shifted = areas - t
    min_shifted = np.min(shifted)
    if min_shifted < -50:
        return -t + penalty
    log_sum = min_shifted + np.log(np.sum(np.exp(shifted - min_shifted)))
    soft_min = t - log_sum / beta
    
    return soft_min + 0.5 * penalty

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

def generate_structured_init(seed=42):
    """
    Generate structured initializations based on known Heilbronn configurations.
    Uses vertices, edge midpoints, and centroid-like points.
    """
    rng = np.random.default_rng(seed)
    best_bary = None
    best_val = -np.inf
    
    # Configuration 1: 3 vertices + edge points + interior
    configs = [
        # Vertices
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
    ]
    
    # Add points along edges and interior
    for config in configs:
        bary = np.array(config, dtype=float)
        
        # Add edge midpoints
        edge_points = [
            [0.5, 0.0],  # midpoint of edge 0-1
            [0.0, 0.5],  # midpoint of edge 0-2
            [0.5, 0.5],  # midpoint of edge 1-2
        ]
        bary = np.vstack([bary, edge_points])
        
        # Add interior points (centroid, and others)
        centroid = [1/3, 1/3]
        bary = np.vstack([bary, [centroid]])
        
        # Need 11 points, we have 7, add 4 more
        while len(bary) < N:
            # Add points in a structured pattern
            r = rng.uniform(0.1, 0.8)
            theta = rng.uniform(0, 2 * np.pi)
            u = 0.5 + r * np.cos(theta) * 0.5
            v = 0.5 + r * np.sin(theta) * 0.5
            if u >= 0 and v >= 0 and u + v <= 1:
                bary = np.vstack([bary, [u, v]])
            else:
                # Retry with smaller radius
                r = rng.uniform(0.05, 0.4)
                u = 1/3 + r * np.cos(theta)
                v = 1/3 + r * np.sin(theta)
                if u >= 0 and v >= 0 and u + v <= 1:
                    bary = np.vstack([bary, [u, v]])
        
        bary = bary[:N]
        bary = clamp_bary(bary)
        val = -hard_min_obj(bary.flatten())
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
    
    # Configuration 2: Grid-based
    for grid_size in [3, 4]:
        pts = []
        step = 1.0 / (grid_size - 1)
        for i in range(grid_size):
            for j in range(grid_size - i):
                u = i * step
                v = j * step
                if u + v <= 1 + 1e-9:
                    pts.append([u, v])
        
        if len(pts) >= N:
            # Select N points from grid
            for trial in range(10):
                if len(pts) == N:
                    bary = np.array(pts)
                else:
                    idx = rng.choice(len(pts), N, replace=False)
                    bary = np.array(pts)[idx]
                bary = clamp_bary(bary)
                val = -hard_min_obj(bary.flatten())
                if val > best_val:
                    best_val = val
                    best_bary = bary.copy()
    
    # Configuration 3: Perturbed regular hexagon + center
    for trial in range(20):
        bary = np.zeros((N, 2))
        # Center
        bary[0] = [1/3, 1/3]
        # 6 points around center
        for k in range(6):
            angle = k * np.pi / 3
            r = 0.35 + rng.uniform(-0.05, 0.05)
            u = 1/3 + r * np.cos(angle)
            v = 1/3 + r * np.sin(angle)
            if u >= 0 and v >= 0 and u + v <= 1:
                bary[k+1] = [u, v]
            else:
                bary[k+1] = [1/3 + 0.2 * np.cos(angle), 1/3 + 0.2 * np.sin(angle)]
        # 4 more points
        for k in range(4):
            r = rng.uniform(0.1, 0.4)
            angle = rng.uniform(0, 2*np.pi)
            u = 1/3 + r * np.cos(angle)
            v = 1/3 + r * np.sin(angle)
            if u >= 0 and v >= 0 and u + v <= 1:
                bary[k+7] = [u, v]
            else:
                bary[k+7] = [0.2, 0.2]
        
        bary = clamp_bary(bary)
        val = -hard_min_obj(bary.flatten())
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
    
    # Configuration 4: Known good Heilbronn-like layout
    # Based on known optimal/near-optimal configurations
    known_layouts = [
        # Layout with points near vertices and edges
        [
            [0.01, 0.01], [0.99, 0.01], [0.01, 0.99],
            [0.5, 0.02], [0.02, 0.5], [0.48, 0.48],
            [0.33, 0.33], [0.67, 0.1], [0.1, 0.67],
            [0.8, 0.15], [0.15, 0.8]
        ],
        # More spread out
        [
            [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
            [0.5, 0.0], [0.0, 0.5], [0.5, 0.5],
            [0.25, 0.25], [0.75, 0.25], [0.25, 0.75],
            [0.5, 0.25], [0.25, 0.5]
        ],
    ]
    
    for layout in known_layouts:
        bary = np.array(layout, dtype=float)
        bary = clamp_bary(bary)
        val = -hard_min_obj(bary.flatten())
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
    
    if best_bary is None:
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        best_bary = np.column_stack([u, v])
    
    return best_bary

def refine_local(bary_init, time_limit=5.0):
    """
    Multi-stage refinement using active-set coordinate ascent and epigraph optimization.
    """
    start_time = time.time()
    bounds = [(0, 1)] * (N * 2)
    
    bary_current = bary_init.copy()
    best_val = -hard_min_obj(bary_current.flatten())
    
    # Stage 1: Active-set coordinate ascent (targeted at minimum-area triangles)
    t1 = min(2.5, time_limit * 0.45)
    if time.time() - start_time < t1:
        bary_active = active_set_coordinate_ascent(bary_current, time_limit=t1, max_sweeps=40)
        val_active = -hard_min_obj(bary_active.flatten())
        if val_active > best_val:
            best_val = val_active
            bary_current = bary_active
    
    if time.time() - start_time > time_limit * 0.5:
        return bary_current
    
    # Stage 2: Epigraph formulation with L-BFGS-B
    try:
        # Initialize t to current min area
        areas_init = calc_areas_bary(bary_current)
        t_init = np.min(areas_init)
        
        params_init = np.concatenate([bary_current.flatten(), [t_init]])
        bounds_ep = [(0, 1)] * (N * 2) + [(0, 0.5)]
        
        result_ep = minimize(
            epigraph_obj,
            params_init,
            method='L-BFGS-B',
            bounds=bounds_ep,
            options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        
        if result_ep.success or result_ep.fun < epigraph_obj(params_init):
            bary_ep = clamp_bary(result_ep.x[:N*2].reshape(N, 2))
            val_ep = -hard_min_obj(bary_ep.flatten())
            if val_ep > best_val:
                best_val = val_ep
                bary_current = bary_ep
    except Exception:
        pass
    
    if time.time() - start_time > time_limit * 0.7:
        return bary_current
    
    # Stage 3: Another round of active-set ascent with tighter tolerance
    t3 = min(1.5, time_limit * 0.25)
    if time.time() - start_time < time_limit - 0.5:
        bary_active2 = active_set_coordinate_ascent(bary_current, time_limit=t3, max_sweeps=60)
        val_active2 = -hard_min_obj(bary_active2.flatten())
        if val_active2 > best_val:
            best_val = val_active2
            bary_current = bary_active2
    
    # Stage 4: Final SLSQP polish
    if time.time() - start_time < time_limit - 0.2:
        try:
            result_slsqp = minimize(
                hard_min_obj,
                bary_current.flatten(),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-14}
            )
            bary_slsqp = clamp_bary(result_slsqp.x.reshape(N, 2))
            val_slsqp = -hard_min_obj(bary_slsqp.flatten())
            if val_slsqp > best_val:
                best_val = val_slsqp
                bary_current = bary_slsqp
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

    # 1. Structured initialization with active-set refinement
    struct_init = generate_structured_init(seed=42)
    refined_struct = refine_local(struct_init, time_limit=10.0)
    update_best(refined_struct)
    
    # 2. Additional structured inits with different seeds
    rng = np.random.default_rng(123)
    for seed_trial in range(3):
        if time.time() > deadline - 80:
            break
        struct_init2 = generate_structured_init(seed=100 + seed_trial * 7)
        refined_struct2 = refine_local(struct_init2, time_limit=6.0)
        update_best(refined_struct2)
    
    # 3. Random initialization with refinement
    for seed_trial in range(4):
        if time.time() > deadline - 70:
            break
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        rand_init = np.column_stack([u, v])
        refined_rand = refine_local(rand_init, time_limit=5.0)
        update_best(refined_rand)
    
    # 4. Differential Evolution for global exploration
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
                maxiter=20,
                popsize=15,
                tol=1e-6,
                polish=False
            )
            
            de_bary = clamp_bary(result_de.x.reshape(N, 2))
            refined_de = refine_local(de_bary, time_limit=8.0)
            update_best(refined_de)
        except Exception:
            pass
    
    # 5. Basin hopping on best found
    if best_bary is not None and time.time() < deadline - 15:
        bh_time = min(25.0, deadline - start_time - 15.0)
        if bh_time > 5:
            # Simple basin hopping: perturb + refine
            current_bary = best_bary.copy()
            current_val = best_val
            rng_bh = np.random.default_rng(999)
            temperature = 0.03
            pert_scale = 0.08
            
            while time.time() - start_time < deadline - 5:
                # Refine current
                refined = refine_local(current_bary, time_limit=min(3.0, deadline - start_time - 5))
                refined_val = -hard_min_obj(refined.flatten())
                
                if refined_val > current_val:
                    current_bary = refined
                    current_val = refined_val
                    if current_val > best_val:
                        best_val = current_val
                        best_bary = refined.copy()
                
                # Perturb
                noise = rng_bh.normal(0, pert_scale, current_bary.shape)
                perturbed = clamp_bary(current_bary + noise)
                perturbed_val = -hard_min_obj(perturbed.flatten())
                
                # Metropolis
                delta = perturbed_val - current_val
                if delta > 0 or np.log(rng_bh.random()) < delta / temperature:
                    current_bary = perturbed
                    current_val = perturbed_val
                
                temperature *= 0.93
                pert_scale *= 0.97
                
                if temperature < 0.005:
                    break
    
    # 6. Final high-precision polish
    if best_bary is not None and time.time() < deadline - 3:
        final_time = min(5.0, deadline - start_time - 1.0)
        if final_time > 1:
            final_bary = active_set_coordinate_ascent(best_bary, time_limit=final_time, max_sweeps=100)
            update_best(final_bary)
        
        if best_bary is not None and time.time() < deadline - 1:
            try:
                bounds = [(0, 1)] * (N * 2)
                result_final = minimize(
                    hard_min_obj,
                    best_bary.flatten(),
                    method='SLSQP',
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-15}
                )
                update_best(clamp_bary(result_final.x.reshape(N, 2)))
            except Exception:
                pass

    # Convert to cartesian
    cart = bary_to_cart(best_bary)
    
    return cart
