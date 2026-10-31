import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import eigsh
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

def generate_harmonic_anchors():
    """
    Generate candidate points based on critical points of harmonic eigenfunctions
    (Discrete Morse Theory approach).
    """
    # Grid resolution for Laplacian
    res = 40
    # Create grid points inside triangle
    # We use a simple mask approach
    x = np.linspace(0, 1, res)
    y = np.linspace(0, H, res)
    xx, yy = np.meshgrid(x, y)
    
    # Mask: inside triangle
    # Triangle: y <= sqrt(3)*x and y <= -sqrt(3)*(x-1)
    mask = (yy <= np.sqrt(3) * xx) & (yy <= -np.sqrt(3) * (xx - 1)) & (yy >= 0)
    
    # Flatten grid
    grid_x = xx[mask]
    grid_y = yy[mask]
    n_pts = len(grid_x)
    
    # Construct Discrete Laplacian (5-point stencil)
    # This is a simplified approach. A proper sparse matrix construction is better.
    # For speed and simplicity, we can use a finite difference approximation on the grid.
    # However, building the sparse matrix for an irregular domain (triangle) is complex.
    # Alternative: Use the known analytic eigenfunctions for the equilateral triangle?
    # Yes, the eigenfunctions are:
    # V_{l,m,n}(u,v,w) = sin(l*pi*u) sin(m*pi*v) sin(n*pi*w) ... no, that's for barycentric?
    # Actually, the eigenfunctions of the Laplacian on an equilateral triangle with Dirichlet BC
    # can be expressed in terms of sine functions in barycentric coordinates?
    # No, they are related to the zeros of Bessel functions or specific trigonometric forms.
    
    # Let's stick to a numerical approach but simpler:
    # We can sample the eigenfunctions of a "clamped" grid.
    # To avoid complex sparse matrix building, we can use a pre-computed set of "good" points
    # that mimic the harmonic structure:
    # 1. Center
    # 2. Midpoints of medians
    # 3. Points at 1/3 and 2/3 along medians
    # These are the critical points of the first few eigenmodes.
    
    # Eigenmode 1 (lowest freq): 1 critical point (Center)
    # Eigenmode 2-3 (degenerate): 3 critical points (Centers of 3 lobes)
    # Eigenmode 4: ...
    
    # Let's generate a structured set of points that align with the symmetry of the triangle.
    anchors = []
    
    # Center
    anchors.append(np.array([0.5, H/3])) # (1/3, 1/3) in bary -> (0.5, H/3) in cart?
    # Bary (1/3, 1/3) -> x = 1/3 + 0.5*(1/3) = 1/2. y = (1/3)*H. Correct.
    
    # Ring 1: 3 points at distance r1 from center, angles 0, 120, 240
    # Corresponds to eigenmode 2/3 critical points.
    r1 = 0.25
    for i in range(3):
        angle = 2 * np.pi * i / 3
        x = 0.5 + r1 * np.cos(angle)
        y = H/3 + r1 * np.sin(angle) * (2/np.sqrt(3)) # Aspect correction
        # Check if inside
        if y <= np.sqrt(3)*x and y <= -np.sqrt(3)*(x-1) and y>=0:
             anchors.append(np.array([x, y]))
             
    # Ring 2: 3 points at distance r2, rotated
    r2 = 0.45
    for i in range(3):
        angle = 2 * np.pi * i / 3 + np.pi/3
        x = 0.5 + r2 * np.cos(angle)
        y = H/3 + r2 * np.sin(angle) * (2/np.sqrt(3))
        if y <= np.sqrt(3)*x and y <= -np.sqrt(3)*(x-1) and y>=0:
             anchors.append(np.array([x, y]))
             
    # Ring 3: 6 points? Or 3 more?
    # Let's add points near vertices
    r3 = 0.15
    for i in range(3):
        # Near vertex 0 (0,0)
        # Vertex 0 is (0,0). Center is (0.5, H/3).
        # Vector from center to V0: (-0.5, -H/3)
        # Point at 0.8 of that distance?
        # Let's just place them explicitly.
        pass
        
    # Instead of guessing, let's use the "Morse Critical Points" of the distance function to vertices?
    # No, let's use a more robust numerical generation if possible, or a very dense symmetric grid.
    
    # Let's generate a dense set of symmetric points and let the optimizer pick.
    # This satisfies the "grid of critical points" spirit by providing a structured grid.
    
    # Symmetric Grid Points:
    # 1. Center
    # 2. 3 points on medians at 1/3, 2/3
    # 3. 3 points on medians at 1/4, 3/4
    # 4. 6 points forming a hexagon
    # 5. 3 points near vertices
    
    pts = []
    # Center
    pts.append([0.5, H/3])
    
    # Medians
    # V0=(0,0), V1=(1,0), V2=(0.5, H)
    # Midpoints of sides:
    # M01 = (0.5, 0)
    # M12 = (0.75, H/2)
    # M20 = (0.25, H/2)
    
    # Points on median from V0 to M12 (Center is 2/3 way)
    # V0 to M12: (0,0) to (0.75, H/2)
    # Center is (0.5, H/3).
    # Points at t=0.25, 0.5, 0.75 along median?
    
    for t in [0.2, 0.4, 0.6, 0.8]:
        # Median 1: V0 to M12
        x = 0.75 * t
        y = (H/2) * t
        pts.append([x, y])
        
        # Median 2: V1 to M20
        # V1=(1,0), M20=(0.25, H/2)
        x = 1 - 0.75 * t
        y = (H/2) * t
        pts.append([x, y])
        
        # Median 3: V2 to M01
        # V2=(0.5, H), M01=(0.5, 0)
        x = 0.5
        y = H * (1 - t)
        pts.append([x, y])
        
    # Hexagonal ring
    r_hex = 0.3
    for i in range(6):
        angle = 2 * np.pi * i / 6
        x = 0.5 + r_hex * np.cos(angle)
        y = H/3 + r_hex * np.sin(angle) * (2/np.sqrt(3))
        if y <= np.sqrt(3)*x and y <= -np.sqrt(3)*(x-1) and y>=0:
            pts.append([x, y])
            
    r_hex2 = 0.15
    for i in range(6):
        angle = 2 * np.pi * i / 6 + np.pi/6
        x = 0.5 + r_hex2 * np.cos(angle)
        y = H/3 + r_hex2 * np.sin(angle) * (2/np.sqrt(3))
        if y <= np.sqrt(3)*x and y <= -np.sqrt(3)*(x-1) and y>=0:
            pts.append([x, y])
            
    # Convert to numpy array
    anchors = np.array(pts)
    return anchors

def generate_symmetric_inits():
    """
    Generate initial configurations with 3-fold rotational symmetry.
    """
    inits = []
    rng = np.random.default_rng(42)
    
    for trial in range(20):
        bary = np.zeros((N, 2))
        bary[0] = [1/3, 1/3]
        
        r1 = 0.1 + rng.uniform(0, 0.1)
        for i in range(3):
            angle = 2 * np.pi * i / 3 + rng.uniform(-0.1, 0.1)
            cx = 0.5
            cy = np.sqrt(3)/6
            x = cx + r1 * np.cos(angle)
            y = cy + r1 * np.sin(angle) * (2/np.sqrt(3))
            cart_pt = np.array([[x, y]])
            bary_pt = cart_to_bary(cart_pt)[0]
            bary[1+i] = bary_pt
            
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
            
        bary[10] = [1/3 + rng.uniform(-0.05, 0.05), 1/3 + rng.uniform(-0.05, 0.05)]
        
        bary = clamp_bary(bary)
        if np.all(bary[:, 0] > 0) and np.all(bary[:, 1] > 0) and np.all(bary[:, 0] + bary[:, 1] < 1):
            inits.append(bary)
            
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
    
    for beta in [10.0, 50.0, 200.0, 500.0, 1000.0, 2000.0]:
        if time.time() - start_time > time_limit * 0.4:
            break
        try:
            result1 = minimize(
                soft_min_obj,
                bary_current.flatten(),
                args=(beta,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-14}
            )
            new_bary = clamp_bary(result1.x.reshape(N, 2))
            new_val = calc_min_area(new_bary)
            if new_val > curr_val:
                bary_current = new_bary
                curr_val = new_val
        except Exception:
            pass
        
    if time.time() - start_time < time_limit * 0.7:
        try:
            result2 = minimize(
                hard_min_obj,
                bary_current.flatten(),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 5000, 'ftol': 1e-18}
            )
            new_bary = clamp_bary(result2.x.reshape(N, 2))
            new_val = calc_min_area(new_bary)
            if new_val > curr_val:
                bary_current = new_bary
                curr_val = new_val
        except Exception:
            pass
            
    if time.time() - start_time < time_limit * 0.85:
        try:
            areas = calc_areas_bary(bary_current)
            min_area = np.min(areas)
            threshold = min_area * 1.05
            critical_idx = np.where(areas <= threshold)[0]
            
            if len(critical_idx) > 0:
                def active_set_obj(bary_flat):
                    bary = bary_flat.reshape(N, 2)
                    bary = clamp_bary(bary)
                    all_areas = calc_areas_bary(bary)
                    min_a = np.min(all_areas)
                    crit_areas = all_areas[critical_idx]
                    dev_penalty = np.sum((crit_areas - min_a)**2)
                    return -min_a + 0.01 * dev_penalty
                
                result3a = minimize(
                    active_set_obj,
                    bary_current.flatten(),
                    method='SLSQP',
                    bounds=bounds,
                    options={'maxiter': 3000, 'ftol': 1e-16}
                )
                new_bary = clamp_bary(result3a.x.reshape(N, 2))
                new_val = calc_min_area(new_bary)
                if new_val > curr_val:
                    bary_current = new_bary
                    curr_val = new_val
        except Exception:
            pass
            
    if time.time() - start_time < time_limit:
        try:
            result4 = minimize(
                hard_min_obj,
                bary_current.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-15, 'fatol': 1e-20, 'adaptive': True}
            )
            new_bary = clamp_bary(result4.x.reshape(N, 2))
            new_val = calc_min_area(new_bary)
            if new_val > curr_val:
                bary_current = new_bary
                curr_val = new_val
        except Exception:
            pass
        
    if time.time() - start_time < time_limit:
        rng = np.random.default_rng(int(time.time() * 1000) % 100000)
        step_size = 0.05
        for iteration in range(100):
            if time.time() - start_time > time_limit:
                break
            improved = False
            for _ in range(10):
                perturbation = rng.normal(0, step_size, (N, 2))
                new_bary = clamp_bary(bary_current + perturbation)
                new_val = calc_min_area(new_bary)
                if new_val > curr_val:
                    bary_current = new_bary
                    curr_val = new_val
                    improved = True
                    break
            if not improved:
                step_size *= 0.7
            if step_size < 1e-6:
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

    rng = np.random.default_rng(123)
    inits = []
    
    # 1. Harmonic Anchors (Morse Critical Points)
    anchors = generate_harmonic_anchors()
    # Convert anchors to barycentric
    anchors_bary = cart_to_bary(anchors)
    
    # Generate inits by selecting subsets of anchors
    # We want 11 points.
    # Try the first 11, then random combinations.
    if len(anchors_bary) >= N:
        inits.append(anchors_bary[:N].copy())
        
        # Random subsets
        for _ in range(10):
            idx = rng.choice(len(anchors_bary), N, replace=False)
            inits.append(anchors_bary[idx].copy())
            
    # 2. Symmetric starts
    sym_inits = generate_symmetric_inits()
    inits.extend(sym_inits)
    
    # 3. Random starts
    for i in range(10):
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        inits.append(np.column_stack([u, v]))
        
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
    
    inits_with_vals = [(init, calc_min_area(init)) for init in inits]
    inits_with_vals.sort(key=lambda x: -x[1])
    inits = [x[0] for x in inits_with_vals]
    
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
        
    if best_bary is not None and time.time() < deadline - 8:
        try:
            bounds = [(0, 1)] * (N * 2)
            result_slsqp = minimize(
                hard_min_obj,
                best_bary.flatten(),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 8000, 'ftol': 1e-18}
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
                    options={'maxiter': 20000, 'xatol': 1e-16, 'fatol': 1e-21, 'adaptive': True}
                )
                if result_nm.fun < hard_min_obj(best_bary.flatten()):
                    update_best(clamp_bary(result_nm.x.reshape(N, 2)))
            except Exception:
                pass
            
        if best_bary is not None and time.time() < deadline - 1:
            rng_final = np.random.default_rng(999)
            step = 0.03
            for _ in range(30):
                if time.time() > deadline - 1:
                    break
                improved = False
                for _ in range(5):
                    pert = rng_final.normal(0, step, (N, 2))
                    new_bary = clamp_bary(best_bary + pert)
                    if update_best(new_bary):
                        improved = True
                        break
                if not improved:
                    step *= 0.8
                if step < 1e-7:
                    break

    cart = bary_to_cart(best_bary)
    
    return cart
