# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _hex_vertices(cx, cy, side, angle_deg):
    """Compute 6 vertices of a regular hexagon centered at (cx,cy) with given side and angle."""
    a = np.radians(angle_deg)
    angles = a + np.arange(6) * np.pi / 3.0
    return np.column_stack([
        cx + side * np.cos(angles),
        cy + side * np.sin(angles)
    ])

def _min_outer_side(vertices_all):
    """
    Given array of all inner hexagon vertices (shape (72,2)), compute minimum outer hex side s
    (centered at origin, angle 0) that contains all vertices.
    """
    k = np.arange(6)
    angles = (k + 0.5) * np.pi / 3
    m = np.column_stack([np.cos(angles), np.sin(angles)])
    
    dots = vertices_all @ m.T
    max_dot = np.max(dots, axis=1)
    global_max = np.max(max_dot)
    
    if global_max < 0:
        global_max = 0.0
    
    s_min = (2.0 / np.sqrt(3.0)) * global_max
    return s_min

def _check_nonoverlap(centers, angles_deg, side=1.0, tol=1e-6):
    """
    Check if all pairs of hexagons are non-overlapping using SAT.
    Returns True if no overlap, False otherwise.
    """
    n = len(centers)
    vertices_list = []
    normals_list = []
    for i in range(n):
        verts = _hex_vertices(centers[i,0], centers[i,1], side, angles_deg[i])
        vertices_list.append(verts)
        normals = []
        for e in range(6):
            p1 = verts[e]
            p2 = verts[(e+1) % 6]
            edge = p2 - p1
            nx, ny = edge[1], -edge[0]
            norm = np.sqrt(nx*nx + ny*ny)
            if norm > 1e-12:
                normals.append(np.array([nx/norm, ny/norm]))
        normals_list.append(normals)
    
    for i in range(n):
        for j in range(i+1, n):
            v1 = vertices_list[i]
            v2 = vertices_list[j]
            overlap = True
            for hex_idx in range(2):
                verts = [v1, v2][hex_idx]
                normals = normals_list[hex_idx]
                for normal in normals:
                    proj1 = v1 @ normal
                    proj2 = v2 @ normal
                    min1, max1 = np.min(proj1), np.max(proj1)
                    min2, max2 = np.min(proj2), np.max(proj2)
                    if max1 + tol < min2 or max2 + tol < min1:
                        overlap = False
                        break
                if not overlap:
                    break
            if overlap:
                return False
    return True

def _compute_all_vertices(centers, angles_deg, side=1.0):
    """Compute all 72 vertices of 12 hexagons."""
    verts = []
    for i in range(len(centers)):
        verts.append(_hex_vertices(centers[i,0], centers[i,1], side, angles_deg[i]))
    return np.vstack(verts)

def _initial_honeycomb():
    """
    Create initial honeycomb-like arrangement of 12 hexagons.
    """
    d = np.sqrt(3.0)
    positions = []
    angles = []
    
    positions.append([0.0, 0.0])
    angles.append(0.0)
    
    for k in range(6):
        angle = k * np.pi / 3
        x = d * np.cos(angle)
        y = d * np.sin(angle)
        positions.append([x, y])
        angles.append(0.0)
    
    d2 = 2.0 * d
    for k in range(5):
        angle = (k * 72 + 30) * np.pi / 180
        x = d2 * np.cos(angle)
        y = d2 * np.sin(angle)
        positions.append([x, y])
        angles.append(0.0)
    
    positions = np.array(positions)
    angles = np.array(angles)
    return positions, angles

def _penetration_depth(centers, angles_deg, side=1.0):
    """
    Compute total squared penetration depth between all pairs using SAT.
    Returns (total_pen, n_overlaps) where total_pen is sum of squared penetrations.
    """
    n = len(centers)
    vertices_list = []
    normals_list = []
    for i in range(n):
        verts = _hex_vertices(centers[i,0], centers[i,1], side, angles_deg[i])
        vertices_list.append(verts)
        normals = []
        for e in range(6):
            p1 = verts[e]
            p2 = verts[(e+1) % 6]
            edge = p2 - p1
            nx, ny = edge[1], -edge[0]
            norm = np.sqrt(nx*nx + ny*ny)
            if norm > 1e-12:
                normals.append(np.array([nx/norm, ny/norm]))
        normals_list.append(normals)
    
    total_pen = 0.0
    n_overlaps = 0
    for i in range(n):
        for j in range(i+1, n):
            v1 = vertices_list[i]
            v2 = vertices_list[j]
            max_pen = 0.0
            for hex_idx in range(2):
                verts = [v1, v2][hex_idx]
                normals = normals_list[hex_idx]
                for normal in normals:
                    proj1 = v1 @ normal
                    proj2 = v2 @ normal
                    min1, max1 = np.min(proj1), np.max(proj1)
                    min2, max2 = np.min(proj2), np.max(proj2)
                    # If separated, this axis shows no overlap
                    if max1 <= min2 or max2 <= min1:
                        continue
                    # Overlap on this axis: penetration is min(max1-min2, max2-min1)
                    pen = min(max1 - min2, max2 - min1)
                    if pen > max_pen:
                        max_pen = pen
            if max_pen > 0:
                total_pen += max_pen * max_pen
                n_overlaps += 1
    return total_pen, n_overlaps

def _pressure_energy(centers, angles_deg, s, side=1.0):
    """
    Pressure-based energy: sum of squared penetrations + pressure term from outer hex.
    """
    all_verts = _compute_all_vertices(centers, angles_deg, side)
    # Check which vertices are outside the outer hexagon of side s at origin, angle 0
    k = np.arange(6)
    angles = (k + 0.5) * np.pi / 3
    m = np.column_stack([np.cos(angles), np.sin(angles)])
    
    dots = all_verts @ m.T  # (72, 6)
    max_dot = np.max(dots, axis=1)  # (72,)
    # For a regular hexagon with side s centered at origin angle 0,
    # the distance from center to each edge is s*sqrt(3)/2
    # A vertex is inside if for all 6 directions, dot <= s*sqrt(3)/2
    edge_dist = s * np.sqrt(3.0) / 2.0
    # Violation: max_dot - edge_dist, positive means outside
    violations = np.maximum(0.0, max_dot - edge_dist)
    pressure_energy = np.sum(violations ** 2)
    return pressure_energy

def _soft_objective(params, side=1.0):
    """
    Smooth objective: s + lambda1 * total_penetration + lambda2 * pressure_energy
    """
    params_2d = params.reshape(12, 3)
    centers = params_2d[:, :2]
    angles_deg = params_2d[:, 2]
    
    all_verts = _compute_all_vertices(centers, angles_deg, side)
    s = _min_outer_side(all_verts)
    
    pen, n_overlaps = _penetration_depth(centers, angles_deg, side)
    pressure = _pressure_energy(centers, angles_deg, s, side)
    
    obj = s + 10.0 * pen + 5.0 * pressure
    return obj, s, pen, pressure

def _gradient_descent(params, deadline, side=1.0, lr=0.05, max_iters=20000):
    """
    Gradient descent on the soft objective using finite differences.
    """
    current = params.copy()
    obj_current, _, _, _ = _soft_objective(current, side)
    best_obj = obj_current
    best_params = current.copy()
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Finite difference gradient
        grad = np.zeros_like(current)
        h = 0.01
        for k in range(len(current)):
            perturbed = current.copy()
            perturbed[k] += h
            obj_plus, _, _, _ = _soft_objective(perturbed, side)
            grad[k] = (obj_plus - obj_current) / h
        
        # Normalize gradient
        gnorm = np.linalg.norm(grad)
        if gnorm < 1e-10:
            break
        
        # Update
        step = lr * grad / gnorm
        new_params = current - step
        
        obj_new, _, _, _ = _soft_objective(new_params, side)
        
        if obj_new < obj_current:
            current = new_params
            obj_current = obj_new
            if obj_current < best_obj:
                best_obj = obj_current
                best_params = current.copy()
        else:
            # Halve learning rate
            lr *= 0.5
            if lr < 1e-8:
                break
    
    return best_params, best_obj

def _logsumexp_surrogate(params, beta=50.0, side=1.0):
    """
    Smooth surrogate for the outer-side objective using log-sum-exp.
    This makes the objective differentiable and suitable for L-BFGS-B.
    """
    params_2d = params.reshape(12, 3)
    centers = params_2d[:, :2]
    angles_deg = params_2d[:, 2]
    
    all_verts = _compute_all_vertices(centers, angles_deg, side)
    
    # Compute the 6 support function values for each vertex
    k = np.arange(6)
    angles = (k + 0.5) * np.pi / 3
    m = np.column_stack([np.cos(angles), np.sin(angles)])
    
    dots = all_verts @ m.T  # (72, 6)
    
    # Soft max over all 72*6 values using log-sum-exp
    # s = (2/sqrt(3)) * max(dots)
    # Use soft-max: s_smooth = (2/sqrt(3)) * (1/beta) * log(sum(exp(beta * dots)))
    max_dots = np.max(dots)
    exp_dots = np.exp(beta * (dots - max_dots))
    log_sum_exp = max_dots + np.log(np.sum(exp_dots)) / beta
    
    s_smooth = (2.0 / np.sqrt(3.0)) * log_sum_exp
    
    # Add penalty for overlaps
    pen, _ = _penetration_depth(centers, angles_deg, side)
    pressure = _pressure_energy(centers, angles_deg, s_smooth, side)
    
    obj = s_smooth + 10.0 * pen + 5.0 * pressure
    return obj

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 600.0
    
    # Start with honeycomb arrangement
    centers, angles_deg = _initial_honeycomb()
    params = np.column_stack([centers, angles_deg]).flatten()
    
    all_verts = _compute_all_vertices(centers, angles_deg)
    best_s = _min_outer_side(all_verts)
    best_params = params.copy()
    
    if not _check_nonoverlap(centers, angles_deg):
        scale = 1.1
        while not _check_nonoverlap(centers * scale, angles_deg) and scale < 3.0:
            scale += 0.05
        centers = centers * scale
        params = np.column_stack([centers, angles_deg]).flatten()
        all_verts = _compute_all_vertices(centers, angles_deg)
        best_s = _min_outer_side(all_verts)
        best_params = params.copy()
    
    # Phase 1: Simulated annealing
    np.random.seed(42)
    T = 0.5
    cooling = 0.9995
    step_size = 0.3
    
    current_params = best_params.copy()
    current_s = best_s
    
    max_iters = 300000
    for it in range(max_iters):
        if time.time() > deadline - 60:
            break
        
        new_params = current_params + np.random.randn(36) * step_size
        
        centers_new = new_params.reshape(12, 3)[:, :2]
        angles_new = new_params.reshape(12, 3)[:, 2]
        
        all_verts_new = _compute_all_vertices(centers_new, angles_new)
        s_new = _min_outer_side(all_verts_new)
        
        if not _check_nonoverlap(centers_new, angles_new):
            s_new += 50.0
        
        delta = s_new - current_s
        
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            current_params = new_params
            current_s = s_new
            if s_new < best_s:
                best_s = s_new
                best_params = new_params.copy()
        
        T *= cooling
        if it % 1000 == 0:
            step_size *= 0.95
            if step_size < 0.01:
                step_size = 0.01
    
    # Phase 2: L-BFGS-B with smooth surrogate
    if time.time() < deadline - 30:
        # Use multiple restarts with different perturbation scales
        for restart in range(8):
            if time.time() > deadline - 30:
                break
            
            # Perturb the best params for restart diversity
            if restart == 0:
                x0 = best_params.copy()
            else:
                pert_scale = 0.1 * (restart + 1)
                x0 = best_params + np.random.randn(36) * pert_scale
            
            # Define objective for L-BFGS-B
            def objective(x):
                return _logsumexp_surrogate(x, beta=50.0)
            
            # Run L-BFGS-B
            result = minimize(objective, x0, method='L-BFGS-B', 
                             options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if result.success:
                centers_opt = result.x.reshape(12, 3)[:, :2]
                angles_opt = result.x.reshape(12, 3)[:, 2]
                
                if _check_nonoverlap(centers_opt, angles_opt):
                    all_verts_opt = _compute_all_vertices(centers_opt, angles_opt)
                    s_opt = _min_outer_side(all_verts_opt)
                    if s_opt < best_s:
                        best_s = s_opt
                        best_params = result.x.copy()
    
    # Phase 3: Final refinement with small steps
    for _ in range(15000):
        if time.time() > deadline - 10:
            break
        trial = best_params + np.random.randn(36) * 0.03
        c = trial.reshape(12,3)[:, :2]
        a = trial.reshape(12,3)[:, 2]
        if not _check_nonoverlap(c, a):
            continue
        s_trial = _min_outer_side(_compute_all_vertices(c, a))
        if s_trial < best_s:
            best_s = s_trial
            best_params = trial
    
    # Phase 4: Very fine polish
    for _ in range(5000):
        if time.time() > deadline - 5:
            break
        trial = best_params + np.random.randn(36) * 0.01
        c = trial.reshape(12,3)[:, :2]
        a = trial.reshape(12,3)[:, 2]
        if not _check_nonoverlap(c, a):
            continue
        s_trial = _min_outer_side(_compute_all_vertices(c, a))
        if s_trial < best_s:
            best_s = s_trial
            best_params = trial
    
    # Phase 5: Ultra-fine polish
    for _ in range(3000):
        if time.time() > deadline - 2:
            break
        trial = best_params + np.random.randn(36) * 0.005
        c = trial.reshape(12,3)[:, :2]
        a = trial.reshape(12,3)[:, 2]
        if not _check_nonoverlap(c, a):
            continue
        s_trial = _min_outer_side(_compute_all_vertices(c, a))
        if s_trial < best_s:
            best_s = s_trial
            best_params = trial
    
    centers_final = best_params.reshape(12, 3)[:, :2]
    angles_final = best_params.reshape(12, 3)[:, 2]
    
    centroid = np.mean(centers_final, axis=0)
    centers_final = centers_final - centroid
    
    all_verts_final = _compute_all_vertices(centers_final, angles_final)
    s_final = _min_outer_side(all_verts_final)
    
    if not _check_nonoverlap(centers_final, angles_final):
        centers_final, angles_final = _initial_honeycomb()
        all_verts_final = _compute_all_vertices(centers_final, angles_final)
        s_final = _min_outer_side(all_verts_final)
    
    inner_hex_data = np.column_stack([centers_final, angles_final])
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, s_final

# EVOLVE-BLOCK-END
