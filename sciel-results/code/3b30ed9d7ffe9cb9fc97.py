import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree, Delaunay
from scipy.spatial.distance import cdist

N = 300

def _hex_init(n_points, seed=0):
    """Generate a hexagonal lattice and select n_points best points."""
    rng = np.random.default_rng(seed)
    best_pts = None
    best_r = 0.0

    # Try multiple rotations and spacings to find a good subset
    for attempt in range(10):
        # Base spacing for packing N circles in radius 1
        # Area per circle ~ pi * r^2. Total area ~ pi.
        # N * pi * r^2 < pi => r < 1/sqrt(N).
        # Spacing d = 2r.
        base_spacing = 2.0 / np.sqrt(n_points) * 0.98
        spacing = base_spacing * (0.97 + 0.01 * attempt)
        
        k = int(1.0 / spacing) + 5
        rot_angle = rng.uniform(0, 2 * np.pi)
        cos_a = np.cos(rot_angle)
        sin_a = np.sin(rot_angle)
        
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                # Hexagonal lattice coordinates
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                
                # Rotate
                x_rot = x * cos_a - y * sin_a
                y_rot = x * sin_a + y * cos_a
                r = np.hypot(x_rot, y_rot)
                
                # Must fit inside unit circle (center distance < 1 - radius)
                if r <= 1.0 - spacing / 2.0:
                    rows.append((x_rot, y_rot, r))

        if len(rows) < n_points:
            continue

        pts = np.array([(r[0], r[1]) for r in rows])
        radii = np.array([r[2] for r in rows])
        
        # Select points closest to center to maximize symmetry and packing density
        order = np.argsort(radii, kind="stable")
        selected = pts[order[:n_points]]
        
        # Add small noise for subsequent attempts to break symmetry
        if attempt > 0:
            selected = selected + rng.normal(0, spacing * 0.02, selected.shape)
            norms = np.linalg.norm(selected, axis=1)
            mask = norms < 1.0
            selected = selected[mask]
            if len(selected) < n_points:
                continue
            selected = selected[:n_points]

        # Compute current radius
        d = cdist(selected, selected)
        np.fill_diagonal(d, np.inf)
        min_pair = d.min() / 2.0
        min_bound = 1.0 - np.linalg.norm(selected, axis=1).max()
        r = min(min_pair, min_bound)

        if r > best_r:
            best_r = r
            best_pts = selected.copy()

    if best_pts is None:
        # Fallback: simple grid or hex without rotation
        spacing = 2.0 / np.sqrt(n_points) * 0.9
        k = int(1.0 / spacing) + 5
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if np.hypot(x, y) <= 1.0 - spacing / 2.0:
                    rows.append((x, y))
        pts = np.array(rows)
        order = np.argsort(np.linalg.norm(pts, axis=1), kind="stable")
        best_pts = pts[order[:n_points]]

    return best_pts

def _compute_radius(pts):
    """Compute the packing radius for given points."""
    norms = np.linalg.norm(pts, axis=1)
    min_bound = 1.0 - norms.max()
    if min_bound <= 0:
        return 0.0
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    min_pair = d.min() / 2.0
    return min(min_pair, min_bound)

def _get_active_edges(pts, t, margin=1.01):
    """
    Get edges where distance < 2*t*margin.
    Also include boundary edges where norm > 1 - t*margin.
    """
    n = len(pts)
    tree = cKDTree(pts)
    pairs = tree.query_pairs(r=2 * t * margin, output_type='ndarray')
    edge_list = [(int(i), int(j)) for i, j in pairs]
    
    # Boundary constraints are active if norm is close to 1-t
    # We don't add them to 'edges' for pairwise gradient, 
    # but we need to know which points are near the boundary for the objective.
    return edge_list

def _objective_and_grad_epigraph(x, T, edge_indices, lambda_pair=100.0, lambda_bound=100.0):
    """
    Epigraph formulation with penalty.
    x = [pts (n*2), t]
    Objective: -t + lambda_pair * sum(max(0, 2t - d_ij)^2) + lambda_bound * sum(max(0, t + norm_i - 1)^2)
    This directly maximizes t while penalizing violations.
    """
    n_pts = N
    pts = x[:n_pts*2].reshape(n_pts, 2)
    t = x[n_pts*2]
    
    # 1. Calculate distances for active edges
    # To save time, we only compute penalties for edges that are close.
    # However, gradients need to be consistent. 
    # Using a fixed set of edges from the last iteration is a common approximation (Active Set Method).
    
    pair_pen = 0.0
    grad_pair = np.zeros_like(pts)
    
    for i, j in edge_indices:
        diff = pts[i] - pts[j]
        d = np.linalg.norm(diff)
        if d < 1e-10:
            continue
        
        violation = 2 * t - d
        if violation > 0:
            # Penalty term: 0.5 * lambda * violation^2
            # d/dt (0.5 * lambda * (2t - d)^2) = lambda * (2t - d) * 2 = 2 * lambda * violation
            # d/dpts (0.5 * lambda * (2t - d)^2) = lambda * (2t - d) * (-2 * diff/d) 
            # Note: d = ||pts_i - pts_j||. grad_d/grad_pts_i = -diff/d. 
            # Chain rule: grad_obj = lambda * violation * 2 * (-diff/d) for i?
            # Let's derive carefully.
            # L = 0.5 * lambda * (2t - d)^2
            # dL/dt = lambda * (2t - d) * 2 = 2 * lambda * v
            # dL/dpts_i = dL/dd * dd/dpts_i
            # dd/dpts_i = (pts_i - pts_j) / d = diff/d
            # So dL/dpts_i = lambda * (2t - d) * 2 * (diff/d)  <-- Wait.
            # Let u = 2t - d. L = 0.5 * lambda * u^2.
            # dL/du = lambda * u.
            # du/dt = 2.
            # du/dpts_i = -1 * d(pts_i - pts_j)/dpts_i = -1 * (1) for the component? 
            # d/dx_i (sqrt((xi-xj)^2)) = (xi-xj)/d.
            # So du/dpts_i = - (diff/d).
            # dL/dpts_i = lambda * u * (-diff/d) = -lambda * v * diff/d.
            
            # We want to MINIMIZE the objective.
            # Objective = -t + Penalty.
            # Grad t: -1 + 2 * lambda * v
            # Grad pts_i: -lambda * v * diff/d
            # Grad pts_j: +lambda * v * diff/d (since diff = pi - pj, d/dpj = -diff/d, so -lambda * v * (-diff/d))
            
            grad_t = 2 * lambda_pair * violation
            grad_vec = -lambda_pair * violation * diff / d
            
            grad_pair[i] += grad_vec
            grad_pair[j] -= grad_vec
            
            pair_pen += 0.5 * lambda_pair * violation * violation
            # Store grad_t contribution separately to avoid looping twice or using a list
            # Since we return a single scalar grad, we need to sum all grad_t contributions.
            # It's inefficient to do this in a loop without accumulating.
            # Let's refactor: accumulate grad_t in a variable.
    
    # Boundary Penalties
    bound_pen = 0.0
    grad_bound = np.zeros_like(pts)
    grad_t_bound = 0.0
    
    norms = np.linalg.norm(pts, axis=1)
    # Only check points near boundary to save time? Or all?
    # For 300 points, checking all is fast.
    for i in range(n_pts):
        v = t + norms[i] - 1.0
        if v > 0:
            # L = 0.5 * lambda_bound * v^2
            # v = t + norm - 1
            # dv/dt = 1
            # dv/dpts = pts / norm
            # dL/dt = lambda_bound * v
            # dL/dpts = lambda_bound * v * pts/norm
            
            grad_t_bound += lambda_bound * v
            grad_bound[i] += lambda_bound * v * pts[i] / norms[i]
            bound_pen += 0.5 * lambda_bound * v * v

    # Recalculate pair grad_t properly
    grad_t_pair = 0.0
    for i, j in edge_indices:
        diff = pts[i] - pts[j]
        d = np.linalg.norm(diff)
        if d < 1e-10: continue
        v = 2 * t - d
        if v > 0:
            grad_t_pair += 2 * lambda_pair * v

    total_grad_t = -1.0 + grad_t_pair + grad_t_bound
    total_grad_pts = grad_pair + grad_bound
    
    obj = -t + pair_pen + bound_pen
    
    grad_x = np.concatenate([total_grad_pts.ravel(), [total_grad_t]])
    
    return obj, grad_x

def _local_polish_epigraph(pts, t_init, deadline):
    """
    Polish using epigraph formulation.
    """
    n_pts = N
    x0 = np.concatenate([pts.ravel(), [t_init]])
    
    # Initial active edges
    edges = _get_active_edges(pts, t_init, margin=1.02)
    
    best_x = x0.copy()
    best_obj = _objective_and_grad_epigraph(best_x, 0, edges)[0] # T not used in this version
    best_r = t_init
    
    # We will run minimize a few times with updated edge sets
    # Start with higher lambda to push violations away, then lower?
    # Or just use a fixed reasonable lambda.
    # Since it's a penalty method, if lambda is too high, it might be stiff.
    # If too low, it might not satisfy constraints.
    # We rely on the final _compute_radius check.
    
    # Phase 1: Strong penalty to fix gross violations
    # Phase 2: Weaker penalty, tighter tolerance, to maximize t
    
    phases = [
        {'lambda_p': 500.0, 'lambda_b': 500.0, 'maxiter': 100, 'ftol': 1e-12},
        {'lambda_p': 100.0, 'lambda_b': 100.0, 'maxiter': 200, 'ftol': 1e-14},
        {'lambda_p': 10.0, 'lambda_b': 10.0, 'maxiter': 300, 'ftol': 1e-16},
    ]
    
    for phase in phases:
        if time.time() > deadline:
            break
            
        # Update edges based on current best
        edges = _get_active_edges(pts, t_init, margin=1.02)
        
        def obj_func(x):
            return _objective_and_grad_epigraph(x, 0, edges, phase['lambda_p'], phase['lambda_b'])[0]
        
        def grad_func(x):
            return _objective_and_grad_epigraph(x, 0, edges, phase['lambda_p'], phase['lambda_b'])[1]
        
        try:
            res = minimize(
                obj_func, 
                x0, 
                jac=grad_func,
                method='L-BFGS-B',
                options={
                    'maxiter': phase['maxiter'],
                    'ftol': phase['ftol'],
                    'gtol': 1e-10,
                    'maxfun': phase['maxiter'] * 10,
                    'disp': False
                }
            )
            
            new_x = res.x
            new_pts = new_x[:n_pts*2].reshape(n_pts, 2)
            new_t = new_x[n_pts*2]
            
            # Check validity
            if np.all(np.isfinite(new_pts)) and np.all(np.linalg.norm(new_pts, axis=1) < 1.0):
                # Calculate true radius to see if it's valid and better
                true_r = _compute_radius(new_pts)
                if true_r > best_r:
                    best_r = true_r
                    best_x = new_x.copy()
                    pts = new_pts
                    t_init = true_r # Use true radius for next edge selection
                    x0 = best_x.copy()
        except Exception:
            pass
            
    return pts, best_r

def _force_relax(pts, n_steps=1000, deadline=None):
    """Fast force-directed relaxation using cKDTree for neighbor search."""
    pos = pts.copy().astype(np.float64)
    best_pos = pos.copy()
    best_r = _compute_radius(best_pos)
    n = len(pos)
    r_est = best_r

    lr = 0.01
    cooldown = 0.999

    for step in range(n_steps):
        if deadline is not None and time.time() > deadline:
            break

        forces = np.zeros_like(pos)
        target = 2.0 * r_est

        try:
            tree = cKDTree(pos)
            pairs = tree.query_pairs(r=target, output_type='ndarray')
            for i, j in pairs:
                diff = pos[i] - pos[j]
                d = np.linalg.norm(diff)
                if d < 1e-10:
                    continue
                excess = target - d
                if excess > 0:
                    # Soft repulsion
                    force_mag = 1.0 * excess
                    unit_vec = diff / d
                    forces[i] += force_mag * unit_vec
                    forces[j] -= force_mag * unit_vec
        except Exception:
            pass

        # Boundary force
        norms = np.linalg.norm(pos, axis=1)
        wall_margin = 0.05
        close_to_wall = norms > (1.0 - wall_margin)
        if np.any(close_to_wall):
            excess_b = np.maximum(norms[close_to_wall] - (1.0 - wall_margin), 0)
            f_mags = 10.0 * excess_b
            directions = -pos[close_to_wall] / norms[close_to_wall, np.newaxis]
            forces[close_to_wall] += f_mags[:, np.newaxis] * directions

        pos = pos + lr * forces

        # Project back into unit disk
        norms = np.linalg.norm(pos, axis=1)
        mask_out = norms > 1.0
        if np.any(mask_out):
            pos[mask_out] = pos[mask_out] / norms[mask_out, np.newaxis] * 0.999

        if step % 20 == 0:
            r = _compute_radius(pos)
            if r > best_r:
                best_r = r
                best_pos = pos.copy()
                r_est = r

        lr *= cooldown

    return best_pos, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    best_pts = None
    best_r = 0.0

    # Phase 1: Initialization and Coarse Relaxation
    seeds = [42, 123, 7, 99, 256, 777, 1234, 5678, 1111, 2222, 3333, 4444]
    
    for seed in seeds:
        if time.time() > deadline - 30.0:
            break

        pts = _hex_init(N, seed=seed)
        r = _compute_radius(pts)

        # Quick relaxation to remove initial overlaps
        relaxed_pts, relaxed_r = _force_relax(pts, n_steps=500, deadline=deadline - 25.0)
        
        if relaxed_r > r:
            pts = relaxed_pts
            r = relaxed_r

        if r > best_r:
            best_r = r
            best_pts = pts.copy()

    if best_pts is None:
        best_pts = _hex_init(N, seed=0)
        best_r = _compute_radius(best_pts)

    # Phase 2: Epigraph Polish
    # Use the best point found so far
    if time.time() < deadline - 5.0:
        polished_pts, polished_r = _local_polish_epigraph(best_pts, best_r, deadline=deadline - 2.0)
        if polished_r > best_r:
            best_r = polished_r
            best_pts = polished_pts.copy()
    
    # Phase 3: Final High-Precision Polish on the result of Phase 2
    # Run again with even tighter tolerances if time permits
    if time.time() < deadline - 1.0:
        polished_pts2, polished_r2 = _local_polish_epigraph(best_pts, best_r, deadline=deadline - 0.5)
        if polished_r2 > best_r:
            best_r = polished_r2
            best_pts = polished_pts2.copy()

    # Final validity check
    if best_pts is None:
        best_pts = _hex_init(N, seed=0)
    
    norms = np.linalg.norm(best_pts, axis=1)
    if not np.all(norms < 1.0) or not np.all(np.isfinite(best_pts)):
        # Fallback
        best_pts = _hex_init(N, seed=0)
        norms = np.linalg.norm(best_pts, axis=1)
        mask = norms < 1.0
        if not np.all(mask):
             # If even fallback fails (shouldn't happen), return zero
             best_pts = np.zeros((N, 2))

    return best_pts
