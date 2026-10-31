"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 26

# EVOLVE-BLOCK-START

def _softplus(x, beta=50.0):
    """Numerically stable softplus."""
    x = np.asarray(x, dtype=float)
    return np.where(x > 20.0/beta, x, np.log1p(np.exp(beta * x)) / beta)

def _soft_max(v, beta=50.0):
    """Smooth approximation of max(0, v) using softplus."""
    return _softplus(v, beta)

def _objective(x, penalty_weight=100.0, beta=50.0):
    """
    Objective: minimize -sum(radii) + penalty * smooth_violations.
    x = [x1, y1, ..., xN, yN, r1, ..., rN]
    """
    pos = x[:2*N].reshape(N, 2)
    radii = np.maximum(x[2*N:], 1e-6)
    
    # Boundary violations
    bv = np.maximum(0.0, radii - pos[:, 0]) + np.maximum(0.0, pos[:, 0] - (1.0 - radii)) + \
         np.maximum(0.0, radii - pos[:, 1]) + np.maximum(0.0, pos[:, 1] - (1.0 - radii))
    boundary_pen = np.sum(bv * bv)
    
    # Overlap violations (vectorized)
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    min_dist_sq = min_dist * min_dist
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    
    viol = np.maximum(0.0, min_dist_sq - dist_sq)
    overlap_pen = np.sum(viol[mask] * viol[mask])
    
    return -np.sum(radii) + penalty_weight * (boundary_pen + overlap_pen)

def _objective_smooth(x, penalty_weight=100.0, beta=30.0):
    """
    Objective with softplus-based smooth penalties.
    """
    pos = x[:2*N].reshape(N, 2)
    radii = np.maximum(x[2*N:], 1e-6)
    
    # Smooth boundary violations
    left_v = radii - pos[:, 0]
    right_v = pos[:, 0] - (1.0 - radii)
    bottom_v = radii - pos[:, 1]
    top_v = pos[:, 1] - (1.0 - radii)
    
    boundary_pen = np.sum(_soft_max(left_v, beta)) + np.sum(_soft_max(right_v, beta)) + \
                   np.sum(_soft_max(bottom_v, beta)) + np.sum(_soft_max(top_v, beta))
    
    # Smooth overlap violations
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    min_dist_sq = min_dist * min_dist
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    
    viol_raw = min_dist_sq - dist_sq
    overlap_pen = np.sum(_soft_max(viol_raw[mask], beta))
    
    return -np.sum(radii) + penalty_weight * (boundary_pen + overlap_pen)

def _is_valid(pos, radii, tol=1e-7):
    """Strict validity check."""
    n = len(radii)
    if np.any(pos[:, 0] - radii < -tol) or np.any(pos[:, 0] + radii > 1.0 + tol):
        return False
    if np.any(pos[:, 1] - radii < -tol) or np.any(pos[:, 1] + radii > 1.0 + tol):
        return False
        
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    
    if np.any(dist_sq[mask] < (min_dist[mask] - tol)**2):
        return False
    return True

def _try_radius_expansion(pos, radii, max_iter=50, step_size=0.005):
    """
    Try to expand radii of circles that are not touching any constraint.
    This is a greedy local improvement.
    """
    pos = pos.copy()
    radii = radii.copy()
    
    for _ in range(max_iter):
        improved = False
        for i in range(N):
            # Check if circle i is touching a boundary
            touching_boundary = False
            if abs(pos[i, 0] - radii[i]) < 1e-5 or abs(pos[i, 0] + radii[i] - 1.0) < 1e-5:
                touching_boundary = True
            if abs(pos[i, 1] - radii[i]) < 1e-5 or abs(pos[i, 1] + radii[i] - 1.0) < 1e-5:
                touching_boundary = True
            
            # Check if circle i is touching any other circle
            touching_circle = False
            for j in range(N):
                if i == j:
                    continue
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < radii[i] + radii[j] + 1e-5:
                    touching_circle = True
                    break
            
            # If not touching anything, try to expand
            if not touching_boundary and not touching_circle:
                # Calculate max possible expansion
                max_expansion = 0.01 # Small step
                
                # Check distance to boundaries
                max_expansion = min(max_expansion, pos[i, 0] - radii[i])
                max_expansion = min(max_expansion, 1.0 - pos[i, 0] - radii[i])
                max_expansion = min(max_expansion, pos[i, 1] - radii[i])
                max_expansion = min(max_expansion, 1.0 - pos[i, 1] - radii[i])
                
                # Check distance to other circles
                for j in range(N):
                    if i == j:
                        continue
                    dx = pos[i, 0] - pos[j, 0]
                    dy = pos[i, 1] - pos[j, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    space = dist - radii[i] - radii[j]
                    max_expansion = min(max_expansion, space)
                
                if max_expansion > 1e-6:
                    # Expand by a fraction of the available space
                    expansion = min(step_size, max_expansion * 0.5)
                    radii[i] += expansion
                    improved = True
        
        if not improved:
            break
    
    return pos, radii

def _post_process(pos, radii):
    """Shrink radii to ensure strict validity."""
    pos = pos.copy()
    radii = radii.copy()
    
    for _ in range(200):
        valid = True
        
        for i in range(N):
            r = radii[i]
            x, y = pos[i]
            max_r = min(x, 1.0 - x, y, 1.0 - y)
            if r > max_r:
                radii[i] = max(max_r * 0.9999, 1e-5)
                valid = False
        
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                sum_r = radii[i] + radii[j]
                if dist < sum_r:
                    k = dist / (sum_r + 1e-12) * 0.9999
                    if k < 1.0:
                        radii[i] = max(radii[i] * k, 1e-5)
                        radii[j] = max(radii[j] * k, 1e-5)
                        valid = False
                        
        if valid:
            break
            
    return pos, radii

def _initial_best_fit():
    """Greedy best-fit initialization."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    pos[0] = [0.5, 0.5]
    radii[0] = 0.25
    
    for i in range(1, N):
        best_x, best_y, best_r = 0.5, 0.5, 0.0
        found = False
        rng = np.random.RandomState(i * 1000 + 42)
        
        for _ in range(5000):
            x = rng.uniform(0.02, 0.98)
            y = rng.uniform(0.02, 0.98)
            
            max_r_wall = min(x, 1.0 - x, y, 1.0 - y)
            max_r_circ = max_r_wall
            for j in range(i):
                dx = x - pos[j, 0]
                dy = y - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                possible_r = dist - radii[j]
                if possible_r < max_r_circ:
                    max_r_circ = possible_r
            
            if max_r_circ > best_r and max_r_circ > 0.003:
                best_r = max_r_circ
                best_x = x
                best_y = y
                found = True
        
        if not found or best_r < 0.003:
            best_x, best_y = 0.5, 0.5
            best_r = 0.003
            
        pos[i] = [best_x, best_y]
        radii[i] = best_r * 0.97
        
    return pos, radii

def _initial_hexagonal():
    """Hexagonal lattice initialization."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    r = 0.11
    dx = r * np.sqrt(3)
    dy = r * 1.5
    
    k = 0
    for row in range(6):
        offset = dx / 2 if row % 2 == 1 else 0
        start_x = r + offset
        y = row * dy + r
        
        for col in range(5):
            if k >= N:
                break
            x = start_x + col * dx
            
            if x + r <= 1.0 and x - r >= 0.0 and y + r <= 1.0 and y - r >= 0.0:
                pos[k] = [x, y]
                radii[k] = r
                k += 1
        
        if k >= N:
            break
            
    while k < N:
        rng = np.random.RandomState(k * 7 + 13)
        for _ in range(100):
            x = rng.uniform(0.1, 0.9)
            y = rng.uniform(0.1, 0.9)
            min_dist = min(x, 1 - x, y, 1 - y)
            for j in range(k):
                d = np.sqrt((x - pos[j, 0])**2 + (y - pos[j, 1])**2)
                min_dist = min(min_dist, d - radii[j])
            if min_dist > 0.02:
                pos[k] = [x, y]
                radii[k] = min_dist * 0.95
                k += 1
                break
        if k < N:
            pos[k] = [0.5, 0.5]
            radii[k] = 0.01
            k += 1
            
    return pos, radii

def _initial_corner_biased(seed=7):
    """Corner-biased initialization."""
    rng = np.random.RandomState(seed)
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    
    corners = [(0.15, 0.15), (0.85, 0.15), (0.15, 0.85), (0.85, 0.85)]
    for i, (cx, cy) in enumerate(corners):
        pos[i] = [cx + rng.uniform(-0.03, 0.03), cy + rng.uniform(-0.03, 0.03)]
        radii[i] = 0.18
    
    for i in range(4, N):
        if rng.random() < 0.4:
            edge = rng.randint(4)
            t = rng.uniform(0.1, 0.9)
            if edge == 0:
                x, y = t, 0.08
            elif edge == 1:
                x, y = t, 0.92
            elif edge == 2:
                x, y = 0.08, t
            else:
                x, y = 0.92, t
        else:
            x, y = rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9)
        
        max_r = min(x, 1.0 - x, y, 1.0 - y)
        for j in range(i):
            dx = x - pos[j, 0]
            dy = y - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            max_r = min(max_r, dist - radii[j])
        
        if max_r < 0.01:
            for attempt in range(200):
                x2 = rng.uniform(0.05, 0.95)
                y2 = rng.uniform(0.05, 0.95)
                max_r2 = min(x2, 1.0 - x2, y2, 1.0 - y2)
                for j in range(i):
                    dx = x2 - pos[j, 0]
                    dy = y2 - pos[j, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    max_r2 = min(max_r2, dist - radii[j])
                if max_r2 > max_r:
                    max_r = max_r2
                    x, y = x2, y2
        
        pos[i] = [x, y]
        radii[i] = max(max_r, 0.01) * 0.95
        
    return pos, radii

def _initial_random(seed):
    """Random initialization with post-processing."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, (N, 2))
    radii = rng.uniform(0.04, 0.12, N)
    pos, radii = _post_process(pos, radii)
    return pos, radii

def _simulated_annealing(pos, radii, n_steps=3000, t0=0.05, t_end=0.001, seed=0):
    """
    Simulated annealing: perturb positions and radii, accept based on energy.
    Uses adaptive step sizes and more aggressive exploration.
    """
    rng = np.random.RandomState(seed)
    pos = pos.copy().astype(float)
    radii = radii.copy().astype(float)
    n = len(radii)
    
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    # Current energy (penalty-based)
    current_energy = _objective(np.concatenate([pos.flatten(), radii]), 100.0)
    
    # Track recent improvements for adaptive step size
    recent_improvements = []
    
    for step in range(n_steps):
        frac = step / max(n_steps - 1, 1)
        temp = t0 * (t_end / t0) ** frac
        
        # Adaptive step size based on recent performance
        if len(recent_improvements) > 50:
            avg_improve = np.mean(recent_improvements[-50:])
            if avg_improve > 0.01:
                step_mult = 0.8
            elif avg_improve < 0.001:
                step_mult = 1.2
            else:
                step_mult = 1.0
        else:
            step_mult = 1.0
        
        # Perturb: move one circle slightly
        i = rng.randint(n)
        base_scale = temp * step_mult
        dx = rng.normal(0, base_scale * 0.015)
        dy = rng.normal(0, base_scale * 0.015)
        dr = rng.normal(0, base_scale * 0.008)
        
        new_x = pos[i, 0] + dx
        new_y = pos[i, 1] + dy
        new_r = radii[i] + dr
        
        # Clamp
        new_x = np.clip(new_x, 0.01, 0.99)
        new_y = np.clip(new_y, 0.01, 0.99)
        new_r = np.clip(new_r, 0.01, 0.4)
        
        # Try the move
        old_x, old_y, old_r = pos[i, 0], pos[i, 1], radii[i]
        pos[i, 0] = new_x
        pos[i, 1] = new_y
        radii[i] = new_r
        
        x_new = np.concatenate([pos.flatten(), radii])
        new_energy = _objective(x_new, 100.0)
        
        delta_e = new_energy - current_energy
        if delta_e < 0 or rng.random() < np.exp(-delta_e / max(temp, 1e-8)):
            current_energy = new_energy
            recent_improvements.append(-delta_e)
        else:
            # Reject
            pos[i, 0] = old_x
            pos[i, 1] = old_y
            radii[i] = old_r
            recent_improvements.append(0.0)
        
        # Periodically check validity and update best
        if step % 50 == 0:
            pos_pp, radii_pp = _post_process(pos, radii)
            if _is_valid(pos_pp, radii_pp):
                score = np.sum(radii_pp)
                if score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
    
    return best_pos, best_radii

def _slsqp_polish(pos, radii, deadline, max_iter=500):
    """
    Polish using SLSQP with explicit inequality constraints.
    """
    x = np.concatenate([pos.flatten(), radii])
    
    bounds = []
    for i in range(N):
        bounds.append((0.01, 0.99))
        bounds.append((0.01, 0.99))
    for i in range(N):
        bounds.append((0.01, 0.45))
    
    # Objective: minimize -sum(radii)
    def obj(x):
        return -np.sum(x[2*N:])
    
    # Constraint function for SLSQP
    cons = []
    for i in range(N):
        # x_i - r_i >= 0
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i] - x[2*N+i]})
        # (1 - x_i) - r_i >= 0
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: (1.0 - x[2*i]) - x[2*N+i]})
        # y_i - r_i >= 0
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i+1] - x[2*N+i]})
        # (1 - y_i) - r_i >= 0
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: (1.0 - x[2*i+1]) - x[2*N+i]})
    
    for i in range(N):
        for j in range(i + 1, N):
            def overlap_con(x, i=i, j=j):
                dx = x[2*i] - x[2*j]
                dy = x[2*i+1] - x[2*j+1]
                dist = np.sqrt(dx*dx + dy*dy)
                return dist - x[2*N+i] - x[2*N+j]
            cons.append({'type': 'ineq', 'fun': overlap_con})
    
    try:
        result = minimize(obj, x, method='SLSQP', bounds=bounds,
                          constraints=cons,
                          options={'maxiter': max_iter, 'ftol': 1e-12})
        if result.success or result.fun < -np.sum(radii):
            pos_out = result.x[:2*N].reshape(N, 2)
            radii_out = np.maximum(result.x[2*N:], 1e-4)
            return pos_out, radii_out
    except Exception:
        pass
    
    return pos, radii

def _newton_contact_graph(pos, radii, deadline, max_iter=200):
    """
    Damped Newton-Raphson on a fixed contact graph (Delaunay triangulation).
    Variables: positions (2N) and radii (N).
    Constraints: for each edge in contact graph, distance = r_i + r_j (tangency),
                 plus boundary contacts for circles touching walls.
    We solve the system F(x) = 0 where F encodes the residual of contact equations.
    """
    pos = pos.copy().astype(float)
    radii = radii.copy().astype(float)
    n = len(radii)
    
    # Build contact graph from Delaunay
    try:
        tri = Delaunay(pos)
        edges = set()
        for simplex in tri.simplices:
            for k in range(3):
                i, j = int(simplex[k]), int(simplex[(k+1) % 3])
                if i < j:
                    edges.add((i, j))
                else:
                    edges.add((j, i))
    except Exception:
        return pos, radii
    
    if len(edges) == 0:
        return pos, radii
    
    # Identify boundary contacts
    boundary_contacts = []  # (circle_idx, axis, side) where side: 0=left, 1=right, 2=bottom, 3=top
    for i in range(n):
        if pos[i, 0] - radii[i] < 1e-6:
            boundary_contacts.append((i, 0, 0))
        if pos[i, 0] + radii[i] > 1.0 - 1e-6:
            boundary_contacts.append((i, 0, 1))
        if pos[i, 1] - radii[i] < 1e-6:
            boundary_contacts.append((i, 1, 0))
        if pos[i, 1] + radii[i] > 1.0 - 1e-6:
            boundary_contacts.append((i, 1, 1))
    
    # We have 3N variables and len(edges) + len(boundary_contacts) constraints.
    # For a rigid packing, we need exactly 3N constraints.
    # If we have fewer, we add "soft" constraints to keep circles from expanding infinitely.
    # If we have more, we select a subset (prefer those closest to actual tangency).
    
    def compute_residual(pos, radii):
        """Compute residual vector for the contact constraints."""
        res = []
        for (i, j) in edges:
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            res.append(dist - radii[i] - radii[j])
        for (i, axis, side) in boundary_contacts:
            if axis == 0:
                if side == 0:
                    res.append(pos[i, 0] - radii[i])
                else:
                    res.append(pos[i, 0] + radii[i] - 1.0)
            else:
                if side == 0:
                    res.append(pos[i, 1] - radii[i])
                else:
                    res.append(pos[i, 1] + radii[i] - 1.0)
        return np.array(res)
    
    def compute_jacobian(pos, radii):
        """Compute the Jacobian of the residual w.r.t. [x1,y1,...,xN,yN,r1,...,rN]."""
        m = len(edges) + len(boundary_contacts)
        J = np.zeros((m, 3 * n))
        row = 0
        for (i, j) in edges:
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < 1e-12:
                dist = 1e-12
            J[row, 2*i] = dx / dist
            J[row, 2*i + 1] = dy / dist
            J[row, 2*j] = -dx / dist
            J[row, 2*j + 1] = -dy / dist
            J[row, 2*n + i] = -1.0
            J[row, 2*n + j] = -1.0
            row += 1
        for (i, axis, side) in boundary_contacts:
            if axis == 0:
                if side == 0:
                    J[row, 2*i] = 1.0
                    J[row, 2*n + i] = -1.0
                else:
                    J[row, 2*i] = 1.0
                    J[row, 2*n + i] = 1.0
            else:
                if side == 0:
                    J[row, 2*i + 1] = 1.0
                    J[row, 2*n + i] = -1.0
                else:
                    J[row, 2*i + 1] = 1.0
                    J[row, 2*n + i] = 1.0
            row += 1
        return J
    
    # Select a subset of constraints if we have more than 3N
    all_constraints = list(edges) + [(i, axis, side) for (i, axis, side) in boundary_contacts]
    
    # Rank constraints by how close they are to actual tangency
    def constraint_residual_val(c):
        if c in edges:
            i, j = c
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            return abs(dist - radii[i] - radii[j])
        else:
            i, axis, side = c
            if axis == 0:
                if side == 0:
                    return abs(pos[i, 0] - radii[i])
                else:
                    return abs(pos[i, 0] + radii[i] - 1.0)
            else:
                if side == 0:
                    return abs(pos[i, 1] - radii[i])
                else:
                    return abs(pos[i, 1] + radii[i] - 1.0)
    
    # Sort by residual (closest to tangency first)
    sorted_constraints = sorted(all_constraints, key=constraint_residual_val)
    
    # Take up to 3N constraints
    selected = sorted_constraints[:min(3*n, len(sorted_constraints))]
    selected_edges = [c for c in selected if c in edges]
    selected_boundary = [c for c in selected if c not in edges]
    
    if len(selected) == 0:
        return pos, radii
    
    # Newton iteration
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    for iteration in range(max_iter):
        if time.time() > deadline - 5:
            break
        
        # Build residual and Jacobian with selected constraints
        def res_func(p, r):
            res = []
            for (i, j) in selected_edges:
                dx = p[i, 0] - p[j, 0]
                dy = p[i, 1] - p[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                res.append(dist - r[i] - r[j])
            for (i, axis, side) in selected_boundary:
                if axis == 0:
                    if side == 0:
                        res.append(p[i, 0] - r[i])
                    else:
                        res.append(p[i, 0] + r[i] - 1.0)
                else:
                    if side == 0:
                        res.append(p[i, 1] - r[i])
                    else:
                        res.append(p[i, 1] + r[i] - 1.0)
            return np.array(res)
        
        def jac_func(p, r):
            m = len(selected_edges) + len(selected_boundary)
            J = np.zeros((m, 3 * n))
            row = 0
            for (i, j) in selected_edges:
                dx = p[i, 0] - p[j, 0]
                dy = p[i, 1] - p[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < 1e-12:
                    dist = 1e-12
                J[row, 2*i] = dx / dist
                J[row, 2*i + 1] = dy / dist
                J[row, 2*j] = -dx / dist
                J[row, 2*j + 1] = -dy / dist
                J[row, 2*n + i] = -1.0
                J[row, 2*n + j] = -1.0
                row += 1
            for (i, axis, side) in selected_boundary:
                if axis == 0:
                    if side == 0:
                        J[row, 2*i] = 1.0
                        J[row, 2*n + i] = -1.0
                    else:
                        J[row, 2*i] = 1.0
                        J[row, 2*n + i] = 1.0
                else:
                    if side == 0:
                        J[row, 2*i + 1] = 1.0
                        J[row, 2*n + i] = -1.0
                    else:
                        J[row, 2*i + 1] = 1.0
                        J[row, 2*n + i] = 1.0
                row += 1
            return J
        
        F = res_func(pos, radii)
        J = jac_func(pos, radii)
        
        # Solve J * delta = -F using least squares if overdetermined
        try:
            delta, residuals, rank, sv = np.linalg.lstsq(J, -F, rcond=None)
        except Exception:
            break
        
        if np.any(np.isnan(delta)) or np.any(np.isinf(delta)):
            break
        
        # Damped Newton step with line search
        alpha = 1.0
        current_norm = np.linalg.norm(F)
        
        for ls_iter in range(20):
            new_pos = pos + delta[:2*n].reshape(n, 2) * alpha
            new_radii = radii + delta[2*n:] * alpha
            
            # Clamp to valid range
            new_pos = np.clip(new_pos, 0.005, 0.995)
            new_radii = np.maximum(new_radii, 0.005)
            
            # Check if this is a valid configuration
            if not _is_valid(new_pos, new_radii, tol=1e-5):
                alpha *= 0.5
                continue
            
            new_F = res_func(new_pos, new_radii)
            new_norm = np.linalg.norm(new_F)
            
            if new_norm < current_norm * (1 - 1e-4 * alpha) or new_norm < 1e-10:
                pos = new_pos
                radii = new_radii
                break
            else:
                alpha *= 0.5
        
        # Update best
        pos_pp, radii_pp = _post_process(pos, radii)
        if _is_valid(pos_pp, radii_pp):
            score = np.sum(radii_pp)
            if score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
        
        # Check convergence
        if np.linalg.norm(F) < 1e-10:
            break
    
    return best_pos, best_radii

def _voronoi_optimize(pos, radii, deadline, max_iter=1000):
    """
    Voronoi-based optimization: use Delaunay triangulation to find
    the Voronoi diagram, then compute the inradius of each cell.
    Optimize positions to maximize the sum of inradii.
    """
    pos = pos.copy().astype(float)
    radii = radii.copy().astype(float)
    
    # Add boundary points to ensure all cells are bounded
    # Create a slightly larger square with corner points
    eps = 0.001
    boundary_points = np.array([
        [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0],
        [0.5, 0.0], [0.5, 1.0], [0.0, 0.5], [1.0, 0.5]
    ])
    
    all_points = np.vstack([pos, boundary_points])
    n_boundary = len(boundary_points)
    
    try:
        tri = Delaunay(all_points)
    except Exception:
        return pos, radii
    
    # Compute inradius for each circle's Voronoi cell
    # For each circle i, find its Voronoi cell and compute the inradius
    def compute_inradii(points, tri):
        n = len(points) - n_boundary
        inradii = np.zeros(n)
        
        for i in range(n):
            # Find the Voronoi cell of point i
            # The Voronoi cell is the intersection of half-planes
            # For simplicity, approximate by finding the minimum distance
            # to any other point or boundary, but use the Delaunay dual
            
            # Find all triangles that contain point i
            # The Voronoi cell vertices are circumcenters of triangles adjacent to i
            cell_vertices = []
            for simplex in tri.simplices:
                if i in simplex:
                    # Compute circumcenter of this triangle
                    p1 = points[simplex[0]]
                    p2 = points[simplex[1]]
                    p3 = points[simplex[2]]
                    
                    # Circumcenter formula
                    ax, ay = p1
                    bx, by = p2
                    cx, cy = p3
                    
                    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
                    if abs(d) < 1e-12:
                        continue
                    
                    ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay) + (cx**2 + cy**2) * (ay - by)) / d
                    uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx) + (cx**2 + cy**2) * (bx - ax)) / d
                    
                    cell_vertices.append([ux, uy])
            
            if len(cell_vertices) < 3:
                inradii[i] = 0.0
                continue
            
            cell_vertices = np.array(cell_vertices)
            
            # Compute the inradius of the polygon formed by cell_vertices
            # The inradius is the minimum distance from the center to any edge
            cx, cy = pos[i]
            
            # Find the minimum distance from (cx, cy) to any edge of the cell
            min_dist = float('inf')
            for k in range(len(cell_vertices)):
                v1 = cell_vertices[k]
                v2 = cell_vertices[(k + 1) % len(cell_vertices)]
                
                # Distance from point to line segment
                dx = v2[0] - v1[0]
                dy = v2[1] - v1[1]
                if dx == 0 and dy == 0:
                    continue
                
                t = ((cx - v1[0]) * dx + (cy - v1[1]) * dy) / (dx*dx + dy*dy)
                t = np.clip(t, 0, 1)
                proj_x = v1[0] + t * dx
                proj_y = v1[1] + t * dy
                dist = np.sqrt((cx - proj_x)**2 + (cy - proj_y)**2)
                min_dist = min(min_dist, dist)
            
            # Also consider boundary constraints
            min_dist = min(min_dist, cx, 1.0 - cx, cy, 1.0 - cy)
            
            inradii[i] = max(min_dist, 0.0)
        
        return inradii
    
    inradii = compute_inradii(all_points, tri)
    current_score = np.sum(inradii)
    
    # Optimize positions to maximize sum of inradii
    # Use gradient ascent with small steps
    rng = np.random.RandomState(42)
    
    for iteration in range(max_iter):
        if time.time() > deadline - 5:
            break
        
        # Compute gradient numerically
        grad = np.zeros((N, 2))
        step_size = 0.001
        
        for i in range(N):
            for d in range(2):
                pos_old = pos[i, d]
                pos[i, d] += step_size
                all_points_new = np.vstack([pos, boundary_points])
                try:
                    tri_new = Delaunay(all_points_new)
                    inradii_new = compute_inradii(all_points_new, tri_new)
                    score_new = np.sum(inradii_new)
                except Exception:
                    pos[i, d] = pos_old
                    continue
                pos[i, d] = pos_old
                
                grad[i, d] = (score_new - current_score) / step_size
        
        # Update positions with gradient ascent
        lr = 0.005
        pos += lr * grad
        
        # Clamp to valid range
        pos = np.clip(pos, 0.02, 0.98)
        
        # Recompute inradii
        all_points = np.vstack([pos, boundary_points])
        try:
            tri = Delaunay(all_points)
            inradii = compute_inradii(all_points, tri)
            new_score = np.sum(inradii)
        except Exception:
            break
        
        if new_score > current_score:
            current_score = new_score
            # Update radii to inradii
            radii = inradii.copy()
        else:
            # If no improvement, reduce learning rate
            lr *= 0.9
    
    return pos, radii

def _polish(pos, radii, deadline):
    """High-precision polishing with multiple optimization stages."""
    x = np.concatenate([pos.flatten(), radii])
    
    bounds = []
    for i in range(N):
        bounds.append((0.01, 0.99))
        bounds.append((0.01, 0.99))
    for i in range(N):
        bounds.append((0.01, 0.5))
    
    pos_out = pos.copy()
    radii_out = radii.copy()
    
    # Stage 1: SLSQP with explicit constraints (most reliable for feasibility)
    if time.time() < deadline - 15:
        pos_slsqp, radii_slsqp = _slsqp_polish(pos, radii, deadline, max_iter=300)
        pos_pp, radii_pp = _post_process(pos_slsqp, radii_slsqp)
        if _is_valid(pos_pp, radii_pp):
            pos_out, radii_out = pos_pp, radii_pp
    
    # Stage 2: L-BFGS-B with high penalty for further refinement
    if time.time() < deadline - 10:
        x = np.concatenate([pos_out.flatten(), radii_out])
        try:
            result = minimize(_objective, x, method='L-BFGS-B', bounds=bounds,
                              args=(5000.0,),
                              options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10})
            pos_opt = result.x[:2*N].reshape(N, 2)
            radii_opt = np.maximum(result.x[2*N:], 1e-4)
            pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
            if _is_valid(pos_pp, radii_pp) and np.sum(radii_pp) > np.sum(radii_out):
                pos_out, radii_out = pos_pp, radii_pp
        except Exception:
            pass
    
    # Stage 3: Newton contact graph optimization
    if time.time() < deadline - 8:
        pos_newton, radii_newton = _newton_contact_graph(pos_out, radii_out, deadline, max_iter=150)
        pos_pp, radii_pp = _post_process(pos_newton, radii_newton)
        if _is_valid(pos_pp, radii_pp) and np.sum(radii_pp) > np.sum(radii_out):
            pos_out, radii_out = pos_pp, radii_pp
    
    # Stage 4: Voronoi-based optimization
    if time.time() < deadline - 8:
        pos_voro, radii_voro = _voronoi_optimize(pos_out, radii_out, deadline, max_iter=300)
        pos_pp, radii_pp = _post_process(pos_voro, radii_voro)
        if _is_valid(pos_pp, radii_pp) and np.sum(radii_pp) > np.sum(radii_out):
            pos_out, radii_out = pos_pp, radii_pp
    
    # Stage 5: SLSQP with explicit constraints again (refinement)
    if time.time() < deadline - 8:
        pos_slsqp2, radii_slsqp2 = _slsqp_polish(pos_out, radii_out, deadline, max_iter=200)
        pos_pp2, radii_pp2 = _post_process(pos_slsqp2, radii_slsqp2)
        if _is_valid(pos_pp2, radii_pp2) and np.sum(radii_pp2) > np.sum(radii_out):
            pos_out, radii_out = pos_pp2, radii_pp2
    
    # Stage 6: Newton contact graph refinement
    if time.time() < deadline - 5:
        pos_newton2, radii_newton2 = _newton_contact_graph(pos_out, radii_out, deadline, max_iter=100)
        pos_pp4, radii_pp4 = _post_process(pos_newton2, radii_newton2)
        if _is_valid(pos_pp4, radii_pp4) and np.sum(radii_pp4) > np.sum(radii_out):
            pos_out, radii_out = pos_pp4, radii_pp4
    
    # Stage 7: Radius expansion
    if time.time() < deadline - 3:
        pos_exp, rad_exp = _try_radius_expansion(pos_out, radii_out, max_iter=100, step_size=0.002)
        pos_pp3, radii_pp3 = _post_process(pos_exp, rad_exp)
        if _is_valid(pos_pp3, radii_pp3):
            pos_out, radii_out = pos_pp3, radii_pp3
    
    return pos_out, radii_out

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Generate multiple initializations
    inits = []
    pos_bf, rad_bf = _initial_best_fit()
    inits.append((pos_bf, rad_bf))
    
    pos_hex, rad_hex = _initial_hexagonal()
    inits.append((pos_hex, rad_hex))
    
    pos_cb, rad_cb = _initial_corner_biased(seed=7)
    inits.append((pos_cb, rad_cb))
    
    for i in range(2):
        pos_r, rad_r = _initial_random(100 + i)
        inits.append((pos_r, rad_r))
    
    best_pos = pos_bf.copy()
    best_radii = rad_bf.copy()
    best_score = np.sum(best_radii)
    
    bounds = []
    for i in range(N):
        bounds.append((0.01, 0.99))
        bounds.append((0.01, 0.99))
    for i in range(N):
        bounds.append((0.01, 0.5))
        
    penalty_stages = [50.0, 500.0, 2000.0]
    
    for idx, (pos_init, rad_init) in enumerate(inits):
        if time.time() > deadline - 20:
            break
            
        x0 = np.concatenate([pos_init.flatten(), rad_init])
        current_x = x0.copy()
        
        # Stage 1: Standard objective
        for p_w in penalty_stages:
            if time.time() > deadline - 10:
                break
            try:
                result = minimize(_objective, current_x, method='L-BFGS-B', bounds=bounds,
                                  args=(p_w,),
                                  options={'maxiter': 800, 'ftol': 1e-12, 'gtol': 1e-8})
                if result.fun < _objective(current_x, p_w) - 1e-10:
                    current_x = result.x
            except Exception:
                pass
            
            pos_opt = current_x[:2*N].reshape(N, 2)
            radii_opt = np.maximum(current_x[2*N:], 1e-4)
            pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
            score = np.sum(radii_pp)
            if _is_valid(pos_pp, radii_pp) and score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
        
        # Stage 2: Smooth objective
        if time.time() < deadline - 10:
            for p_w in [500.0, 2000.0]:
                if time.time() > deadline - 5:
                    break
                try:
                    result = minimize(_objective_smooth, current_x, method='L-BFGS-B', bounds=bounds,
                                      args=(p_w, 30.0),
                                      options={'maxiter': 600, 'ftol': 1e-12, 'gtol': 1e-8})
                    if result.fun < _objective_smooth(current_x, p_w, 30.0) - 1e-10:
                        current_x = result.x
                except Exception:
                    pass
                
                pos_opt = current_x[:2*N].reshape(N, 2)
                radii_opt = np.maximum(current_x[2*N:], 1e-4)
                pos_pp, radii_pp = _post_process(pos_opt, radii_opt)
                score = np.sum(radii_pp)
                if _is_valid(pos_pp, radii_pp) and score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
        
        # Stage 3: Simulated annealing
        if time.time() < deadline - 8:
            try:
                pos_sa, rad_sa = _simulated_annealing(
                    current_x[:2*N].reshape(N, 2),
                    np.maximum(current_x[2*N:], 1e-4),
                    n_steps=2000, t0=0.03, t_end=0.0005,
                    seed=idx * 100
                )
                # Try radius expansion after SA
                pos_exp, rad_exp = _try_radius_expansion(pos_sa, rad_sa, max_iter=50, step_size=0.005)
                pos_pp, radii_pp = _post_process(pos_exp, rad_exp)
                score = np.sum(radii_pp)
                if _is_valid(pos_pp, radii_pp) and score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
            except Exception:
                pass
    
    # Final polish with multiple restarts
    if time.time() < deadline - 15:
        # Primary polish
        try:
            pos_pol, rad_pol = _polish(best_pos, best_radii, deadline)
            pos_pp, radii_pp = _post_process(pos_pol, rad_pol)
            score = np.sum(radii_pp)
            if _is_valid(pos_pp, radii_pp) and score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
        except Exception:
            pass
        
        # Secondary polish from slightly perturbed initial
        if time.time() < deadline - 8:
            rng = np.random.RandomState(42)
            pos_pert = best_pos + rng.normal(0, 0.005, (N, 2))
            pos_pert = np.clip(pos_pert, 0.01, 0.99)
            radii_pert = best_radii.copy()
            try:
                pos_pol2, rad_pol2 = _polish(pos_pert, radii_pert, deadline)
                pos_pp2, radii_pp2 = _post_process(pos_pol2, rad_pol2)
                score2 = np.sum(radii_pp2)
                if _is_valid(pos_pp2, radii_pp2) and score2 > best_score:
                    best_score = score2
                    best_pos = pos_pp2.copy()
                    best_radii = radii_pp2.copy()
            except Exception:
                pass
        
        # Tertiary polish from another perturbation
        if time.time() < deadline - 5:
            rng2 = np.random.RandomState(123)
            pos_pert2 = best_pos + rng2.normal(0, 0.01, (N, 2))
            pos_pert2 = np.clip(pos_pert2, 0.01, 0.99)
            radii_pert2 = best_radii * 0.95
            try:
                pos_pol3, rad_pol3 = _polish(pos_pert2, radii_pert2, deadline)
                pos_pp3, radii_pp3 = _post_process(pos_pol3, rad_pol3)
                score3 = np.sum(radii_pp3)
                if _is_valid(pos_pp3, radii_pp3) and score3 > best_score:
                    best_score = score3
                    best_pos = pos_pp3.copy()
                    best_radii = radii_pp3.copy()
            except Exception:
                pass
    
    # Final validation
    best_pos, best_radii = _post_process(best_pos, best_radii)
    
    # Final radius expansion attempt
    if time.time() < deadline - 1:
        pos_exp, rad_exp = _try_radius_expansion(best_pos, best_radii, max_iter=100, step_size=0.001)
        pos_pp, radii_pp = _post_process(pos_exp, rad_exp)
        if _is_valid(pos_pp, radii_pp) and np.sum(radii_pp) > best_score:
            best_pos = pos_pp
            best_radii = radii_pp
            best_score = np.sum(radii_pp)
            
    if not _is_valid(best_pos, best_radii):
        pos = np.zeros((N, 2))
        radii = np.zeros(N)
        k = 0
        for i in range(5):
            for j in range(5):
                if k < N - 1:
                    pos[k] = [(i + 0.5) / 5.0, (j + 0.5) / 5.0]
                    radii[k] = 0.095
                    k += 1
        pos[k] = [0.5, 0.02]
        radii[k] = 0.01
        best_pos, best_radii = pos, radii
        best_pos, best_radii = _post_process(best_pos, best_radii)
    
    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
