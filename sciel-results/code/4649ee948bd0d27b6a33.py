"""
Pack 50 equal circles in a unit square, maximize common radius.
Strategy: Delaunay-based Contact Graph Identification + SQP Equilibrium Solver.
This approach treats the packing as a rigid body mechanics problem.
It iteratively:
1. Identifies the contact graph (kissing graph) using Delaunay triangulation 
   on the current centers.
2. Solves for the equilibrium configuration where active contacts are exactly 
   at distance 2r and walls are exactly at distance r, using scipy's SLSQP 
   as an SQP-like solver for the constrained problem.
3. Expands r slightly and repeats, or performs a final polish.
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 50
DEADLINE = 145.0  # Safety margin before 150s kill

def _radius(centers):
    """Compute the max common radius for given centers (exact, for scoring)."""
    if not np.all(np.isfinite(centers)):
        return 0.0
    wall_r = np.min(0.5 - np.abs(centers))
    if wall_r < 0:
        return 0.0
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist)
    return min(wall_r, min_pair / 2.0)

def _get_contact_graph(centers, r, tolerance=0.01):
    """
    Identify the contact graph using Delaunay triangulation.
    A pair (i, j) is considered a potential contact if they are neighbors in
    the Delaunay triangulation and their distance is close to 2r.
    Also include wall contacts.
    """
    # Delaunay triangulation
    try:
        tri = Delaunay(centers)
    except Exception:
        # Fallback: simple distance-based graph if Delaunay fails (e.g., collinear)
        tri = None
    
    contacts = set()
    
    if tri is not None:
        for simplex in tri.simplices:
            for i in range(len(simplex)):
                for j in range(i+1, len(simplex)):
                    idx_i, idx_j = simplex[i], simplex[j]
                    d = np.linalg.norm(centers[idx_i] - centers[idx_j])
                    # If distance is close to 2r, it's a contact
                    if d < 2*r + tolerance:
                        contacts.add((min(idx_i, idx_j), max(idx_i, idx_j)))
    
    # Wall contacts
    # A circle touches a wall if its distance to the wall is <= r + tolerance
    # and it's the closest to that wall among its neighbors? 
    # For SQP, we just need to constrain the active ones.
    # We'll handle walls in the constraints directly based on current proximity.
    
    return contacts

def _solve_equilibrium(centers, r, time_limit=10.0):
    """
    Solve for the configuration where the active contacts (identified from 
    the current state) are exactly at distance 2r, and walls are respected.
    We maximize r locally by adjusting centers.
    Actually, for a fixed r, we just need to find a valid configuration.
    But we want to maximize r. 
    Strategy: Fix r, find a configuration where all pairs in the contact graph
    are exactly 2r apart, and no pair is closer.
    This is an equality constrained problem for the contact graph.
    """
    start_time = time.time()
    x0 = centers.ravel()
    
    # Identify active contacts
    contact_set = _get_contact_graph(centers, r, tolerance=0.05)
    
    # Convert to list of indices for constraints
    contact_pairs = list(contact_set)
    
    # Objective: Minimize 0 (feasibility problem) or minimize sum of squared violations?
    # To maximize r, we usually increase r until infeasible.
    # Here, we solve for centers given r.
    # If we find a valid center, we can increase r.
    
    # Constraints:
    # 1. For each contact pair (i, j): ||c_i - c_j||^2 == (2r)^2
    # 2. For each center: -0.5 + r <= c_ik <= 0.5 - r
    
    # SLSQP can handle equality constraints.
    
    def objective(x):
        return 0.0
    
    def gradient(x):
        return np.zeros_like(x)
    
    def constraint_func(x):
        c = x.reshape(N, 2)
        vals = []
        for i, j in contact_pairs:
            d2 = np.sum((c[i] - c[j])**2)
            vals.append(d2 - (2*r)**2)
        return np.array(vals)
    
    def constraint_jacobian(x):
        c = x.reshape(N, 2)
        m = len(contact_pairs)
        jac = np.zeros((m, 2*N))
        for k, (i, j) in enumerate(contact_pairs):
            diff = c[i] - c[j]
            # d/dx_i (||c_i - c_j||^2) = 2 * diff
            # d/dx_j (||c_i - c_j||^2) = -2 * diff
            jac[k, 2*i:2*i+2] = 2 * diff
            jac[k, 2*j:2*j+2] = -2 * diff
        return jac
    
    # Bounds
    bounds = []
    for k in range(N):
        bounds.append((-0.5 + r, 0.5 - r))
        bounds.append((-0.5 + r, 0.5 - r))
    
    # Inequality constraints: non-contact pairs must be >= 2r
    # This is tricky for SLSQP with many pairs. 
    # We rely on the contact graph being the "bottleneck".
    # If a non-contact pair violates, the radius is actually smaller than r.
    # We check this after solving.
    
    # To make it more robust, we can add inequality constraints for the 
    # "near" pairs that are not in the contact graph but close.
    # Let's stick to equality constraints for the contact graph first.
    
    cons = [{
        'type': 'eq',
        'fun': constraint_func,
        'jac': constraint_jacobian
    }]
    
    # Add inequality constraints for all pairs to ensure no overlap?
    # Too many constraints (1225 pairs). SLSQP might struggle.
    # Alternative: Only add constraints for pairs closer than 2.5 * 2r?
    # Let's try without explicit inequality constraints for non-contacts, 
    # relying on the initial guess being close and the contact graph being correct.
    # If it fails, we fall back to a softer method.
    
    try:
        res = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 100, 'ftol': 1e-9, 'disp': False}
        )
        if res.success and time.time() - start_time < time_limit:
            return res.x.reshape(N, 2)
    except Exception:
        pass
    
    # If SQP fails, try a softer approach: minimize sum of squared violations
    # for the contact graph, with penalties for overlaps.
    def soft_objective(x):
        c = x.reshape(N, 2)
        val = 0.0
        for i, j in contact_pairs:
            d2 = np.sum((c[i] - c[j])**2)
            val += (d2 - (2*r)**2)**2
        # Penalty for any overlap
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(c[i] - c[j])
                if d < 2*r:
                    val += (2*r - d)**2 * 10
        # Penalty for going out of bounds
        for k in range(N):
            for axis in range(2):
                if c[k, axis] < -0.5 + r:
                    val += (-0.5 + r - c[k, axis])**2 * 10
                if c[k, axis] > 0.5 - r:
                    val += (c[k, axis] - (0.5 - r))**2 * 10
        return val
    
    try:
        res2 = minimize(
            soft_objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-10, 'disp': False}
        )
        if res2.success and time.time() - start_time < time_limit:
            return res2.x.reshape(N, 2)
    except Exception:
        pass
    
    return centers

def _expand_radius_sq(centers, r_init, r_max=0.075, dr=0.0002, time_limit=30.0):
    """
    Expand radius using SQP equilibrium solver.
    """
    start_time = time.time()
    centers = centers.copy()
    r = r_init
    
    while r < r_max:
        if time.time() - start_time > time_limit:
            break
            
        # Solve for equilibrium at current r
        new_centers = _solve_equilibrium(centers, r, time_limit=min(5.0, time_limit - (time.time() - start_time)))
        
        # Check if the solution is valid (no overlaps, in bounds)
        r_check = _radius(new_centers)
        
        if r_check >= r - 1e-9:
            # Success, we can try to increase r
            centers = new_centers
            r += dr
        else:
            # Failed to find valid configuration at this r
            # Try to repair by slightly decreasing r and re-solving?
            # Or just break.
            # Let's try a small repair step
            for _ in range(5):
                r -= dr * 0.5
                new_centers = _solve_equilibrium(centers, r, time_limit=2.0)
                r_check = _radius(new_centers)
                if r_check >= r - 1e-9:
                    centers = new_centers
                    break
            else:
                break
    
    return centers, _radius(centers)

def _make_grid_rect():
    """Create a 7x8 rectangular grid (56 points), take first 50."""
    nx, ny = 7, 8
    xs = np.linspace(-0.5 + 0.5/nx, 0.5 - 0.5/nx, nx)
    ys = np.linspace(-0.5 + 0.5/ny, 0.5 - 0.5/ny, ny)
    grid = np.array([(x, y) for y in ys for x in xs])[:N]
    return grid

def _make_hexagonal_grid():
    """Create a hexagonal packing of 50 points."""
    spacing = 0.142
    dy = spacing * np.sqrt(3) / 2
    points = []
    for j in range(12):
        for i in range(12):
            x = i * spacing - 0.5
            y = j * dy - 0.5
            if j % 2 != 0:
                x += spacing / 2
            points.append((x, y))
    points = np.array(points)[:N]
    points -= np.mean(points, axis=0)
    max_ext = np.max(np.abs(points))
    if max_ext > 0.45:
        points *= (0.45 / max_ext)
    return points

def construct():
    start_time = time.time()
    
    best_centers = None
    best_r = 0.0
    
    # Phase 1: Generate initializations
    inits = []
    inits.append(_make_grid_rect())
    inits.append(_make_hexagonal_grid())
    
    np.random.seed(42)
    for _ in range(2):
        g = _make_grid_rect().copy()
        g += np.random.randn(N, 2) * 0.01
        g = np.clip(g, -0.45, 0.45)
        inits.append(g)
    
    # Phase 2: Radius expansion with SQP
    expand_budget = 0.6 * DEADLINE
    per_expand = expand_budget / len(inits)
    
    for i, init in enumerate(inits):
        time_left = start_time + DEADLINE - time.time()
        if time_left < 5:
            break
        alloc = min(per_expand, time_left - 5)
        if alloc < 5:
            break
        
        expanded, r_exp = _expand_radius_sq(init, r_init=0.06, r_max=0.075, dr=0.0002, time_limit=alloc)
        r = _radius(expanded)
        if r > best_r:
            best_r = r
            best_centers = expanded
    
    if best_centers is None:
        best_centers = inits[0]
        best_r = _radius(best_centers)
    
    # Phase 3: Final Polish
    # Use a generic SLSQP on the soft-min objective to polish the best candidate
    time_left = DEADLINE - 5 - (time.time() - start_time)
    if time_left > 10:
        x0 = best_centers.ravel()
        
        def soft_min_radius(x, beta=100.0):
            c = x.reshape(N, 2)
            c1 = 0.5 - c[:, 0]
            c2 = 0.5 + c[:, 0]
            c3 = 0.5 - c[:, 1]
            c4 = 0.5 + c[:, 1]
            diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
            dist = np.sqrt(np.sum(diff ** 2, axis=2))
            iu = np.triu_indices(N, 1)
            pair_dists = dist[iu] / 2.0
            all_c = np.concatenate([c1, c2, c3, c4, pair_dists])
            y = -beta * all_c
            max_y = np.max(y)
            sum_exp = np.sum(np.exp(y - max_y))
            log_sum_exp = max_y + np.log(sum_exp)
            return -log_sum_exp / beta
        
        def objective(x):
            return -soft_min_radius(x)
            
        try:
            res = minimize(
                objective, x0,
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': 200, 'maxfun': 1000, 'ftol': 1e-12, 'gtol': 1e-10, 'disp': False}
            )
            if res.x is not None:
                polished = res.x.reshape(N, 2)
                r_polished = _radius(polished)
                if r_polished > best_r:
                    best_r = r_polished
                    best_centers = polished
        except Exception:
            pass
    
    # Final safety
    best_centers = np.clip(best_centers, -0.5, 0.5)
    r_final = _radius(best_centers)
    if r_final <= 0 or not np.all(np.isfinite(best_centers)):
        best_centers = _make_grid_rect()
        
    return best_centers
