"""
Strategy: Topological Graph-Based Packing Optimization.

This approach differs from the previous random-walk local search by:
1. Explicitly identifying the "contact graph" (pairs of circles at minimum distance 
   and circles touching walls).
2. Formulating the problem as a constraint satisfaction / optimization problem where 
   we maximize the radius subject to non-overlap and boundary constraints.
3. Using a gradient-based optimizer (SLSQP) on a reduced set of variables or with 
   tight constraints to push the packing into a "rigid" configuration where many 
   circles are in contact. This allows the optimizer to explore the configuration 
   space of rigid packings, which is often where global optima reside.
4. Iteratively perturbing the contact graph (by forcing some contacts to break or 
   new ones to form) to escape local minima.
"""

import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import distance

N = 50

# EVOLVE-BLOCK-START

def _score(cents):
    """Compute the packing radius."""
    n = len(cents)
    # Wall distances
    wall_min = np.min(np.minimum(0.5 - np.abs(cents[:, 0]), 0.5 - np.abs(cents[:, 1])))
    # Pairwise distances
    if n > 1:
        diff = cents[:, np.newaxis, :] - cents[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        pair_min = np.min(dists) / 2.0
    else:
        pair_min = 0.5
    return min(wall_min, pair_min)

def _get_contact_graph(cents, r, tol=1e-4):
    """
    Identify pairs of circles that are in contact (distance approx 2r)
    and circles touching the wall (distance to wall approx r).
    Returns:
        pair_contacts: list of (i, j) indices
        wall_contacts: list of (i, wall_index) where wall is 0:Left, 1:Right, 2:Bottom, 3:Top
    """
    n = len(cents)
    pair_contacts = []
    wall_contacts = []
    
    # Pairwise
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(cents[i] - cents[j])
            if abs(d - 2*r) < tol:
                pair_contacts.append((i, j))
                
    # Wall
    for i in range(n):
        x, y = cents[i]
        if abs(x - (-0.5 + r)) < tol: wall_contacts.append((i, 0)) # Left
        if abs(x - (0.5 - r)) < tol: wall_contacts.append((i, 1))  # Right
        if abs(y - (-0.5 + r)) < tol: wall_contacts.append((i, 2)) # Bottom
        if abs(y - (0.5 - r)) < tol: wall_contacts.append((i, 3))  # Top
        
    return pair_contacts, wall_contacts

def _objective_and_grad(params, n, fixed_indices=None):
    """
    Objective: Minimize -r (maximize r) subject to constraints.
    Here we use a soft penalty approach for optimization stability, 
    or better, we fix r and check feasibility, then binary search for max r?
    Actually, SLSQP can handle variable r directly.
    
    Let's define variables as positions of all N circles (2N vars) + r (1 var).
    Total 2N+1 vars.
    
    Constraints:
    1. |c_i - c_j| >= 2r  => |c_i - c_j|^2 - (2r)^2 >= 0
    2. -0.5 + r <= c_ix <= 0.5 - r
    3. -0.5 + r <= c_iy <= 0.5 - r
    """
    cents = params[:2*n].reshape(n, 2)
    r = params[2*n]
    
    # Objective: Minimize -r
    obj = -r
    
    # Gradient
    grad = np.zeros_like(params)
    grad[2*n] = -1.0
    
    return obj, grad

def _constraints(params, n):
    """
    Returns list of constraint dicts for scipy minimize.
    Inequality constraints: g(x) >= 0
    """
    cents = params[:2*n].reshape(n, 2)
    r = params[2*n]
    
    cons = []
    
    # Pairwise non-overlap
    # For performance, we might only enforce constraints on "nearby" pairs 
    # or all pairs. N=50 is small enough for all pairs (1225 pairs).
    for i in range(n):
        for j in range(i+1, n):
            def c_func(p, i=i, j=j):
                ci = p[2*i:2*i+2]
                cj = p[2*j:2*j+2]
                r_val = p[2*n]
                return np.dot(ci-cj, ci-cj) - 4*r_val*r_val
            def c_grad(p, i=i, j=j):
                g = np.zeros_like(p)
                ci = p[2*i:2*i+2]
                cj = p[2*j:2*j+2]
                r_val = p[2*n]
                diff = ci - cj
                g[2*i:2*i+2] = 2*diff
                g[2*j:2*j+2] = -2*diff
                g[2*n] = -8*r_val
                return g
            cons.append({'type': 'ineq', 'fun': c_func, 'jac': c_grad})
            
    # Wall constraints
    # Left: x_i >= -0.5 + r  => x_i - r + 0.5 >= 0
    # Right: x_i <= 0.5 - r  => 0.5 - r - x_i >= 0
    # Bottom: y_i >= -0.5 + r => y_i - r + 0.5 >= 0
    # Top: y_i <= 0.5 - r    => 0.5 - r - y_i >= 0
    
    for i in range(n):
        idx_x = 2*i
        idx_y = 2*i + 1
        idx_r = 2*n
        
        # Left
        def c_left(p, ix=idx_x):
            return p[ix] - p[idx_r] + 0.5
        def g_left(p, ix=idx_x):
            g = np.zeros_like(p)
            g[ix] = 1.0
            g[idx_r] = -1.0
            return g
        cons.append({'type': 'ineq', 'fun': c_left, 'jac': g_left})
        
        # Right
        def c_right(p, ix=idx_x):
            return 0.5 - p[idx_r] - p[ix]
        def g_right(p, ix=idx_x):
            g = np.zeros_like(p)
            g[ix] = -1.0
            g[idx_r] = -1.0
            return g
        cons.append({'type': 'ineq', 'fun': c_right, 'jac': g_right})
        
        # Bottom
        def c_bot(p, iy=idx_y):
            return p[iy] - p[idx_r] + 0.5
        def g_bot(p, iy=idx_y):
            g = np.zeros_like(p)
            g[iy] = 1.0
            g[idx_r] = -1.0
            return g
        cons.append({'type': 'ineq', 'fun': c_bot, 'jac': g_bot})
        
        # Top
        def c_top(p, iy=idx_y):
            return 0.5 - p[idx_r] - p[iy]
        def g_top(p, iy=idx_y):
            g = np.zeros_like(p)
            g[iy] = -1.0
            g[idx_r] = -1.0
            return g
        cons.append({'type': 'ineq', 'fun': c_top, 'jac': g_top})
        
    return cons

def _construct_hexagonal():
    """Create a good initial hexagonal packing."""
    best_cents = None
    best_r = 0
    
    for n_rows in range(5, 9):
        base = N // n_rows
        extra = N % n_rows
        row_counts = [base + (1 if i < extra else 0) for i in range(n_rows)]
        
        max_rc = max(row_counts)
        # Estimate r
        # Horizontal fit: (max_rc - 1) * 2r + 2r <= 1 -> r <= 1 / (2*max_rc)
        # Vertical fit: (n_rows - 1) * sqrt(3)*r + 2r <= 1 -> r <= 1 / (2 + (n_rows-1)*sqrt(3))
        r_horiz = 1.0 / (2.0 * max_rc)
        r_vert = 1.0 / (2.0 + (n_rows - 1) * np.sqrt(3.0))
        r = min(r_horiz, r_vert)
        
        if r <= best_r:
            continue
            
        centers = []
        total_y = (n_rows - 1) * np.sqrt(3.0) * r
        y_start = -0.5 + r + total_y / 2.0
        
        for row_idx in range(n_rows):
            rc = row_counts[row_idx]
            y = y_start + row_idx * np.sqrt(3.0) * r
            x_span = (rc - 1) * 2 * r
            x_start = -0.5 + r + x_span / 2.0
            offset = np.sqrt(3.0) * r * 0.5 * (row_idx % 2)
            for col in range(rc):
                x = x_start + col * 2 * r + offset
                centers.append([x, y])
                
        centers = np.array(centers[:N])
        r_actual = _score(centers)
        
        if r_actual > best_r:
            best_r = r_actual
            best_cents = centers.copy()
            
    if best_cents is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_cents = grid[:N]
        best_r = _score(best_cents)
        
    return best_cents, best_r

def _optimize_with_graph(cents, time_limit=20.0):
    """
    Optimize the packing using SLSQP, potentially fixing some circles 
    based on a contact graph to reduce dimensionality or guide search.
    For now, we use full optimization but with a good start.
    """
    start_time = time.time()
    n = len(cents)
    
    # Initial parameters
    x0 = np.zeros(2*n + 1)
    x0[:2*n] = cents.flatten()
    # Estimate initial r from score
    r0 = _score(cents) * 0.99 # Start slightly smaller to be feasible
    x0[2*n] = r0
    
    cons = _constraints(x0, n)
    
    # Bounds for positions and r
    # r > 0, positions in [-0.5, 0.5]
    bounds = [(None, None)] * (2*n) + [(0, 0.5)]
    
    try:
        # SLSQP is slow but effective for small N with tight constraints
        # Max iterations to stay within time limit
        res = minimize(
            lambda p: _objective_and_grad(p, n)[0], 
            x0, 
            jac=lambda p: _objective_and_grad(p, n)[1],
            method='SLSQP', 
            bounds=bounds, 
            constraints=cons,
            options={'maxiter': 200, 'ftol': 1e-12, 'disp': False}
        )
        
        if res.success:
            new_cents = res.x[:2*n].reshape(n, 2)
            new_r = res.x[2*n]
            # Validate
            actual_r = _score(new_cents)
            if actual_r > r0:
                return new_cents, actual_r
    except Exception:
        pass
        
    return cents, r0

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    # 1. Initial Hexagonal Packing
    cents, r0 = _construct_hexagonal()
    best_cents = cents.copy()
    best_r = r0
    
    # 2. Global Optimization Phase
    # We try to optimize the whole set. 
    # Since SLSQP can be slow, we limit it.
    remaining = deadline - (time.time() - start_time)
    if remaining > 10:
        try:
            opt_cents, opt_r = _optimize_with_graph(best_cents, time_limit=min(remaining - 5, 60))
            if opt_r > best_r:
                best_r = opt_r
                best_cents = opt_cents
        except Exception:
            pass

    # 3. Local Refinement / Perturbation Phase
    # If we have time, do a few rounds of random restarts or perturbations
    # on the best solution found, re-optimizing locally.
    remaining = deadline - (time.time() - start_time)
    rng = np.random.default_rng(42)
    
    while remaining > 10:
        # Perturb current best
        perturbation = rng.normal(0, 0.001, size=best_cents.shape)
        new_cents = best_cents + perturbation
        # Clip to square
        new_cents[:, 0] = np.clip(new_cents[:, 0], -0.5, 0.5)
        new_cents[:, 1] = np.clip(new_cents[:, 1], -0.5, 0.5)
        
        # Quick local optimization on perturbed start
        sub_deadline = time.time() + min(remaining - 5, 10)
        try:
            opt_cents, opt_r = _optimize_with_graph(new_cents, time_limit=sub_deadline - time.time())
            if opt_r > best_r:
                best_r = opt_r
                best_cents = opt_cents
        except Exception:
            pass
            
        remaining = deadline - (time.time() - start_time)
        
    return best_cents

# EVOLVE-BLOCK-END
