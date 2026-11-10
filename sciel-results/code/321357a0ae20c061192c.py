import numpy as np
import time
from scipy.optimize import linprog, minimize

def _is_valid_vectorized(centers, sides, tol=1e-7):
    """
    Vectorized validity check.
    centers: (N, 2)
    sides: (N,)
    """
    n = len(sides)
    if centers.shape != (n, 2) or sides.shape != (n,):
        return False
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if np.any(sides <= 0):
        return False
        
    # Boundary check
    if np.any(centers[:, 0] - sides/2 < -tol): return False
    if np.any(centers[:, 0] + sides/2 > 1 + tol): return False
    if np.any(centers[:, 1] - sides/2 < -tol): return False
    if np.any(centers[:, 1] + sides/2 > 1 + tol): return False
    
    # Overlap check
    dx = np.abs(centers[:, 0][:, np.newaxis] - centers[:, 0][np.newaxis, :])
    dy = np.abs(centers[:, 1][:, np.newaxis] - centers[:, 1][np.newaxis, :])
    sep = (sides[:, np.newaxis] + sides[np.newaxis, :]) / 2.0 - tol
    mask = ~np.eye(n, dtype=bool)
    
    if np.any((dx < sep) & (dy < sep) & mask):
        return False
        
    return True

def _initial_packing_grid():
    """
    Returns the known good packing with sum 3.0.
    2 squares of 0.5, 8 squares of 0.25.
    """
    centers = np.zeros((10, 2))
    sides = np.zeros(10)
    
    centers[0] = [0.25, 0.25]
    sides[0] = 0.5
    centers[1] = [0.75, 0.25]
    sides[1] = 0.5
    
    for k in range(8):
        row = k // 4
        col = k % 4
        cy = 0.625 + row * 0.25
        cx = 0.125 + col * 0.25
        centers[2+k] = [cx, cy]
        sides[2+k] = 0.25
        
    return centers, sides

def _get_topology(centers, sides):
    """
    Extracts a valid topology from a given packing.
    Returns a list of relations for each pair (i, j) with i < j.
    Relation: 0 (left), 1 (right), 2 (below), 3 (above).
    """
    n = len(sides)
    pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    relations = []
    for i, j in pairs:
        cx_i, cy_i = centers[i]
        cx_j, cy_j = centers[j]
        s_i, s_j = sides[i], sides[j]
        
        # Check which non-overlap condition is satisfied.
        # We need to pick one that is true.
        # If they touch, multiple might be true. We pick the first one found.
        
        # i left of j: cx_i + s_i/2 <= cx_j - s_j/2
        if cx_i + s_i/2 <= cx_j - s_j/2 + 1e-9:
            relations.append(0)
        # i right of j: cx_i - s_i/2 >= cx_j + s_j/2
        elif cx_i - s_i/2 >= cx_j + s_j/2 - 1e-9:
            relations.append(1)
        # i below j: cy_i + s_i/2 <= cy_j - s_j/2
        elif cy_i + s_i/2 <= cy_j - s_j/2 + 1e-9:
            relations.append(2)
        # i above j: cy_i - s_i/2 >= cy_j + s_j/2
        elif cy_i - s_i/2 >= cy_j + s_j/2 - 1e-9:
            relations.append(3)
        else:
            # Should not happen if packing is valid.
            # Default to left
            relations.append(0)
            
    return relations

def _solve_lp(topology, n=10, deadline=None):
    """
    Solves the LP for a given topology.
    Returns (centers, sides, value) or None if infeasible or timeout.
    """
    # Variables: cx_0, cy_0, s_0, cx_1, cy_1, s_1, ...
    # Index: 3*i + 0 for cx, 3*i + 1 for cy, 3*i + 2 for s
    
    # Objective: maximize sum(s_i) => minimize -sum(s_i)
    c = np.zeros(3*n)
    for i in range(n):
        c[3*i + 2] = -1.0
        
    # Bounds
    # cx, cy in [0, 1]
    # s in [0.01, 1.0] (0.01 to avoid 0)
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0)) # cx
        bounds.append((0.0, 1.0)) # cy
        bounds.append((0.01, 1.0)) # s
        
    # Constraints A_ub x <= b_ub
    
    A_ub_rows = []
    b_ub_vals = []
    
    # 1. Boundary constraints
    # cx_i - s_i/2 >= 0  => -cx_i + 0.5*s_i <= 0
    # 1 - cx_i - s_i/2 >= 0 => cx_i + 0.5*s_i <= 1
    # Same for y
    
    for i in range(n):
        # -cx_i + 0.5*s_i <= 0
        row = np.zeros(3*n)
        row[3*i] = -1.0
        row[3*i+2] = 0.5
        A_ub_rows.append(row)
        b_ub_vals.append(0.0)
        
        # cx_i + 0.5*s_i <= 1
        row = np.zeros(3*n)
        row[3*i] = 1.0
        row[3*i+2] = 0.5
        A_ub_rows.append(row)
        b_ub_vals.append(1.0)
        
        # -cy_i + 0.5*s_i <= 0
        row = np.zeros(3*n)
        row[3*i+1] = -1.0
        row[3*i+2] = 0.5
        A_ub_rows.append(row)
        b_ub_vals.append(0.0)
        
        # cy_i + 0.5*s_i <= 1
        row = np.zeros(3*n)
        row[3*i+1] = 1.0
        row[3*i+2] = 0.5
        A_ub_rows.append(row)
        b_ub_vals.append(1.0)
        
    # 2. Topology constraints
    pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    for k, (i, j) in enumerate(pairs):
        rel = topology[k]
        row = np.zeros(3*n)
        
        if rel == 0: # i left of j: cx_i + s_i/2 <= cx_j - s_j/2
            # cx_i - cx_j + 0.5*s_i + 0.5*s_j <= 0
            row[3*i] = 1.0
            row[3*j] = -1.0
            row[3*i+2] = 0.5
            row[3*j+2] = 0.5
            b_val = 0.0
        elif rel == 1: # i right of j: cx_i - s_i/2 >= cx_j + s_j/2
            # -cx_i + cx_j - 0.5*s_i - 0.5*s_j <= 0
            row[3*i] = -1.0
            row[3*j] = 1.0
            row[3*i+2] = -0.5
            row[3*j+2] = -0.5
            b_val = 0.0
        elif rel == 2: # i below j: cy_i + s_i/2 <= cy_j - s_j/2
            # cy_i - cy_j + 0.5*s_i + 0.5*s_j <= 0
            row[3*i+1] = 1.0
            row[3*j+1] = -1.0
            row[3*i+2] = 0.5
            row[3*j+2] = 0.5
            b_val = 0.0
        elif rel == 3: # i above j: cy_i - s_i/2 >= cy_j + s_j/2
            # -cy_i + cy_j - 0.5*s_i - 0.5*s_j <= 0
            row[3*i+1] = -1.0
            row[3*j+1] = 1.0
            row[3*i+2] = -0.5
            row[3*j+2] = -0.5
            b_val = 0.0
        else:
            continue
            
        A_ub_rows.append(row)
        b_ub_vals.append(b_val)
        
    if not A_ub_rows:
        A_ub = np.array([])
        b_ub = np.array([])
    else:
        A_ub = np.array(A_ub_rows)
        b_ub = np.array(b_ub_vals)
        
    # Solve
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    except Exception:
        return None
        
    if not res.success:
        return None
        
    x = res.x
    centers = np.zeros((n, 2))
    sides = np.zeros(n)
    for i in range(n):
        centers[i, 0] = x[3*i]
        centers[i, 1] = x[3*i+1]
        sides[i] = x[3*i+2]
        
    val = np.sum(sides)
    return centers, sides, val

def _nlp_optimize(centers_init, sides_init, n=10, max_iter=500):
    """
    Uses SLSQP to optimize the packing starting from an initial configuration.
    Returns (centers, sides, value) or None.
    """
    def objective_and_penalty(x):
        # x is flat array of length 3*n: [cx0, cy0, s0, cx1, cy1, s1, ...]
        centers = x[:2*n].reshape(n, 2)
        sides = x[2*n:]
        
        # Objective: -sum(sides)
        obj = -np.sum(sides)
        
        # Penalty for boundary violations
        pen = 0.0
        for i in range(n):
            if centers[i, 0] - sides[i]/2 < 0: pen += (centers[i, 0] - sides[i]/2)**2
            if centers[i, 0] + sides[i]/2 > 1: pen += (centers[i, 0] + sides[i]/2 - 1)**2
            if centers[i, 1] - sides[i]/2 < 0: pen += (centers[i, 1] - sides[i]/2)**2
            if centers[i, 1] + sides[i]/2 > 1: pen += (centers[i, 1] + sides[i]/2 - 1)**2
            if sides[i] <= 0: pen += sides[i]**2 + 1.0 # Ensure positive
            
        # Penalty for overlaps
        for i in range(n):
            for j in range(i+1, n):
                dx = np.abs(centers[i, 0] - centers[j, 0])
                dy = np.abs(centers[i, 1] - centers[j, 1])
                req_dx = (sides[i] + sides[j]) / 2.0
                req_dy = (sides[i] + sides[j]) / 2.0
                
                # If dx < req_dx and dy < req_dy, they overlap
                # Penalty is the square of the "penetration" depth
                # Approximation: if both are violated, penalty is proportional to min(violation_x, violation_y)
                # A simpler differentiable penalty:
                vx = max(0, req_dx - dx)
                vy = max(0, req_dy - dy)
                # Product is 0 if no overlap, positive if overlap
                pen += (vx * vy) ** 2
                
        return obj + 100.0 * pen

    x0 = np.zeros(3*n)
    for i in range(n):
        x0[3*i] = centers_init[i, 0]
        x0[3*i+1] = centers_init[i, 1]
        x0[3*i+2] = sides_init[i]
        
    # Bounds
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0)) # cx
        bounds.append((0.0, 1.0)) # cy
        bounds.append((0.01, 1.0)) # s
        
    try:
        res = minimize(objective_and_penalty, x0, method='SLSQP', bounds=bounds, options={'maxiter': max_iter, 'ftol': 1e-12})
    except Exception:
        return None
        
    if not res.success:
        return None
        
    x = res.x
    centers = x[:2*n].reshape(n, 2)
    sides = x[2*n:]
    
    # Check validity
    if not _is_valid_vectorized(centers, sides, tol=1e-7):
        return None
        
    val = np.sum(sides)
    return centers, sides, val

def squares_in_square10() -> tuple:
    n = 10
    tol = 1e-7
    start_time = time.time()
    deadline = start_time + 590 # Safety margin
    
    # Initial best
    best_centers, best_sides = _initial_packing_grid()
    best_val = np.sum(best_sides)
    
    # Get initial topology
    current_topology = _get_topology(best_centers, best_sides)
    
    # 1. LP refinement of initial topology
    res = _solve_lp(current_topology, n)
    if res:
        c_new, s_new, val_new = res
        if val_new > best_val:
            best_centers, best_sides, best_val = c_new, s_new, val_new
            current_topology = _get_topology(best_centers, best_sides)
            
    # 2. Local search on topologies + NLP exploration
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try flipping each pair's relation
        n_pairs = len(current_topology)
        for k in range(n_pairs):
            if time.time() > deadline:
                break
                
            original_rel = current_topology[k]
            for new_rel in [0, 1, 2, 3]:
                if new_rel == original_rel:
                    continue
                    
                # Try new topology with LP
                new_topology = current_topology.copy()
                new_topology[k] = new_rel
                
                res = _solve_lp(new_topology, n)
                if res is None:
                    continue
                    
                c_new, s_new, val_new = res
                
                if _is_valid_vectorized(c_new, s_new, tol=1e-7):
                    if val_new > best_val + 1e-9:
                        best_centers = c_new
                        best_sides = s_new
                        best_val = val_new
                        current_topology = new_topology
                        improved = True
                        break # Restart pair loop
            if improved:
                break
                
        # If LP local search didn't improve, try NLP from best solution
        if not improved and time.time() < deadline:
            # Try a few random perturbations of the best solution for NLP
            for _ in range(5):
                if time.time() > deadline:
                    break
                
                # Perturb centers and sides slightly
                noise_c = np.random.normal(0, 0.01, size=(n, 2))
                noise_s = np.random.normal(0, 0.01, size=n)
                c_pert = best_centers + noise_c
                s_pert = np.maximum(0.01, best_sides + noise_s)
                
                # Clip to bounds
                c_pert[:, 0] = np.clip(c_pert[:, 0], 0.0, 1.0)
                c_pert[:, 1] = np.clip(c_pert[:, 1], 0.0, 1.0)
                s_pert = np.clip(s_pert, 0.01, 1.0)
                
                res_nlp = _nlp_optimize(c_pert, s_pert, n, max_iter=200)
                if res_nlp:
                    c_nlp, s_nlp, val_nlp = res_nlp
                    if val_nlp > best_val + 1e-9:
                        best_centers = c_nlp
                        best_sides = s_nlp
                        best_val = val_nlp
                        # Extract new topology and continue LP search from here
                        current_topology = _get_topology(best_centers, best_sides)
                        improved = True
                        break
                        
    # Final check
    if not _is_valid_vectorized(best_centers, best_sides, tol=1e-7):
        best_centers, best_sides = _initial_packing_grid()
        
    return best_centers, best_sides

# EVOLVE-BLOCK-END
