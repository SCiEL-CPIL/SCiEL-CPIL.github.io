import numpy as np
import time
from scipy.optimize import minimize, linprog
import itertools

N = 26

def _is_valid(x, y, r, tol=1e-9):
    """Check if circles are within bounds and disjoint."""
    if np.any(r < 0):
        return False
    # Bounds check
    if np.any(x < r - tol) or np.any(x > 1 - r + tol):
        return False
    if np.any(y < r - tol) or np.any(y > 1 - r + tol):
        return False
    # Overlap check
    for i in range(N):
        for j in range(i + 1, N):
            dist_sq = (x[i] - x[j])**2 + (y[i] - y[j])**2
            rad_sum = r[i] + r[j]
            if dist_sq < rad_sum**2 - tol:
                return False
    return True

def _energy(x, y, r, beta=1.0):
    """
    Energy function for Simulated Annealing.
    We want to maximize sum(r), so we minimize -sum(r) + penalties.
    Penalties are soft constraints.
    """
    energy = -np.sum(r)
    
    # Boundary penalties
    # x < r -> r - x > 0
    # x > 1-r -> x - (1-r) > 0
    viol = np.maximum(r - x, 0) + np.maximum(x - (1 - r), 0)
    viol += np.maximum(r - y, 0) + np.maximum(y - (1 - r), 0)
    energy += beta * np.sum(viol**2)
    
    # Overlap penalties
    # dist < ri + rj -> (ri + rj - dist) > 0
    for i in range(N):
        for j in range(i + 1, N):
            dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
            overlap = r[i] + r[j] - dist
            if overlap > 0:
                energy += beta * overlap**2
                
    return energy

def _objective_smooth(v, lam_ov=50.0, lam_box=50.0, delta=1e-4):
    """
    Smooth objective for L-BFGS-B.
    v: [x0, y0, r0, x1, y1, r1, ...]
    """
    x = v[0::3]
    y = v[1::3]
    r = v[2::3]
    
    loss = -np.sum(r)
    
    # Boundary
    diff_l = r - x
    diff_r = x + r - 1
    diff_b = r - y
    diff_t = y + r - 1
    
    for diff in [diff_l, diff_r, diff_b, diff_t]:
        # Softplus squared approximation for max(0, diff)^2
        # Using a smooth approximation: (0.5 * (diff + sqrt(diff^2 + delta^2)))^2
        # Actually, simpler: just use penalty on violation magnitude
        # P = lam * (max(0, diff))^2
        # Smooth max: 0.5 * (diff + sqrt(diff^2 + delta^2))
        # Let's use the standard smooth max penalty
        smooth_max = 0.5 * (diff + np.sqrt(diff**2 + delta**2))
        # We only want to penalize positive violations, but smooth_max is always positive.
        # If diff is large negative, smooth_max ~ 0.
        # If diff is large positive, smooth_max ~ diff.
        # This works.
        loss += lam_box * np.sum(smooth_max**2)
        
    # Overlap
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx**2 + dy**2 + delta**2)
            overlap = r[i] + r[j] - dist
            smooth_max = 0.5 * (overlap + np.sqrt(overlap**2 + delta**2))
            loss += lam_ov * smooth_max**2
            
    return loss

def _grad_smooth(v, lam_ov=50.0, lam_box=50.0, delta=1e-4):
    """
    Gradient of the smooth objective.
    """
    x = v[0::3]
    y = v[1::3]
    r = v[2::3]
    
    grad = np.zeros_like(v)
    # Base gradient from -sum(r)
    grad[2::3] = -1.0
    
    # Boundary gradients
    # f(d) = 0.5 * (d + sqrt(d^2 + delta^2))
    # f'(d) = 0.5 * (1 + d/sqrt(d^2 + delta^2))
    # Penalty term: lam * f(d)^2
    # dP/dd = 2 * lam * f(d) * f'(d)
    
    def add_boundary_grad(grad_idx_var, diff, sign):
        # sign: +1 if diff = var - const, -1 if diff = const - var
        # Here we define diff such that positive diff is violation.
        # L: diff = r - x. d(diff)/dx = -1, d(diff)/dr = 1
        # R: diff = x + r - 1. d(diff)/dx = 1, d(diff)/dr = 1
        # B: diff = r - y. d(diff)/dy = -1, d(diff)/dr = 1
        # T: diff = y + r - 1. d(diff)/dy = 1, d(diff)/dr = 1
        
        s = np.sqrt(diff**2 + delta**2)
        f = 0.5 * (diff + s)
        fp = 0.5 * (1 + diff / s)
        dP_dd = 2 * lam_box * f * fp
        
        # Update gradients
        # dP/dvar = dP_dd * d(diff)/dvar
        
        # This function is a bit abstract, let's do it explicitly for clarity and speed
        
    # Explicit boundary updates
    diff_l = r - x
    s_l = np.sqrt(diff_l**2 + delta**2)
    f_l = 0.5 * (diff_l + s_l)
    fp_l = 0.5 * (1 + diff_l / s_l)
    dP_dd_l = 2 * lam_box * f_l * fp_l
    grad[2::3] += dP_dd_l # dr
    grad[0::3] -= dP_dd_l # dx

    diff_r = x + r - 1
    s_r = np.sqrt(diff_r**2 + delta**2)
    f_r = 0.5 * (diff_r + s_r)
    fp_r = 0.5 * (1 + diff_r / s_r)
    dP_dd_r = 2 * lam_box * f_r * fp_r
    grad[2::3] += dP_dd_r # dr
    grad[0::3] += dP_dd_r # dx

    diff_b = r - y
    s_b = np.sqrt(diff_b**2 + delta**2)
    f_b = 0.5 * (diff_b + s_b)
    fp_b = 0.5 * (1 + diff_b / s_b)
    dP_dd_b = 2 * lam_box * f_b * fp_b
    grad[2::3] += dP_dd_b # dr
    grad[1::3] -= dP_dd_b # dy

    diff_t = y + r - 1
    s_t = np.sqrt(diff_t**2 + delta**2)
    f_t = 0.5 * (diff_t + s_t)
    fp_t = 0.5 * (1 + diff_t / s_t)
    dP_dd_t = 2 * lam_box * f_t * fp_t
    grad[2::3] += dP_dd_t # dr
    grad[1::3] += dP_dd_t # dy
    
    # Overlap gradients
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx**2 + dy**2 + delta**2
            dist = np.sqrt(dist_sq)
            overlap = r[i] + r[j] - dist
            
            s = np.sqrt(overlap**2 + delta**2)
            f = 0.5 * (overlap + s)
            fp = 0.5 * (1 + overlap / s)
            dP_dd = 2 * lam_ov * f * fp
            
            # d(overlap)/dr_i = 1, d(overlap)/dr_j = 1
            grad[2*i] += dP_dd
            grad[2*j] += dP_dd
            
            # d(overlap)/dx_i = -dx/dist, d(overlap)/dx_j = dx/dist
            # d(overlap)/dy_i = -dy/dist, d(overlap)/dy_j = dy/dist
            gx = dx / dist
            gy = dy / dist
            
            grad[i] -= dP_dd * gx
            grad[j] += dP_dd * gx
            grad[i+1] -= dP_dd * gy
            grad[j+1] += dP_dd * gy
            
    return grad

def _maximize_radii_lp(x, y):
    """
    Maximize sum(r) subject to:
    r_i + r_j <= dist_ij
    r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i
    r_i >= 0
    """
    # Variables: r_0, ..., r_{N-1}
    c = -np.ones(N) # Minimize -sum(r)
    
    # Inequality constraints A_ub @ r <= b_ub
    A_ub = []
    b_ub = []
    
    # Overlap constraints
    for i in range(N):
        for j in range(i + 1, N):
            dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
            row = np.zeros(N)
            row[i] = 1
            row[j] = 1
            A_ub.append(row)
            b_ub.append(dist)
            
    # Boundary constraints
    for i in range(N):
        for bound in [x[i], 1 - x[i], y[i], 1 - y[i]]:
            row = np.zeros(N)
            row[i] = 1
            A_ub.append(row)
            b_ub.append(bound)
            
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    bounds = [(0, None)] * N
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return res.x
        else:
            # Fallback: small radii
            return np.full(N, 0.01)
    except Exception:
        return np.full(N, 0.01)

def construct():
    start_time = time.time()
    time_limit = 140.0 # Safety margin
    
    best_score = -np.inf
    best_result = None
    
    # 1. Greedy Initialization
    # Place circles one by one to maximize distance to existing circles and boundaries.
    # This creates a "Voronoi-like" distribution.
    rng = np.random.default_rng(42)
    
    def greedy_init():
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        
        # Start with one circle in center
        x[0] = 0.5
        y[0] = 0.5
        r[0] = 0.2 # Initial guess
        
        for k in range(1, N):
            # Candidate points: grid + random
            candidates_x = np.linspace(0.1, 0.9, 20)
            candidates_y = np.linspace(0.1, 0.9, 20)
            cx, cy = np.meshgrid(candidates_x, candidates_y)
            cx = cx.flatten()
            cy = cy.flatten()
            
            # Add random points
            cx = np.append(cx, rng.uniform(0.05, 0.95, 200))
            cy = np.append(cy, rng.uniform(0.05, 0.95, 200))
            
            best_x = 0.5
            best_y = 0.5
            best_min_dist = -1
            
            for px, py in zip(cx, cy):
                # Distance to boundary
                d_bound = min(px, 1-px, py, 1-py)
                
                # Distance to existing circles
                min_d_circ = np.inf
                for i in range(k):
                    d = np.sqrt((px - x[i])**2 + (py - y[i])**2)
                    # We want center to be far from center of existing circle
                    # But we also need space for radius.
                    # Simple heuristic: maximize distance to centers
                    min_d_circ = min(min_d_circ, d)
                
                # Score: balance boundary and circle distance
                # We want to be far from boundaries AND far from other circles
                score = min(d_bound, min_d_circ)
                
                if score > best_min_dist:
                    best_min_dist = score
                    best_x = px
                    best_y = py
                    
            x[k] = best_x
            y[k] = best_y
            # Estimate radius as half the minimum distance to constraint
            r[k] = best_min_dist * 0.8
            
        return x, y, r

    x0, y0, r0 = greedy_init()
    
    # 2. Simulated Annealing
    # We will run SA for a fixed number of steps or time.
    # State: x, y, r
    # Move: perturb one coordinate of one circle
    
    x, y, r = x0.copy(), y0.copy(), r0.copy()
    current_energy = _energy(x, y, r, beta=10.0)
    best_energy = current_energy
    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
    
    T = 0.1
    cooling = 0.999
    
    # Determine steps based on time
    # SA is fast, we can do many steps
    max_steps = 20000
    
    for step in range(max_steps):
        if time.time() - start_time > time_limit * 0.5:
            break
            
        # Pick a random circle and coordinate
        i = rng.integers(0, N)
        coord = rng.integers(0, 3)
        
        old_val = x[i] if coord == 0 else (y[i] if coord == 1 else r[i])
        
        # Perturbation size
        step_size = 0.05 * T + 0.001
        delta = rng.normal(0, step_size)
        
        new_val = old_val + delta
        
        # Bounds for x, y: [0, 1], r: [0, 0.5]
        if coord == 0:
            new_val = np.clip(new_val, 0.0, 1.0)
            x_new = x.copy(); x_new[i] = new_val
        elif coord == 1:
            new_val = np.clip(new_val, 0.0, 1.0)
            y_new = y.copy(); y_new[i] = new_val
        else:
            new_val = np.clip(new_val, 0.0, 0.5)
            r_new = r.copy(); r_new[i] = new_val
            
        # Calculate new energy
        # To save time, we could do incremental energy, but N=26 is small.
        if coord == 0:
            e_new = _energy(x_new, y, r, beta=10.0)
        elif coord == 1:
            e_new = _energy(x, y_new, r, beta=10.0)
        else:
            e_new = _energy(x, y, r_new, beta=10.0)
            
        delta_e = e_new - current_energy
        if delta_e < 0 or rng.random() < np.exp(-delta_e / T):
            # Accept
            if coord == 0:
                x = x_new
            elif coord == 1:
                y = y_new
            else:
                r = r_new
            current_energy = e_new
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_x = x.copy()
                best_y = y.copy()
                best_r = r.copy()
                
        T *= cooling
        
        # Check validity of best solution periodically
        if step % 1000 == 0:
            if _is_valid(best_x, best_y, best_r):
                score = np.sum(best_r)
                if score > best_score:
                    best_score = score
                    best_result = np.column_stack((best_x, best_y, best_r))

    # Use the best SA result as start for polishing
    if best_result is not None:
        x, y, r = best_result[:, 0], best_result[:, 1], best_result[:, 2]
    else:
        x, y, r = best_x, best_y, best_r
        
    # 3. Gradient Polishing (L-BFGS-B)
    # We use the smooth penalty objective
    v0 = np.zeros(3*N)
    v0[0::3] = x
    v0[1::3] = y
    v0[2::3] = r
    
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 0.5))
        
    try:
        res = minimize(_objective_smooth, v0, jac=_grad_smooth, method='L-BFGS-B', 
                       bounds=bounds, options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
        v_res = res.x
        x_res = v_res[0::3]
        y_res = v_res[1::3]
        r_res = v_res[2::3]
        
        if _is_valid(x_res, y_res, r_res):
            score = np.sum(r_res)
            if score > best_score:
                best_score = score
                best_result = np.column_stack((x_res, y_res, r_res))
    except Exception:
        pass
        
    # 4. Final Radius Maximization (LP)
    # Use the positions from the best valid result found so far
    if best_result is not None:
        x_final = best_result[:, 0]
        y_final = best_result[:, 1]
        r_final = _maximize_radii_lp(x_final, y_final)
        
        # Check if LP result is valid (it should be by construction, but check for numerical issues)
        if _is_valid(x_final, y_final, r_final):
            score = np.sum(r_final)
            if score > best_score:
                best_score = score
                best_result = np.column_stack((x_final, y_final, r_final))
    else:
        # Fallback if no valid result found
        # Create a grid
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) < N:
                    cx = (i + 0.5) / side
                    cy = (j + 0.5) / side
                    cr = 0.5 / side * 0.9
                    cells.append([cx, cy, cr])
        best_result = np.array(cells)
        
    return best_result

if __name__ == "__main__":
    result = construct()
    print(result.shape)
    print(np.sum(result[:, 2]))
