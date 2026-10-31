"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, linprog

def _softplus(x, beta=50.0):
    """Numerically stable softplus."""
    x = np.asarray(x, dtype=float)
    return np.where(x > 20.0/beta, x, np.log1p(np.exp(beta * x)) / beta)

def _energy(pos, beta=50.0):
    """
    Smooth energy: sum of soft violations.
    pos: (N, 2) array of circle centers.
    Returns (energy, grad) where grad is (N, 2).
    """
    n = len(pos)
    energy = 0.0
    grad = np.zeros((n, 2))
    
    # Boundary violations: circle must stay in [0,1]^2
    # For each circle, r is determined by LP later; here we use a target radius
    # Actually, for position optimization, we minimize overlap + boundary violations
    # using a fixed "effective" radius estimate.
    # Better: use a smooth penalty that encourages circles to be well-separated
    # and inside the square, then solve LP for radii.
    
    # Pairwise separation: penalize when circles are too close
    # Use a target minimum distance based on average radius
    # We'll use a smooth max(0, r_i + r_j - dist_ij) penalty
    # But we don't know r_i yet... 
    
    # Alternative approach: minimize sum of squared distances to boundaries
    # and maximize minimum pairwise distance.
    
    # Let's use: energy = sum_i [softplus(r_est - x_i) + softplus(r_est - y_i) 
    #                          + softplus(r_est - (1-x_i)) + softplus(r_est - (1-y_i))]
    #              + sum_{i<j} softplus(r_est - dist_ij)
    # where r_est is an estimated radius (e.g., 0.1 for 26 circles in unit square)
    
    # Actually, a cleaner approach: just minimize a smooth function that
    # penalizes boundary violations and close pairs, with a fixed scale.
    # Then solve LP for optimal radii given those positions.
    
    # Use a fixed reference radius for the energy landscape
    r_ref = 0.12  # typical radius for 26 circles
    
    # Boundary energy
    for i in range(n):
        x, y = pos[i]
        # x >= r_ref, y >= r_ref, x <= 1-r_ref, y <= 1-r_ref
        # softplus of violation (positive when violated)
        v1 = r_ref - x
        v2 = r_ref - y
        v3 = x - (1.0 - r_ref)
        v4 = y - (1.0 - r_ref)
        
        e_b = _softplus(v1, beta) + _softplus(v2, beta) + _softplus(v3, beta) + _softplus(v4, beta)
        energy += e_b
        
        # Gradients: d/dx softplus(v) = sigmoid(beta*v)
        s1 = 1.0 / (1.0 + np.exp(-beta * v1))
        s2 = 1.0 / (1.0 + np.exp(-beta * v2))
        s3 = 1.0 / (1.0 + np.exp(-beta * v3))
        s4 = 1.0 / (1.0 + np.exp(-beta * v4))
        
        grad[i, 0] += -s1 + s3
        grad[i, 1] += -s2 + s4
    
    # Pairwise energy
    for i in range(n):
        for j in range(i+1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx*dx + dy*dy) + 1e-10
            v = 2.0 * r_ref - dist
            e_p = _softplus(v, beta)
            energy += e_p
            
            s = 1.0 / (1.0 + np.exp(-beta * v))
            # d/dpos_i of softplus(2*r_ref - dist)
            # = s * (-d(dist)/dpos_i) = s * (dx/dist, dy/dist)
            grad[i, 0] += s * (dx / dist)
            grad[i, 1] += s * (dy / dist)
            grad[j, 0] += -s * (dx / dist)
            grad[j, 1] += -s * (dy / dist)
    
    return energy, grad

def _solve_lp_radii(pos, max_iter=200):
    """
    Given positions, solve LP to maximize sum of radii subject to:
    - r_i >= 0
    - r_i <= x_i, r_i <= y_i, r_i <= 1-x_i, r_i <= 1-y_i
    - r_i + r_j <= dist(i,j) for all pairs
    Returns array of radii (N,) or None if infeasible.
    """
    n = len(pos)
    # Variables: r_0, ..., r_{n-1}
    # Objective: maximize sum(r) => minimize -sum(r)
    c = -np.ones(n)
    
    # Inequality constraints: A_ub @ r <= b_ub
    # Boundary: for each i:
    #   -r_i <= -x_i  =>  r_i >= x_i  ... wait, we need r_i <= min(x_i, y_i, 1-x_i, 1-y_i)
    # So: r_i <= x_i, r_i <= y_i, r_i <= 1-x_i, r_i <= 1-y_i
    # In standard form A_ub @ r <= b_ub:
    #   r_i <= x_i  =>  [0..1..0] @ r <= x_i
    #   r_i <= y_i  =>  [0..1..0] @ r <= y_i
    #   r_i <= 1-x_i => [0..1..0] @ r <= 1-x_i
    #   r_i <= 1-y_i => [0..1..0] @ r <= 1-y_i
    
    A_ub = []
    b_ub = []
    
    for i in range(n):
        x, y = pos[i]
        # r_i <= min(x, y, 1-x, 1-y)
        # Add four constraints
        for bound_val in [x, y, 1.0-x, 1.0-y]:
            row = np.zeros(n)
            row[i] = 1.0
            A_ub.append(row)
            b_ub.append(bound_val)
    
    # Pairwise: r_i + r_j <= dist_ij
    for i in range(n):
        for j in range(i+1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(dist)
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    bounds = [(0, None)] * n
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            radii = np.maximum(res.x, 1e-8)
            return radii
        else:
            return None
    except Exception:
        return None

def _is_feasible(v, tol=1e-6):
    """Check if flat array (x,y,r)*N is feasible."""
    for i in range(N):
        x, y, r = v[3*i], v[3*i+1], v[3*i+2]
        if x < r - tol or y < r - tol or x + r > 1 + tol or y + r > 1 + tol:
            return False
    for i in range(N):
        for j in range(i+1, N):
            dx = v[3*i] - v[3*j]
            dy = v[3*i+1] - v[3*j+1]
            dist_sq = dx*dx + dy*dy
            r_sum = v[3*i+2] + v[3*j+2]
            if dist_sq < (r_sum - tol)**2:
                return False
    return True

def _format_result(pos, radii):
    result = np.zeros((N, 3))
    for i in range(N):
        result[i, 0] = pos[i, 0]
        result[i, 1] = pos[i, 1]
        result[i, 2] = radii[i]
    return result

def _initial_hexagonal():
    """Generate a hexagonal packing initial position."""
    pos = np.zeros((N, 2))
    # Use a hexagonal grid pattern
    row_height = 0.17
    col_width = 0.19
    row = 0
    col = 0
    idx = 0
    while idx < N and row < 10:
        offset = 0.0 if row % 2 == 0 else col_width / 2.0
        x = col_width / 2.0 + offset + col * col_width
        y = row_height / 2.0 + row * row_height
        if x < 1.0 and y < 1.0:
            pos[idx, 0] = x
            pos[idx, 1] = y
            idx += 1
        col += 1
        if col > 10:
            col = 0
            row += 1
    # Fill remaining with random
    while idx < N:
        pos[idx, 0] = np.random.uniform(0.1, 0.9)
        pos[idx, 1] = np.random.uniform(0.1, 0.9)
        idx += 1
    return pos

def _initial_grid():
    """Generate a grid initial position."""
    pos = np.zeros((N, 2))
    side = 6
    idx = 0
    for i in range(side):
        for j in range(side):
            if idx >= N:
                break
            pos[idx, 0] = (i + 0.5) / side
            pos[idx, 1] = (j + 0.5) / side
            idx += 1
    return pos

def _initial_corner_biased():
    """Place circles biased toward corners and edges for better packing."""
    pos = np.zeros((N, 2))
    # Corners first
    corners = [(0.12, 0.12), (0.88, 0.12), (0.12, 0.88), (0.88, 0.88)]
    for i, (cx, cy) in enumerate(corners):
        pos[i, 0] = cx
        pos[i, 1] = cy
    # Edge midpoints
    edges = [(0.5, 0.12), (0.12, 0.5), (0.88, 0.5), (0.5, 0.88)]
    for i, (cx, cy) in enumerate(edges):
        pos[4+i, 0] = cx
        pos[4+i, 1] = cy
    # Center region
    remaining = N - 8
    for i in range(remaining):
        angle = 2 * np.pi * i / remaining
        r = 0.25
        pos[8+i, 0] = 0.5 + r * np.cos(angle)
        pos[8+i, 1] = 0.5 + r * np.sin(angle)
    return pos

def _optimize_positions(pos_init, beta=50.0, maxiter=500, time_limit=30.0):
    """Optimize positions using L-BFGS-B on smooth energy."""
    start = time.time()
    
    def obj_and_grad(flat):
        pos = flat.reshape(N, 2)
        e, g = _energy(pos, beta)
        return e, g.flatten()
    
    bounds = [(0.0, 1.0)] * (2 * N)
    
    try:
        res = minimize(obj_and_grad, pos_init.flatten(), method='L-BFGS-B',
                       jac=True, bounds=bounds,
                       options={'maxiter': maxiter, 'ftol': 1e-8, 'gtol': 1e-6,
                                'maxfun': 5000})
        if res.success or res.status == 0:
            return res.x.reshape(N, 2)
        else:
            # Still return the result if it's better
            return res.x.reshape(N, 2)
    except Exception:
        return pos_init

def _anneal_positions(pos, beta_start=20.0, beta_end=100.0, steps=200, temp_start=0.05, temp_end=0.001):
    """Simulated annealing on positions."""
    current = pos.copy()
    current_energy, _ = _energy(current, beta_start)
    best = current.copy()
    best_energy = current_energy
    
    n_steps = steps
    for step in range(n_steps):
        if time.time() - start_global > deadline_global - 10:
            break
        t = step / max(n_steps - 1, 1)
        beta = beta_start + (beta_end - beta_start) * t
        temp = temp_start + (temp_end - temp_start) * t
        
        # Propose move: perturb a random circle
        new_pos = current.copy()
        i = np.random.randint(N)
        new_pos[i, 0] += np.random.normal(0, temp)
        new_pos[i, 1] += np.random.normal(0, temp)
        # Clip to [0,1]
        new_pos[i, 0] = np.clip(new_pos[i, 0], 0.0, 1.0)
        new_pos[i, 1] = np.clip(new_pos[i, 1], 0.0, 1.0)
        
        new_energy, _ = _energy(new_pos, beta)
        delta = new_energy - current_energy
        
        if delta < 0 or np.random.random() < np.exp(-delta / max(temp, 1e-10)):
            current = new_pos
            current_energy = new_energy
            if current_energy < best_energy:
                best = current.copy()
                best_energy = current_energy
    
    return best

def construct():
    global start_global, deadline_global
    start_global = time.time()
    deadline_global = start_global + 140.0
    
    np.random.seed(12345)
    
    best_pos = None
    best_radii = None
    best_value = 0.0
    
    # Try multiple initial configurations
    inits = [
        ("hex", _initial_hexagonal()),
        ("grid", _initial_grid()),
        ("corner", _initial_corner_biased()),
    ]
    
    for name, pos_init in inits:
        if time.time() > deadline_global - 20:
            break
        
        # Optimize positions
        pos_opt = _optimize_positions(pos_init, beta=50.0, maxiter=300, time_limit=30.0)
        
        # Anneal for refinement
        pos_anneal = _anneal_positions(pos_opt, beta_start=30.0, beta_end=80.0, steps=150, temp_start=0.03, temp_end=0.001)
        
        # Use whichever is better
        e_opt, _ = _energy(pos_opt, beta=50.0)
        e_ann, _ = _energy(pos_anneal, beta=50.0)
        pos_best = pos_opt if e_opt <= e_ann else pos_anneal
        
        # Solve LP for radii
        radii = _solve_lp_radii(pos_best)
        if radii is not None:
            val = np.sum(radii)
            # Verify feasibility
            v = np.zeros(N*3)
            for i in range(N):
                v[3*i] = pos_best[i, 0]
                v[3*i+1] = pos_best[i, 1]
                v[3*i+2] = radii[i]
            if _is_feasible(v, tol=1e-5):
                if val > best_value:
                    best_value = val
                    best_pos = pos_best.copy()
                    best_radii = radii.copy()
    
    # If no valid result, use fallback
    if best_pos is None:
        pos_init = _initial_grid()
        radii = _solve_lp_radii(pos_init)
        if radii is None:
            # Last resort: uniform small circles
            pos_init = _initial_grid()
            radii = np.full(N, 0.07)
            # Check feasibility, shrink if needed
            for _ in range(50):
                feasible = True
                for i in range(N):
                    x, y, r = pos_init[i,0], pos_init[i,1], radii[i]
                    if x < r or y < r or x+r > 1 or y+r > 1:
                        radii[i] = min(x, y, 1-x, 1-y) * 0.99
                        feasible = False
                for i in range(N):
                    for j in range(i+1, N):
                        dx = pos_init[i,0] - pos_init[j,0]
                        dy = pos_init[i,1] - pos_init[j,1]
                        dist = np.sqrt(dx*dx + dy*dy)
                        if dist < radii[i] + radii[j]:
                            scale = dist / (radii[i] + radii[j]) * 0.99
                            radii[i] *= scale
                            radii[j] *= scale
                            feasible = False
                if feasible:
                    break
            best_pos = pos_init
            best_radii = radii
            best_value = np.sum(radii)
    
    return _format_result(best_pos, best_radii)

# EVOLVE-BLOCK-END
