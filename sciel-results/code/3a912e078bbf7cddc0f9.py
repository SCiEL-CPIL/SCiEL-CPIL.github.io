import numpy as np
import time
from scipy.optimize import linprog

N = 26
_DEADLINE = 145.0  # Safety margin from 150s limit

def is_valid(x, y, r, tol=1e-5):
    if np.any(r < 0): return False
    if np.any(x < r - tol) or np.any(x > 1 - r + tol): return False
    if np.any(y < r - tol) or np.any(y > 1 - r + tol): return False
    
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    min_dist = r[:, None] + r[None, :]
    if np.any(dist < min_dist - tol):
        return False
    return True

def solve_radii_lp(x, y, epsilon=1e-9):
    """
    Solves the Linear Programming problem to maximize sum(r)
    given fixed centers x, y.
    """
    n = len(x)
    # Objective: maximize sum(r) => minimize -sum(r)
    c = -np.ones(n)
    
    A_rows = []
    b_vals = []
    
    # Pairwise constraints: r_i + r_j <= d_ij - epsilon
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
            if d < epsilon:
                return None
            row = np.zeros(n)
            row[i] = 1
            row[j] = 1
            A_rows.append(row)
            b_vals.append(d - epsilon)
            
    # Boundary constraints: r_i <= min(x_i, 1-x_i, y_i, 1-y_i) - epsilon
    b_i = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
    for i in range(n):
        row = np.zeros(n)
        row[i] = 1
        A_rows.append(row)
        b_vals.append(b_i[i] - epsilon)
        
    A_ub = np.array(A_rows)
    b_ub = np.array(b_vals)
    bounds = [(0, None)] * n
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            r = res.x
            # Verify validity
            if is_valid(x, y, r):
                return r
    except Exception:
        pass
        
    return None

def compute_forces(x, y, r, alpha, beta):
    """
    Computes forces for x, y, r.
    Returns fx, fy, fr where we do x -= lr * fx (gradient descent on energy).
    Energy = -sum(r) + alpha * overlap_penalty + beta * bound_penalty
    """
    N = len(r)
    fx = np.zeros(N)
    fy = np.zeros(N)
    fr = np.zeros(N)
    
    # Gradient of -sum(r)
    fr -= 1.0
    
    # Overlap forces
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    min_dist = r[:, None] + r[None, :]
    
    dist_safe = np.where(dist > 1e-6, dist, 1e-6)
    
    overlap = np.maximum(0, min_dist - dist)
    np.fill_diagonal(overlap, 0)
    
    ux = dx / dist_safe
    uy = dy / dist_safe
    
    # Force on i from j is proportional to overlap * direction
    # Gradient of penalty term alpha * sum(overlap^2)
    # d/dx_i (overlap_ij^2) = 2 * overlap_ij * d(overlap_ij)/dx_i
    # overlap_ij = r_i + r_j - dist_ij
    # d(dist_ij)/dx_i = (x_i - x_j) / dist_ij = ux_ij
    # d(overlap_ij)/dx_i = -ux_ij
    # So term is 2 * overlap_ij * (-ux_ij)
    # We want to minimize energy, so force = -gradient? 
    # Wait, standard GD: x_new = x - lr * grad(energy)
    # grad_x(energy) = grad_x(alpha * sum overlap^2) = alpha * sum_j 2 * overlap_ij * (-ux_ij)
    # So fx = - grad_x(energy) = alpha * sum_j 2 * overlap_ij * ux_ij
    # This pushes i away from j (ux is direction from j to i).
    
    fx += 2.0 * alpha * np.sum(overlap * ux, axis=1)
    fy += 2.0 * alpha * np.sum(overlap * uy, axis=1)
    
    # Gradient w.r.t r_i
    # d(overlap_ij)/dr_i = 1
    # grad_r(energy) = alpha * sum_j 2 * overlap_ij * 1
    # fr = - grad_r(energy) = - alpha * sum_j 2 * overlap_ij
    fr -= 2.0 * alpha * np.sum(overlap, axis=1)
    
    # Boundary forces
    # bx1 = max(0, r - x)  (violation if r > x, i.e. x < r)
    # bx2 = max(0, r - (1-x)) (violation if r > 1-x, i.e. x > 1-r)
    # by1 = max(0, r - y)
    # by2 = max(0, r - (1-y))
    
    bx1 = np.maximum(0, r - x)
    bx2 = np.maximum(0, r - (1 - x))
    by1 = np.maximum(0, r - y)
    by2 = np.maximum(0, r - (1 - y))
    
    # Energy term: beta * (bx1^2 + bx2^2 + by1^2 + by2^2)
    # grad_x(bx1^2) = 2*bx1 * d(bx1)/dx = 2*bx1 * (-1) = -2*bx1
    # grad_x(bx2^2) = 2*bx2 * d(bx2)/dx = 2*bx2 * (1) = 2*bx2
    # grad_x(energy) = beta * (-2*bx1 + 2*bx2)
    # fx = - grad_x(energy) = beta * (2*bx1 - 2*bx2)
    fx += 2.0 * beta * (bx1 - bx2)
    
    # grad_y(by1^2) = 2*by1 * (-1) = -2*by1
    # grad_y(by2^2) = 2*by2 * (1) = 2*by2
    # fy = beta * (2*by1 - 2*by2)
    fy += 2.0 * beta * (by1 - by2)
    
    # grad_r(bx1^2) = 2*bx1 * 1 = 2*bx1
    # grad_r(bx2^2) = 2*bx2 * 1 = 2*bx2
    # ...
    # grad_r(energy) = beta * (2*bx1 + 2*bx2 + 2*by1 + 2*by2)
    # fr = - grad_r(energy) = - beta * (2*bx1 + 2*bx2 + 2*by1 + 2*by2)
    fr -= 2.0 * beta * (bx1 + bx2 + by1 + by2)
    
    return fx, fy, fr

def force_relaxation(x, y, r, start_time, deadline):
    """
    Force-directed relaxation with adaptive step size and LP refinement.
    Circles repel via overlap springs and are pushed toward boundaries.
    """
    best_score = 0.0
    best_config = None
    
    # Adaptive parameters
    alpha = 10.0  # Overlap penalty weight
    beta = 10.0   # Boundary penalty weight
    lr = 0.1      # Learning rate
    
    # Track for adaptive step
    prev_energy = None
    lr_decay = 0.995
    lr_min = 0.001
    
    n_steps = 2000
    
    for step in range(n_steps):
        if time.time() > deadline - 5:
            break
        
        # Compute forces (negative gradient)
        fx, fy, fr = compute_forces(x, y, r, alpha, beta)
        
        # Propose new state
        x_new = x + lr * fx
        y_new = y + lr * fy
        r_new = r + lr * fr
        
        # Clip to valid ranges
        x_new = np.clip(x_new, 0.001, 0.999)
        y_new = np.clip(y_new, 0.001, 0.999)
        r_new = np.maximum(r_new, 0.0005)
        
        # Calculate energy for adaptive step
        def calc_energy(x, y, r, a, b):
            e = -np.sum(r)
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            dist = np.sqrt(dx**2 + dy**2)
            min_dist = r[:, None] + r[None, :]
            overlap = np.maximum(0, min_dist - dist)
            np.fill_diagonal(overlap, 0)
            e += a * np.sum(overlap**2)
            
            bx1 = np.maximum(0, r - x)
            bx2 = np.maximum(0, r - (1 - x))
            by1 = np.maximum(0, r - y)
            by2 = np.maximum(0, r - (1 - y))
            e += b * (np.sum(bx1**2) + np.sum(bx2**2) + np.sum(by1**2) + np.sum(by2**2))
            return e
        
        E_new = calc_energy(x_new, y_new, r_new, alpha, beta)
        
        if prev_energy is not None:
            # Adaptive step: decrease lr if energy increases
            if E_new > prev_energy:
                lr *= lr_decay
                # Revert if too bad
                if E_new > prev_energy * 1.5:
                    x_new, y_new, r_new = x, y, r
                    E_new = prev_energy
            else:
                # Slightly increase lr if improving
                lr = min(lr * 1.01, 0.1)
        
        prev_energy = E_new
        x, y, r = x_new, y_new, r_new
        
        # Check validity and update best
        if is_valid(x, y, r):
            score = np.sum(r)
            if score > best_score:
                best_score = score
                best_config = (x.copy(), y.copy(), r.copy())
        
        # Periodically try LP refinement
        if step % 100 == 0 and step > 0:
            r_lp = solve_radii_lp(x, y)
            if r_lp is not None:
                if is_valid(x, y, r_lp):
                    score_lp = np.sum(r_lp)
                    if score_lp > best_score:
                        best_score = score_lp
                        best_config = (x.copy(), y.copy(), r_lp.copy())
        
        # Gradually increase penalty weights
        if step % 200 == 0 and step > 0:
            alpha *= 1.2
            beta *= 1.2
    
    return best_score, best_config

def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE  # Safety margin
    
    best_score = 0.0
    best_config = None
    
    n_restarts = 30
    
    for restart in range(n_restarts):
        if time.time() > deadline - 15:
            break
        
        # Initialization strategies
        if restart % 4 == 0:
            # Grid init
            side = int(np.ceil(np.sqrt(N)))
            xs = []
            ys = []
            for i in range(side):
                for j in range(side):
                    if len(xs) < N:
                        xs.append((i + 0.5) / side)
                        ys.append((j + 0.5) / side)
            x = np.array(xs[:N])
            y = np.array(ys[:N])
        elif restart % 4 == 1:
            # Random init
            x = np.random.uniform(0.1, 0.9, N)
            y = np.random.uniform(0.1, 0.9, N)
        elif restart % 4 == 2:
            # Perturbed grid
            side = int(np.ceil(np.sqrt(N)))
            xs = []
            ys = []
            for i in range(side):
                for j in range(side):
                    if len(xs) < N:
                        xs.append((i + 0.5) / side)
                        ys.append((j + 0.5) / side)
            x = np.array(xs[:N])
            y = np.array(ys[:N])
            x += np.random.normal(0, 0.05, N)
            y += np.random.normal(0, 0.05, N)
            x = np.clip(x, 0.05, 0.95)
            y = np.clip(y, 0.05, 0.95)
        else:
            # Hexagonal-like init
            rows = 5
            cols = 6
            xs = []
            ys = []
            for i in range(rows):
                for j in range(cols):
                    if len(xs) < N:
                        xs.append((j + 0.5 + (0.5 if i % 2 else 0)) / cols)
                        ys.append((i + 0.5) / rows)
            x = np.array(xs[:N])
            y = np.array(ys[:N])
            x += np.random.normal(0, 0.02, N)
            y += np.random.normal(0, 0.02, N)
            x = np.clip(x, 0.05, 0.95)
            y = np.clip(y, 0.05, 0.95)
            
        # Initial radii: small
        r = np.full(N, 0.03)
        
        # Run force relaxation
        score, config = force_relaxation(x, y, r, start_time, deadline)
        
        if config is not None:
            if score > best_score:
                best_score = score
                best_config = config
                # Try LP refinement
                bx, by, br = config
                r_lp = solve_radii_lp(bx, by)
                if r_lp is not None:
                    score_lp = np.sum(r_lp)
                    if score_lp > best_score:
                        best_score = score_lp
                        best_config = (bx.copy(), by.copy(), r_lp.copy())

    # Fallback
    if best_config is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    bx, by, br = best_config
    return np.column_stack((bx, by, br))
