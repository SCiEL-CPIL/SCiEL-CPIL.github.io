"""Circle packing in unit square - 26 circles maximizing sum of radii.
Strategy: Simulated annealing over a smooth energy (negative sum radii + soft
collision/boundary penalties), then SLSQP polish on the best SA result."""

import numpy as np
import time
import random
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

def _is_valid(circles, tol=1e-7):
    """Check validity: inside unit square, non-overlapping, positive radii."""
    n = len(circles)
    for i in range(n):
        x, y, r = circles[i]
        if r < -tol:
            return False
        if x - r < -tol or x + r > 1.0 + tol:
            return False
        if y - r < -tol or y + r > 1.0 + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi, ri = circles[i]
            xj, yj, rj = circles[j]
            dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
            if dist < ri + rj - tol:
                return False
    return True


def _post_process(circles, tol=1e-7):
    """Shrink circles slightly to ensure strict validity, clip to bounds."""
    out = []
    for x, y, r in circles:
        r = max(0.0, r - tol)
        x = min(1.0 - r, max(r, x))
        y = min(1.0 - r, max(r, y))
        out.append((x, y, r))
    return out


def _sum_radii(circles):
    return sum(c[2] for c in circles)


def _softplus(x, beta=100.0):
    """Numerically stable softplus: log(1 + exp(beta*x))/beta."""
    x = np.asarray(x, dtype=float)
    bx = beta * x
    result = np.zeros_like(x, dtype=float)
    mask_large = bx > 50
    mask_small = bx < -50
    mask_mid = ~(mask_large | mask_small)
    result[mask_large] = x[mask_large]
    result[mask_mid] = np.log1p(np.exp(bx[mask_mid])) / beta
    # mask_small: result stays 0
    return result


def _energy(params, beta=50.0, penalty_weight=10.0):
    """
    Energy = -sum(radii) + penalty_weight * (collision_penalty + boundary_penalty)
    params: array of shape (N, 3) with columns (x, y, r)
    """
    x = params[:, 0]
    y = params[:, 1]
    r = params[:, 2]
    
    # Objective: minimize negative sum of radii
    obj = -np.sum(r)
    
    # Collision penalty: for each pair, penalize if dist < ri + rj
    coll_pen = 0.0
    n = len(r)
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy)
            overlap = (r[i] + r[j]) - dist
            if overlap > 0:
                coll_pen += _softplus(overlap, beta)
    
    # Boundary penalty: penalize if circle extends outside [0,1]^2
    bound_pen = 0.0
    for i in range(n):
        # left
        if x[i] - r[i] < 0:
            bound_pen += _softplus(-(x[i] - r[i]), beta)
        # right
        if x[i] + r[i] > 1.0:
            bound_pen += _softplus((x[i] + r[i]) - 1.0, beta)
        # bottom
        if y[i] - r[i] < 0:
            bound_pen += _softplus(-(y[i] - r[i]), beta)
        # top
        if y[i] + r[i] > 1.0:
            bound_pen += _softplus((y[i] + r[i]) - 1.0, beta)
    
    total_penalty = penalty_weight * (coll_pen + bound_pen)
    return obj + total_penalty


def _init_hexagonal():
    """Hexagonal packing initial layout for 26 circles."""
    rows_counts = [5, 5, 5, 5, 6]
    circles = []
    r = 0.08
    row_spacing = r * np.sqrt(3)
    col_spacing = 2 * r
    
    row_idx = 0
    for count in rows_counts:
        offset = 0 if row_idx % 2 == 0 else col_spacing / 2
        total_width = col_spacing * (count - 1) + offset
        start_x = (1.0 - total_width) / 2.0
        y = r + row_spacing * row_idx + 0.02
        for col in range(count):
            x = start_x + offset + col * col_spacing
            if len(circles) < N:
                circles.append([x, y, r])
        row_idx += 1
    
    circles = circles[:N]
    while len(circles) < N:
        circles.append([0.5, 0.5, 0.01])
    return np.array(circles, dtype=float)


def _init_grid():
    """Simple grid initialization."""
    side = int(np.ceil(np.sqrt(N)))
    circles = []
    for k in range(N):
        i = k % side
        j = k // side
        x = (i + 0.5) / side
        y = (j + 0.5) / side
        r = 0.45 / side
        circles.append([x, y, r])
    return np.array(circles, dtype=float)


def _init_corners():
    """Corner-biased initialization: place circles near corners and center."""
    circles = []
    # 4 corners
    corners = [(0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9)]
    for cx, cy in corners:
        circles.append([cx, cy, 0.06])
    # Edge midpoints
    edges = [(0.5, 0.08), (0.5, 0.92), (0.08, 0.5), (0.92, 0.5)]
    for ex, ey in edges:
        circles.append([ex, ey, 0.05])
    # Center and inner
    inner = [(0.5, 0.5), (0.3, 0.3), (0.7, 0.3), (0.3, 0.7), (0.7, 0.7),
             (0.5, 0.3), (0.5, 0.7), (0.3, 0.5), (0.7, 0.5),
             (0.4, 0.4), (0.6, 0.4), (0.4, 0.6), (0.6, 0.6),
             (0.35, 0.65), (0.65, 0.35), (0.35, 0.35), (0.65, 0.65)]
    for ix, iy in inner:
        if len(circles) < N:
            circles.append([ix, iy, 0.04])
    circles = circles[:N]
    while len(circles) < N:
        circles.append([0.5, 0.5, 0.01])
    return np.array(circles, dtype=float)


def _init_random(seed=None):
    """Random initialization."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    circles = []
    for _ in range(N):
        x = random.uniform(0.1, 0.9)
        y = random.uniform(0.1, 0.9)
        r = random.uniform(0.02, 0.08)
        circles.append([x, y, r])
    return np.array(circles, dtype=float)


def _simulated_annealing(params, deadline, n_restarts=3):
    """
    Simulated annealing over the smooth energy landscape.
    Returns best params found.
    """
    best_params = params.copy()
    best_energy = _energy(params)
    
    for restart in range(n_restarts):
        if time.time() > deadline - 15.0:
            break
        
        # Reset or perturb current best
        if restart == 0:
            current = params.copy()
        else:
            current = best_params + np.random.normal(0, 0.02, size=best_params.shape)
            current[:, 0] = np.clip(current[:, 0], 0, 1)
            current[:, 1] = np.clip(current[:, 1], 0, 1)
            current[:, 2] = np.clip(current[:, 2], 0.005, 0.4)
        
        current_energy = _energy(current)
        
        # Annealing schedule
        T_start = 0.05
        T_end = 0.001
        n_steps = 3000
        
        for step in range(n_steps):
            if time.time() > deadline - 10.0:
                break
            
            T = T_start * (T_end / T_start) ** (step / n_steps)
            
            # Generate perturbation
            # Move a random circle or adjust radius
            idx = np.random.randint(0, N)
            move_type = np.random.random()
            
            new_params = current.copy()
            if move_type < 0.4:
                # Move position
                new_params[idx, 0] += np.random.normal(0, 0.02)
                new_params[idx, 1] += np.random.normal(0, 0.02)
            elif move_type < 0.7:
                # Adjust radius
                new_params[idx, 2] += np.random.normal(0, 0.01)
            else:
                # Move and adjust
                new_params[idx, 0] += np.random.normal(0, 0.015)
                new_params[idx, 1] += np.random.normal(0, 0.015)
                new_params[idx, 2] += np.random.normal(0, 0.008)
            
            # Clamp
            new_params[:, 0] = np.clip(new_params[:, 0], 0, 1)
            new_params[:, 1] = np.clip(new_params[:, 1], 0, 1)
            new_params[:, 2] = np.clip(new_params[:, 2], 0.005, 0.4)
            
            new_energy = _energy(new_params)
            delta_e = new_energy - current_energy
            
            if delta_e < 0 or np.random.random() < np.exp(-delta_e / T):
                current = new_params
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_params = current.copy()
    
    return best_params, best_energy


def _slsqp_polish(circles, deadline):
    """SLSQP polish with pairwise constraints."""
    def params_to_circles(params):
        return [(params[3*i], params[3*i+1], params[3*i+2]) for i in range(N)]
    
    def objective(params):
        return -sum(params[3*i+2] for i in range(N))
    
    def _jac_overlap(p, i, j):
        jac = np.zeros(3 * N)
        xi, yi, ri = p[3*i], p[3*i+1], p[3*i+2]
        xj, yj, rj = p[3*j], p[3*j+1], p[3*j+2]
        jac[3*i] = 2 * (xi - xj)
        jac[3*j] = -2 * (xi - xj)
        jac[3*i+1] = 2 * (yi - yj)
        jac[3*j+1] = -2 * (yi - yj)
        jac[3*i+2] = -2 * (ri + rj)
        jac[3*j+2] = -2 * (ri + rj)
        return jac
    
    cons = []
    for i in range(N):
        cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i] - p[3*i+2], 
                     'jac': lambda p, i=i: np.array([1.0 if k==3*i else (-1.0 if k==3*i+2 else 0.0) for k in range(3*N)])})
        cons.append({'type': 'ineq', 'fun': lambda p, i=i: 1.0 - p[3*i] - p[3*i+2], 
                     'jac': lambda p, i=i: np.array([(-1.0 if k==3*i else (-1.0 if k==3*i+2 else 0.0)) for k in range(3*N)])})
        cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i+1] - p[3*i+2], 
                     'jac': lambda p, i=i: np.array([1.0 if k==3*i+1 else (-1.0 if k==3*i+2 else 0.0) for k in range(3*N)])})
        cons.append({'type': 'ineq', 'fun': lambda p, i=i: 1.0 - p[3*i+1] - p[3*i+2], 
                     'jac': lambda p, i=i: np.array([(-1.0 if k==3*i+1 else (-1.0 if k==3*i+2 else 0.0)) for k in range(3*N)])})
        cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i+2], 
                     'jac': lambda p, i=i: np.array([1.0 if k==3*i+2 else 0.0 for k in range(3*N)])})
    
    for i in range(N):
        for j in range(i + 1, N):
            cons.append({
                'type': 'ineq',
                'fun': lambda p, i=i, j=j: (p[3*i]-p[3*j])**2 + (p[3*i+1]-p[3*j+1])**2 - (p[3*i+2]+p[3*j+2])**2,
                'jac': lambda p, i=i, j=j: _jac_overlap(p, i, j)
            })
    
    params = np.array([val for circle in circles for val in circle])
    bounds = [(0, 1), (0, 1), (0, 0.5)] * N
    
    try:
        result = minimize(
            objective, params, method='SLSQP',
            bounds=bounds, constraints=cons,
            options={'maxiter': 100, 'ftol': 1e-10}
        )
        if result.success:
            new_circles = params_to_circles(result.x)
            if _is_valid(new_circles):
                return new_circles
    except Exception:
        pass
    return circles


def _local_search(circles, deadline, max_iters=100):
    """Local search: try moving/growing individual circles."""
    circles = [list(c) for c in circles]
    best_sum = _sum_radii(circles)
    
    for it in range(max_iters):
        if time.time() > deadline - 10.0:
            break
        improved = False
        for i in range(N):
            x, y, r = circles[i]
            perturbations = []
            for dx in [-0.02, -0.01, 0, 0.01, 0.02]:
                for dy in [-0.02, -0.01, 0, 0.01, 0.02]:
                    for dr in [-0.01, 0.005, 0.01, 0.02]:
                        if dx == 0 and dy == 0 and dr == 0:
                            continue
                        perturbations.append((dx, dy, dr))
            random.shuffle(perturbations)
            
            for dx, dy, dr in perturbations:
                new_x = x + dx
                new_y = y + dy
                new_r = max(0.005, r + dr)
                if new_x - new_r < -1e-9 or new_x + new_r > 1 + 1e-9:
                    continue
                if new_y - new_r < -1e-9 or new_y + new_r > 1 + 1e-9:
                    continue
                valid = True
                for j in range(N):
                    if j == i:
                        continue
                    xj, yj, rj = circles[j]
                    dist = np.sqrt((new_x - xj)**2 + (new_y - yj)**2)
                    if dist < new_r + rj - 1e-9:
                        valid = False
                        break
                if not valid:
                    continue
                new_sum = best_sum - r + new_r
                if new_sum > best_sum + 1e-12:
                    circles[i] = [new_x, new_y, new_r]
                    best_sum = new_sum
                    improved = True
                    break
        if not improved:
            break
    
    return [tuple(c) for c in circles], best_sum


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    np.random.seed(42)
    random.seed(42)
    
    # Multiple diverse initializations
    inits = [
        _init_hexagonal(),
        _init_grid(),
        _init_corners(),
        _init_random(42),
        _init_random(123),
        _init_random(777),
    ]
    
    best_params = inits[0].copy()
    best_sum = _sum_radii(best_params.tolist())
    
    for init_params in inits:
        if time.time() > deadline - 30.0:
            break
        
        # Simulated annealing
        sa_params, sa_energy = _simulated_annealing(init_params, deadline, n_restarts=2)
        
        # Convert to list of tuples
        sa_circles = [tuple(sa_params[i]) for i in range(N)]
        sa_sum = _sum_radii(sa_circles)
        
        # SLSQP polish
        if time.time() < deadline - 20.0:
            polished = _slsqp_polish(sa_circles, deadline)
            polished_sum = _sum_radii(polished)
            if polished_sum > sa_sum and _is_valid(polished):
                sa_circles = polished
                sa_sum = polished_sum
        
        # Local search
        if time.time() < deadline - 15.0:
            refined, refined_sum = _local_search(sa_circles, deadline, max_iters=80)
            if refined_sum > sa_sum and _is_valid(refined):
                sa_circles = refined
                sa_sum = refined_sum
        
        if sa_sum > best_sum and _is_valid(sa_circles):
            best_circles = sa_circles
            best_sum = sa_sum
    
    # Final polish on best
    if time.time() < deadline - 10.0:
        polished = _slsqp_polish(best_circles, deadline)
        polished_sum = _sum_radii(polished)
        if polished_sum > best_sum and _is_valid(polished):
            best_circles = polished
            best_sum = polished_sum
    
    # Final local search
    if time.time() < deadline - 5.0:
        refined, refined_sum = _local_search(best_circles, deadline, max_iters=50)
        if refined_sum > best_sum and _is_valid(refined):
            best_circles = refined
            best_sum = refined_sum
    
    # Post-process
    best_circles = _post_process(best_circles)
    
    # Verify and return
    if not _is_valid(best_circles):
        best_circles = _post_process(_init_grid().tolist())
    
    return np.array(best_circles)

# EVOLVE-BLOCK-END
