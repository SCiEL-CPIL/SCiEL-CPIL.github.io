"""Circle packing in unit square - 26 circles maximizing sum of radii.
Strategy: Smooth penalty L-BFGS-B + contact-graph QP radius optimization.
1. Use differentiable soft-max/soft-plus penalties for collisions and boundaries.
2. Optimize positions and radii with L-BFGS-B from diverse starts.
3. Iteratively fix positions, solve convex QP for optimal radii on contact graph.
4. Polish with tight-constraint SLSQP at the end.
"""

import numpy as np
import time
import random
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 26

# EVOLVE-BLOCK-START

def _is_valid(circles, tol=1e-8):
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


def _post_process(circles, tol=1e-9):
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


def _softplus(x, beta=1.0):
    """Numerically stable softplus."""
    return np.logaddexp(0, beta * x) / beta


def _soft_max(x, beta=1.0):
    """Soft max of a vector."""
    if len(x) == 0:
        return 0.0
    max_x = np.max(x)
    if max_x * beta > 500:
        return max_x / beta
    return (max_x + np.log(np.sum(np.exp(beta * (x - max_x))))) / beta


def _objective(params, beta=10.0):
    """
    Penalty-based objective. Minimizes penalty (equivalent to maximizing sum of radii).
    params: (x_0, y_0, r_0, x_1, y_1, r_1, ...)
    """
    n = len(params) // 3
    penalty = 0.0
    
    # Boundary penalties
    for i in range(n):
        x, y, r = params[3*i], params[3*i+1], params[3*i+2]
        # x - r >= 0, x + r <= 1, y - r >= 0, y + r <= 1
        # Penalty if violated
        penalty += _softplus(-(x - r), beta)
        penalty += _softplus((x + r) - 1.0, beta)
        penalty += _softplus(-(y - r), beta)
        penalty += _softplus((y + r) - 1.0, beta)
    
    # Collision penalties
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi, ri = params[3*i], params[3*i+1], params[3*i+2]
            xj, yj, rj = params[3*j], params[3*j+1], params[3*j+2]
            dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
            # dist >= ri + rj
            penalty += _softplus((ri + rj) - dist, beta)
    
    # Objective: minimize penalty - sum(radii) * large_weight
    # Actually, we want to maximize sum(radii) subject to constraints.
    # So we minimize: penalty - sum(radii)
    # But penalty should be large enough to dominate.
    sum_r = sum(params[3*i+2] for i in range(n))
    return penalty - sum_r


def _objective_smooth(params, beta=10.0):
    """
    Smooth objective for L-BFGS-B.
    We maximize sum(radii) - lambda * penalty.
    So we minimize: -sum(radii) + lambda * penalty.
    """
    n = len(params) // 3
    penalty = 0.0
    
    for i in range(n):
        x, y, r = params[3*i], params[3*i+1], params[3*i+2]
        penalty += _softplus(-(x - r), beta)
        penalty += _softplus((x + r) - 1.0, beta)
        penalty += _softplus(-(y - r), beta)
        penalty += _softplus((y + r) - 1.0, beta)
    
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi, ri = params[3*i], params[3*i+1], params[3*i+2]
            xj, yj, rj = params[3*j], params[3*j+1], params[3*j+2]
            dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
            penalty += _softplus((ri + rj) - dist, beta)
    
    sum_r = sum(params[3*i+2] for i in range(n))
    return -sum_r + penalty


def _initial_hexagonal():
    """Hexagonal packing initial layout for 26 circles."""
    circles = []
    r = 0.075
    rows = [5, 5, 5, 5, 6]
    row_spacing = r * np.sqrt(3)
    col_spacing = 2 * r
    
    row_idx = 0
    for count in rows:
        offset = 0 if row_idx % 2 == 0 else col_spacing / 2
        total_width = col_spacing * (count - 1) + offset
        start_x = (1.0 - total_width) / 2.0
        y = r + row_spacing * row_idx + 0.03
        for col in range(count):
            x = start_x + offset + col * col_spacing
            if len(circles) < N:
                circles.append((x, y, r))
        row_idx += 1
    
    circles = circles[:N]
    while len(circles) < N:
        circles.append((0.5, 0.5, 0.01))
    return circles


def _initial_grid():
    """Simple grid initialization."""
    side = int(np.ceil(np.sqrt(N)))
    circles = []
    for k in range(N):
        i = k % side
        j = k // side
        x = (i + 0.5) / side
        y = (j + 0.5) / side
        r = 0.4 / side
        circles.append((x, y, r))
    return circles


def _initial_corner_biased():
    """Initialization with circles at corners and edges."""
    circles = []
    # Corners
    circles.append((0.08, 0.08, 0.08))
    circles.append((0.92, 0.08, 0.08))
    circles.append((0.08, 0.92, 0.08))
    circles.append((0.92, 0.92, 0.08))
    
    # Edges
    circles.append((0.5, 0.05, 0.06))
    circles.append((0.5, 0.95, 0.06))
    circles.append((0.05, 0.5, 0.06))
    circles.append((0.95, 0.5, 0.06))
    
    # Fill rest randomly
    random.seed(42)
    while len(circles) < N:
        x = random.uniform(0.15, 0.85)
        y = random.uniform(0.15, 0.85)
        r = random.uniform(0.04, 0.07)
        circles.append((x, y, r))
    return circles


def _initial_random(seed=None):
    """Random initialization."""
    if seed is not None:
        random.seed(seed)
    circles = []
    for _ in range(N):
        x = random.uniform(0.1, 0.9)
        y = random.uniform(0.1, 0.9)
        r = random.uniform(0.03, 0.08)
        circles.append((x, y, r))
    return circles


def _initial_best_fit():
    """Greedy best-fit initialization."""
    circles = []
    random.seed(0)
    for _ in range(N):
        best_pos = None
        best_r = 0.0
        for _ in range(100):
            x = random.uniform(0.05, 0.95)
            y = random.uniform(0.05, 0.95)
            r = 0.01
            while r < 0.4:
                valid = True
                for cx, cy, cr in circles:
                    dist = np.sqrt((x-cx)**2 + (y-cy)**2)
                    if dist < r + cr:
                        valid = False
                        break
                if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                    valid = False
                if valid:
                    r += 0.01
                else:
                    break
            if r > best_r:
                best_r = r
                best_pos = (x, y, r - 0.01)
        if best_pos:
            circles.append(best_pos)
    return circles


def _incremental_energy(params, beta=10.0):
    """Compute energy incrementally (for speed)."""
    n = len(params) // 3
    penalty = 0.0
    
    for i in range(n):
        x, y, r = params[3*i], params[3*i+1], params[3*i+2]
        penalty += _softplus(-(x - r), beta)
        penalty += _softplus((x + r) - 1.0, beta)
        penalty += _softplus(-(y - r), beta)
        penalty += _softplus((y + r) - 1.0, beta)
    
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi, ri = params[3*i], params[3*i+1], params[3*i+2]
            xj, yj, rj = params[3*j], params[3*j+1], params[3*j+2]
            dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
            penalty += _softplus((ri + rj) - dist, beta)
    
    return penalty


def _full_energy(params, beta=10.0):
    """Compute full energy."""
    return _incremental_energy(params, beta)


def _slsqp_solve(params, bounds, constraints, maxiter=200):
    """Solve SLSQP optimization problem."""
    def objective(p):
        return -sum(p[3*i+2] for i in range(N))
    
    def jac(p):
        g = np.zeros(3*N)
        g[2::3] = -1.0
        return g
    
    try:
        result = minimize(
            objective, params, jac=jac,
            method='SLSQP', bounds=bounds, constraints=constraints,
            options={'maxiter': maxiter, 'ftol': 1e-10}
        )
        return result.x
    except Exception:
        return params


def _local_search(circles, deadline, max_iters=50):
    """Local search using SLSQP with pairwise and boundary constraints."""
    params = np.array([val for c in circles for val in c])
    
    # Bounds
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.001, 0.5)] * N
    
    # Constraints
    constraints = []
    
    # Boundary constraints
    for i in range(N):
        constraints.append({
            'type': 'ineq',
            'fun': lambda p, i=i: p[3*i] - p[3*i+2],
            'jac': lambda p, i=i: np.array([1.0 if k==3*i else (-1.0 if k==3*i+2 else 0.0) for k in range(3*N)])
        })
        constraints.append({
            'type': 'ineq',
            'fun': lambda p, i=i: 1.0 - p[3*i] - p[3*i+2],
            'jac': lambda p, i=i: np.array([(-1.0 if k==3*i else (-1.0 if k==3*i+2 else 0.0)) for k in range(3*N)])
        })
        constraints.append({
            'type': 'ineq',
            'fun': lambda p, i=i: p[3*i+1] - p[3*i+2],
            'jac': lambda p, i=i: np.array([1.0 if k==3*i+1 else (-1.0 if k==3*i+2 else 0.0) for k in range(3*N)])
        })
        constraints.append({
            'type': 'ineq',
            'fun': lambda p, i=i: 1.0 - p[3*i+1] - p[3*i+2],
            'jac': lambda p, i=i: np.array([(-1.0 if k==3*i+1 else (-1.0 if k==3*i+2 else 0.0)) for k in range(3*N)])
        })
    
    # Overlap constraints
    for i in range(N):
        for j in range(i+1, N):
            def overlap_fun(p, i=i, j=j):
                dx = p[3*i] - p[3*j]
                dy = p[3*i+1] - p[3*j+1]
                return dx*dx + dy*dy - (p[3*i+2] + p[3*j+2])**2
            
            def overlap_jac(p, i=i, j=j):
                jac = np.zeros(3*N)
                dx = p[3*i] - p[3*j]
                dy = p[3*i+1] - p[3*j+1]
                sum_r = p[3*i+2] + p[3*j+2]
                jac[3*i] = 2*dx
                jac[3*j] = -2*dx
                jac[3*i+1] = 2*dy
                jac[3*j+1] = -2*dy
                jac[3*i+2] = -2*sum_r
                jac[3*j+2] = -2*sum_r
                return jac
            
            constraints.append({'type': 'ineq', 'fun': overlap_fun, 'jac': overlap_jac})
    
    # Optimize
    new_params = _slsqp_solve(params, bounds, constraints)
    
    # Convert back to circles
    new_circles = [(new_params[3*i], new_params[3*i+1], new_params[3*i+2]) for i in range(N)]
    
    return new_circles


def _simulated_annealing(circles, deadline, max_iters=200):
    """Simulated annealing for global optimization."""
    if time.time() > deadline - 10.0:
        return circles
    
    current = list(circles)
    best = list(circles)
    best_sum = _sum_radii(best)
    
    temp = 0.03
    cooling = 0.95
    
    for it in range(max_iters):
        if time.time() > deadline - 10.0:
            break
        
        # Generate perturbation
        perturbed = []
        for x, y, r in current:
            nx = x + random.gauss(0, temp * 0.5)
            ny = y + random.gauss(0, temp * 0.5)
            nr = r + random.gauss(0, temp * 0.3)
            nr = max(0.01, min(0.4, nr))
            nx = max(nr, min(1.0 - nr, nx))
            ny = max(nr, min(1.0 - nr, ny))
            perturbed.append((nx, ny, nr))
        
        # Local refine
        refined = _local_search(perturbed, deadline)
        new_sum = _sum_radii(refined)
        
        if new_sum > best_sum:
            best = refined
            best_sum = new_sum
            current = refined
        else:
            delta = _sum_radii(current) - new_sum
            if delta < 1e-8:
                continue
            prob = np.exp(-delta / temp)
            if random.random() < prob:
                current = refined
        
        temp *= cooling
        if temp < 0.001:
            break
    
    return best


def _lbfgsb_solve(circles, deadline, beta_start=5.0, beta_end=50.0, max_iter=100):
    """
    L-BFGS-B solver with annealing beta for smooth penalty.
    """
    params = np.array([val for c in circles for val in c])
    
    # Bounds: x in [0,1], y in [0,1], r in [0.001, 0.5]
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.001, 0.5)] * N
    
    current_params = params.copy()
    best_params = params.copy()
    best_obj = _objective_smooth(current_params, beta_start)
    
    num_steps = 30
    for step in range(num_steps):
        if time.time() > deadline - 5.0:
            break
        
        # Anneal beta
        beta = beta_start * (beta_end / beta_start) ** (step / num_steps)
        
        # Define objective with current beta
        def obj(p):
            return _objective_smooth(p, beta)
        
        try:
            result = minimize(
                obj, current_params,
                method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            new_params = result.x
            new_obj = _objective_smooth(new_params, beta)
            
            if new_obj < best_obj:
                best_obj = new_obj
                best_params = new_params.copy()
            
            current_params = new_params
        except Exception:
            pass
    
    return best_params


def _radius_growth_pass(circles, deadline):
    """
    Fix positions, maximize radii via SLSQP with hard constraints.
    This squeezes additional sum-of-radii by growing circles into available gaps.
    """
    params = np.array([val for c in circles for val in c])
    
    # Fix positions, optimize radii
    # Objective: maximize sum(r_i)
    # Constraints: dist(i,j) >= r_i + r_j, and boundary constraints
    
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.001, 0.5)] * N
    
    constraints = []
    
    # Boundary constraints (positions fixed, so these are just r_i <= min(x_i, 1-x_i, y_i, 1-y_i))
    for i in range(N):
        x, y = params[3*i], params[3*i+1]
        max_r = min(x, 1-x, y, 1-y)
        # r_i <= max_r
        constraints.append({
            'type': 'ineq',
            'fun': lambda p, i=i, max_r=max_r: max_r - p[3*i+2],
            'jac': lambda p, i=i: np.array([-1.0 if k==3*i+2 else 0.0 for k in range(3*N)])
        })
    
    # Overlap constraints
    for i in range(N):
        for j in range(i+1, N):
            xi, yi = params[3*i], params[3*i+1]
            xj, yj = params[3*j], params[3*j+1]
            dist = np.sqrt((xi-xj)**2 + (yi-yj)**2)
            
            def overlap_fun(p, i=i, j=j, dist=dist):
                return dist - (p[3*i+2] + p[3*j+2])
            
            def overlap_jac(p, i=i, j=j):
                jac = np.zeros(3*N)
                jac[3*i+2] = -1.0
                jac[3*j+2] = -1.0
                return jac
            
            constraints.append({'type': 'ineq', 'fun': overlap_fun, 'jac': overlap_jac})
    
    def objective(p):
        return -sum(p[3*i+2] for i in range(N))
    
    def jac(p):
        g = np.zeros(3*N)
        g[2::3] = -1.0
        return g
    
    try:
        result = minimize(
            objective, params, jac=jac,
            method='SLSQP', bounds=bounds, constraints=constraints,
            options={'maxiter': 200, 'ftol': 1e-12}
        )
        new_params = result.x
        # Only update radii, keep positions
        for i in range(N):
            params[3*i+2] = new_params[3*i+2]
    except Exception:
        pass
    
    return [(params[3*i], params[3*i+1], params[3*i+2]) for i in range(N)]


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Multiple diverse initializations
    inits = [
        _initial_hexagonal(),
        _initial_grid(),
        _initial_corner_biased(),
        _initial_random(42),
        _initial_random(123),
        _initial_random(777),
        _initial_random(999),
        _initial_random(1234),
        _initial_best_fit(),
    ]
    
    best_circles = inits[0]
    best_sum = _sum_radii(best_circles)
    
    # L-BFGS-B on each initialization
    for init_circles in inits:
        if time.time() > deadline - 30.0:
            break
        
        # L-BFGS-B refine
        params = np.array([val for c in init_circles for val in c])
        refined_params = _lbfgsb_solve(init_circles, deadline)
        refined_circles = [(refined_params[3*i], refined_params[3*i+1], refined_params[3*i+2]) for i in range(N)]
        refined_sum = _sum_radii(refined_circles)
        
        if refined_sum > best_sum and _is_valid(refined_circles):
            best_circles = refined_circles
            best_sum = refined_sum
    
    # Simulated annealing for global exploration
    if time.time() < deadline - 20.0:
        sa_result = _simulated_annealing(best_circles, deadline)
        sa_sum = _sum_radii(sa_result)
        
        if sa_sum > best_sum and _is_valid(sa_result):
            best_circles = sa_result
            best_sum = sa_sum
    
    # Final polish with SLSQP
    if time.time() < deadline - 10.0:
        final = _local_search(best_circles, deadline)
        final_sum = _sum_radii(final)
        
        if final_sum > best_sum and _is_valid(final):
            best_circles = final
            best_sum = final_sum
    
    # Radius growth pass
    if time.time() < deadline - 5.0:
        grown = _radius_growth_pass(best_circles, deadline)
        grown_sum = _sum_radii(grown)
        
        if grown_sum > best_sum and _is_valid(grown):
            best_circles = grown
            best_sum = grown_sum
    
    # Final SLSQP polish
    if time.time() < deadline - 2.0:
        final2 = _local_search(best_circles, deadline)
        final2_sum = _sum_radii(final2)
        
        if final2_sum > best_sum and _is_valid(final2):
            best_circles = final2
            best_sum = final2_sum
    
    # Post-process
    best_circles = _post_process(best_circles)
    
    # Verify and return
    if not _is_valid(best_circles):
        # Fallback
        best_circles = _post_process(_initial_grid())
    
    return np.array(best_circles)

# EVOLVE-BLOCK-END
