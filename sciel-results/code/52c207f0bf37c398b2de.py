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
    x = np.asarray(x, dtype=float)
    return np.where(x > 20/beta, x, np.log1p(np.exp(x * beta)) / beta)


def _soft_max(x, beta=1.0):
    """Differentiable approximation of max(0, x)."""
    return _softplus(x, beta)


def _objective(p, beta=50.0):
    """
    Smooth objective: -sum(r_i) + penalty for collisions + penalty for boundary violations.
    p: flat array of shape (3*N,) with [x0,y0,r0, x1,y1,r1, ...]
    """
    n = N
    x = p[0::3]
    y = p[1::3]
    r = p[2::3]
    
    # Base: maximize sum of radii
    base = -np.sum(r)
    
    # Collision penalty: for each pair, penalize if dist < r_i + r_j
    collision_penalty = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy + 1e-12)
            violation = (r[i] + r[j]) - dist
            collision_penalty += _soft_max(violation, beta)
    
    # Boundary penalty
    boundary_penalty = 0.0
    for i in range(n):
        boundary_penalty += _soft_max(r[i] - x[i], beta)       # x < r
        boundary_penalty += _soft_max(x[i] + r[i] - 1.0, beta) # x + r > 1
        boundary_penalty += _soft_max(r[i] - y[i], beta)       # y < r
        boundary_penalty += _soft_max(y[i] + r[i] - 1.0, beta) # y + r > 1
    
    return base + 10.0 * (collision_penalty + boundary_penalty)


def _objective_grad(p, beta=50.0):
    """Gradient of the smooth objective."""
    n = N
    x = p[0::3].copy()
    y = p[1::3].copy()
    r = p[2::3].copy()
    
    grad = np.zeros(3 * n)
    
    # Base gradient: d(-sum r)/dr_i = -1
    grad[2::3] = -1.0
    
    # Collision gradient
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy + 1e-12)
            violation = (r[i] + r[j]) - dist
            
            # d(soft_max(violation))/d(violation) = sigmoid(beta * violation)
            bv = beta * violation
            if bv > 20:
                sig = 1.0
            elif bv < -20:
                sig = 0.0
            else:
                sig = 1.0 / (1.0 + np.exp(-bv))
            
            if sig < 1e-10:
                continue
            
            # d(violation)/d(x_i) = -dx/dist, d(violation)/d(x_j) = dx/dist
            # d(violation)/d(r_i) = 1, d(violation)/d(r_j) = 1
            factor = 10.0 * sig
            grad[3*i] += factor * (-dx / dist)
            grad[3*j] += factor * (dx / dist)
            grad[3*i+1] += factor * (-dy / dist)
            grad[3*j+1] += factor * (dy / dist)
            grad[3*i+2] += factor
            grad[3*j+2] += factor
    
    # Boundary gradient
    for i in range(n):
        # x < r: violation = r_i - x_i
        v1 = r[i] - x[i]
        bv1 = beta * v1
        sig1 = 1.0 if bv1 > 20 else (0.0 if bv1 < -20 else 1.0/(1.0+np.exp(-bv1)))
        if sig1 > 1e-10:
            grad[3*i] += 10.0 * sig1
            grad[3*i+2] += 10.0 * sig1
        
        # x + r > 1: violation = x_i + r_i - 1
        v2 = x[i] + r[i] - 1.0
        bv2 = beta * v2
        sig2 = 1.0 if bv2 > 20 else (0.0 if bv2 < -20 else 1.0/(1.0+np.exp(-bv2)))
        if sig2 > 1e-10:
            grad[3*i] += 10.0 * sig2
            grad[3*i+2] += 10.0 * sig2
        
        # y < r: violation = r_i - y_i
        v3 = r[i] - y[i]
        bv3 = beta * v3
        sig3 = 1.0 if bv3 > 20 else (0.0 if bv3 < -20 else 1.0/(1.0+np.exp(-bv3)))
        if sig3 > 1e-10:
            grad[3*i+1] += 10.0 * sig3
            grad[3*i+2] += 10.0 * sig3
        
        # y + r > 1: violation = y_i + r_i - 1
        v4 = y[i] + r[i] - 1.0
        bv4 = beta * v4
        sig4 = 1.0 if bv4 > 20 else (0.0 if bv4 < -20 else 1.0/(1.0+np.exp(-bv4)))
        if sig4 > 1e-10:
            grad[3*i+1] += 10.0 * sig4
            grad[3*i+2] += 10.0 * sig4
    
    return grad


def _lbgfbs_b_optimize(p, deadline, beta=50.0, maxiter=500):
    """Optimize using L-BFGS-B with smooth penalty objective."""
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.001, 0.45)] * N
    
    try:
        result = minimize(
            _objective, p, jac=_objective_grad,
            args=(beta,),
            method='L-BFGS-B', bounds=bounds,
            options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
        )
        return result.x
    except Exception:
        return p


def _build_contact_graph(circles, margin=0.01):
    """Build contact graph: pairs of circles that are close (nearly touching)."""
    n = N
    contacts = []
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi, ri = circles[i]
            xj, yj, rj = circles[j]
            dist = np.sqrt((xi-xj)**2 + (yi-yj)**2)
            if dist < ri + rj + margin:
                contacts.append((i, j, dist))
    return contacts


def _solve_qp_for_radii(circles, deadline):
    """
    Fix positions, solve convex QP to maximize sum of radii.
    Constraints: for each pair, r_i + r_j <= dist(i,j)
    For each circle, r_i <= min(x_i, 1-x_i, y_i, 1-y_i)
    r_i >= 0
    """
    n = N
    x = np.array([c[0] for c in circles])
    y = np.array([c[1] for c in circles])
    
    # Upper bounds on radii from boundaries
    r_max = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    r_max = np.maximum(r_max, 0.0)
    
    # Build constraint matrix for pairwise constraints
    # r_i + r_j <= dist_ij for all pairs
    pairs = []
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
            pairs.append((i, j))
            dists.append(d)
    
    if len(pairs) == 0:
        return r_max
    
    m = len(pairs)
    A = np.zeros((m, n))
    b = np.array(dists)
    for k, (i, j) in enumerate(pairs):
        A[k, i] = 1.0
        A[k, j] = 1.0
    
    # Maximize sum(r) subject to A @ r <= b, 0 <= r <= r_max
    # Equivalent to minimize -sum(r)
    c = -np.ones(n)
    
    # Use scipy linprog (LP relaxation - since constraints are linear in r with fixed positions)
    from scipy.optimize import linprog
    bounds_lp = [(0.0, r_max[i]) for i in range(n)]
    
    try:
        res = linprog(c, A_ub=A, b_ub=b, bounds=bounds_lp, method='highs')
        if res.success:
            return res.x
    except Exception:
        pass
    
    # Fallback: simple iterative clipping
    r = r_max.copy()
    for _ in range(100):
        for k, (i, j) in enumerate(pairs):
            if r[i] + r[j] > dists[k] + 1e-12:
                excess = (r[i] + r[j] - dists[k]) / 2.0
                r[i] = max(0.0, r[i] - excess)
                r[j] = max(0.0, r[j] - excess)
        # Re-clip to boundary
        r = np.minimum(r, r_max)
    return r


def _update_positions(circles, r_new, deadline):
    """After updating radii, adjust positions to maintain validity."""
    n = N
    new_circles = []
    for i in range(n):
        x, y, r_old = circles[i]
        r = r_new[i]
        # Clip position to stay in bounds
        x = min(1.0 - r, max(r, x))
        y = min(1.0 - r, max(r, y))
        new_circles.append((x, y, r))
    return new_circles


def _contact_graph_optimize(circles, deadline, max_rounds=20):
    """Iteratively: fix positions -> solve QP for radii -> update positions -> repeat."""
    current = list(circles)
    best = list(circles)
    best_sum = _sum_radii(best)
    
    for rnd in range(max_rounds):
        if time.time() > deadline - 15.0:
            break
        
        # Solve QP for optimal radii given current positions
        r_opt = _solve_qp_for_radii(current, deadline)
        
        # Update circles with new radii
        new_circles = _update_positions(current, r_opt, deadline)
        new_sum = _sum_radii(new_circles)
        
        # Check validity
        if _is_valid(new_circles):
            if new_sum > best_sum:
                best = new_circles
                best_sum = new_sum
            current = new_circles
        else:
            # If invalid, slightly shrink and retry
            shrunken = [(x, y, max(0.0, r * 0.95)) for x, y, r in new_circles]
            shrunken = _update_positions(current, np.array([r*0.95 for _,_,r in new_circles]), deadline)
            if _is_valid(shrunken):
                current = shrunken
            else:
                break
        
        # Small position perturbation to escape local optima
        if rnd % 3 == 2:
            perturbed = []
            for x, y, r in current:
                nx = x + random.gauss(0, 0.005)
                ny = y + random.gauss(0, 0.005)
                nx = min(1.0 - r, max(r, nx))
                ny = min(1.0 - r, max(r, ny))
                perturbed.append((nx, ny, r))
            if _is_valid(perturbed):
                current = perturbed
    
    return best


def _init_hexagonal():
    """Hexagonal packing initial layout for 26 circles."""
    circles = []
    r = 0.07
    rows = [5, 5, 5, 5, 6]
    row_spacing = r * np.sqrt(3)
    col_spacing = 2 * r
    
    row_idx = 0
    for count in rows:
        offset = 0 if row_idx % 2 == 0 else col_spacing / 2
        total_width = col_spacing * (count - 1) + offset
        start_x = (1.0 - total_width) / 2.0
        y = r + row_spacing * row_idx + 0.02
        for col in range(count):
            x = start_x + offset + col * col_spacing
            if len(circles) < N:
                circles.append((x, y, r))
        row_idx += 1
    
    circles = circles[:N]
    while len(circles) < N:
        circles.append((0.5, 0.5, 0.01))
    return circles


def _init_grid():
    """Simple grid initialization."""
    side = int(np.ceil(np.sqrt(N)))
    circles = []
    for k in range(N):
        i = k % side
        j = k // side
        x = (i + 0.5) / side
        y = (j + 0.5) / side
        r = 0.38 / side
        circles.append((x, y, r))
    return circles


def _init_corner_biased():
    """Initialization biased toward corners and edges."""
    circles = []
    # Place some circles near corners
    corners = [(0.08, 0.08), (0.92, 0.08), (0.08, 0.92), (0.92, 0.92)]
    for cx, cy in corners:
        circles.append((cx, cy, 0.08))
    
    # Edge midpoints
    edges = [(0.5, 0.05), (0.5, 0.95), (0.05, 0.5), (0.95, 0.5)]
    for ex, ey in edges:
        circles.append((ex, ey, 0.06))
    
    random.seed(42)
    while len(circles) < N:
        x = random.uniform(0.15, 0.85)
        y = random.uniform(0.15, 0.85)
        r = random.uniform(0.04, 0.07)
        circles.append((x, y, r))
    return circles


def _init_random(seed=None):
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


def _slsqp_polish(circles, deadline):
    """Final SLSQP polish with hard constraints."""
    params = np.array([val for c in circles for val in c])
    n = N
    dim = 3 * n
    
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.001, 0.45)] * n
    
    constraints = []
    
    # Boundary constraints
    for i in range(n):
        constraints.append({
            'type': 'ineq',
            'fun': lambda p, i=i: p[3*i] - p[3*i+2],
            'jac': lambda p, i=i: np.array([1.0 if k==3*i else (-1.0 if k==3*i+2 else 0.0) for k in range(dim)])
        })
        constraints.append({
            'type': 'ineq',
            'fun': lambda p, i=i: 1.0 - p[3*i] - p[3*i+2],
            'jac': lambda p, i=i: np.array([(-1.0 if k==3*i else (-1.0 if k==3*i+2 else 0.0)) for k in range(dim)])
        })
        constraints.append({
            'type': 'ineq',
            'fun': lambda p, i=i: p[3*i+1] - p[3*i+2],
            'jac': lambda p, i=i: np.array([1.0 if k==3*i+1 else (-1.0 if k==3*i+2 else 0.0) for k in range(dim)])
        })
        constraints.append({
            'type': 'ineq',
            'fun': lambda p, i=i: 1.0 - p[3*i+1] - p[3*i+2],
            'jac': lambda p, i=i: np.array([(-1.0 if k==3*i+1 else (-1.0 if k==3*i+2 else 0.0)) for k in range(dim)])
        })
    
    # Overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            constraints.append({
                'type': 'ineq',
                'fun': lambda p, i=i, j=j: (p[3*i]-p[3*j])**2 + (p[3*i+1]-p[3*j+1])**2 - (p[3*i+2]+p[3*j+2])**2,
                'jac': lambda p, i=i, j=j: (lambda dx, dy, sr: np.array([
                    2*dx if k==3*i else (-2*dx if k==3*j else (2*dy if k==3*i+1 else (-2*dy if k==3*j+1 else (-2*sr if k in (3*i+2, 3*j+2) else 0.0))))
                    for k in range(dim)
                ]))(p[3*i]-p[3*j], p[3*i+1]-p[3*j+1], p[3*i+2]+p[3*j+2])
            })
    
    def objective(p):
        return -sum(p[3*i+2] for i in range(n))
    
    def jac(p):
        g = np.zeros(dim)
        g[2::3] = -1.0
        return g
    
    try:
        result = minimize(
            objective, params, jac=jac,
            method='SLSQP', bounds=bounds, constraints=constraints,
            options={'maxiter': 300, 'ftol': 1e-12}
        )
        if result.success:
            new_circles = [(result.x[3*i], result.x[3*i+1], result.x[3*i+2]) for i in range(n)]
            return new_circles
    except Exception:
        pass
    
    return circles


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Multiple diverse initializations
    inits = [
        _init_hexagonal(),
        _init_grid(),
        _init_corner_biased(),
        _init_random(42),
        _init_random(123),
        _init_random(777),
    ]
    
    best_circles = inits[0]
    best_sum = _sum_radii(best_circles)
    
    # Phase 1: L-BFGS-B smooth optimization from each init
    for idx, init_circles in enumerate(inits):
        if time.time() > deadline - 40.0:
            break
        
        params = np.array([val for c in init_circles for val in c])
        
        # Start with lower beta (smoother), increase for final
        for beta in [10.0, 30.0, 80.0]:
            if time.time() > deadline - 40.0:
                break
            params = _lbgfbs_b_optimize(params, deadline, beta=beta, maxiter=300)
        
        new_circles = [(params[3*i], params[3*i+1], params[3*i+2]) for i in range(N)]
        new_sum = _sum_radii(new_circles)
        
        if new_sum > best_sum and _is_valid(new_circles):
            best_circles = new_circles
            best_sum = new_sum
    
    # Phase 2: Contact-graph QP optimization (fix positions, optimize radii)
    if time.time() < deadline - 30.0:
        cg_result = _contact_graph_optimize(best_circles, deadline, max_rounds=30)
        cg_sum = _sum_radii(cg_result)
        if cg_sum > best_sum and _is_valid(cg_result):
            best_circles = cg_result
            best_sum = cg_sum
    
    # Phase 3: Another L-BFGS-B pass with high beta on the best
    if time.time() < deadline - 20.0:
        params = np.array([val for c in best_circles for val in c])
        for beta in [50.0, 100.0, 200.0]:
            if time.time() > deadline - 20.0:
                break
            params = _lbgfbs_b_optimize(params, deadline, beta=beta, maxiter=200)
        new_circles = [(params[3*i], params[3*i+1], params[3*i+2]) for i in range(N)]
        new_sum = _sum_radii(new_circles)
        if new_sum > best_sum and _is_valid(new_circles):
            best_circles = new_circles
            best_sum = new_sum
    
    # Phase 4: Contact-graph QP again on the refined result
    if time.time() < deadline - 15.0:
        cg_result2 = _contact_graph_optimize(best_circles, deadline, max_rounds=20)
        cg_sum2 = _sum_radii(cg_result2)
        if cg_sum2 > best_sum and _is_valid(cg_result2):
            best_circles = cg_result2
            best_sum = cg_sum2
    
    # Phase 5: SLSQP polish
    if time.time() < deadline - 5.0:
        polished = _slsqp_polish(best_circles, deadline)
        pol_sum = _sum_radii(polished)
        if pol_sum > best_sum and _is_valid(polished):
            best_circles = polished
            best_sum = pol_sum
    
    # Post-process
    best_circles = _post_process(best_circles)
    
    # Verify and return
    if not _is_valid(best_circles):
        # Fallback: use grid init
        best_circles = _post_process(_init_grid())
    
    return np.array(best_circles)

# EVOLVE-BLOCK-END
