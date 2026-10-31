"""Circle packing in unit square - 26 circles maximizing sum of radii.
Strategy: Hard-constraint operator alternation.
  - grow_radii: given fixed positions, find max feasible radii via iterative
    feasibility (binary-search on sum + per-circle radius growth with hard
    collision/boundary checks). This is a QP-like radius-growth operator.
  - relax_positions: given fixed radii, move positions to relieve congestion
    (force-directed repulsion) so that grow_radii can expand more.
  - Alternate these two operators (both maintain hard feasibility) with
    multiple restarts and perturbations, then SLSQP polish.
This is structurally different from penalty-based SA: the search space is
restricted to feasible configurations and the "energy" is purely sum(r)."""

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


def _max_feasible_radius(i, circles, tol=1e-10):
    """Compute the maximum radius for circle i given all other circles fixed."""
    x, y, _ = circles[i]
    # Boundary constraints
    max_r = min(x, 1.0 - x, y, 1.0 - y)
    # Collision constraints with all other circles
    for j in range(N):
        if j == i:
            continue
        xj, yj, rj = circles[j]
        dist = np.sqrt((x - xj)**2 + (y - yj)**2)
        if dist > 0:
            max_r = min(max_r, dist - rj)
        else:
            # Degenerate: circles at same position
            max_r = 0.0
    return max(0.0, max_r)


def _grow_radii(circles, deadline, tol=1e-10):
    """
    Grow all radii to their maximum feasible values given fixed positions.
    Uses iterative relaxation: repeatedly set each radius to its max feasible
    value (considering current other radii). This converges to a local maximum
    of sum(radii) for fixed positions.
    """
    circles = [list(c) for c in circles]
    # Ensure initial feasibility by shrinking if needed
    for it in range(200):
        if time.time() > deadline - 15.0:
            break
        changed = False
        for i in range(N):
            max_r = _max_feasible_radius(i, circles, tol)
            if max_r > circles[i][2] + 1e-12:
                circles[i][2] = max_r
                changed = True
            elif circles[i][2] > max_r + 1e-12:
                # Infeasible, shrink
                circles[i][2] = max(0.0, max_r)
                changed = True
        if not changed:
            break
    return [tuple(c) for c in circles]


def _relax_positions(circles, deadline, n_iters=500, lr=0.005):
    """
    Move positions to relieve congestion (force-directed repulsion).
    Given fixed radii, apply repulsive forces between overlapping/near-overlapping
    circles and boundary forces, then grow radii again.
    Maintains hard feasibility by only accepting moves that don't violate constraints.
    """
    circles = [list(c) for c in circles]
    x = np.array([c[0] for c in circles])
    y = np.array([c[1] for c in circles])
    r = np.array([c[2] for c in circles])
    
    for it in range(n_iters):
        if time.time() > deadline - 15.0:
            break
        
        # Compute forces
        fx = np.zeros(N)
        fy = np.zeros(N)
        
        for i in range(N):
            # Boundary forces
            if x[i] - r[i] < 0.02:
                fx[i] += (0.02 - (x[i] - r[i])) * 5.0
            if x[i] + r[i] > 0.98:
                fx[i] -= ((x[i] + r[i]) - 0.98) * 5.0
            if y[i] - r[i] < 0.02:
                fy[i] += (0.02 - (y[i] - r[i])) * 5.0
            if y[i] + r[i] > 0.98:
                fy[i] -= ((y[i] + r[i]) - 0.98) * 5.0
            
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = r[i] + r[j]
                if dist < min_dist + 0.03:
                    if dist < 1e-10:
                        # Degenerate
                        dx = random.uniform(-0.01, 0.01)
                        dy = random.uniform(-0.01, 0.01)
                        dist = np.sqrt(dx*dx + dy*dy)
                    force = (min_dist + 0.03 - dist) * 10.0
                    fx[i] += force * dx / dist
                    fy[i] += force * dy / dist
                    fx[j] -= force * dx / dist
                    fy[j] -= force * dy / dist
        
        # Apply forces with step size
        step = lr * (1.0 - it / n_iters) + 0.001
        new_x = x + step * fx
        new_y = y + step * fy
        
        # Clamp to bounds
        new_x = np.clip(new_x, r, 1.0 - r)
        new_y = np.clip(new_y, r, 1.0 - r)
        
        # Check if move improves (or maintains) feasibility
        # Try the move; if it causes overlap, reject
        valid = True
        for i in range(N):
            for j in range(i + 1, N):
                dist = np.sqrt((new_x[i]-new_x[j])**2 + (new_y[i]-new_y[j])**2)
                if dist < r[i] + r[j] - 1e-10:
                    valid = False
                    break
            if not valid:
                break
        # Also check boundaries
        if valid:
            for i in range(N):
                if new_x[i] - r[i] < -1e-10 or new_x[i] + r[i] > 1.0 + 1e-10:
                    valid = False
                    break
                if new_y[i] - r[i] < -1e-10 or new_y[i] + r[i] > 1.0 + 1e-10:
                    valid = False
                    break
        
        if valid:
            x = new_x
            y = new_y
            circles = [(x[i], y[i], r[i]) for i in range(N)]
        else:
            # Halve step and retry
            step *= 0.5
            new_x = x + step * fx
            new_y = y + step * fy
            new_x = np.clip(new_x, r, 1.0 - r)
            new_y = np.clip(new_y, r, 1.0 - r)
            valid = True
            for i in range(N):
                for j in range(i + 1, N):
                    dist = np.sqrt((new_x[i]-new_x[j])**2 + (new_y[i]-new_y[j])**2)
                    if dist < r[i] + r[j] - 1e-10:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                x = new_x
                y = new_y
                circles = [(x[i], y[i], r[i]) for i in range(N)]
    
    return circles


def _perturb_positions(circles, scale=0.03):
    """Randomly perturb positions slightly (maintaining boundary feasibility)."""
    circles = [list(c) for c in circles]
    for i in range(N):
        dx = random.gauss(0, scale)
        dy = random.gauss(0, scale)
        new_x = circles[i][0] + dx
        new_y = circles[i][1] + dy
        r = circles[i][2]
        new_x = max(r, min(1.0 - r, new_x))
        new_y = max(r, min(1.0 - r, new_y))
        circles[i][0] = new_x
        circles[i][1] = new_y
    return circles


def _perturb_radii(circles, scale=0.01):
    """Slightly shrink radii to allow position relaxation."""
    circles = [list(c) for c in circles]
    for i in range(N):
        circles[i][2] = max(0.005, circles[i][2] * (1.0 - random.uniform(0, 0.1)))
    return circles


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
    corners = [(0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9)]
    for cx, cy in corners:
        circles.append([cx, cy, 0.06])
    edges = [(0.5, 0.08), (0.5, 0.92), (0.08, 0.5), (0.92, 0.5)]
    for ex, ey in edges:
        circles.append([ex, ey, 0.05])
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


def _operator_alternation(circles, deadline, n_rounds=30):
    """
    Alternate grow_radii and relax_positions operators.
    Both maintain hard feasibility. This is the core search mechanism.
    """
    circles = [tuple(c) for c in circles]
    best_circles = circles
    best_sum = _sum_radii(circles)
    
    for rnd in range(n_rounds):
        if time.time() > deadline - 20.0:
            break
        
        # Grow radii (positions fixed)
        circles = _grow_radii(circles, deadline)
        s = _sum_radii(circles)
        if s > best_sum:
            best_sum = s
            best_circles = circles
        
        # Relax positions (radii fixed)
        circles = _relax_positions(circles, deadline, n_iters=200)
        s = _sum_radii(circles)
        if s > best_sum:
            best_sum = s
            best_circles = circles
        
        # Occasionally perturb to escape local optima
        if rnd % 5 == 4 and rnd < n_rounds - 5:
            circles = _perturb_positions(circles, scale=0.02)
            circles = _perturb_radii(circles)
            # Re-feasibilize
            circles = _grow_radii(circles, deadline)
            s = _sum_radii(circles)
            if s > best_sum:
                best_sum = s
                best_circles = circles
    
    return best_circles, best_sum


def _slsqp_polish(circles, deadline):
    """SLSQP polish with pairwise constraints."""
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
            options={'maxiter': 80, 'ftol': 1e-10}
        )
        if result.success:
            new_circles = [(result.x[3*i], result.x[3*i+1], result.x[3*i+2]) for i in range(N)]
            if _is_valid(new_circles):
                return new_circles
    except Exception:
        pass
    return circles


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
        _init_random(999),
    ]
    
    best_circles = None
    best_sum = 0.0
    
    for init_params in inits:
        if time.time() > deadline - 30.0:
            break
        
        circles = [tuple(init_params[i]) for i in range(N)]
        
        # First: grow radii to make feasible and maximize for these positions
        circles = _grow_radii(circles, deadline)
        
        # Operator alternation search
        circles, s = _operator_alternation(circles, deadline, n_rounds=25)
        
        # SLSQP polish
        if time.time() < deadline - 20.0:
            polished = _slsqp_polish(circles, deadline)
            polished_sum = _sum_radii(polished)
            if polished_sum > s and _is_valid(polished):
                circles = polished
                s = polished_sum
        
        # Final grow after polish
        circles = _grow_radii(circles, deadline)
        s = _sum_radii(circles)
        
        if s > best_sum and _is_valid(circles):
            best_circles = circles
            best_sum = s
    
    # Final polish on best
    if time.time() < deadline - 10.0 and best_circles is not None:
        polished = _slsqp_polish(best_circles, deadline)
        polished_sum = _sum_radii(polished)
        if polished_sum > best_sum and _is_valid(polished):
            best_circles = polished
            best_sum = polished_sum
    
    # Final grow
    if best_circles is not None:
        best_circles = _grow_radii(best_circles, deadline)
        best_sum = _sum_radii(best_circles)
    
    # Post-process
    if best_circles is not None:
        best_circles = _post_process(best_circles)
    else:
        best_circles = _post_process(_init_grid().tolist())
    
    # Verify and return
    if not _is_valid(best_circles):
        best_circles = _post_process(_init_grid().tolist())
    
    return np.array(best_circles)

# EVOLVE-BLOCK-END
