"""Circle packing in unit square: multi-phase optimization with grow-push and polish."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint

N = 26
DEADLINE = 140.0  # safety margin before 150s kill


def _is_valid(x, tol=1e-6):
    """Check if solution is valid: all circles inside unit square, non-overlapping."""
    x_c = x[:N]
    y_c = x[N:2*N]
    r = x[2*N:]
    
    # Check bounds
    if np.any(x_c < -tol) or np.any(x_c > 1 + tol):
        return False
    if np.any(y_c < -tol) or np.any(y_c > 1 + tol):
        return False
    if np.any(r < -tol):
        return False
    
    # Check boundary: circle must be inside [0,1]^2
    for i in range(N):
        if x_c[i] - r[i] < -tol or x_c[i] + r[i] > 1 + tol:
            return False
        if y_c[i] - r[i] < -tol or y_c[i] + r[i] > 1 + tol:
            return False
    
    # Check pairwise distances
    for i in range(N):
        for j in range(i + 1, N):
            dx = x_c[i] - x_c[j]
            dy = y_c[i] - y_c[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - tol:
                return False
    
    return True


def _objective(x):
    """Negative sum of radii (minimize negative to maximize)."""
    r = x[2*N:]
    return -np.sum(r)


def _constraints_func(x):
    """All constraints as g(x) >= 0."""
    x_c = x[:N]
    y_c = x[N:2*N]
    r = x[2*N:]
    
    cons = []
    
    # Boundary constraints
    for i in range(N):
        cons.append(x_c[i] - r[i])
        cons.append(1 - x_c[i] - r[i])
        cons.append(y_c[i] - r[i])
        cons.append(1 - y_c[i] - r[i])
        cons.append(r[i] - 0.001)  # small positive radius
    
    # Pairwise distance constraints
    for i in range(N):
        for j in range(i + 1, N):
            dx = x_c[i] - x_c[j]
            dy = y_c[i] - y_c[j]
            dist_sq = dx * dx + dy * dy
            min_dist = r[i] + r[j]
            cons.append(dist_sq - min_dist * min_dist)
    
    return np.array(cons)


def _jacobian_constraints(x):
    """Jacobian of all constraints."""
    x_c = x[:N]
    y_c = x[N:2*N]
    r = x[2*N:]
    
    n_constraints = 5 * N + N * (N - 1) // 2
    jac = np.zeros((n_constraints, 3 * N))
    
    idx = 0
    # Boundary constraints
    for i in range(N):
        # x_i - r_i >= 0
        jac[idx, i] = 1.0
        jac[idx, 2*N + i] = -1.0
        idx += 1
        # 1 - x_i - r_i >= 0
        jac[idx, i] = -1.0
        jac[idx, 2*N + i] = -1.0
        idx += 1
        # y_i - r_i >= 0
        jac[idx, N + i] = 1.0
        jac[idx, 2*N + i] = -1.0
        idx += 1
        # 1 - y_i - r_i >= 0
        jac[idx, N + i] = -1.0
        jac[idx, 2*N + i] = -1.0
        idx += 1
        # r_i - 0.001 >= 0
        jac[idx, 2*N + i] = 1.0
        idx += 1
    
    # Pairwise constraints
    for i in range(N):
        for j in range(i + 1, N):
            dx = x_c[i] - x_c[j]
            dy = y_c[i] - y_c[j]
            min_dist = r[i] + r[j]
            
            jac[idx, i] = 2 * dx
            jac[idx, j] = -2 * dx
            jac[idx, N + i] = 2 * dy
            jac[idx, N + j] = -2 * dy
            jac[idx, 2*N + i] = -2 * min_dist
            jac[idx, 2*N + j] = -2 * min_dist
            idx += 1
    
    return jac


def _gradient_objective(x):
    """Gradient of objective."""
    grad = np.zeros(3 * N)
    grad[2*N:] = -1.0
    return grad


def _make_corner_heavy():
    """Initial guess with large corner circles."""
    positions = []
    radii = []
    
    # 4 corner circles
    corner_r = 0.17
    corners = [(corner_r, corner_r), (1 - corner_r, corner_r),
               (corner_r, 1 - corner_r), (1 - corner_r, 1 - corner_r)]
    for x, y in corners:
        positions.append((x, y))
        radii.append(corner_r)
    
    # 8 edge circles
    edge_r = 0.11
    positions.extend([(0.3, 1 - edge_r), (0.7, 1 - edge_r)])
    radii.extend([edge_r, edge_r])
    positions.extend([(0.3, edge_r), (0.7, edge_r)])
    radii.extend([edge_r, edge_r])
    positions.extend([(edge_r, 0.3), (edge_r, 0.7)])
    radii.extend([edge_r, edge_r])
    positions.extend([(1 - edge_r, 0.3), (1 - edge_r, 0.7)])
    radii.extend([edge_r, edge_r])
    
    # Interior circles in hexagonal pattern
    interior_r = 0.075
    cx, cy = 0.5, 0.5
    # Hexagonal arrangement
    offsets = [(0, 0), (0.2, 0), (-0.2, 0), (0.1, 0.17), (-0.1, 0.17),
               (0.1, -0.17), (-0.1, -0.17), (0.3, 0.17), (-0.3, 0.17),
               (0.3, -0.17), (-0.3, -0.17), (0.2, 0.34), (-0.2, 0.34),
               (0.2, -0.34), (-0.2, -0.34)]
    
    for dx, dy in offsets:
        if len(positions) >= N:
            break
        x = np.clip(cx + dx, 0.05, 0.95)
        y = np.clip(cy + dy, 0.05, 0.95)
        positions.append((x, y))
        radii.append(interior_r)
    
    # Fill remaining
    while len(positions) < N:
        positions.append((0.5, 0.5))
        radii.append(0.04)
    
    x = np.array([p[0] for p in positions[:N]])
    y = np.array([p[1] for p in positions[:N]])
    r = np.array(radii[:N])
    
    return np.concatenate([x, y, r])


def _make_hexagonal():
    """Hexagonal packing initialization."""
    positions = []
    radii = []
    
    # Hexagonal grid
    r = 0.09
    dx = 2 * r * np.sqrt(3) / 2
    dy = 2 * r * 1.5
    
    row = 0
    while len(positions) < N:
        x_offset = 0 if row % 2 == 0 else dx / 2
        col = 0
        while len(positions) < N:
            x = r + x_offset + col * dx
            y = r + row * dy
            if x <= 1 - r and y <= 1 - r:
                positions.append((x, y))
                radii.append(r)
            col += 1
        row += 1
    
    # If we have more than N, trim
    positions = positions[:N]
    radii = radii[:N]
    
    # If we have fewer, add more
    while len(positions) < N:
        positions.append((0.5, 0.5))
        radii.append(0.05)
    
    x = np.array([p[0] for p in positions[:N]])
    y = np.array([p[1] for p in positions[:N]])
    r_arr = np.array(radii[:N])
    
    return np.concatenate([x, y, r_arr])


def _make_grid():
    """Grid initialization."""
    positions = []
    radii = []
    
    # 5x5 grid
    for i in range(5):
        for j in range(5):
            if len(positions) >= N:
                break
            x = 0.1 + i * 0.2
            y = 0.1 + j * 0.2
            positions.append((x, y))
            radii.append(0.08)
    
    while len(positions) < N:
        positions.append((0.5, 0.5))
        radii.append(0.05)
    
    x = np.array([p[0] for p in positions[:N]])
    y = np.array([p[1] for p in positions[:N]])
    r_arr = np.array(radii[:N])
    
    return np.concatenate([x, y, r_arr])


def _make_random(seed=42):
    """Random initialization."""
    rng = np.random.default_rng(seed)
    
    x = rng.uniform(0.1, 0.9, N)
    y = rng.uniform(0.1, 0.9, N)
    r = rng.uniform(0.05, 0.12, N)
    
    return np.concatenate([x, y, r])


def _grow_and_push(x, num_iters=50, growth_rate=0.002, push_strength=0.5):
    """Grow radii and push circles apart to resolve overlaps."""
    x_c = x[:N].copy()
    y_c = x[N:2*N].copy()
    r = x[2*N:].copy()
    
    for it in range(num_iters):
        # Grow radii slightly
        r *= (1 + growth_rate)
        
        # Push circles apart
        for _ in range(3):  # Multiple push iterations
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x_c[i] - x_c[j]
                    dy = y_c[i] - y_c[j]
                    dist = np.sqrt(dx * dx + dy * dy)
                    min_dist = r[i] + r[j]
                    
                    if dist < min_dist and dist > 1e-10:
                        # Push apart
                        overlap = min_dist - dist
                        push_x = (dx / dist) * overlap * push_strength * 0.5
                        push_y = (dy / dist) * overlap * push_strength * 0.5
                        x_c[i] += push_x
                        y_c[i] += push_y
                        x_c[j] -= push_x
                        y_c[j] -= push_y
                    elif dist < min_dist:
                        # Handle near-zero distance
                        x_c[i] += 0.001
                        x_c[j] -= 0.001
        
        # Push away from boundaries
        for i in range(N):
            if x_c[i] < r[i]:
                x_c[i] = r[i]
            if x_c[i] > 1 - r[i]:
                x_c[i] = 1 - r[i]
            if y_c[i] < r[i]:
                y_c[i] = r[i]
            if y_c[i] > 1 - r[i]:
                y_c[i] = 1 - r[i]
    
    return np.concatenate([x_c, y_c, r])


def _polish(x, deadline):
    """High-precision polish using SLSQP with tighter tolerances."""
    if time.time() > deadline - 5:
        return x
    
    def slsqp_objective(x):
        return -np.sum(x[2*N:])
    
    def slsqp_constraints(x):
        cons = []
        x_c = x[:N]
        y_c = x[N:2*N]
        r = x[2*N:]
        
        # Boundary
        for i in range(N):
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[i] - x[2*N + i]})
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[i] - x[2*N + i]})
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[N + i] - x[2*N + i]})
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[N + i] - x[2*N + i]})
        
        # Pairwise
        for i in range(N):
            for j in range(i + 1, N):
                cons.append({
                    'type': 'ineq',
                    'fun': lambda x, i=i, j=j:
                        (x[i] - x[j])**2 + (x[N + i] - x[N + j])**2 -
                        (x[2*N + i] + x[2*N + j])**2
                })
        return cons
    
    slsqp_bounds = [(0, 1)] * N + [(0, 1)] * N + [(0.001, 0.5)] * N
    
    try:
        res = minimize(
            slsqp_objective,
            x,
            method='SLSQP',
            bounds=slsqp_bounds,
            constraints=slsqp_constraints(),
            options={'maxiter': 500, 'ftol': 1e-14}
        )
        
        if res.fun < _objective(x) and _is_valid(res.x):
            return res.x
    except Exception:
        pass
    
    return x


def _trust_constr_optimize(x, deadline):
    """Run trust-constr optimization."""
    if time.time() > deadline - 10:
        return x
    
    bounds = []
    for i in range(N):
        bounds.append((0, 1))
    for i in range(N):
        bounds.append((0, 1))
    for i in range(N):
        bounds.append((0.001, 0.5))
    
    constraint = NonlinearConstraint(
        _constraints_func,
        0, np.inf,
        jac=_jacobian_constraints
    )
    
    try:
        res = minimize(
            _objective,
            x,
            method='trust-constr',
            jac=_gradient_objective,
            bounds=bounds,
            constraints=[constraint],
            options={'maxiter': 300, 'gtol': 1e-8, 'xtol': 1e-10}
        )
        
        if res.fun < _objective(x) and _is_valid(res.x):
            return res.x
    except Exception:
        pass
    
    return x


def _simulated_annealing(x, deadline, max_iters=1000):
    """Simulated annealing to escape local minima."""
    if time.time() > deadline - 20:
        return x
    
    current_x = x.copy()
    current_obj = _objective(current_x)
    best_x = current_x.copy()
    best_obj = current_obj
    
    T = 0.05  # Initial temperature
    cooling_rate = 0.995
    
    rng = np.random.default_rng(123)
    
    for it in range(max_iters):
        if time.time() > deadline - 10:
            break
        
        # Propose a perturbation
        x_new = current_x.copy()
        # Perturb positions
        x_new[:N] += rng.normal(0, 0.02, N)
        x_new[N:2*N] += rng.normal(0, 0.02, N)
        # Perturb radii
        x_new[2*N:] *= (1 + rng.normal(0, 0.05, N))
        # Clip to valid ranges
        x_new[:N] = np.clip(x_new[:N], 0, 1)
        x_new[N:2*N] = np.clip(x_new[N:2*N], 0, 1)
        x_new[2*N:] = np.clip(x_new[2*N:], 0.001, 0.5)
        
        # Check validity with penalty
        x_c = x_new[:N]
        y_c = x_new[N:2*N]
        r = x_new[2*N:]
        
        penalty = 0
        # Boundary penalty
        for i in range(N):
            if x_c[i] < r[i]:
                penalty += (r[i] - x_c[i]) ** 2
            if x_c[i] > 1 - r[i]:
                penalty += (x_c[i] - (1 - r[i])) ** 2
            if y_c[i] < r[i]:
                penalty += (r[i] - y_c[i]) ** 2
            if y_c[i] > 1 - r[i]:
                penalty += (y_c[i] - (1 - r[i])) ** 2
        
        # Overlap penalty
        for i in range(N):
            for j in range(i + 1, N):
                dx = x_c[i] - x_c[j]
                dy = y_c[i] - y_c[j]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    penalty += (min_dist - dist) ** 2
        
        new_obj = _objective(x_new) + penalty * 10
        
        # Acceptance criterion
        if new_obj < current_obj:
            current_x = x_new
            current_obj = new_obj
            if _is_valid(x_new) and new_obj < best_obj:
                best_x = x_new.copy()
                best_obj = new_obj
        else:
            # Accept with probability exp(-delta / T)
            delta = new_obj - current_obj
            if delta < 0:
                current_x = x_new
                current_obj = new_obj
            else:
                if np.log(rng.random()) < -delta / T:
                    current_x = x_new
                    current_obj = new_obj
        
        T *= cooling_rate
    
    return best_x


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Generate multiple initializations
    initializations = [
        _make_corner_heavy(),
        _make_hexagonal(),
        _make_grid(),
        _make_random(42),
        _make_random(123),
    ]
    
    best_x = initializations[0].copy()
    best_obj = _objective(best_x)
    
    # Try each initialization
    for init_idx, x0 in enumerate(initializations):
        if time.time() > deadline - 30:
            break
        
        x = x0.copy()
        
        # Grow and push phase
        if time.time() < deadline - 25:
            x = _grow_and_push(x, num_iters=30, growth_rate=0.003, push_strength=0.6)
        
        # Trust-constr optimization
        if time.time() < deadline - 15:
            x = _trust_constr_optimize(x, deadline)
        
        # Simulated annealing
        if time.time() < deadline - 10:
            x = _simulated_annealing(x, deadline, max_iters=300)
        
        # Final polish
        if time.time() < deadline - 5:
            x = _polish(x, deadline)
        
        # Update best
        if _is_valid(x):
            obj = _objective(x)
            if obj < best_obj:
                best_x = x.copy()
                best_obj = obj
    
    # Final polish on the best solution
    if time.time() < deadline - 3:
        best_x = _polish(best_x, deadline)
    
    # Ensure validity
    if not _is_valid(best_x):
        # Fallback to a valid solution
        x_c = best_x[:N]
        y_c = best_x[N:2*N]
        r = best_x[2*N:]
        
        # Shrink radii to make valid
        for i in range(N):
            r[i] = min(r[i], x_c[i], 1 - x_c[i], y_c[i], 1 - y_c[i])
            r[i] = max(r[i], 0.001)
        
        best_x = np.concatenate([x_c, y_c, r])
    
    x_c = best_x[:N]
    y_c = best_x[N:2*N]
    r = best_x[2*N:]
    
    result = np.column_stack([x_c, y_c, r])
    return result
