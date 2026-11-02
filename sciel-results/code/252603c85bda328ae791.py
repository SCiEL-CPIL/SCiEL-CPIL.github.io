"""Circle packing in unit square: 26 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 26

# EVOLVE-BLOCK-START

def _validate(xyz):
    """Check if configuration is valid: circles inside unit square and non-overlapping."""
    n = xyz.shape[0]
    x = xyz[:, 0]
    y = xyz[:, 1]
    r = xyz[:, 2]
    if np.any(r < -1e-9):
        return False
    if np.any(x - r < -1e-7) or np.any(x + r > 1 + 1e-7):
        return False
    if np.any(y - r < -1e-7) or np.any(y + r > 1 + 1e-7):
        return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-7:
                return False
    return True

def _get_delaunay_edges(centers):
    """Get Delaunay triangulation edges for the given centers."""
    try:
        tri = Delaunay(centers)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                a, b = simplex[i], simplex[(i + 1) % 3]
                if a > b:
                    a, b = b, a
                edges.add((a, b))
        return list(edges)
    except Exception:
        return [(i, j) for i in range(N) for j in range(i + 1, N)]

def _compute_penalty_and_grad(params, edges, penalty_weight=50.0):
    """
    Compute penalty and gradient for Delaunay-based objective.
    params: [x_0, y_0, r_0, ..., x_25, y_25, r_25]
    Returns: (value, grad)
    """
    x = params[0::3].copy()
    y = params[1::3].copy()
    r = params[2::3].copy()
    
    penalty = 0.0
    grad = np.zeros_like(params)
    
    # Boundary penalties
    for i in range(N):
        # x - r >= 0
        v = x[i] - r[i]
        if v < 0:
            p = v * v
            penalty += p
            grad[3*i] += -2*v
            grad[3*i+2] += 2*v
        # x + r <= 1
        v = x[i] + r[i] - 1
        if v > 0:
            p = v * v
            penalty += p
            grad[3*i] += 2*v
            grad[3*i+2] += 2*v
        # y - r >= 0
        v = y[i] - r[i]
        if v < 0:
            p = v * v
            penalty += p
            grad[3*i+1] += -2*v
            grad[3*i+2] += 2*v
        # y + r <= 1
        v = y[i] + r[i] - 1
        if v > 0:
            p = v * v
            penalty += p
            grad[3*i+1] += 2*v
            grad[3*i+2] += 2*v
        # r >= 0
        if r[i] < 0:
            p = r[i] * r[i]
            penalty += p
            grad[3*i+2] += 2*r[i]
    
    # Overlap penalties for Delaunay edges
    for i, j in edges:
        dx = x[i] - x[j]
        dy = y[i] - y[j]
        dist_sq = dx * dx + dy * dy
        dist = np.sqrt(dist_sq) if dist_sq > 1e-12 else 1e-12
        overlap = r[i] + r[j] - dist
        if overlap > 0:
            p = overlap * overlap
            penalty += p
            # d(overlap)/dx_i = -dx/dist, d(overlap)/dx_j = dx/dist
            # d(overlap)/dr_i = 1, d(overlap)/dr_j = 1
            gp = 2 * overlap
            grad[3*i] += gp * (-dx / dist)
            grad[3*i+1] += gp * (-dy / dist)
            grad[3*i+2] += gp
            grad[3*j] += gp * (dx / dist)
            grad[3*j+1] += gp * (dy / dist)
            grad[3*j+2] += gp
    
    # Objective: -sum(r) + penalty_weight * penalty
    value = -np.sum(r) + penalty_weight * penalty
    # Gradient: -1 for r components, penalty_weight * grad_penalty for others
    grad[2::3] -= 1.0
    grad *= penalty_weight
    # But we need to be careful: the -sum(r) contributes -1 to each r gradient
    # So total grad = penalty_weight * grad_penalty - e_r
    # Let me redo this more carefully
    
    # Actually, let me recompute:
    # f = -sum(r) + w * P
    # df/dr_i = -1 + w * dP/dr_i
    # df/dx_i = w * dP/dx_i
    # df/dy_i = w * dP/dy_i
    
    grad = np.zeros_like(params)
    for i in range(N):
        grad[3*i+2] = -1.0  # from -sum(r)
    
    # Add penalty gradient contributions
    for i in range(N):
        v = x[i] - r[i]
        if v < 0:
            gp = -2 * v
            grad[3*i] += penalty_weight * gp
            grad[3*i+2] += penalty_weight * (-gp)  # d(v)/dr = -1
        v = x[i] + r[i] - 1
        if v > 0:
            gp = 2 * v
            grad[3*i] += penalty_weight * gp
            grad[3*i+2] += penalty_weight * gp
        v = y[i] - r[i]
        if v < 0:
            gp = -2 * v
            grad[3*i+1] += penalty_weight * gp
            grad[3*i+2] += penalty_weight * (-gp)
        v = y[i] + r[i] - 1
        if v > 0:
            gp = 2 * v
            grad[3*i+1] += penalty_weight * gp
            grad[3*i+2] += penalty_weight * gp
        if r[i] < 0:
            gp = 2 * r[i]
            grad[3*i+2] += penalty_weight * gp
    
    for i, j in edges:
        dx = x[i] - x[j]
        dy = y[i] - y[j]
        dist_sq = dx * dx + dy * dy
        dist = np.sqrt(dist_sq) if dist_sq > 1e-12 else 1e-12
        overlap = r[i] + r[j] - dist
        if overlap > 0:
            gp = 2 * overlap
            grad[3*i] += penalty_weight * gp * (-dx / dist)
            grad[3*i+1] += penalty_weight * gp * (-dy / dist)
            grad[3*i+2] += penalty_weight * gp
            grad[3*j] += penalty_weight * gp * (dx / dist)
            grad[3*j+1] += penalty_weight * gp * (dy / dist)
            grad[3*j+2] += penalty_weight * gp
    
    return value, grad

def _objective_delaunay(params, edges_cache, penalty_weight=50.0):
    """Objective using Delaunay-based constraints."""
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    centers = np.column_stack([x, y])
    edges = _get_delaunay_edges(centers)
    edges_cache['edges'] = edges
    
    penalty = 0.0
    for i in range(N):
        v = x[i] - r[i]
        if v < 0:
            penalty += v * v
        v = x[i] + r[i] - 1
        if v > 0:
            penalty += v * v
        v = y[i] - r[i]
        if v < 0:
            penalty += v * v
        v = y[i] + r[i] - 1
        if v > 0:
            penalty += v * v
        if r[i] < 0:
            penalty += r[i] * r[i]
    
    for i, j in edges:
        dx = x[i] - x[j]
        dy = y[i] - y[j]
        dist = np.sqrt(dx * dx + dy * dy)
        overlap = r[i] + r[j] - dist
        if overlap > 0:
            penalty += overlap * overlap
    
    return -np.sum(r) + penalty_weight * penalty

def _objective_full(params, penalty_weight=100.0):
    """Full pairwise objective for final polish."""
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    penalty = 0.0
    for i in range(N):
        v = x[i] - r[i]
        if v < 0:
            penalty += v * v
        v = x[i] + r[i] - 1
        if v > 0:
            penalty += v * v
        v = y[i] - r[i]
        if v < 0:
            penalty += v * v
        v = y[i] + r[i] - 1
        if v > 0:
            penalty += v * v
        if r[i] < 0:
            penalty += r[i] * r[i]
    
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            overlap = r[i] + r[j] - dist
            if overlap > 0:
                penalty += overlap * overlap
    
    return -np.sum(r) + penalty_weight * penalty

def _gauss_newton_step(params, edges, step_size=0.01):
    """
    Perform a Gauss-Newton-like step on the contact constraints.
    For each tight constraint (overlap or boundary violation), compute
    the direction that reduces the violation most efficiently.
    """
    x = params[0::3].copy()
    y = params[1::3].copy()
    r = params[2::3].copy()
    
    dx = np.zeros_like(x)
    dy = np.zeros_like(y)
    dr = np.zeros_like(r)
    
    # Boundary constraints
    for i in range(N):
        # x_i - r_i >= 0: if violated, push x right or r left
        v = x[i] - r[i]
        if v < 0:
            # Increase x or decrease r
            dx[i] += -v
            dr[i] += v  # decrease r
        # x_i + r_i <= 1
        v = x[i] + r[i] - 1
        if v > 0:
            dx[i] -= v
            dr[i] -= v
        # y_i - r_i >= 0
        v = y[i] - r[i]
        if v < 0:
            dy[i] += -v
            dr[i] += v
        # y_i + r_i <= 1
        v = y[i] + r[i] - 1
        if v > 0:
            dy[i] -= v
            dr[i] -= v
        # r_i >= 0
        if r[i] < 0:
            dr[i] += -r[i]
    
    # Overlap constraints for Delaunay edges
    for i, j in edges:
        ddx = x[i] - x[j]
        ddy = y[i] - y[j]
        dist_sq = ddx * ddx + ddy * ddy
        dist = np.sqrt(dist_sq) if dist_sq > 1e-12 else 1e-12
        overlap = r[i] + r[j] - dist
        if overlap > 0:
            # Separate along the line connecting centers
            ux = ddx / dist
            uy = ddy / dist
            # Push apart: i moves in +u direction, j in -u direction
            # Also reduce radii
            sep = overlap / 2
            dx[i] += sep * ux
            dy[i] += sep * uy
            dx[j] -= sep * ux
            dy[j] -= sep * uy
            dr[i] -= sep
            dr[j] -= sep
    
    return np.array([dx, dy, dr]).reshape(3, N).T.flatten()

def _optimize_multi_phase(xyz, time_budget=30.0):
    """
    Multi-phase optimization:
    1. Gauss-Newton relaxation to fix overlaps
    2. L-BFGS-B with Delaunay penalty
    3. L-BFGS-B with full penalty for polish
    4. Final Gauss-Newton polish
    """
    start_time = time.time()
    params = xyz.flatten().copy()
    best_params = params.copy()
    best_score = np.sum(params[2::3])
    
    edges_cache = {'edges': None}
    
    bounds = []
    for i in range(N):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((0, 0.5))
    
    # Phase 0: Gauss-Newton relaxation
    for _ in range(50):
        if time.time() - start_time > time_budget * 0.1:
            break
        centers = np.column_stack([params[0::3], params[1::3]])
        edges = _get_delaunay_edges(centers)
        step = _gauss_newton_step(params, edges, step_size=0.05)
        # Damped update
        alpha = 0.5
        new_params = params + alpha * step
        # Clip to bounds
        for i in range(N):
            new_params[3*i] = np.clip(new_params[3*i], 0, 1)
            new_params[3*i+1] = np.clip(new_params[3*i+1], 0, 1)
            new_params[3*i+2] = np.clip(new_params[3*i+2], 0, 0.5)
        # Accept if score doesn't decrease too much
        new_score = np.sum(new_params[2::3])
        if new_score >= best_score - 0.01:
            params = new_params
            if new_score > best_score:
                best_score = new_score
                best_params = params.copy()
    
    # Phase 1: L-BFGS-B with low penalty
    if time.time() - start_time < time_budget * 0.4:
        try:
            result = minimize(lambda p: _objective_delaunay(p, edges_cache, 10.0),
                              params, method='L-BFGS-B',
                              bounds=bounds,
                              options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8})
            if result.fun < _objective_delaunay(params, edges_cache, 10.0):
                params = result.x
                score = np.sum(params[2::3])
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
        except Exception:
            pass
    
    # Phase 2: L-BFGS-B with medium penalty
    if time.time() - start_time < time_budget * 0.65:
        try:
            result = minimize(lambda p: _objective_delaunay(p, edges_cache, 50.0),
                              params, method='L-BFGS-B',
                              bounds=bounds,
                              options={'maxiter': 400, 'ftol': 1e-12, 'gtol': 1e-10})
            if result.fun < _objective_delaunay(params, edges_cache, 50.0):
                params = result.x
                score = np.sum(params[2::3])
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
        except Exception:
            pass
    
    # Phase 3: Full pairwise polish
    if time.time() - start_time < time_budget * 0.85:
        try:
            result = minimize(lambda p: _objective_full(p, 200.0),
                              params, method='L-BFGS-B',
                              bounds=bounds,
                              options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12})
            if result.fun < _objective_full(params, 200.0):
                params = result.x
                score = np.sum(params[2::3])
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
        except Exception:
            pass
    
    # Phase 4: Final Gauss-Newton polish
    for _ in range(30):
        if time.time() - start_time > time_budget:
            break
        centers = np.column_stack([params[0::3], params[1::3]])
        edges = _get_delaunay_edges(centers)
        step = _gauss_newton_step(params, edges, step_size=0.02)
        alpha = 0.3
        new_params = params + alpha * step
        for i in range(N):
            new_params[3*i] = np.clip(new_params[3*i], 0, 1)
            new_params[3*i+1] = np.clip(new_params[3*i+1], 0, 1)
            new_params[3*i+2] = np.clip(new_params[3*i+2], 0, 0.5)
        new_score = np.sum(new_params[2::3])
        if new_score >= best_score - 0.005:
            params = new_params
            if new_score > best_score:
                best_score = new_score
                best_params = params.copy()
    
    return best_params.reshape(N, 3)

def _initial_config_grid():
    """Simple grid initial configuration."""
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side * 0.9
            cells.append([x, y, r])
    return np.array(cells[:N])

def _initial_config_hex():
    """Hexagonal-like initial configuration."""
    side = 5
    positions = []
    for i in range(side):
        for j in range(side):
            if j % 2 == 1:
                x = (i + 0.5) / side + 0.1 / side
            else:
                x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side * 0.95
            positions.append([x, y, r])
    positions.append([0.5, 0.5, 0.5 / side * 0.5])
    return np.array(positions[:N])

def _initial_config_dense_center():
    """Dense center with smaller outer circles."""
    positions = []
    positions.append([0.5, 0.5, 0.12])
    for k in range(6):
        angle = 2 * np.pi * k / 6
        x = 0.5 + 0.15 * np.cos(angle)
        y = 0.5 + 0.15 * np.sin(angle)
        positions.append([x, y, 0.09])
    for k in range(12):
        angle = 2 * np.pi * k / 12 + np.pi / 12
        x = 0.5 + 0.25 * np.cos(angle)
        y = 0.5 + 0.25 * np.sin(angle)
        positions.append([x, y, 0.07])
    for k in range(7):
        angle = 2 * np.pi * k / 7
        x = 0.5 + 0.38 * np.cos(angle)
        y = 0.5 + 0.38 * np.sin(angle)
        positions.append([x, y, 0.055])
    return np.array(positions[:N])

def _initial_config_mixed():
    """Mixed sizes: a few large, many small."""
    positions = []
    positions.append([0.3, 0.3, 0.11])
    positions.append([0.7, 0.3, 0.11])
    positions.append([0.5, 0.7, 0.11])
    for k in range(8):
        angle = 2 * np.pi * k / 8
        x = 0.5 + 0.28 * np.cos(angle)
        y = 0.5 + 0.28 * np.sin(angle)
        positions.append([x, y, 0.065])
    np.random.seed(123)
    for k in range(15):
        x = np.random.uniform(0.08, 0.92)
        y = np.random.uniform(0.08, 0.92)
        r = np.random.uniform(0.03, 0.05)
        positions.append([x, y, r])
    return np.array(positions[:N])

def _fix_config(xyz):
    """Try to fix an invalid configuration by reducing radii and adjusting positions."""
    xyz_fixed = xyz.copy()
    for _ in range(100):
        if _validate(xyz_fixed):
            break
        xyz_fixed[:, 2] *= 0.95
        for i in range(N):
            x, y, r = xyz_fixed[i]
            xyz_fixed[i, 0] = np.clip(x, r, 1 - r)
            xyz_fixed[i, 1] = np.clip(y, r, 1 - r)
    return xyz_fixed

def construct():
    """
    Construct 26 circles with disjoint interiors inside [0,1]^2,
    maximizing the sum of their radii.
    """
    global_start = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    best_xyz = _initial_config_grid()
    best_score = np.sum(best_xyz[:, 2])
    
    # Generate initial configurations
    initial_configs = []
    initial_configs.append(_initial_config_grid())
    initial_configs.append(_initial_config_hex())
    initial_configs.append(_initial_config_dense_center())
    initial_configs.append(_initial_config_mixed())
    
    # Random configurations
    for seed in [42, 123, 456]:
        np.random.seed(seed)
        x = np.random.uniform(0.1, 0.9, N)
        y = np.random.uniform(0.1, 0.9, N)
        r = np.random.uniform(0.05, 0.15, N)
        for i in range(N):
            x[i] = np.clip(x[i], r[i], 1 - r[i])
            y[i] = np.clip(y[i], r[i], 1 - r[i])
        initial_configs.append(np.column_stack([x, y, r]))
    
    # Corner-biased random
    np.random.seed(789)
    x = np.concatenate([
        np.random.uniform(0.05, 0.3, 7),
        np.random.uniform(0.7, 0.95, 7),
        np.random.uniform(0.05, 0.3, 6),
        np.random.uniform(0.7, 0.95, 6)
    ])
    y = np.concatenate([
        np.random.uniform(0.05, 0.3, 7),
        np.random.uniform(0.05, 0.3, 7),
        np.random.uniform(0.7, 0.95, 6),
        np.random.uniform(0.7, 0.95, 6)
    ])
    r = np.random.uniform(0.06, 0.12, N)
    for i in range(N):
        x[i] = np.clip(x[i], r[i], 1 - r[i])
        y[i] = np.clip(y[i], r[i], 1 - r[i])
    initial_configs.append(np.column_stack([x, y, r]))
    
    # Optimize from each initial configuration
    n_configs = len(initial_configs)
    for idx, xyz0 in enumerate(initial_configs):
        elapsed = time.time() - global_start
        remaining = deadline - elapsed
        if remaining < 10.0:
            break
        
        time_budget = min(18.0, remaining - 5.0)
        if time_budget < 4.0:
            break
        
        xyz_opt = _optimize_multi_phase(xyz0, time_budget=time_budget)
        
        if not _validate(xyz_opt):
            xyz_opt = _fix_config(xyz_opt)
        
        score = np.sum(xyz_opt[:, 2])
        if score > best_score:
            best_score = score
            best_xyz = xyz_opt
    
    # Final validation
    if not _validate(best_xyz):
        best_xyz = _fix_config(best_xyz)
    
    if not _validate(best_xyz):
        best_xyz = _initial_config_grid()
    
    return best_xyz

# EVOLVE-BLOCK-END
