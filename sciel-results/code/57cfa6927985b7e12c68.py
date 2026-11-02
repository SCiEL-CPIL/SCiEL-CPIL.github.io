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

def _slsqp_objective(params):
    """Objective: minimize -sum(r) for SLSQP."""
    r = params[2::3]
    return -np.sum(r)

def _slsqp_constraints(params):
    """
    Build constraint functions for SLSQP.
    Each constraint is of the form: f(params) >= 0
    """
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    cons = []
    
    # Wall constraints: x_i - r_i >= 0, etc.
    for i in range(N):
        # x_i - r_i >= 0
        cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i] - p[3*i+2]})
        # 1 - x_i - r_i >= 0
        cons.append({'type': 'ineq', 'fun': lambda p, i=i: 1 - p[3*i] - p[3*i+2]})
        # y_i - r_i >= 0
        cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i+1] - p[3*i+2]})
        # 1 - y_i - r_i >= 0
        cons.append({'type': 'ineq', 'fun': lambda p, i=i: 1 - p[3*i+1] - p[3*i+2]})
    
    # Overlap constraints: dist(i,j) - r_i - r_j >= 0
    # Use Delaunay to reduce the number of constraints
    centers = np.column_stack([x, y])
    edges = _get_delaunay_edges(centers)
    
    # Add all Delaunay edges
    for i, j in edges:
        cons.append({
            'type': 'ineq',
            'fun': lambda p, i=i, j=j: np.sqrt((p[3*i]-p[3*j])**2 + (p[3*i+1]-p[3*j+1])**2) - p[3*i+2] - p[3*j+2]
        })
    
    # Also add a few extra close pairs for safety
    for i in range(N):
        for j in range(i + 1, N):
            if (i, j) in edges or (j, i) in edges:
                continue
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx*dx + dy*dy
            # Only add if potentially close
            if dist_sq < 4.0 * (r[i] + r[j])**2:
                cons.append({
                    'type': 'ineq',
                    'fun': lambda p, i=i, j=j: np.sqrt((p[3*i]-p[3*j])**2 + (p[3*i+1]-p[3*j+1])**2) - p[3*i+2] - p[3*j+2]
                })
    
    return cons

def _penalty_objective(params, weight=50.0):
    """Penalty-based objective for L-BFGS-B."""
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    penalty = 0.0
    
    # Boundary penalties
    for i in range(N):
        if x[i] - r[i] < 0:
            penalty += (x[i] - r[i]) ** 2
        if x[i] + r[i] > 1:
            penalty += (x[i] + r[i] - 1) ** 2
        if y[i] - r[i] < 0:
            penalty += (y[i] - r[i]) ** 2
        if y[i] + r[i] > 1:
            penalty += (y[i] + r[i] - 1) ** 2
        if r[i] < 0:
            penalty += r[i] ** 2
    
    # Overlap penalties using Delaunay
    centers = np.column_stack([x, y])
    edges = _get_delaunay_edges(centers)
    
    for i, j in edges:
        dx = x[i] - x[j]
        dy = y[i] - y[j]
        dist = np.sqrt(dx * dx + dy * dy)
        overlap = r[i] + r[j] - dist
        if overlap > 0:
            penalty += overlap ** 2
    
    # Extra close pairs
    for i in range(N):
        for j in range(i + 1, N):
            if (i, j) in edges or (j, i) in edges:
                continue
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx * dx + dy * dy
            if dist_sq < 4.0 * (r[i] + r[j])**2:
                dist = np.sqrt(dist_sq)
                overlap = r[i] + r[j] - dist
                if overlap > 0:
                    penalty += overlap ** 2
    
    return -np.sum(r) + weight * penalty

def _full_penalty_objective(params, weight=100.0):
    """Full pairwise penalty objective for final polish."""
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    penalty = 0.0
    
    for i in range(N):
        if x[i] - r[i] < 0:
            penalty += (x[i] - r[i]) ** 2
        if x[i] + r[i] > 1:
            penalty += (x[i] + r[i] - 1) ** 2
        if y[i] - r[i] < 0:
            penalty += (y[i] - r[i]) ** 2
        if y[i] + r[i] > 1:
            penalty += (y[i] + r[i] - 1) ** 2
        if r[i] < 0:
            penalty += r[i] ** 2
    
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            overlap = r[i] + r[j] - dist
            if overlap > 0:
                penalty += overlap ** 2
    
    return -np.sum(r) + weight * penalty

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

def _initial_config_rings():
    """Concentric rings initial configuration."""
    positions = []
    positions.append([0.5, 0.5, 0.10])
    for k in range(8):
        angle = 2 * np.pi * k / 8
        x = 0.5 + 0.22 * np.cos(angle)
        y = 0.5 + 0.22 * np.sin(angle)
        positions.append([x, y, 0.10 * 0.9])
    for k in range(10):
        angle = 2 * np.pi * k / 10 + np.pi / 10
        x = 0.5 + 0.35 * np.cos(angle)
        y = 0.5 + 0.35 * np.sin(angle)
        positions.append([x, y, 0.08 * 0.9])
    for k in range(8):
        angle = 2 * np.pi * k / 8
        x = 0.5 + 0.48 * np.cos(angle)
        y = 0.5 + 0.48 * np.sin(angle)
        positions.append([x, y, 0.06 * 0.9])
    return np.array(positions[:N])

def _slsqp_optimize(xyz, time_budget=20.0):
    """
    Optimize using SLSQP with explicit inequality constraints.
    Returns the best valid configuration found.
    """
    start_time = time.time()
    params = xyz.flatten().copy()
    
    bounds = []
    for i in range(N):
        bounds.append((0, 1))  # x
        bounds.append((0, 1))  # y
        bounds.append((0, 0.5))  # r
    
    best_params = params.copy()
    best_score = np.sum(params[2::3])
    
    try:
        cons = _slsqp_constraints(params)
        result = minimize(_slsqp_objective, params, method='SLSQP',
                          bounds=bounds, constraints=cons,
                          options={'maxiter': 300, 'ftol': 1e-10})
        if result.success or result.fun < _slsqp_objective(params):
            candidate = result.x
            # Validate and adjust
            xyz_cand = candidate.reshape(N, 3)
            if _validate(xyz_cand):
                if np.sum(xyz_cand[:, 2]) > best_score:
                    best_params = candidate
                    best_score = np.sum(candidate[2::3])
            else:
                # Try to fix
                xyz_fixed = xyz_cand.copy()
                for _ in range(50):
                    if _validate(xyz_fixed):
                        break
                    xyz_fixed[:, 2] *= 0.97
                    for i in range(N):
                        x, y, r = xyz_fixed[i]
                        xyz_fixed[i, 0] = np.clip(x, r, 1 - r)
                        xyz_fixed[i, 1] = np.clip(y, r, 1 - r)
                if _validate(xyz_fixed) and np.sum(xyz_fixed[:, 2]) > best_score:
                    best_params = xyz_fixed.flatten()
                    best_score = np.sum(xyz_fixed[:, 2])
    except Exception:
        pass
    
    return best_params.reshape(N, 3)

def _penalty_optimize(xyz, time_budget=20.0):
    """
    Optimize using penalty-based L-BFGS-B.
    """
    start_time = time.time()
    params = xyz.flatten().copy()
    best_params = params.copy()
    best_score = np.sum(params[2::3])
    
    bounds = []
    for i in range(N):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((0, 0.5))
    
    # Phase 1: Delaunay-based penalty
    try:
        result = minimize(_penalty_objective, best_params, method='L-BFGS-B',
                          bounds=bounds, args=(50.0,),
                          options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
        if result.fun < _penalty_objective(best_params, 50.0):
            best_params = result.x
            best_score = np.sum(best_params[2::3])
    except Exception:
        pass
    
    if time.time() - start_time > time_budget:
        return best_params.reshape(N, 3)
    
    # Phase 2: Full pairwise polish
    try:
        result = minimize(_full_penalty_objective, best_params, method='L-BFGS-B',
                          bounds=bounds, args=(100.0,),
                          options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10})
        if result.fun < _full_penalty_objective(best_params, 100.0):
            best_params = result.x
            best_score = np.sum(best_params[2::3])
    except Exception:
        pass
    
    return best_params.reshape(N, 3)

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
    deadline = 110.0  # Safety margin before 120s limit
    
    best_xyz = _initial_config_grid()
    best_score = np.sum(best_xyz[:, 2])
    
    # Generate initial configurations
    initial_configs = []
    initial_configs.append(_initial_config_grid())
    initial_configs.append(_initial_config_hex())
    initial_configs.append(_initial_config_rings())
    
    # Random configurations
    np.random.seed(42)
    for trial in range(4):
        x = np.random.uniform(0.1, 0.9, N)
        y = np.random.uniform(0.1, 0.9, N)
        r = np.random.uniform(0.05, 0.15, N)
        for i in range(N):
            x[i] = np.clip(x[i], r[i], 1 - r[i])
            y[i] = np.clip(y[i], r[i], 1 - r[i])
        initial_configs.append(np.column_stack([x, y, r]))
    
    # Optimize from each initial configuration
    for idx, xyz0 in enumerate(initial_configs):
        elapsed = time.time() - global_start
        remaining = deadline - elapsed
        if remaining < 5.0:
            break
        
        # Try SLSQP first (structural constraint approach)
        time_budget_slsqp = min(15.0, remaining - 3.0)
        if time_budget_slsqp > 2.0:
            xyz_slsqp = _slsqp_optimize(xyz0, time_budget=time_budget_slsqp)
            score = np.sum(xyz_slsqp[:, 2])
            if score > best_score:
                best_score = score
                best_xyz = xyz_slsqp
        
        elapsed = time.time() - global_start
        remaining = deadline - elapsed
        if remaining < 5.0:
            break
        
        # Then try penalty-based L-BFGS-B (refinement)
        time_budget_pen = min(15.0, remaining - 3.0)
        if time_budget_pen > 2.0:
            xyz_pen = _penalty_optimize(xyz0, time_budget=time_budget_pen)
            score = np.sum(xyz_pen[:, 2])
            if score > best_score:
                best_score = score
                best_xyz = xyz_pen
    
    # Final polish on best configuration
    elapsed = time.time() - global_start
    remaining = deadline - elapsed
    if remaining > 10.0:
        xyz_final = _penalty_optimize(best_xyz, time_budget=min(10.0, remaining - 2.0))
        score = np.sum(xyz_final[:, 2])
        if score > best_score:
            best_score = score
            best_xyz = xyz_final
    
    # Final validation
    if not _validate(best_xyz):
        best_xyz = _fix_config(best_xyz)
    
    if not _validate(best_xyz):
        best_xyz = _initial_config_grid()
    
    return best_xyz

# EVOLVE-BLOCK-END
