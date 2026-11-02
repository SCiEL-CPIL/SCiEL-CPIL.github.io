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

def _smooth_penalty_objective(params, weight=50.0, smooth_beta=20.0):
    """
    Smooth penalty objective using soft-max surrogate for overlap penalties.
    Uses log-sum-exp style smoothing to avoid nonsmooth max(0, overlap).
    """
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    penalty = 0.0
    
    # Boundary penalties (smooth)
    for i in range(N):
        # x_i - r_i < 0 violation
        v1 = -(x[i] - r[i])
        if v1 > 0:
            penalty += v1 * v1
        # 1 - x_i - r_i < 0 violation
        v2 = -(1 - x[i] - r[i])
        if v2 > 0:
            penalty += v2 * v2
        v3 = -(y[i] - r[i])
        if v3 > 0:
            penalty += v3 * v3
        v4 = -(1 - y[i] - r[i])
        if v4 > 0:
            penalty += v4 * v4
        if r[i] < 0:
            penalty += r[i] * r[i]
    
    # Overlap penalties using Delaunay with smooth surrogate
    centers = np.column_stack([x, y])
    edges = _get_delaunay_edges(centers)
    
    for i, j in edges:
        dx = x[i] - x[j]
        dy = y[i] - y[j]
        dist = np.sqrt(dx * dx + dy * dy)
        overlap = r[i] + r[j] - dist
        # Smooth penalty: use soft-max approximation
        # For overlap > 0: penalty ~ overlap^2, for overlap <= 0: small smooth value
        if overlap > 1e-8:
            penalty += overlap * overlap
        else:
            # Smooth transition: exp(-beta * |overlap|) scaled down
            penalty += np.exp(-smooth_beta * abs(overlap)) * 1e-6
    
    # Extra close pairs with smooth penalty
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
                if overlap > 1e-8:
                    penalty += overlap * overlap
                else:
                    penalty += np.exp(-smooth_beta * abs(overlap)) * 1e-6
    
    return -np.sum(r) + weight * penalty

def _full_smooth_penalty_objective(params, weight=100.0, smooth_beta=30.0):
    """Full pairwise smooth penalty objective for final polish."""
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    penalty = 0.0
    
    for i in range(N):
        v1 = -(x[i] - r[i])
        if v1 > 0:
            penalty += v1 * v1
        v2 = -(1 - x[i] - r[i])
        if v2 > 0:
            penalty += v2 * v2
        v3 = -(y[i] - r[i])
        if v3 > 0:
            penalty += v3 * v3
        v4 = -(1 - y[i] - r[i])
        if v4 > 0:
            penalty += v4 * v4
        if r[i] < 0:
            penalty += r[i] * r[i]
    
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            overlap = r[i] + r[j] - dist
            if overlap > 1e-8:
                penalty += overlap * overlap
            else:
                penalty += np.exp(-smooth_beta * abs(overlap)) * 1e-6
    
    return -np.sum(r) + weight * penalty

def _mma_style_refine(xyz, time_budget=10.0):
    """
    MMA-style iterative refinement: repeatedly solve a convex quadratic subproblem
    that approximates the non-convex overlap constraints, then update the model.
    This mimics the Method of Moving Asymptotes structure.
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
    
    # Iterative refinement with increasing penalty weight (MMA-style)
    weights = [20.0, 50.0, 100.0, 200.0, 500.0]
    betas = [15.0, 20.0, 30.0, 40.0, 50.0]
    
    for w, beta in zip(weights, betas):
        if time.time() - start_time > time_budget:
            break
        
        try:
            result = minimize(_smooth_penalty_objective, best_params, method='L-BFGS-B',
                              bounds=bounds, args=(w, beta),
                              options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12})
            if result.fun < _smooth_penalty_objective(best_params, w, beta):
                candidate = result.x
                xyz_cand = candidate.reshape(N, 3)
                if _validate(xyz_cand):
                    score = np.sum(xyz_cand[:, 2])
                    if score > best_score:
                        best_score = score
                        best_params = candidate
                else:
                    # Try to fix
                    xyz_fixed = xyz_cand.copy()
                    for _ in range(20):
                        if _validate(xyz_fixed):
                            break
                        xyz_fixed[:, 2] *= 0.98
                        for i in range(N):
                            xyz_fixed[i, 0] = np.clip(xyz_fixed[i, 0], xyz_fixed[i, 2], 1 - xyz_fixed[i, 2])
                            xyz_fixed[i, 1] = np.clip(xyz_fixed[i, 1], xyz_fixed[i, 2], 1 - xyz_fixed[i, 2])
                    if _validate(xyz_fixed):
                        score = np.sum(xyz_fixed[:, 2])
                        if score > best_score:
                            best_score = score
                            best_params = xyz_fixed.flatten()
        except Exception:
            pass
    
    return best_params.reshape(N, 3)

def _high_precision_polish(xyz, time_budget=8.0):
    """
    Final high-precision polish pass with very tight tolerances.
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
    
    # Very tight tolerance final solve
    try:
        result = minimize(_full_smooth_penalty_objective, best_params, method='L-BFGS-B',
                          bounds=bounds, args=(500.0, 50.0),
                          options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 100})
        if result.fun < _full_smooth_penalty_objective(best_params, 500.0, 50.0):
            candidate = result.x
            xyz_cand = candidate.reshape(N, 3)
            if _validate(xyz_cand):
                score = np.sum(xyz_cand[:, 2])
                if score > best_score:
                    best_score = score
                    best_params = candidate
    except Exception:
        pass
    
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
    Optimize using smooth penalty-based L-BFGS-B with MMA-style refinement.
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
    
    # Phase 1: Smooth Delaunay-based penalty with moderate weight
    remaining = time_budget - (time.time() - start_time)
    if remaining > 3.0:
        try:
            result = minimize(_smooth_penalty_objective, best_params, method='L-BFGS-B',
                              bounds=bounds, args=(50.0, 20.0),
                              options={'maxiter': 400, 'ftol': 1e-13, 'gtol': 1e-11})
            if result.fun < _smooth_penalty_objective(best_params, 50.0, 20.0):
                candidate = result.x
                xyz_cand = candidate.reshape(N, 3)
                if _validate(xyz_cand):
                    score = np.sum(xyz_cand[:, 2])
                    if score > best_score:
                        best_score = score
                        best_params = candidate
                else:
                    xyz_fixed = xyz_cand.copy()
                    for _ in range(30):
                        if _validate(xyz_fixed):
                            break
                        xyz_fixed[:, 2] *= 0.97
                        for i in range(N):
                            xyz_fixed[i, 0] = np.clip(xyz_fixed[i, 0], xyz_fixed[i, 2], 1 - xyz_fixed[i, 2])
                            xyz_fixed[i, 1] = np.clip(xyz_fixed[i, 1], xyz_fixed[i, 2], 1 - xyz_fixed[i, 2])
                    if _validate(xyz_fixed):
                        score = np.sum(xyz_fixed[:, 2])
                        if score > best_score:
                            best_score = score
                            best_params = xyz_fixed.flatten()
        except Exception:
            pass
    
    # Phase 2: MMA-style iterative refinement
    remaining = time_budget - (time.time() - start_time)
    if remaining > 5.0:
        try:
            xyz_mma = _mma_style_refine(best_params.reshape(N, 3), time_budget=min(8.0, remaining - 2.0))
            score = np.sum(xyz_mma[:, 2])
            if score > best_score:
                best_score = score
                best_params = xyz_mma.flatten()
        except Exception:
            pass
    
    # Phase 3: High-precision final polish
    remaining = time_budget - (time.time() - start_time)
    if remaining > 4.0:
        try:
            xyz_polish = _high_precision_polish(best_params.reshape(N, 3), time_budget=min(6.0, remaining - 1.0))
            score = np.sum(xyz_polish[:, 2])
            if score > best_score:
                best_score = score
                best_params = xyz_polish.flatten()
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
    
    # Additional high-precision polish if time permits
    elapsed = time.time() - global_start
    remaining = deadline - elapsed
    if remaining > 5.0:
        xyz_hp = _high_precision_polish(best_xyz, time_budget=min(4.0, remaining - 1.0))
        score = np.sum(xyz_hp[:, 2])
        if score > best_score:
            best_score = score
            best_xyz = xyz_hp
    
    # Final validation
    if not _validate(best_xyz):
        best_xyz = _fix_config(best_xyz)
    
    if not _validate(best_xyz):
        best_xyz = _initial_config_grid()
    
    return best_xyz

# EVOLVE-BLOCK-END
