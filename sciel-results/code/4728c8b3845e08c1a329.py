"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# EVOLVE-BLOCK-END

def _validate_and_fix(xyz):
    """Ensure circles are inside [0,1]^2 with positive radii. Returns (xyz, valid)."""
    xyz = np.array(xyz, dtype=float).copy()
    if xyz.shape != (N, 3):
        return None, False
    # Clip positions to [0,1]
    xyz[:, 0] = np.clip(xyz[:, 0], 0.0, 1.0)
    xyz[:, 1] = np.clip(xyz[:, 1], 0.0, 1.0)
    # Ensure radii positive
    xyz[:, 2] = np.clip(xyz[:, 2], 1e-6, 0.5)
    return xyz, True

def _check_validity(xyz, tol=1e-7):
    """Check if all circles are disjoint and inside the unit square."""
    if xyz is None:
        return False
    for i in range(N):
        x, y, r = xyz[i]
        if x - r < -tol or x + r > 1 + tol or y - r < -tol or y + r > 1 + tol or r <= 0:
            return False
    for i in range(N):
        for j in range(i+1, N):
            dx = xyz[i, 0] - xyz[j, 0]
            dy = xyz[i, 1] - xyz[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < xyz[i, 2] + xyz[j, 2] - tol:
                return False
    return True

def _score(xyz):
    """Compute the objective: sum of radii."""
    if xyz is None:
        return 0.0
    return float(np.sum(xyz[:, 2]))

def _make_grid_initial():
    """Grid-based initial configuration."""
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side - 0.001
            cells.append([x, y, r])
    return np.array(cells[:N])

def _make_hex_initial():
    """Hexagonal packing initial configuration."""
    # Hexagonal lattice for better packing
    rows = int(np.ceil(N / 4))
    cells = []
    for row in range(rows):
        for col in range(4):
            if len(cells) >= N:
                break
            x = (col + 0.5 + (row % 2) * 0.5) / 4.0
            y = (row + 0.5) / float(rows)
            r = 0.22
            cells.append([x, y, r])
    return np.array(cells[:N])

def _make_random_initial(seed):
    """Random initial configuration."""
    rng = np.random.RandomState(seed)
    xyz = np.zeros((N, 3))
    xyz[:, 0] = rng.uniform(0.1, 0.9, N)
    xyz[:, 1] = rng.uniform(0.1, 0.9, N)
    xyz[:, 2] = rng.uniform(0.05, 0.15, N)
    return xyz

def _objective_and_grad(params, beta=100.0):
    """
    Objective: minimize -sum(radii) + penalty for violations.
    params: flattened array of size 3N (x, y, r for each circle)
    """
    xyz = params.reshape(N, 3)
    x = xyz[:, 0]
    y = xyz[:, 1]
    r = xyz[:, 2]
    
    # Main objective: negative sum of radii
    obj = -np.sum(r)
    grad_obj = np.zeros_like(params)
    grad_obj[2::3] = -1.0  # gradient of -sum(r) w.r.t. r
    
    # Penalty terms
    penalty = 0.0
    grad_pen = np.zeros_like(params)
    
    # Circle-boundary constraints: r <= x, r <= 1-x, r <= y, r <= 1-y
    # Use smooth max: softplus(beta * violation)
    for i in range(N):
        # x_i - r_i >= 0  => violation = r_i - x_i
        v1 = r[i] - x[i]
        if v1 > 0:
            p = np.log1p(np.exp(beta * v1)) / beta
            g = 1.0 / (1 + np.exp(-beta * v1))
            penalty += p
            grad_pen[3*i] -= g      # d/dx_i
            grad_pen[3*i+2] += g    # d/dr_i
        
        v2 = r[i] - (1 - x[i])
        if v2 > 0:
            p = np.log1p(np.exp(beta * v2)) / beta
            g = 1.0 / (1 + np.exp(-beta * v2))
            penalty += p
            grad_pen[3*i] += g
            grad_pen[3*i+2] += g
        
        v3 = r[i] - y[i]
        if v3 > 0:
            p = np.log1p(np.exp(beta * v3)) / beta
            g = 1.0 / (1 + np.exp(-beta * v3))
            penalty += p
            grad_pen[3*i+1] -= g
            grad_pen[3*i+2] += g
        
        v4 = r[i] - (1 - y[i])
        if v4 > 0:
            p = np.log1p(np.exp(beta * v4)) / beta
            g = 1.0 / (1 + np.exp(-beta * v4))
            penalty += p
            grad_pen[3*i+1] += g
            grad_pen[3*i+2] += g
    
    # Circle-circle constraints: dist(i,j) >= r_i + r_j
    for i in range(N):
        for j in range(i+1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy) + 1e-10
            rsum = r[i] + r[j]
            violation = rsum - dist
            if violation > 0:
                p = np.log1p(np.exp(beta * violation)) / beta
                g = 1.0 / (1 + np.exp(-beta * violation))
                penalty += p
                # Gradients w.r.t. positions
                gx = -g * dx / dist
                gy = -g * dy / dist
                grad_pen[3*i] += gx
                grad_pen[3*i+1] += gy
                grad_pen[3*j] -= gx
                grad_pen[3*j+1] -= gy
                # Gradients w.r.t. radii
                grad_pen[3*i+2] += g
                grad_pen[3*j+2] += g
    
    total_obj = obj + penalty
    total_grad = grad_obj + grad_pen
    return total_obj, total_grad

def _optimize_from_initial(xyz_init, deadline, max_iter=200):
    """Run L-BFGS-B optimization from a given initial configuration."""
    params = xyz_init.flatten().copy()
    
    # Progressive beta values for annealing
    betas = [50.0, 100.0, 200.0, 500.0]
    best_xyz = xyz_init.copy()
    best_score = _score(xyz_init)
    
    for beta in betas:
        if time.time() > deadline - 5.0:
            break
        
        def obj_grad(p):
            return _objective_and_grad(p, beta=beta)
        
        try:
            res = minimize(
                obj_grad,
                params,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if res.success or res.fun < obj_grad(params)[0]:
                params = res.x
                xyz_trial = params.reshape(N, 3)
                xyz_trial, valid = _validate_and_fix(xyz_trial)
                if valid and _check_validity(xyz_trial):
                    s = _score(xyz_trial)
                    if s > best_score:
                        best_score = s
                        best_xyz = xyz_trial.copy()
        except Exception:
            pass
    
    return best_xyz, best_score

def _high_precision_polish(xyz, deadline):
    """Final polish with very high beta for tight packing."""
    if time.time() > deadline - 10.0:
        return xyz, _score(xyz)
    
    params = xyz.flatten().copy()
    best_xyz = xyz.copy()
    best_score = _score(xyz)
    
    for beta in [1000.0, 2000.0, 5000.0]:
        if time.time() > deadline - 5.0:
            break
        
        def obj_grad(p):
            return _objective_and_grad(p, beta=beta)
        
        try:
            res = minimize(
                obj_grad,
                params,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            params = res.x
            xyz_trial = params.reshape(N, 3)
            xyz_trial, valid = _validate_and_fix(xyz_trial)
            if valid and _check_validity(xyz_trial):
                s = _score(xyz_trial)
                if s > best_score:
                    best_score = s
                    best_xyz = xyz_trial.copy()
        except Exception:
            pass
    
    return best_xyz, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin: 10s before 150s kill
    
    # Generate multiple initial configurations
    inits = [
        _make_grid_initial(),
        _make_hex_initial(),
        _make_random_initial(42),
        _make_random_initial(123),
        _make_random_initial(999),
    ]
    
    best_xyz = None
    best_score = 0.0
    
    # Try each initial configuration
    for i, xyz_init in enumerate(inits):
        if time.time() > deadline - 15.0:
            break
        
        xyz_init, valid = _validate_and_fix(xyz_init)
        if not valid:
            continue
        
        xyz_res, score = _optimize_from_initial(xyz_init, deadline, max_iter=150)
        if score > best_score:
            best_score = score
            best_xyz = xyz_res
    
    # Final high-precision polish
    if best_xyz is not None and time.time() < deadline - 10.0:
        best_xyz, best_score = _high_precision_polish(best_xyz, deadline)
    
    if best_xyz is None:
        # Fallback to grid
        best_xyz = _make_grid_initial()
    
    # Final validation
    best_xyz, valid = _validate_and_fix(best_xyz)
    if not valid or not _check_validity(best_xyz):
        best_xyz = _make_grid_initial()
    
    return best_xyz

# EVOLVE-BLOCK-END
