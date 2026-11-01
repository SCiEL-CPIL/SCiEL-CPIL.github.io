import numpy as np
import time
from scipy.optimize import minimize

N = 32

def _validity_check(x, y, r):
    """Check if circles are valid (inside square, no overlaps)."""
    if np.any(x < r) or np.any(x > 1 - r) or np.any(y < r) or np.any(y > 1 - r):
        return False
    for i in range(N):
        for j in range(i + 1, N):
            d2 = (x[i] - x[j])**2 + (y[i] - y[j])**2
            if d2 < (r[i] + r[j])**2 - 1e-9:
                return False
    return True

def _score(x, y, r):
    """Sum of radii."""
    return np.sum(r)

def _hex_seed():
    """Generate hexagonal lattice positions for 32 circles."""
    positions = []
    # Hexagonal packing with staggered rows
    row_spacing = 0.18
    col_spacing = 0.18
    for row in range(8):
        y = 0.1 + row * row_spacing
        if y > 0.95:
            break
        offset = 0.09 if row % 2 == 1 else 0.0
        for col in range(8):
            x = 0.1 + col * col_spacing + offset
            if x > 0.95:
                break
            positions.append((x, y))
            if len(positions) >= N:
                break
        if len(positions) >= N:
            break
    # Pad if needed
    while len(positions) < N:
        positions.append((0.5, 0.5))
    return np.array(positions[:N])

def _smooth_penalty(v, penalty_weight=100.0):
    """
    Smooth penalty function for L-BFGS-B optimization.
    Uses soft-min to create differentiable penalty for constraint violations.
    """
    x = v[:N]
    y = v[N:2*N]
    r = v[2*N:]
    
    # Objective: maximize sum of radii
    obj = -np.sum(r)
    
    # Boundary penalties (smooth)
    # x - r >= 0, 1 - x - r >= 0, y - r >= 0, 1 - y - r >= 0
    boundary_violations = np.concatenate([
        r - x,
        r + x - 1,
        r - y,
        r + y - 1
    ])
    # Soft penalty: sum of max(0, violation)^2, approximated smoothly
    boundary_penalty = np.sum(np.maximum(boundary_violations, 0)**2)
    
    # Overlap penalties (smooth)
    overlap_penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dr = r[i] + r[j]
            d2 = dx*dx + dy*dy
            # Violation: (r_i + r_j)^2 - d^2 > 0
            violation = dr*dr - d2
            # Smooth penalty
            overlap_penalty += np.maximum(violation, 0)**2
    
    return obj + penalty_weight * (boundary_penalty + overlap_penalty)

def _smooth_gradient(v, penalty_weight=100.0):
    """Gradient of smooth penalty function."""
    x = v[:N].copy()
    y = v[N:2*N].copy()
    r = v[2*N:].copy()
    
    grad = np.zeros_like(v)
    
    # Gradient of -sum(r)
    grad[2*N:] = -1.0
    
    # Boundary gradients
    # For x - r >= 0: violation = r - x, grad w.r.t. x = -1, w.r.t. r = 1
    # For 1 - x - r >= 0: violation = r + x - 1, grad w.r.t. x = 1, w.r.t. r = 1
    for i in range(N):
        # Left boundary: r_i - x_i
        viol = r[i] - x[i]
        if viol > 0:
            grad[i] += 2 * penalty_weight * viol * (-1)
            grad[2*N + i] += 2 * penalty_weight * viol * 1
        
        # Right boundary: r_i + x_i - 1
        viol = r[i] + x[i] - 1
        if viol > 0:
            grad[i] += 2 * penalty_weight * viol * 1
            grad[2*N + i] += 2 * penalty_weight * viol * 1
        
        # Bottom boundary: r_i - y_i
        viol = r[i] - y[i]
        if viol > 0:
            grad[N + i] += 2 * penalty_weight * viol * (-1)
            grad[2*N + i] += 2 * penalty_weight * viol * 1
        
        # Top boundary: r_i + y_i - 1
        viol = r[i] + y[i] - 1
        if viol > 0:
            grad[N + i] += 2 * penalty_weight * viol * 1
            grad[2*N + i] += 2 * penalty_weight * viol * 1
    
    # Overlap gradients
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dr = r[i] + r[j]
            d2 = dx*dx + dy*dy
            violation = dr*dr - d2
            if violation > 0:
                # d(violation)/dx_i = -2*dx, d/dx_j = 2*dx
                # d(violation)/dy_i = -2*dy, d/dy_j = 2*dy
                # d(violation)/dr_i = 2*dr, d/dr_j = 2*dr
                factor = 2 * penalty_weight * violation
                grad[i] += factor * (-2*dx)
                grad[j] += factor * (2*dx)
                grad[N + i] += factor * (-2*dy)
                grad[N + j] += factor * (2*dy)
                grad[2*N + i] += factor * (2*dr)
                grad[2*N + j] += factor * (2*dr)
    
    return grad

def _repulsion_optimize(init_pos, max_iter=500, penalty_start=10.0, penalty_end=1000.0):
    """
    Optimize using L-BFGS-B with smooth penalty.
    """
    r0 = np.full(N, 0.05)
    v0 = np.concatenate([init_pos[:, 0], init_pos[:, 1], r0])
    
    bounds = []
    for i in range(N):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((0.01, 0.45))
    
    best_v = v0.copy()
    best_score = -np.inf
    
    # Annealing penalty weight
    n_anneal = 5
    for k in range(n_anneal):
        penalty_w = penalty_start * (penalty_end / penalty_start) ** (k / max(n_anneal - 1, 1))
        
        def obj_func(v, pw=penalty_w):
            return _smooth_penalty(v, pw)
        
        def grad_func(v, pw=penalty_w):
            return _smooth_gradient(v, pw)
        
        try:
            res = minimize(obj_func, v0, jac=grad_func, method='L-BFGS-B',
                           bounds=bounds, options={'maxiter': max_iter // n_anneal, 'ftol': 1e-8, 'gtol': 1e-6})
            v0 = res.x
            
            # Check validity and score
            x, y, r = v0[:N], v0[N:2*N], v0[2*N:]
            if _validity_check(x, y, r):
                sc = _score(x, y, r)
                if sc > best_score:
                    best_score = sc
                    best_v = v0.copy()
        except Exception:
            pass
    
    return best_v, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 105.0  # Safety margin
    
    best_score = 0.0
    best_sol = None
    
    # Initializations
    inits = []
    
    # 1. Grid 6x6
    side = 6
    cells = [((i + 0.5) / side, (j + 0.5) / side) for i in range(side) for j in range(side)]
    inits.append(np.array(cells[:N]))
    
    # 2. Grid 8x4
    side_x, side_y = 8, 4
    cells = [((i + 0.5) / side_x, (j + 0.5) / side_y) for i in range(side_x) for j in range(side_y)]
    inits.append(np.array(cells[:N]))
    
    # 3. Hexagonal lattice
    inits.append(_hex_seed())
    
    # 4. Random
    rng = np.random.default_rng(42)
    inits.append(rng.uniform(0.1, 0.9, (N, 2)))
    
    # 5. Perturbed Grid
    base = inits[0].copy()
    base += rng.normal(0, 0.01, (N, 2))
    inits.append(np.clip(base, 0.05, 0.95))
    
    # 6. Perturbed Hex
    hex_base = _hex_seed()
    hex_base += rng.normal(0, 0.02, (N, 2))
    inits.append(np.clip(hex_base, 0.05, 0.95))
    
    for init_pos in inits:
        if time.time() > deadline:
            break
        
        v_opt, sc = _repulsion_optimize(init_pos, max_iter=300, penalty_start=5.0, penalty_end=500.0)
        
        if sc > best_score:
            best_score = sc
            best_sol = np.column_stack([v_opt[:N], v_opt[N:2*N], v_opt[2*N:]])
    
    # Final polish with SLSQP on best solution
    if best_sol is not None and time.time() < deadline - 5.0:
        v0 = best_sol.flatten()
        r = v0[2*N:]
        
        def objective(v):
            return -np.sum(v[2*N:])
        
        def constraint_func(v):
            x = v[:N]
            y = v[N:2*N]
            r = v[2*N:]
            
            c = np.empty(4*N + N*(N-1)//2)
            idx = 0
            c[idx:idx+N] = x - r
            idx += N
            c[idx:idx+N] = 1 - x - r
            idx += N
            c[idx:idx+N] = y - r
            idx += N
            c[idx:idx+N] = 1 - y - r
            idx += N
            
            for i in range(N):
                for j in range(i+1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dr = r[i] + r[j]
                    c[idx] = dx*dx + dy*dy - dr*dr
                    idx += 1
            return c
        
        bounds = []
        for i in range(N):
            bounds.append((0, 1))
            bounds.append((0, 1))
            bounds.append((0, 0.5))
        
        try:
            res = minimize(objective, v0, method='SLSQP', bounds=bounds,
                           constraints=[{'type': 'ineq', 'fun': constraint_func}],
                           options={'maxiter': 100, 'ftol': 1e-8})
            
            if res.success:
                x, y, r = res.x[:N], res.x[N:2*N], res.x[2*N:]
                if _validity_check(x, y, r):
                    sc = _score(x, y, r)
                    if sc > best_score:
                        best_score = sc
                        best_sol = np.column_stack([x, y, r])
        except Exception:
            pass
    
    if best_sol is None:
        # Fallback to seed
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_sol
