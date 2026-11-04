"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

DEADLINE_SECONDS = 115.0  # safety margin before 120s limit


def _soft_min_penalty(dists, r1, r2, beta):
    """
    Soft penalty for overlap: for each pair, if dist < r1+r2, penalty = softplus(beta*(r1+r2 - dist))
    This is smooth and encourages separation.
    """
    # dists shape (n, n), r1, r2 shape (n, n)
    overlap = r1 + r2 - dists
    # softplus: log(1 + exp(x)) / beta approximates max(0, x) as beta -> inf
    # Use stable version
    max_val = np.max(overlap)
    if max_val > 50:
        # overflow risk; use approximation
        penalty = np.where(overlap > 0, overlap, 0)
    else:
        penalty = np.logaddexp(0, beta * overlap) / beta
    return penalty


def _objective(params, n, beta, wall_deadline):
    """
    params: 3n vector: x[0:n], y[n:2n], r[2n:3n]
    Returns (value, grad) or just value.
    Objective to MINIMIZE: -sum(r) + penalty for overlaps + penalty for boundary violations
    """
    x = params[0:n]
    y = params[n:2*n]
    r = params[2*n:3*n]
    
    # Boundary constraints: circles must be inside [0,1]^2
    # x - r >= 0, x + r <= 1, y - r >= 0, y + r <= 1
    boundary_penalty = 0.0
    boundary_grad_x = np.zeros(n)
    boundary_grad_y = np.zeros(n)
    boundary_grad_r = np.zeros(n)
    
    # x - r >= 0 => penalty if x - r < 0
    viol_x_min = np.maximum(0, r - x)
    boundary_penalty += np.sum(viol_x_min) ** 2
    boundary_grad_x -= 2 * viol_x_min
    boundary_grad_r += 2 * viol_x_min
    
    viol_x_max = np.maximum(0, x + r - 1)
    boundary_penalty += np.sum(viol_x_max) ** 2
    boundary_grad_x += 2 * viol_x_max
    boundary_grad_r += 2 * viol_x_max
    
    viol_y_min = np.maximum(0, r - y)
    boundary_penalty += np.sum(viol_y_min) ** 2
    boundary_grad_y -= 2 * viol_y_min
    boundary_grad_r += 2 * viol_y_min
    
    viol_y_max = np.maximum(0, y + r - 1)
    boundary_penalty += np.sum(viol_y_max) ** 2
    boundary_grad_y += 2 * viol_y_max
    boundary_grad_r += 2 * viol_y_max
    
    # Overlap penalty
    if n > 1:
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dists = np.sqrt(dx**2 + dy**2 + 1e-12)
        r_sum = r[:, None] + r[None, :]
        
        overlap = r_sum - dists
        # Only consider upper triangle to avoid double counting, but for gradient we need all
        # Use softplus for smoothness
        # penalty = sum over i<j of softplus(beta * overlap_ij)
        # For simplicity, use full matrix but divide by 2
        max_overlap = np.max(overlap)
        if max_overlap > 50:
            soft_pen = np.where(overlap > 0, overlap, 0)
            soft_grad = np.where(overlap > 0, 1, 0)
        else:
            soft_pen = np.logaddexp(0, beta * overlap) / beta
            soft_grad = np.exp(beta * overlap) / (1 + np.exp(beta * overlap))
        
        # Mask diagonal
        mask = ~np.eye(n, dtype=bool)
        overlap_pen = np.sum(soft_pen[mask]) / 2.0
        overlap_grad = np.zeros((n, n))
        overlap_grad[mask] = soft_grad[mask]
        # Symmetrize for gradient contribution
        overlap_grad = (overlap_grad + overlap_grad.T) / 2.0
        overlap_grad -= np.diag(np.diag(overlap_grad))
        
        # Gradient of overlap penalty w.r.t. x, y, r
        # d/dx_i sum softplus(beta*(r_i + r_j - dist_ij)) 
        # = sum_j soft_grad_ij * beta * (-dx_ij / dist_ij)
        # where dx_ij = x_i - x_j
        dists_safe = np.where(dists < 1e-10, 1e-10, dists)
        gx = -beta * np.sum(overlap_grad * dx / dists_safe, axis=1)
        gy = -beta * np.sum(overlap_grad * dy / dists_safe, axis=1)
        gr = beta * np.sum(overlap_grad, axis=1)
        
        # Combine: total overlap penalty contribution to objective
        boundary_penalty += overlap_pen
        boundary_grad_x += gx
        boundary_grad_y += gy
        boundary_grad_r += gr
    
    # Objective: -sum(r) + boundary_penalty + overlap_pen
    value = -np.sum(r) + boundary_penalty
    grad = np.concatenate([boundary_grad_x, boundary_grad_y, boundary_grad_r - np.ones(n)])
    
    return value, grad


def _objective_value_only(params, n, beta):
    """Value only, for quick checks."""
    x = params[0:n]
    y = params[n:2*n]
    r = params[2*n:3*n]
    
    # Check validity
    if np.any(x < -0.1) or np.any(x > 1.1) or np.any(y < -0.1) or np.any(y > 1.1) or np.any(r < -0.1):
        return 1e10
    
    # Boundary violations
    if np.any(x - r < -1e-6) or np.any(x + r > 1 + 1e-6) or np.any(y - r < -1e-6) or np.any(y + r > 1 + 1e-6):
        return 1e10
    
    # Overlap check
    for i in range(n):
        for j in range(i+1, n):
            d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
            if d < r[i] + r[j] - 1e-6:
                return 1e10
    
    return -np.sum(r)


def _is_valid(params, n, tol=1e-7):
    x = params[0:n]
    y = params[n:2*n]
    r = params[2*n:3*n]
    if np.any(r < -tol):
        return False
    if np.any(x - r < -tol) or np.any(x + r > 1 + tol) or np.any(y - r < -tol) or np.any(y + r > 1 + tol):
        return False
    for i in range(n):
        for j in range(i+1, n):
            d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
            if d < r[i] + r[j] - tol:
                return False
    return True


def _extract_best(params, n):
    """Extract (x, y, r) from params, ensuring validity."""
    x = params[0:n].copy()
    y = params[n:2*n].copy()
    r = params[2*n:3*n].copy()
    # Clip to valid range
    r = np.maximum(r, 1e-6)
    x = np.clip(x, r, 1 - r)
    y = np.clip(y, r, 1 - r)
    return x, y, r


def construct():
    start_time = time.time()
    n = N
    best_value = -1e10
    best_params = None
    
    # Initial guesses: various configurations
    inits = []
    
    # 1. Grid initialization (5x6 or 6x5)
    for grid_shape in [(5, 6), (6, 5), (4, 7), (7, 4)]:
        rows, cols = grid_shape
        total = rows * cols
        if total < n:
            continue
        x = np.zeros(n)
        y = np.zeros(n)
        r = np.zeros(n)
        for i in range(n):
            row = i // cols
            col = i % cols
            x[i] = (col + 0.5) / cols
            y[i] = (row + 0.5) / rows
            r[i] = min(0.5 / cols, 0.5 / rows) * 0.95
        inits.append(np.concatenate([x, y, r]))
    
    # 2. Hexagonal packing
    # Approximate hexagonal arrangement
    hex_inits = []
    for offset in [0, 0.1, 0.2]:
        x = np.zeros(n)
        y = np.zeros(n)
        r = np.zeros(n)
        # Place circles in hexagonal pattern
        idx = 0
        for row in range(10):
            for col in range(10):
                if idx >= n:
                    break
                x[idx] = col * 0.12 + (0.06 if row % 2 == 0 else 0.12) + offset * 0.01
                y[idx] = row * 0.11 + offset * 0.01
                r[idx] = 0.055
                idx += 1
                if idx >= n:
                    break
        if idx == n:
            hex_inits.append(np.concatenate([x, y, r]))
    inits.extend(hex_inits)
    
    # 3. Random initializations
    rng = np.random.default_rng(42)
    for _ in range(5):
        x = rng.uniform(0.1, 0.9, n)
        y = rng.uniform(0.1, 0.9, n)
        r = rng.uniform(0.03, 0.08, n)
        inits.append(np.concatenate([x, y, r]))
    
    # 4. Concentric rings
    for ring_r in [0.15, 0.2]:
        x = np.zeros(n)
        y = np.zeros(n)
        r = np.zeros(n)
        # Center circle
        x[0] = 0.5
        y[0] = 0.5
        r[0] = ring_r * 0.5
        # Ring of circles
        count = n - 1
        angle_step = 2 * np.pi / count
        for i in range(1, n):
            angle = (i - 1) * angle_step
            x[i] = 0.5 + ring_r * 0.8 * np.cos(angle)
            y[i] = 0.5 + ring_r * 0.8 * np.sin(angle)
            r[i] = ring_r * 0.3
        inits.append(np.concatenate([x, y, r]))
    
    # Optimize each initialization
    for init_idx, init_params in enumerate(inits):
        if time.time() - start_time > DEADLINE_SECONDS - 5:
            break
        
        # Multi-stage beta annealing
        current_params = init_params.copy()
        
        for beta in [10, 50, 200, 1000, 5000]:
            if time.time() - start_time > DEADLINE_SECONDS - 5:
                break
            
            # Try L-BFGS-B
            try:
                result = minimize(
                    _objective,
                    current_params,
                    args=(n, beta),
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if result.success or result.fun < 1e8:
                    current_params = result.x
            except Exception:
                pass
            
            # Check if valid
            if _is_valid(current_params, n):
                val = -_objective_value_only(current_params, n, 10000)
                if val > best_value:
                    best_value = val
                    best_params = current_params.copy()
        
        # Final high-precision polish
        if time.time() - start_time < DEADLINE_SECONDS - 2:
            for beta in [5000, 20000]:
                try:
                    result = minimize(
                        _objective,
                        current_params,
                        args=(n, beta),
                        method='L-BFGS-B',
                        jac=True,
                        options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
                    )
                    if result.success or result.fun < 1e8:
                        current_params = result.x
                        if _is_valid(current_params, n):
                            val = -_objective_value_only(current_params, n, 10000)
                            if val > best_value:
                                best_value = val
                                best_params = current_params.copy()
                except Exception:
                    pass
    
    # Fallback: use the best valid result or a simple grid
    if best_params is not None:
        x, y, r = _extract_best(best_params, n)
        # Final validation
        if _is_valid(np.concatenate([x, y, r]), n):
            return np.column_stack([x, y, r])
    
    # Fallback: simple grid
    side = int(np.ceil(np.sqrt(n)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    return np.array(cells[:n])
# EVOLVE-BLOCK-END
