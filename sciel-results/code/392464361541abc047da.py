"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _dist_to_boundary(x, y):
    """Minimum distance from point (x,y) to boundary of [0,1]^2."""
    return min(x, y, 1.0 - x, 1.0 - y)

def _compute_radii(pos):
    """Given positions (n,2), compute max feasible radii for each circle."""
    n = len(pos)
    radii = np.zeros(n)
    for i in range(n):
        # Distance to boundary
        r_boundary = _dist_to_boundary(pos[i, 0], pos[i, 1])
        # Distance to other circles
        r_others = np.inf
        for j in range(n):
            if i != j:
                d = np.linalg.norm(pos[i] - pos[j])
                r_others = min(r_others, d / 2.0)
        radii[i] = min(r_boundary, r_others)
    return radii

def _objective_sum_radii(pos):
    """Negative sum of radii (for minimization)."""
    radii = _compute_radii(pos)
    return -np.sum(radii)

def _objective_grad(pos):
    """Gradient of negative sum of radii using finite differences."""
    n = len(pos)
    grad = np.zeros_like(pos)
    eps = 1e-7
    f0 = _objective_sum_radii(pos)
    for i in range(n):
        for d in range(2):
            pos_plus = pos.copy()
            pos_plus[i, d] += eps
            f_plus = _objective_sum_radii(pos_plus)
            grad[i, d] = (f_plus - f0) / eps
    return grad

def _is_valid(pos, radii, tol=1e-9):
    """Check if circles are valid: inside square and non-overlapping."""
    n = len(pos)
    for i in range(n):
        x, y = pos[i]
        r = radii[i]
        if x - r < -tol or x + r > 1.0 + tol or y - r < -tol or y + r > 1.0 + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pos[i] - pos[j])
            if d < radii[i] + radii[j] - tol:
                return False
    return True

def _initial_hexagonal(n):
    """Generate initial positions in a hexagonal-like pattern."""
    # Use a grid with hexagonal offset
    cols = int(np.ceil(np.sqrt(n * 1.2)))
    rows = int(np.ceil(n / cols))
    
    pos = []
    for row in range(rows):
        for col in range(cols):
            if len(pos) >= n:
                break
            x = (col + 0.5 + 0.5 * (row % 2)) / (cols + 0.5)
            y = (row + 0.5) / (rows + 0.5)
            # Clamp to [0.05, 0.95]
            x = np.clip(x, 0.05, 0.95)
            y = np.clip(y, 0.05, 0.95)
            pos.append([x, y])
    
    # If we don't have enough, add random points
    while len(pos) < n:
        pos.append([np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9)])
    
    return np.array(pos[:n])

def _refine_positions(pos, max_iter=200, deadline=None):
    """Refine positions using L-BFGS-B to maximize sum of radii."""
    n = len(pos)
    
    # Bounds: keep circles well inside the square
    bounds = [(0.01, 0.99)] * (2 * n)
    
    result = minimize(
        _objective_sum_radii,
        pos,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': max_iter, 'maxfun': max_iter * 10, 'ftol': 1e-12, 'gtol': 1e-10}
    )
    
    if deadline is not None:
        if time.time() > deadline:
            return pos  # Return current position if we're out of time
    
    return result.x.reshape(n, 2)

def _simulated_annealing(pos, n_iter=500, t_start=0.05, t_end=0.001, deadline=None):
    """Simulated annealing to escape local minima."""
    n = len(pos)
    current_pos = pos.copy()
    current_radii = _compute_radii(current_pos)
    current_score = np.sum(current_radii)
    
    best_pos = current_pos.copy()
    best_score = current_score
    
    for i in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        t = t_start * (t_end / t_start) ** (i / max(n_iter - 1, 1))
        
        # Perturb a random circle
        idx = np.random.randint(n)
        new_pos = current_pos.copy()
        new_pos[idx] += np.random.normal(0, t, size=2)
        # Clamp to valid range
        new_pos[idx, 0] = np.clip(new_pos[idx, 0], 0.02, 0.98)
        new_pos[idx, 1] = np.clip(new_pos[idx, 1], 0.02, 0.98)
        
        new_radii = _compute_radii(new_pos)
        new_score = np.sum(new_radii)
        
        delta = new_score - current_score
        if delta > 0 or np.random.random() < np.exp(delta / t):
            current_pos = new_pos
            current_radii = new_radii
            current_score = new_score
            if current_score > best_score:
                best_pos = current_pos.copy()
                best_score = current_score
    
    return best_pos

def _slsqp_polish(pos, max_iter=100, deadline=None):
    """Final polish using SLSQP with explicit constraints."""
    n = len(pos)
    
    # Variables: positions (2n) and radii (n)
    x0 = np.concatenate([pos.flatten(), _compute_radii(pos)])
    
    def obj(x):
        p = x[:2*n].reshape(n, 2)
        r = x[2*n:]
        # Penalize invalid configurations
        penalty = 0.0
        for i in range(n):
            if p[i, 0] - r[i] < 0:
                penalty += (p[i, 0] - r[i]) ** 2
            if p[i, 0] + r[i] > 1:
                penalty += (p[i, 0] + r[i] - 1) ** 2
            if p[i, 1] - r[i] < 0:
                penalty += (p[i, 1] - r[i]) ** 2
            if p[i, 1] + r[i] > 1:
                penalty += (p[i, 1] + r[i] - 1) ** 2
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(p[i] - p[j])
                if d < r[i] + r[j]:
                    penalty += (r[i] + r[j] - d) ** 2
        return -np.sum(r) + 1e6 * penalty
    
    # Constraints: non-overlapping and inside square
    constraints = []
    for i in range(n):
        # Circle i inside square
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i] - x[2*n+i], 'jac': None})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2*i] - x[2*n+i], 'jac': None})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i+1] - x[2*n+i], 'jac': None})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2*i+1] - x[2*n+i], 'jac': None})
    
    # Non-overlapping constraints
    for i in range(n):
        for j in range(i + 1, n):
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=i, j=j: np.linalg.norm(x[2*i:2*i+2] - x[2*j:2*j+2]) - x[2*n+i] - x[2*n+j],
                'jac': None
            })
    
    # Radius non-negativity
    bounds = [(0.001, 0.5)] * (3 * n)
    
    try:
        result = minimize(
            obj,
            x0,
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-12}
        )
        
        if result.success or result.fun < obj(x0):
            new_pos = result.x[:2*n].reshape(n, 2)
            new_radii = result.x[2*n:]
            # Recompute radii to ensure feasibility
            final_radii = _compute_radii(new_pos)
            return new_pos, final_radii
    except Exception:
        pass
    
    return pos, _compute_radii(pos)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    np.random.seed(42)
    
    # Generate multiple initial configurations
    best_pos = None
    best_radii = None
    best_score = -np.inf
    
    n_configs = 5
    
    for config_idx in range(n_configs):
        if time.time() > deadline:
            break
        
        # Different initializations
        if config_idx == 0:
            pos = _initial_hexagonal(N)
        elif config_idx == 1:
            # Grid with slight perturbation
            side = int(np.ceil(np.sqrt(N)))
            pos = []
            for i in range(side):
                for j in range(side):
                    if len(pos) >= N:
                        break
                    x = (i + 0.5) / side + np.random.uniform(-0.02, 0.02)
                    y = (j + 0.5) / side + np.random.uniform(-0.02, 0.02)
                    x = np.clip(x, 0.05, 0.95)
                    y = np.clip(y, 0.05, 0.95)
                    pos.append([x, y])
            while len(pos) < N:
                pos.append([np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9)])
            pos = np.array(pos[:N])
        elif config_idx == 2:
            # Random uniform
            pos = np.random.uniform(0.1, 0.9, size=(N, 2))
        elif config_idx == 3:
            # Hexagonal with different aspect
            pos = _initial_hexagonal(N)
            pos = pos + np.random.normal(0, 0.03, size=(N, 2))
            pos[:, 0] = np.clip(pos[:, 0], 0.05, 0.95)
            pos[:, 1] = np.clip(pos[:, 1], 0.05, 0.95)
        else:
            # Fibonacci-like spiral
            golden_angle = np.pi * (3 - np.sqrt(5))
            pos = []
            for i in range(N):
                r = 0.45 * np.sqrt(i / N)
                theta = i * golden_angle
                x = 0.5 + r * np.cos(theta)
                y = 0.5 + r * np.sin(theta)
                x = np.clip(x, 0.05, 0.95)
                y = np.clip(y, 0.05, 0.95)
                pos.append([x, y])
            pos = np.array(pos)
        
        # Refine with L-BFGS-B
        pos = _refine_positions(pos, max_iter=150, deadline=deadline)
        
        if time.time() > deadline:
            break
        
        # Simulated annealing
        pos = _simulated_annealing(pos, n_iter=300, t_start=0.03, t_end=0.001, deadline=deadline)
        
        if time.time() > deadline:
            break
        
        # Final refinement
        pos = _refine_positions(pos, max_iter=100, deadline=deadline)
        
        # SLSQP polish
        if time.time() < deadline - 5:
            pos, radii = _slsqp_polish(pos, max_iter=50, deadline=deadline)
        else:
            radii = _compute_radii(pos)
        
        score = np.sum(radii)
        
        if score > best_score:
            best_score = score
            best_pos = pos.copy()
            best_radii = radii.copy()
    
    # Ensure we have a valid result
    if best_pos is None:
        # Fallback to grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    # Final validation and adjustment
    radii = _compute_radii(best_pos)
    
    # Build result array
    result = np.zeros((N, 3))
    for i in range(N):
        result[i] = [best_pos[i, 0], best_pos[i, 1], radii[i]]
    
    return result
# EVOLVE-BLOCK-END
