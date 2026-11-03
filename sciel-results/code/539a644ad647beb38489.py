"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _objective(params, n_circles):
    """Objective: minimize negative sum of radii plus penalties for violations."""
    x = params[:n_circles]
    y = params[n_circles:2*n_circles]
    r = params[2*n_circles:]
    
    # Negative sum of radii (we minimize, so negate)
    obj = -np.sum(r)
    
    # Penalty for circles going out of bounds
    boundary_penalty = 0.0
    for i in range(n_circles):
        boundary_penalty += max(0, r[i] - x[i])**2
        boundary_penalty += max(0, x[i] - (1 - r[i]))**2
        boundary_penalty += max(0, r[i] - y[i])**2
        boundary_penalty += max(0, y[i] - (1 - r[i]))**2
    
    # Penalty for overlapping circles
    overlap_penalty = 0.0
    for i in range(n_circles):
        for j in range(i+1, n_circles):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy)
            min_dist = r[i] + r[j]
            if dist < min_dist:
                overlap_penalty += (min_dist - dist)**2
    
    return obj + 100.0 * (boundary_penalty + overlap_penalty)

def _gradient(params, n_circles):
    """Compute gradient of the objective."""
    x = params[:n_circles].copy()
    y = params[n_circles:2*n_circles].copy()
    r = params[2*n_circles:].copy()
    
    grad = np.zeros_like(params)
    
    # Gradient of -sum(r)
    grad[2*n_circles:] = -1.0
    
    # Boundary penalties
    for i in range(n_circles):
        # x[i] > 1 - r[i]
        if x[i] > 1 - r[i]:
            grad[i] += 200.0 * (x[i] - (1 - r[i]))
            grad[2*n_circles + i] += 200.0 * (1 - r[i] - x[i])
        # x[i] < r[i]
        if x[i] < r[i]:
            grad[i] += 200.0 * (r[i] - x[i])
            grad[2*n_circles + i] += -200.0 * (r[i] - x[i])
        # y[i] > 1 - r[i]
        if y[i] > 1 - r[i]:
            grad[n_circles + i] += 200.0 * (y[i] - (1 - r[i]))
            grad[2*n_circles + i] += 200.0 * (1 - r[i] - y[i])
        # y[i] < r[i]
        if y[i] < r[i]:
            grad[n_circles + i] += 200.0 * (r[i] - y[i])
            grad[2*n_circles + i] += -200.0 * (r[i] - y[i])
    
    # Overlap penalties
    for i in range(n_circles):
        for j in range(i+1, n_circles):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy)
            min_dist = r[i] + r[j]
            if dist < min_dist and dist > 1e-10:
                penalty = (min_dist - dist)**2
                dpen_ddist = -2 * (min_dist - dist)
                # d(dist)/d(x_i) = dx/dist
                grad[i] += 200.0 * dpen_ddist * (dx / dist)
                grad[n_circles + i] += 200.0 * dpen_ddist * (dy / dist)
                grad[j] += 200.0 * dpen_ddist * (-dx / dist)
                grad[n_circles + j] += 200.0 * dpen_ddist * (-dy / dist)
                grad[2*n_circles + i] += 200.0 * dpen_ddist * 1.0
                grad[2*n_circles + j] += 200.0 * dpen_ddist * 1.0
    
    return grad

def _enforce_constraints(params, n_circles):
    """Project parameters to satisfy constraints approximately."""
    x = params[:n_circles].copy()
    y = params[n_circles:2*n_circles].copy()
    r = np.maximum(params[2*n_circles:], 0.01)
    
    # Clamp radii
    r = np.clip(r, 0.01, 0.49)
    
    # Adjust positions to be within bounds
    for i in range(n_circles):
        x[i] = np.clip(x[i], r[i], 1 - r[i])
        y[i] = np.clip(y[i], r[i], 1 - r[i])
    
    return np.concatenate([x, y, r])

def _initial_params(seed):
    """Generate initial parameters with random positions and small radii."""
    rng = np.random.RandomState(seed)
    x = rng.uniform(0.1, 0.9, N)
    y = rng.uniform(0.1, 0.9, N)
    r = rng.uniform(0.02, 0.08, N)
    return np.concatenate([x, y, r])

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    best_params = None
    best_score = -np.inf
    
    # Try multiple restarts
    for restart in range(200):
        if time.time() > deadline:
            break
        
        params = _initial_params(restart * 7 + 42)
        
        # Run L-BFGS-B optimization
        try:
            result = minimize(
                _objective,
                params,
                args=(N,),
                method='L-BFGS-B',
                jac=_gradient,
                options={'maxiter': 200, 'gtol': 1e-6}
            )
            
            if result.success or result.status in [0, 1, 2]:
                final_params = _enforce_constraints(result.x, N)
                score = -_objective(final_params, N)  # Convert back to maximization
                score = np.sum(final_params[2*N:])  # Sum of radii
                
                if score > best_score:
                    best_score = score
                    best_params = final_params.copy()
        except Exception:
            continue
    
    if best_params is None:
        # Fallback to simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    # Extract final result
    x = best_params[:N]
    y = best_params[N:2*N]
    r = best_params[2*N:]
    
    return np.column_stack([x, y, r])

# EVOLVE-BLOCK-END
