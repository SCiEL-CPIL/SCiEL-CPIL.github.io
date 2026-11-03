import numpy as np
import time
from scipy.optimize import minimize

N = 300

def construct():
    start_time = time.time()
    deadline = 115.0  # Leave 5s margin

    # --- Step 1: Initialization ---
    # Hexagonal lattice initialization
    def hexagonal_positions(n_target):
        # Estimate spacing
        area_per_circle = np.pi / n_target
        spacing = np.sqrt(area_per_circle / (np.sqrt(3)/2))
        
        best_points = None
        # Try a range of spacings to ensure we get enough points
        for mult in np.linspace(0.8, 1.2, 20):
            s = spacing * mult
            a = s
            b = s * np.sqrt(3) / 2
            k = int(1.0 / s) + 5
            points = []
            for j in range(-k, k + 1):
                y = j * b
                for i in range(-k, k + 1):
                    x = (i + 0.5 * (j % 2)) * a
                    if x*x + y*y <= 0.98**2:
                        points.append((x, y))
            
            points = np.array(points)
            if len(points) == 0:
                continue
            
            # If we have more than n_target, select the ones closest to center? 
            # Or just take the first n_target? 
            # Better: select n_target points that are most spread out?
            # Simple heuristic: sort by distance to center and take first n_target?
            # This clusters them in the center. 
            # Better: random subset?
            # For packing, we want them spread out. 
            # The hexagonal grid is already spread out. 
            # If we have more, we can just take the first n_target from the grid generation order?
            # Grid generation order is row by row. 
            # Let's just take the first n_target.
            
            if len(points) >= n_target:
                points = points[:n_target]
                if best_points is None or len(points) == n_target:
                    best_points = points
                if len(points) == n_target:
                    break
        
        if best_points is None or len(best_points) != n_target:
            # Fallback: Spiral
            angles = np.linspace(0, 2*np.pi, n_target, endpoint=False)
            radii = np.linspace(0.05, 0.95, n_target)
            best_points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        
        return best_points

    positions = hexagonal_positions(N)
    
    # Precompute pair indices
    i_idx, j_idx = np.triu_indices(N, k=1)
    n_pairs = len(i_idx)

    # --- Step 2: Optimization ---
    # Objective: Minimize sum of repulsive potentials
    # V_pair(r) = 1 / (r^2 + eps)^2
    # V_bound(r) = 1 / ((1 - r) + eps)^2
    
    # We will anneal eps. Start with larger eps (smoother), end with smaller eps (sharper).
    eps_list = [0.1, 0.05, 0.01, 0.001, 0.0001]
    
    best_pos = positions.copy()
    best_score = 0.0
    
    def compute_score(pos):
        # Calculate the actual packing radius for scoring
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        min_pair_dist = np.min(dist) / 2.0
        boundary = 1.0 - np.linalg.norm(pos, axis=1)
        min_boundary = np.min(boundary)
        return min(min_pair_dist, min_boundary)

    current_pos = positions.copy()
    
    for eps in eps_list:
        if time.time() - start_time > deadline - 10:
            break
            
        def objective_and_grad(x):
            pos = x.reshape(N, 2)
            
            # Pairwise distances
            diff = pos[i_idx] - pos[j_idx]
            dist_sq = np.sum(diff * diff, axis=1)
            
            # Boundary distances
            rad_sq = np.sum(pos * pos, axis=1)
            rad = np.sqrt(rad_sq)
            
            # Potentials
            # V_pair = 1 / (dist_sq + eps)^2
            denom_pair = dist_sq + eps
            V_pair = 1.0 / (denom_pair * denom_pair)
            
            # V_bound = 1 / ((1 - rad) + eps)^2
            gap_bound = 1.0 - rad + eps
            V_bound = 1.0 / (gap_bound * gap_bound)
            
            obj_val = np.sum(V_pair) + np.sum(V_bound)
            
            # Gradients
            # d V_pair / d x_i = -4 * diff * (dist_sq + eps)^(-3)
            # d V_pair / d x_j = 4 * diff * (dist_sq + eps)^(-3)
            
            grad_pair_factor = -4.0 / (denom_pair ** 3)
            grad_pair = grad_pair_factor[:, np.newaxis] * diff
            
            grad = np.zeros_like(pos)
            np.add.at(grad, i_idx, grad_pair)
            np.add.at(grad, j_idx, -grad_pair)
            
            # d V_bound / d x_i = 2 * (1 - rad + eps)^(-3) * (x_i / rad)
            # Note: d(1-rad)/dx = -x/rad. d(1/((1-rad)+eps)^2)/dx = -2((1-rad)+eps)^-3 * (-x/rad) = 2((1-rad)+eps)^-3 * (x/rad)
            
            safe_rad = np.where(rad > 1e-9, rad, 1e-9)
            grad_bound_factor = 2.0 / (gap_bound ** 3)
            grad_bound = grad_bound_factor[:, np.newaxis] * (pos / safe_rad[:, np.newaxis])
            
            grad += grad_bound
            
            return obj_val, grad

        def objective(x):
            val, _ = objective_and_grad(x)
            return val

        def grad(x):
            _, g = objective_and_grad(x)
            return g

        try:
            res = minimize(
                objective,
                current_pos.flatten(),
                method='L-BFGS-B',
                jac=grad,
                options={'maxiter': 100, 'gtol': 1e-8, 'ftol': 1e-8}
            )
            
            new_pos = res.x.reshape(N, 2)
            
            # Validate and fix if necessary
            norms = np.linalg.norm(new_pos, axis=1)
            mask = norms >= 1.0
            if np.any(mask):
                new_pos[mask] = new_pos[mask] * (0.999 / norms[mask])
                
            new_score = compute_score(new_pos)
            
            if new_score > best_score:
                best_score = new_score
                best_pos = new_pos.copy()
                current_pos = new_pos.copy()
                
        except Exception:
            pass

    # Final Polish with a very small eps if time permits
    if time.time() - start_time < deadline - 5:
        eps = 1e-6
        def objective_and_grad_final(x):
            pos = x.reshape(N, 2)
            diff = pos[i_idx] - pos[j_idx]
            dist_sq = np.sum(diff * diff, axis=1)
            rad = np.linalg.norm(pos, axis=1)
            
            denom_pair = dist_sq + eps
            V_pair = 1.0 / (denom_pair * denom_pair)
            
            gap_bound = 1.0 - rad + eps
            V_bound = 1.0 / (gap_bound * gap_bound)
            
            obj_val = np.sum(V_pair) + np.sum(V_bound)
            
            grad_pair_factor = -4.0 / (denom_pair ** 3)
            grad_pair = grad_pair_factor[:, np.newaxis] * diff
            
            grad = np.zeros_like(pos)
            np.add.at(grad, i_idx, grad_pair)
            np.add.at(grad, j_idx, -grad_pair)
            
            safe_rad = np.where(rad > 1e-9, rad, 1e-9)
            grad_bound_factor = 2.0 / (gap_bound ** 3)
            grad_bound = grad_bound_factor[:, np.newaxis] * (pos / safe_rad[:, np.newaxis])
            
            grad += grad_bound
            
            return obj_val, grad

        def objective_final(x):
            val, _ = objective_and_grad_final(x)
            return val

        def grad_final(x):
            _, g = objective_and_grad_final(x)
            return g

        try:
            res = minimize(
                objective_final,
                current_pos.flatten(),
                method='L-BFGS-B',
                jac=grad_final,
                options={'maxiter': 100, 'gtol': 1e-10, 'ftol': 1e-10}
            )
            new_pos = res.x.reshape(N, 2)
            norms = np.linalg.norm(new_pos, axis=1)
            mask = norms >= 1.0
            if np.any(mask):
                new_pos[mask] = new_pos[mask] * (0.999 / norms[mask])
            
            new_score = compute_score(new_pos)
            if new_score > best_score:
                best_score = new_score
                best_pos = new_pos.copy()
        except Exception:
            pass

    return best_pos

def _validate_and_fix(pos):
    """Ensure positions are valid: finite, inside unit disk, no coincident points."""
    pos = np.array(pos, dtype=np.float64)
    if pos.shape != (N, 2):
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        radii = np.linspace(0.1, 0.9, N)
        pos = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    
    if not np.all(np.isfinite(pos)):
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        radii = np.linspace(0.1, 0.9, N)
        pos = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        
    norms = np.linalg.norm(pos, axis=1)
    mask = norms >= 1.0
    if np.any(mask):
        pos[mask] = pos[mask] * (0.999 / norms[mask])
        
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)
    if min_dist < 1e-4:
        for i in range(N):
            for j in range(i+1, N):
                if dist[i,j] < 1e-4:
                    pos[i] += np.random.randn(2) * 1e-3
                    pos[j] -= np.random.randn(2) * 1e-3
                    n = np.linalg.norm(pos[i])
                    if n > 0.999:
                        pos[i] = pos[i] * (0.999 / n)
                    n = np.linalg.norm(pos[j])
                    if n > 0.999:
                        pos[j] = pos[j] * (0.999 / n)
                        
    return pos

_original_construct = construct

def construct():
    try:
        pos = _original_construct()
        pos = _validate_and_fix(pos)
        return pos
    except Exception:
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        radii = np.linspace(0.05, 0.95, N)
        pos = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        return pos
