"""Circle packing in unit square - force-directed + optimization hybrid."""
import numpy as np
from scipy.optimize import minimize
import time

N = 26

def construct():
    start_time = time.time()
    deadline = 120.0  # safety margin before 150s kill
    
    best_score = -np.inf
    best_result = None
    
    def compute_score(arr):
        """Compute sum of radii, return -inf if invalid."""
        if arr.shape[0] != N:
            return -np.inf
        x = arr[:, 0]
        y = arr[:, 1]
        r = arr[:, 2]
        # Check inside unit square
        if np.any(x < 0) or np.any(x > 1) or np.any(y < 0) or np.any(y > 1):
            return -np.inf
        if np.any(r <= 0):
            return -np.inf
        if np.any(x + r > 1) or np.any(x - r < 0) or np.any(y + r > 1) or np.any(y - r < 0):
            return -np.inf
        # Check disjoint interiors
        for i in range(N):
            for j in range(i + 1, N):
                dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                if dist < r[i] + r[j] - 1e-9:
                    return -np.inf
        return np.sum(r)
    
    def try_optimize(x0, maxiter=200):
        """Run optimization from initial guess, return best valid result."""
        nonlocal best_score, best_result
        
        def objective(params):
            """Minimize negative sum of radii subject to constraints."""
            x = params[0:N]
            y = params[N:2*N]
            r = params[2*N:3*N]
            
            # Soft penalty for constraint violations
            penalty = 0.0
            # Wall constraints
            penalty += np.sum(np.maximum(0, -x - r + 1e-8))**2
            penalty += np.sum(np.maximum(0, x + r - 1 + 1e-8))**2
            penalty += np.sum(np.maximum(0, -y - r + 1e-8))**2
            penalty += np.sum(np.maximum(0, y + r - 1 + 1e-8))**2
            # Non-overlap constraints
            for i in range(N):
                for j in range(i + 1, N):
                    dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                    overlap = r[i] + r[j] - dist
                    if overlap > 0:
                        penalty += overlap**2
            
            return -np.sum(r) + 100.0 * penalty
        
        def gradient(params):
            """Compute gradient numerically or analytically."""
            x = params[0:N]
            y = params[N:2*N]
            r = params[2*N:3*N]
            
            grad = np.zeros_like(params)
            
            # Objective gradient: -1 for each radius
            grad[2*N:3*N] = -1.0
            
            # Penalty gradients
            # Wall penalties
            # -x - r + eps > 0 => penalty term (max(0, -x-r+eps))^2
            mask = (-x - r + 1e-8) > 0
            grad[0:N][mask] += 2 * 100.0 * (-x[mask] - r[mask] + 1e-8) * (-1)
            grad[2*N:3*N][mask] += 2 * 100.0 * (-x[mask] - r[mask] + 1e-8) * (-1)
            
            mask = (x + r - 1 + 1e-8) > 0
            grad[0:N][mask] += 2 * 100.0 * (x[mask] + r[mask] - 1 + 1e-8) * (1)
            grad[2*N:3*N][mask] += 2 * 100.0 * (x[mask] + r[mask] - 1 + 1e-8) * (1)
            
            mask = (-y - r + 1e-8) > 0
            grad[N:2*N][mask] += 2 * 100.0 * (-y[mask] - r[mask] + 1e-8) * (-1)
            grad[2*N:3*N][mask] += 2 * 100.0 * (-y[mask] - r[mask] + 1e-8) * (-1)
            
            mask = (y + r - 1 + 1e-8) > 0
            grad[N:2*N][mask] += 2 * 100.0 * (y[mask] + r[mask] - 1 + 1e-8) * (1)
            grad[2*N:3*N][mask] += 2 * 100.0 * (y[mask] + r[mask] - 1 + 1e-8) * (1)
            
            # Non-overlap penalties
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx**2 + dy**2) + 1e-12
                    overlap = r[i] + r[j] - dist
                    if overlap > 0:
                        # d/dx_i of overlap^2 = 2*overlap * (-dx/dist)
                        grad[0:N][i] += 2 * 100.0 * overlap * (-dx / dist)
                        grad[0:N][j] += 2 * 100.0 * overlap * (dx / dist)
                        grad[N:2*N][i] += 2 * 100.0 * overlap * (-dy / dist)
                        grad[N:2*N][j] += 2 * 100.0 * overlap * (dy / dist)
                        grad[2*N:3*N][i] += 2 * 100.0 * overlap * (1)
                        grad[2*N:3*N][j] += 2 * 100.0 * overlap * (1)
            
            return grad
        
        # Use L-BFGS-B with numerical gradient as fallback
        try:
            result = minimize(
                objective, x0, method='L-BFGS-B',
                jac=gradient,
                options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            # Convert back to (x, y, r) array
            x = result.x[0:N]
            y = result.x[N:2*N]
            r = result.x[2*N:3*N]
            
            # Clip to valid range
            r = np.maximum(r, 1e-6)
            x = np.clip(x, r, 1 - r)
            y = np.clip(y, r, 1 - r)
            
            arr = np.column_stack([x, y, r])
            score = compute_score(arr)
            
            if score > best_score:
                best_score = score
                best_result = arr.copy()
                
        except Exception:
            pass
    
    # Initial configurations to try
    rng = np.random.default_rng(42)
    
    # Strategy 1: Grid-based initialization
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) < N:
                cx = (i + 0.5) / side
                cy = (j + 0.5) / side
                r = 0.5 / side * 0.95
                cells.append([cx, cy, r])
    
    if len(cells) >= N:
        x0 = np.array(cells[:N])
        params = np.concatenate([x0[:, 0], x0[:, 1], x0[:, 2]])
        try_optimize(params, maxiter=300)
    
    # Strategy 2: Random initialization with varied radii
    for trial in range(10):
        if time.time() - start_time > deadline - 10:
            break
        
        # Random positions and small radii
        x = rng.uniform(0.1, 0.9, N)
        y = rng.uniform(0.1, 0.9, N)
        r = rng.uniform(0.02, 0.06, N)
        
        # Ensure initial feasibility by shrinking if needed
        for i in range(N):
            for j in range(i + 1, N):
                dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                if dist < r[i] + r[j]:
                    scale = dist / (r[i] + r[j] + 1e-6)
                    r[i] *= scale * 0.9
                    r[j] *= scale * 0.9
            r[i] = min(r[i], x[i], 1 - x[i], y[i], 1 - y[i])
            r[i] = max(r[i], 0.005)
        
        params = np.concatenate([x, y, r])
        try_optimize(params, maxiter=200)
    
    # Strategy 3: Multi-stage polish - try to expand radii from a good solution
    if best_result is not None and time.time() - start_time < deadline - 5:
        # Take best result and try to increase radii
        for expand_trial in range(5):
            if time.time() - start_time > deadline - 5:
                break
            
            x = best_result[:, 0].copy()
            y = best_result[:, 1].copy()
            r = best_result[:, 2].copy()
            
            # Slightly increase radii and re-optimize
            r *= 1.02
            params = np.concatenate([x, y, r])
            
            def objective_expand(params):
                x = params[0:N]
                y = params[N:2*N]
                r = params[2*N:3*N]
                
                penalty = 0.0
                penalty += np.sum(np.maximum(0, -x - r + 1e-8))**2
                penalty += np.sum(np.maximum(0, x + r - 1 + 1e-8))**2
                penalty += np.sum(np.maximum(0, -y - r + 1e-8))**2
                penalty += np.sum(np.maximum(0, y + r - 1 + 1e-8))**2
                for i in range(N):
                    for j in range(i + 1, N):
                        dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                        overlap = r[i] + r[j] - dist
                        if overlap > 0:
                            penalty += overlap**2
                
                return -np.sum(r) + 100.0 * penalty
            
            try:
                result = minimize(
                    objective_expand, params, method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                
                x = result.x[0:N]
                y = result.x[N:2*N]
                r = result.x[2*N:3*N]
                r = np.maximum(r, 1e-6)
                x = np.clip(x, r, 1 - r)
                y = np.clip(y, r, 1 - r)
                
                arr = np.column_stack([x, y, r])
                score = compute_score(arr)
                
                if score > best_score:
                    best_score = score
                    best_result = arr.copy()
                    
            except Exception:
                pass
    
    # Final validation and return
    if best_result is not None:
        # Final polish: ensure strict validity
        x = best_result[:, 0]
        y = best_result[:, 1]
        r = best_result[:, 2]
        
        # Tighten constraints slightly
        for i in range(N):
            r[i] = min(r[i], x[i] - 1e-7, 1 - x[i] - 1e-7, y[i] - 1e-7, 1 - y[i] - 1e-7)
            r[i] = max(r[i], 1e-6)
        
        # Check and fix any remaining overlaps by shrinking
        for i in range(N):
            for j in range(i + 1, N):
                dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                if dist < r[i] + r[j] + 1e-7:
                    # Shrink both slightly
                    shrink = (r[i] + r[j] + 1e-7 - dist) / 2 + 1e-8
                    r[i] = max(r[i] - shrink, 1e-6)
                    r[j] = max(r[j] - shrink, 1e-6)
        
        best_result = np.column_stack([x, y, r])
        final_score = compute_score(best_result)
        if final_score > 0:
            return best_result
    
    # Fallback: return grid
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) < N:
                cx = (i + 0.5) / side
                cy = (j + 0.5) / side
                r = 0.5 / side * 0.9
                cells.append([cx, cy, r])
    return np.array(cells[:N])
