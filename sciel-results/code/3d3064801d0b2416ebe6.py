"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    
    best_value = -np.inf
    best_result = None
    
    def compute_value(x, w, h):
        """Compute sum of radii and validate constraints."""
        if len(x) != 3 * N:
            return -np.inf
        xs = x[0::3]
        ys = x[1::3]
        rs = x[2::3]
        
        # Check bounds
        if np.any(rs < 1e-9):
            return -np.inf
        if np.any(xs - rs < -1e-8) or np.any(xs + rs > w + 1e-8):
            return -np.inf
        if np.any(ys - rs < -1e-8) or np.any(ys + rs > h + 1e-8):
            return -np.inf
        
        # Check pairwise distances
        for i in range(N):
            for j in range(i + 1, N):
                dist = np.sqrt((xs[i] - xs[j])**2 + (ys[i] - ys[j])**2)
                if dist < rs[i] + rs[j] - 1e-8:
                    return -np.inf
        
        return np.sum(rs)
    
    def objective(x, w, h):
        """Minimize negative sum of radii."""
        val = compute_value(x, w, h)
        if val < 0:
            return 1e6
        return -val
    
    def try_optimization(w, h, x0, time_budget):
        """Try to optimize with given rectangle dimensions."""
        nonlocal best_value, best_result
        
        if time.time() - start_time > deadline:
            return
        
        # Initial value
        val = compute_value(x0, w, h)
        if val > best_value:
            best_value = val
            best_result = x0.copy()
        
        # Try SLSQP optimization
        try:
            res = minimize(
                objective,
                x0,
                args=(w, h),
                method='SLSQP',
                bounds=[(1e-9, w/2)] * N + [(1e-9, h/2)] * N + [(1e-9, min(w,h)/2)] * N,
                options={'maxiter': 50, 'ftol': 1e-10, 'disp': False}
            )
            
            if time.time() - start_time > deadline:
                return
            
            val = compute_value(res.x, w, h)
            if val > best_value:
                best_value = val
                best_result = res.x.copy()
                
        except Exception:
            pass
    
    # Try different aspect ratios
    aspect_ratios = [1.0, 1.2, 0.8, 1.5, 0.667, 2.0, 0.5, 1.333, 0.75]
    
    for ar in aspect_ratios:
        if time.time() - start_time > deadline:
            break
        
        # Perimeter = 2(w + h) = 4, so w + h = 2
        # w/h = ar, so w = ar*h, ar*h + h = 2, h = 2/(ar+1), w = 2*ar/(ar+1)
        h = 2.0 / (ar + 1)
        w = 2.0 * ar / (ar + 1)
        
        # Create initial configuration: grid-like
        # Use 5x4 grid for 21 circles (20 in grid, 1 extra)
        nx, ny = 5, 4
        x0 = np.zeros(3 * N)
        
        idx = 0
        for i in range(nx):
            for j in range(ny):
                if idx >= N:
                    break
                x0[3*idx] = (i + 0.5) * w / nx
                x0[3*idx + 1] = (j + 0.5) * h / ny
                x0[3*idx + 2] = min(w/nx, h/ny) * 0.45
                idx += 1
        
        # Fill remaining
        while idx < N:
            x0[3*idx] = w / 2
            x0[3*idx + 1] = h / 2
            x0[3*idx + 2] = 0.1
            idx += 1
        
        try_optimization(w, h, x0, 10)
    
    # If we have a result, format it
    if best_result is not None:
        result = np.zeros((N, 3))
        result[:, 0] = best_result[0::3]
        result[:, 1] = best_result[1::3]
        result[:, 2] = best_result[2::3]
        return result
    else:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
# EVOLVE-BLOCK-END
