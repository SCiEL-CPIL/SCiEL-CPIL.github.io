# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

N = 26

def _objective(v, penalty=1000.0):
    """
    Objective: -sum(r) + penalty * sum(violations^2)
    """
    x = v[:N]
    y = v[N:2*N]
    r = v[2*N:]
    
    # Base objective: minimize negative sum of radii
    obj = -np.sum(r)
    
    # Penalty terms
    pen = 0.0
    
    # Boundary constraints
    # x - r >= 0  => violation = max(0, r - x)
    # 1 - x - r >= 0 => violation = max(0, x + r - 1)
    # y - r >= 0
    # 1 - y - r >= 0
    
    # Vectorized boundary violations
    v_x_low = np.maximum(0.0, r - x)
    v_x_high = np.maximum(0.0, x + r - 1.0)
    v_y_low = np.maximum(0.0, r - y)
    v_y_high = np.maximum(0.0, y + r - 1.0)
    
    pen += np.sum(v_x_low**2) + np.sum(v_x_high**2) + np.sum(v_y_low**2) + np.sum(v_y_high**2)
    
    # Circle-circle constraints
    # d_ij >= r_i + r_j => violation = max(0, r_i + r_j - d_ij)
    # We can compute this in a loop for clarity, or vectorize.
    # With N=26, N^2/2 = 325 pairs. Loop is fine.
    for i in range(N):
        for j in range(i+1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx*dx + dy*dy + 1e-12)
            viol = r[i] + r[j] - d
            if viol > 0:
                pen += viol * viol
                
    return obj + penalty * pen

def _objective_grad(v, penalty=1000.0):
    """
    Gradient of the penalized objective.
    """
    x = v[:N]
    y = v[N:2*N]
    r = v[2*N:]
    
    grad = np.zeros(3*N)
    
    # Gradient of -sum(r)
    grad[2*N:] = -1.0
    
    # Gradient of penalty terms
    # d/dx (max(0, r-x)^2) = 2(r-x) * (-1) if r-x > 0 else 0
    # d/dx (max(0, x+r-1)^2) = 2(x+r-1) * (1) if x+r-1 > 0 else 0
    
    v_x_low = r - x
    v_x_high = x + r - 1.0
    v_y_low = r - y
    v_y_high = y + r - 1.0
    
    # Mask for active constraints
    mask_x_low = v_x_low > 0
    mask_x_high = v_x_high > 0
    mask_y_low = v_y_low > 0
    mask_y_high = v_y_high > 0
    
    # Grads for x
    grad[:N][mask_x_low] += penalty * 2 * v_x_low[mask_x_low] * (-1)
    grad[:N][mask_x_high] += penalty * 2 * v_x_high[mask_x_high] * (1)
    
    # Grads for y
    grad[N:2*N][mask_y_low] += penalty * 2 * v_y_low[mask_y_low] * (-1)
    grad[N:2*N][mask_y_high] += penalty * 2 * v_y_high[mask_y_high] * (1)
    
    # Grads for r
    # d/dr (max(0, r-x)^2) = 2(r-x) * 1
    # d/dr (max(0, x+r-1)^2) = 2(x+r-1) * 1
    # d/dr (max(0, r-y)^2) = 2(r-y) * 1
    # d/dr (max(0, y+r-1)^2) = 2(y+r-1) * 1
    grad[2*N:][mask_x_low] += penalty * 2 * v_x_low[mask_x_low]
    grad[2*N:][mask_x_high] += penalty * 2 * v_x_high[mask_x_high]
    grad[2*N:][mask_y_low] += penalty * 2 * v_y_low[mask_y_low]
    grad[2*N:][mask_y_high] += penalty * 2 * v_y_high[mask_y_high]
    
    # Circle-circle gradients
    # f_ij = max(0, r_i + r_j - d_ij)^2
    # d f / d x_i = 2 * viol * d(r_i+r_j-d)/dx_i = 2 * viol * (-dx/d)
    # d f / d x_j = 2 * viol * (dx/d)
    # d f / d r_i = 2 * viol * 1
    # d f / d r_j = 2 * viol * 1
    
    for i in range(N):
        for j in range(i+1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx*dx + dy*dy + 1e-12)
            viol = r[i] + r[j] - d
            if viol > 0:
                # Factor 2 * penalty * viol
                factor = 2.0 * penalty * viol
                # Gradient wrt x_i, x_j
                # d(r_i+r_j-d)/dx_i = -dx/d
                # d(r_i+r_j-d)/dx_j = dx/d
                gx = dx / d
                gy = dy / d
                
                grad[i] += factor * (-gx)
                grad[j] += factor * (gx)
                grad[N+i] += factor * (-gy)
                grad[N+j] += factor * (gy)
                grad[2*N+i] += factor * 1.0
                grad[2*N+j] += factor * 1.0
                
    return grad

def _make_initial_guess(seed=0):
    """Create a feasible initial guess using a jittered grid."""
    rng = np.random.RandomState(seed)
    side = int(np.ceil(np.sqrt(N)))
    # 26 circles, side=6 (36 cells). We use 26 of them.
    # Or side=5 (25 cells) + 1? No, 6x6 grid is better for spacing.
    # Let's just use a 5x5 grid (25) and one extra? 
    # Actually, a 6x6 grid gives more room.
    # Let's place them in the first N cells of a 6x6 grid.
    
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) >= N:
                break
            cx = (i + 0.5) / side
            cy = (j + 0.5) / side
            cx += rng.uniform(-0.01, 0.01)
            cy += rng.uniform(-0.01, 0.01)
            # Max radius for this grid spacing is 0.5/side.
            # Use 0.45/side to be safe.
            r = 0.45 / side
            cells.append((cx, cy, r))
            
    x = np.array([c[0] for c in cells[:N]])
    y = np.array([c[1] for c in cells[:N]])
    r = np.array([c[2] for c in cells[:N]])
    
    return np.concatenate([x, y, r])

def construct():
    """Pack N circles in unit square maximizing sum of radii."""
    best_value = -np.inf
    best_v = None
    
    # Try multiple initial guesses
    for seed in range(10):
        v0 = _make_initial_guess(seed)
        
        # Ensure initial guess is strictly feasible
        # Clip radii to be small enough for the grid
        # x, y are in [0,1]. r must be <= min(x, 1-x, y, 1-y)
        # For the grid, this is satisfied by construction if r is small.
        
        try:
            result = minimize(
                _objective,
                v0,
                method='L-BFGS-B',
                jac=_objective_grad,
                bounds=[(0.0, 1.0)] * (2 * N) + [(0.001, 0.5)] * N,
                options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
            )
            
            v = result.x
            x = v[:N]
            y = v[N:2*N]
            r = v[2*N:]
            
            # Post-processing: Enforce feasibility strictly
            # If the penalty method didn't fully converge to feasibility,
            # we might need to shrink radii or adjust positions.
            # A simple way: check constraints. If violated, reduce r slightly.
            
            feasible = True
            # Check boundaries
            for i in range(N):
                if x[i] < r[i] - 1e-6 or x[i] > 1.0 - r[i] + 1e-6:
                    feasible = False
                if y[i] < r[i] - 1e-6 or y[i] > 1.0 - r[i] + 1e-6:
                    feasible = False
            
            # Check circle-circle
            for i in range(N):
                for j in range(i+1, N):
                    d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                    if d < r[i] + r[j] - 1e-6:
                        feasible = False
            
            if feasible:
                val = np.sum(r)
                if val > best_value:
                    best_value = val
                    best_v = np.column_stack([x, y, r])
            else:
                # If not feasible, try to repair by reducing radii
                # This is a heuristic repair.
                # We can just skip it if we trust the penalty enough,
                # but for safety, let's try to reduce r until feasible.
                # This is expensive, so maybe just skip if not feasible?
                # The penalty should have driven it to feasible.
                # If it's not, the penalty might be too low or optimizer stuck.
                # Let's try a quick local search with higher penalty?
                # Or just accept if it's "close" and fix by shrinking?
                
                # Let's try to fix by scaling down radii slightly
                # and re-checking.
                scale = 1.0
                while scale > 0.1:
                    r_new = r * scale
                    # Check if r_new is feasible with current x,y
                    # We need to check if x,y are within bounds for r_new
                    # and circles don't overlap.
                    # This is O(N^2).
                    
                    ok = True
                    for i in range(N):
                        if x[i] < r_new[i] - 1e-6 or x[i] > 1.0 - r_new[i] + 1e-6:
                            ok = False
                            break
                        if y[i] < r_new[i] - 1e-6 or y[i] > 1.0 - r_new[i] + 1e-6:
                            ok = False
                            break
                    if ok:
                        for i in range(N):
                            for j in range(i+1, N):
                                d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                                if d < r_new[i] + r_new[j] - 1e-6:
                                    ok = False
                                    break
                            if not ok:
                                break
                    
                    if ok:
                        val = np.sum(r_new)
                        if val > best_value:
                            best_value = val
                            best_v = np.column_stack([x, y, r_new])
                        break
                    scale *= 0.95

        except Exception:
            continue
    
    if best_v is None:
        # Fallback: grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.45 / side) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_v
# EVOLVE-BLOCK-END
