import numpy as np
import time
from scipy.optimize import minimize

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    best_val = 0
    best_sol = None
    
    # Helper function to evaluate a solution
    def evaluate(z):
        x = z[0::3]
        y = z[1::3]
        r = z[2::3]
        
        # Check boundaries
        # x in [r, 1-r], y in [r, 1-r]
        # If violated, it's invalid, but for optimization we use penalties.
        # For the final score, we need valid circles.
        
        # Calculate sum of radii
        val = np.sum(r)
        
        # Check validity for the "best" tracking
        # A solution is valid if:
        # 1. x_i >= r_i, x_i <= 1 - r_i
        # 2. y_i >= r_i, y_i <= 1 - r_i
        # 3. dist(i,j) >= r_i + r_j
        
        # Fast check
        if np.any(x < r) or np.any(x > 1 - r):
            return val, False
        if np.any(y < r) or np.any(y > 1 - r):
            return val, False
        
        # Check overlaps
        # Vectorized distance check is O(N^2)
        # For N=26, this is small (325 pairs)
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < r[i] + r[j] - 1e-9: # Small tolerance
                    return val, False
        
        return val, True

    def objective(z):
        x = z[0::3]
        y = z[1::3]
        r = z[2::3]
        
        # Objective: maximize sum(r) -> minimize -sum(r)
        obj = -np.sum(r)
        
        # Penalty for overlaps
        pen = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                req = r[i] + r[j]
                if dist < req:
                    # Quadratic penalty
                    pen += (req - dist)**2
        
        # Penalty for boundaries
        # x in [r, 1-r] -> x-r >= 0, 1-r-x >= 0
        # y in [r, 1-r] -> y-r >= 0, 1-r-y >= 0
        bx = np.minimum(0, x - r)
        bx2 = np.minimum(0, 1 - r - x)
        by = np.minimum(0, y - r)
        by2 = np.minimum(0, 1 - r - y)
        
        bpen = np.sum(bx**2) + np.sum(bx2**2) + np.sum(by**2) + np.sum(by2**2)
        
        # Weights
        w_overlap = 10.0
        w_bound = 10.0
        
        return obj + w_overlap * pen + w_bound * bpen

    def gradient(z):
        # Numerical gradient is safer and N is small
        # Or implement analytical
        # Let's use numerical for simplicity and robustness
        eps = 1e-8
        grad = np.zeros_like(z)
        f0 = objective(z)
        for i in range(len(z)):
            z_old = z[i]
            z[i] = z_old + eps
            f1 = objective(z)
            z[i] = z_old - eps
            f2 = objective(z)
            z[i] = z_old
            grad[i] = (f1 - f2) / (2 * eps)
        return grad

    # Initialization: Hexagonal Packing
    def init_hex():
        # Approximate hexagonal packing for 26 circles
        # Try to fit a hexagonal grid
        # Radius r, distance between centers 2r
        # Rows of 5, 6, 5, 6, 4? Or 6,5,6,5,4?
        # Let's just generate a grid of hex centers and pick the closest 26 to center
        
        # Start with a guess for radius
        r_guess = 0.1
        dist = 2 * r_guess
        hex_h = dist * np.sqrt(3) / 2
        
        centers = []
        # Generate a wide grid of potential centers
        for i in range(-10, 10):
            for j in range(-10, 10):
                x = i * dist + (j % 2) * dist / 2
                y = j * hex_h
                centers.append((x, y))
        
        centers = np.array(centers)
        # Shift to center of unit square
        centers += 0.5
        
        # Sort by distance to center (0.5, 0.5)
        dists = np.sum((centers - 0.5)**2, axis=1)
        idx = np.argsort(dists)
        
        # Take first N centers
        selected = centers[idx[:N]]
        
        # Initial positions
        x = selected[:, 0]
        y = selected[:, 1]
        
        # Initial radii: slightly smaller than r_guess to avoid immediate overlap
        r = np.full(N, r_guess * 0.9)
        
        z = np.zeros(3 * N)
        z[0::3] = x
        z[1::3] = y
        z[2::3] = r
        
        return z

    # Run optimization
    num_restarts = 5
    
    for k in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Perturb initial solution
        if k == 0:
            z0 = init_hex()
        else:
            z0 = init_hex()
            # Add random noise
            noise = np.random.normal(0, 0.01, 3*N)
            z0 = z0 + noise
            # Clip radii to positive
            z0[2::3] = np.maximum(z0[2::3], 0.01)
            # Clip positions to [0,1]
            z0[0::3] = np.clip(z0[0::3], 0.01, 0.99)
            z0[1::3] = np.clip(z0[1::3], 0.01, 0.99)
        
        bounds = []
        for i in range(N):
            bounds.append((0.0, 1.0)) # x
            bounds.append((0.0, 1.0)) # y
            bounds.append((0.0, 0.5)) # r
        
        # Use Nelder-Mead as it doesn't need gradients and is robust
        # L-BFGS-B is faster but might get stuck if penalty is not smooth enough
        # Let's try L-BFGS-B first, fallback to Nelder-Mead if it fails
        try:
            res = minimize(objective, z0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 200, 'ftol': 1e-8})
            z_opt = res.x
        except Exception:
            res = minimize(objective, z0, method='Nelder-Mead', bounds=bounds, options={'maxiter': 500})
            z_opt = res.x
        
        # Evaluate
        val, valid = evaluate(z_opt)
        
        # If invalid, try to project/fix? Or just discard.
        # A common issue is tiny overlaps.
        # Let's do a final polish: if invalid, reduce radii slightly until valid?
        # Or just skip.
        
        if valid:
            if val > best_val:
                best_val = val
                best_sol = z_opt.copy()
        
        # If invalid, we can try to repair
        else:
            # Simple repair: reduce all radii by 10% and re-optimize?
            # Or just project to valid space.
            # Let's try a simple local search to fix validity
            # This might be too slow. Let's just skip invalid solutions for now.
            # But wait, if we get close, we can tweak.
            
            # Let's try to manually adjust if it's close.
            # Check max violation
            x = z_opt[0::3]
            y = z_opt[1::3]
            r = z_opt[2::3]
            
            # Just reduce radii by a small factor to ensure validity
            # This is a heuristic
            r_new = r * 0.95
            z_fixed = z_opt.copy()
            z_fixed[2::3] = r_new
            
            # Check if fixed is valid
            val_fix, valid_fix = evaluate(z_fixed)
            if valid_fix:
                if val_fix > best_val:
                    best_val = val_fix
                    best_sol = z_fixed.copy()

    # Final fallback: Grid
    if best_sol is None:
        # Fallback to grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best_sol = np.array(cells[:N]).flatten()
    
    # Format output
    # best_sol is (3*N)
    # Output shape (N, 3)
    out = best_sol.reshape(N, 3)
    
    return out
