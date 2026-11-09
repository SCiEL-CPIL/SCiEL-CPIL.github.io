import numpy as np
from scipy.optimize import minimize
import time

N = 26

def construct():
    start_time = time.time()
    time_limit = 110.0  # Leave 10s safety margin
    
    best_score = -np.inf
    best_result = None
    
    # Initial guess: Grid layout
    # 5x5 grid covers 25, we need 26. 
    # A 5x5 grid of circles with radius 0.1 has diameter 0.2. 
    # Spacing 0.2 fits in 1.0.
    # Let's start with a grid and then optimize.
    
    def objective_and_grad(z):
        # z is shape (3*N,)
        x = z[0::3]
        y = z[1::3]
        r = z[2::3]
        
        # Sum of radii to maximize -> minimize -sum
        val = -np.sum(r)
        grad = np.zeros_like(z)
        grad[2::3] = -1.0
        
        # Penalty for overlaps
        # d_ij = sqrt((xi-xj)^2 + (yi-yj)^2)
        # violation = max(0, ri+rj - d_ij)
        # penalty = sum(violation^2)
        # grad_penalty wrt xi: 2 * violation * (xi-xj)/d_ij
        
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                if d < 1e-12:
                    d = 1e-12
                    dx = 1e-12
                    dy = 0
                s = r[i] + r[j]
                viol = s - d
                if viol > 0:
                    # Penalty term: viol^2
                    # d/dxi (viol^2) = 2*viol * d/dxi(s-d) = 2*viol * (-dx/d)
                    # d/dri (viol^2) = 2*viol * 1
                    factor = 2.0 * viol / d
                    val += viol * viol
                    grad[0::3][i] -= factor * dx
                    grad[0::3][j] += factor * dx
                    grad[1::3][i] -= factor * dy
                    grad[1::3][j] += factor * dy
                    grad[2::3][i] += 2.0 * viol
                    grad[2::3][j] += 2.0 * viol
        
        # Penalty for boundaries
        # Left: max(0, r - x)^2
        # Right: max(0, r - (1-x))^2
        # Bottom: max(0, r - y)^2
        # Top: max(0, r - (1-y))^2
        
        for i in range(N):
            # Left
            v = r[i] - x[i]
            if v > 0:
                val += v*v
                grad[0::3][i] += 2*v * (-1) # d/dx (r-x)^2 = 2(r-x)(-1)
                grad[2::3][i] += 2*v * (1)  # d/dr (r-x)^2 = 2(r-x)(1)
            
            # Right
            v = r[i] - (1 - x[i])
            if v > 0:
                val += v*v
                grad[0::3][i] += 2*v * (1)  # d/dx (r-1+x)^2 = 2(r-1+x)(1)
                grad[2::3][i] += 2*v * (1)
            
            # Bottom
            v = r[i] - y[i]
            if v > 0:
                val += v*v
                grad[1::3][i] += 2*v * (-1)
                grad[2::3][i] += 2*v * (1)
            
            # Top
            v = r[i] - (1 - y[i])
            if v > 0:
                val += v*v
                grad[1::3][i] += 2*v * (1)
                grad[2::3][i] += 2*v * (1)
        
        return val, grad

    # Multiplier for penalty to enforce constraints strongly
    # We can increase it if violations persist, but fixed is usually okay for local opt
    # Let's use a penalty weight of 100.0 to make violations very costly
    penalty_weight = 100.0
    
    def weighted_objective(z):
        val, grad = objective_and_grad(z)
        return val * penalty_weight, grad * penalty_weight

    # Initial guess
    def initial_guess():
        # 5x5 grid
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 0.01
                cells.append((x, y, r))
        # Add 26th circle randomly or in a gap
        # For now, just take first 26 from a larger grid or randomize
        # Let's use a 6x5 grid or just randomize the grid slightly
        # Actually, a 5x5 grid has 25 spots. The 26th can be placed in the center of a 2x2 block?
        # Better: Use a random initialization close to grid
        z0 = []
        for i in range(N):
            # Map index to grid position
            row = i // 5
            col = i % 5
            # If i >= 25, wrap around or randomize
            if i >= 25:
                # Place in a random cell with small radius
                r_row = np.random.randint(0, 5)
                r_col = np.random.randint(0, 5)
                x = (r_col + 0.5) / 5 + np.random.uniform(-0.05, 0.05)
                y = (r_row + 0.5) / 5 + np.random.uniform(-0.05, 0.05)
                r = 0.05
            else:
                x = (col + 0.5) / 5 + np.random.uniform(-0.02, 0.02)
                y = (row + 0.5) / 5 + np.random.uniform(-0.02, 0.02)
                r = 0.09
            z0.extend([x, y, r])
        return np.array(z0)

    # Run optimization
    num_restarts = 10
    
    for k in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
            
        z0 = initial_guess()
        # Bounds: x,y in [0,1], r in [0, 0.5]
        bounds = []
        for i in range(N):
            bounds.append((0.0, 1.0)) # x
            bounds.append((0.0, 1.0)) # y
            bounds.append((0.0, 0.5)) # r
            
        try:
            res = minimize(weighted_objective, z0, method='L-BFGS-B', jac=True, bounds=bounds, options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
            
            # Extract result
            z_opt = res.x
            x = z_opt[0::3]
            y = z_opt[1::3]
            r = z_opt[2::3]
            
            # Check validity
            valid = True
            for i in range(N):
                if x[i] < -1e-5 or x[i] > 1 + 1e-5 or y[i] < -1e-5 or y[i] > 1 + 1e-5 or r[i] < -1e-5:
                    valid = False
                    break
                # Boundary check
                if r[i] > x[i] + 1e-5 or r[i] > 1 - x[i] + 1e-5 or r[i] > y[i] + 1e-5 or r[i] > 1 - y[i] + 1e-5:
                    valid = False
                    break
            
            if valid:
                for i in range(N):
                    for j in range(i+1, N):
                        d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                        if d < r[i] + r[j] - 1e-5:
                            valid = False
                            break
                    if not valid: break
            
            if valid:
                # Clip small numerical errors
                x = np.clip(x, 0, 1)
                y = np.clip(y, 0, 1)
                r = np.clip(r, 0, 0.5)
                # Ensure boundaries are strictly satisfied by shrinking r if needed
                for i in range(N):
                    r[i] = min(r[i], x[i], 1-x[i], y[i], 1-y[i])
                
                score = np.sum(r)
                if score > best_score:
                    best_score = score
                    best_result = np.column_stack((x, y, r))
                    
        except Exception:
            pass
            
    if best_result is not None:
        return best_result
    else:
        # Fallback to seed
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
