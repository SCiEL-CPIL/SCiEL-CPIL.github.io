import numpy as np
from scipy.optimize import minimize
import time

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    best_score = -np.inf
    best_result = None
    
    # Try different aspect ratios
    # W is width, H = 2 - W
    # Square is W=1, H=1.
    # Elongated: W=1.5, H=0.5 or W=0.5, H=1.5
    aspect_ratios = [1.0, 1.2, 0.8, 1.5, 0.66, 2.0, 0.5]
    
    for W in aspect_ratios:
        if time.time() > deadline:
            break
            
        H = 2.0 - W
        if H <= 0: continue
        
        # Initialize grid
        # 5x5 grid is 25 cells, we need 21.
        # Let's use a 5x4 grid (20) + 1? Or just 5x5 and remove 4.
        # Better: 5x5 grid, pick 21 best spaced?
        # Simple: 5x5 grid, take first 21.
        xs = []
        ys = []
        side = 5
        for i in range(side):
            for j in range(side):
                xs.append((i + 0.5) * W / side)
                ys.append((j + 0.5) * H / side)
        
        # We have 25 points, take 21.
        # To be symmetric, maybe take all except 4 corners?
        # Let's just take the first 21.
        x0 = np.array(xs[:N])
        y0 = np.array(ys[:N])
        r0 = np.full(N, 0.1)
        
        # Force Directed Layout
        # Variables: x, y, r
        # We will iterate to expand r and resolve overlaps.
        
        x = x0.copy()
        y = y0.copy()
        r = r0.copy()
        
        iterations = 1000
        dt = 0.01
        
        for it in range(iterations):
            if time.time() > deadline:
                break
            
            # Calculate forces
            fx = np.zeros(N)
            fy = np.zeros(N)
            
            # Overlap forces
            for i in range(N):
                for j in range(i+1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx*dx + dy*dy)
                    if d < 1e-6:
                        dx = 1e-6
                        dy = 0
                        d = 1e-6
                    overlap = (r[i] + r[j]) - d
                    if overlap > 0:
                        # Repulsion proportional to overlap
                        force = overlap * 10.0
                        fx[i] += force * dx / d
                        fy[i] += force * dy / d
                        fx[j] -= force * dx / d
                        fy[j] -= force * dy / d
            
            # Wall forces
            for i in range(N):
                # Left
                if x[i] < r[i]:
                    fx[i] += (r[i] - x[i]) * 10.0
                # Right
                if x[i] > W - r[i]:
                    fx[i] -= (x[i] - (W - r[i])) * 10.0
                # Bottom
                if y[i] < r[i]:
                    fy[i] += (r[i] - y[i]) * 10.0
                # Top
                if y[i] > H - r[i]:
                    fy[i] -= (y[i] - (H - r[i])) * 10.0
            
            # Update positions
            x += fx * dt
            y += fy * dt
            
            # Clamp positions to valid range (approx)
            x = np.clip(x, 0.01, W - 0.01)
            y = np.clip(y, 0.01, H - 0.01)
            
            # Expand radii
            # For each circle, max radius is min distance to walls and other circles
            for i in range(N):
                min_dist = min(x[i], W - x[i], y[i], H - y[i])
                for j in range(N):
                    if i == j: continue
                    d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                    # Max r_i such that r_i + r_j <= d
                    # r_i <= d - r_j
                    min_dist = min(min_dist, d - r[j])
                
                # Grow radius towards min_dist, but don't exceed current + step
                # Also ensure r is positive
                target_r = max(0.01, min_dist)
                # Smooth growth
                r[i] = r[i] + 0.1 * (target_r - r[i])
        
        # Polish with SLSQP
        # Variables: x, y, r, W
        # We fixed W in the outer loop, but let's allow it to vary slightly in polish?
        # No, keep W fixed for this start to maintain aspect ratio strategy.
        
        def objective(params):
            # params: x, y, r
            x_p = params[:N]
            y_p = params[N:2*N]
            r_p = params[2*N:3*N]
            
            # Penalty for constraints
            penalty = 0.0
            
            # Walls
            for i in range(N):
                if x_p[i] < r_p[i]: penalty += (r_p[i] - x_p[i])**2
                if x_p[i] > W - r_p[i]: penalty += (x_p[i] - (W - r_p[i]))**2
                if y_p[i] < r_p[i]: penalty += (r_p[i] - y_p[i])**2
                if y_p[i] > H - r_p[i]: penalty += (y_p[i] - (H - r_p[i]))**2
            
            # Overlaps
            for i in range(N):
                for j in range(i+1, N):
                    d = np.sqrt((x_p[i]-x_p[j])**2 + (y_p[i]-y_p[j])**2)
                    overlap = (r_p[i] + r_p[j]) - d
                    if overlap > 0:
                        penalty += overlap**2
            
            return -np.sum(r_p) + 100.0 * penalty
        
        x0_polish = np.concatenate([x, y, r])
        
        try:
            res = minimize(objective, x0_polish, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-9})
            if res.success or res.fun < 1e-6: # Check if penalty is small
                x_opt = res.x[:N]
                y_opt = res.x[N:2*N]
                r_opt = res.x[2*N:3*N]
                
                # Validate and clamp
                valid = True
                for i in range(N):
                    if x_opt[i] < r_opt[i] or x_opt[i] > W - r_opt[i]:
                        valid = False
                    if y_opt[i] < r_opt[i] or y_opt[i] > H - r_opt[i]:
                        valid = False
                for i in range(N):
                    for j in range(i+1, N):
                        d = np.sqrt((x_opt[i]-x_opt[j])**2 + (y_opt[i]-y_opt[j])**2)
                        if d < r_opt[i] + r_opt[j] - 1e-9:
                            valid = False
                
                if valid:
                    score = np.sum(r_opt)
                    if score > best_score:
                        best_score = score
                        best_result = np.column_stack([x_opt, y_opt, r_opt])
        except Exception:
            pass

    if best_result is None:
        # Fallback to seed
        side = 5
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_result
