import numpy as np
import time
from scipy.optimize import minimize

N = 21
TIME_LIMIT = 140  # seconds

def construct():
    start_time = time.time()
    best_score = -np.inf
    best_config = None
    
    # We will try a few random restarts
    # Each restart uses Nelder-Mead to optimize positions and radii
    # Fixed w=1, h=1 for simplicity and speed, as square is likely optimal for N=21
    
    w = 1.0
    h = 1.0
    
    # Number of restarts depends on time available
    # Each optimization might take 1-5 seconds
    # Let's aim for 10-20 restarts if fast, fewer if slow.
    
    for restart in range(50):  # Max 50 restarts
        if time.time() - start_time > TIME_LIMIT - 10:
            break
            
        # Initialize
        # Grid initialization is a good start
        side = int(np.ceil(np.sqrt(N)))
        xs = (np.arange(N) % side + 0.5) / side
        ys = (np.arange(N) // side + 0.5) / side
        # Add small random noise
        xs += np.random.uniform(-0.01, 0.01, N)
        ys += np.random.uniform(-0.01, 0.01, N)
        
        # Initial radii: small, e.g., 0.5/side
        r_init = np.full(N, 0.5 / side * 0.9)
        
        # Flatten variables: [x1, y1, r1, ..., xN, yN, rN]
        x0 = np.concatenate([xs, ys, r_init])
        
        def objective(vars):
            x = vars[0:N]
            y = vars[N:2*N]
            r = vars[2*N:3*N]
            
            # Ensure radii are positive
            r = np.maximum(r, 0)
            
            # Sum of radii
            sum_r = np.sum(r)
            
            # Overlap Penalty
            overlap_pen = 0.0
            for i in range(N):
                for j in range(i+1, N):
                    d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                    min_d = r[i] + r[j]
                    if d < min_d:
                        overlap_pen += (min_d - d)**2
            
            # Boundary Penalty
            bound_pen = 0.0
            for i in range(N):
                # Left
                if x[i] < r[i]:
                    bound_pen += (r[i] - x[i])**2
                # Right
                if x[i] > w - r[i]:
                    bound_pen += (x[i] - (w - r[i]))**2
                # Bottom
                if y[i] < r[i]:
                    bound_pen += (r[i] - y[i])**2
                # Top
                if y[i] > h - r[i]:
                    bound_pen += (y[i] - (h - r[i]))**2
            
            # Total Penalty
            penalty = 100.0 * (overlap_pen + bound_pen)
            
            return -sum_r + penalty
        
        # Run optimization
        # Nelder-Mead is derivative-free
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6})
        except Exception:
            continue
            
        x_opt = res.x
        x = x_opt[0:N]
        y = x_opt[N:2*N]
        r = x_opt[2*N:3*N]
        
        # Post-processing: Ensure validity
        # Check constraints
        valid = True
        for i in range(N):
            if x[i] < r[i] or x[i] > w - r[i] or y[i] < r[i] or y[i] > h - r[i]:
                # Try to shrink radius to fit boundary
                r[i] = min(r[i], x[i], w-x[i], y[i], h-y[i])
                if r[i] < 0:
                    valid = False
                    break
            
        for i in range(N):
            for j in range(i+1, N):
                d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                min_d = r[i] + r[j]
                if d < min_d:
                    # Overlap! We need to shrink radii or move circles.
                    # Simple fix: shrink both radii slightly? 
                    # Better: This indicates the optimizer failed to find a feasible solution.
                    # We can try a global shrink factor.
                    pass
        
        # Global shrink to ensure no overlaps
        # Calculate max overlap violation
        max_viol = 0.0
        for i in range(N):
            for j in range(i+1, N):
                d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                min_d = r[i] + r[j]
                if d < min_d:
                    max_viol = max(max_viol, (min_d - d) / 2)
        
        if max_viol > 0:
            # Shrink all radii by max_viol + small epsilon
            # This is a conservative fix
            shrink = max_viol + 1e-5
            r = r - shrink
            # Ensure non-negative
            r = np.maximum(r, 0)
            
        # Check validity again
        if not np.all(r > 0):
            continue
            
        score = np.sum(r)
        if score > best_score:
            best_score = score
            best_config = np.column_stack((x, y, r))
            
    if best_config is None:
        # Fallback to grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_config
