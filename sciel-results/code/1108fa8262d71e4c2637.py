"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    deadline = time.time() + 120.0  # Safety margin before 150s limit
    
    # Rectangle with perimeter 4: 2*(w+h) = 4, so w+h = 2
    # Try different aspect ratios
    best_result = None
    best_score = 0.0
    
    # Try several aspect ratios for the rectangle
    aspect_ratios = [1.0, 1.2, 1.5, 2.0, 0.8, 0.6, 1.3, 1.7, 2.5, 0.5]
    
    for ar in aspect_ratios:
        if time.time() > deadline - 30:  # Leave 30s margin
            break
        
        # w + h = 2, h = w * ar, so w*(1+ar) = 2, w = 2/(1+ar)
        w = 2.0 / (1.0 + ar)
        h = 2.0 - w
        
        # Initial layout: grid-based
        n_cols = max(1, int(np.round(np.sqrt(N * w / h))))
        n_rows = max(1, int(np.ceil(N / n_cols)))
        
        # Ensure we have at least N cells
        while n_cols * n_rows < N:
            if n_cols <= n_rows:
                n_cols += 1
            else:
                n_rows += 1
        
        # Generate initial positions
        xs = []
        ys = []
        radii = []
        
        for i in range(N):
            col = i % n_cols
            row = i // n_cols
            x = (col + 0.5) * w / n_cols
            y = (row + 0.5) * h / n_rows
            xs.append(x)
            ys.append(y)
        
        xs = np.array(xs)
        ys = np.array(ys)
        
        # Initial radius estimate
        min_dist = np.inf
        for i in range(N):
            for j in range(i+1, N):
                d = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
                min_dist = min(min_dist, d)
        
        # Also consider distance to walls
        for i in range(N):
            min_dist = min(min_dist, xs[i], w - xs[i], ys[i], h - ys[i])
        
        r0 = 0.4 * min_dist
        radii = np.full(N, r0)
        
        # Define optimization variables: [x1, y1, r1, x2, y2, r2, ...]
        def unpack(vars_):
            x = vars_[0::3]
            y = vars_[1::3]
            r = vars_[2::3]
            return x, y, r
        
        def objective(vars_):
            x, y, r = unpack(vars_)
            # Maximize sum of radii, so minimize negative
            return -np.sum(r)
        
        def constraints(vars_):
            x, y, r = unpack(vars_)
            cons = []
            
            # Circles must be inside rectangle
            for i in range(N):
                cons.append({'type': 'ineq', 'fun': lambda i=i, x=x, r=r: x[i] - r[i]})
                cons.append({'type': 'ineq', 'fun': lambda i=i, x=x, w=w, r=r: w - x[i] - r[i]})
                cons.append({'type': 'ineq', 'fun': lambda i=i, y=y, r=r: y[i] - r[i]})
                cons.append({'type': 'ineq', 'fun': lambda i=i, y=y, h=h, r=r: h - y[i] - r[i]})
            
            # Circles must not overlap
            for i in range(N):
                for j in range(i+1, N):
                    cons.append({'type': 'ineq', 'fun': lambda i=i, j=j, x=x, y=y, r=r: 
                                 np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2) - r[i] - r[j]})
            
            # Radii must be positive
            for i in range(N):
                cons.append({'type': 'ineq', 'fun': lambda i=i, r=r: r[i]})
            
            return cons
        
        # Use SLSQP for constrained optimization
        x0 = np.zeros(3*N)
        for i in range(N):
            x0[3*i] = xs[i]
            x0[3*i+1] = ys[i]
            x0[3*i+2] = radii[i]
        
        try:
            result = minimize(
                objective, x0, method='SLSQP',
                constraints=constraints,
                options={'maxiter': 200, 'ftol': 1e-6}
            )
            
            if result.success or result.fun < 0:
                x, y, r = unpack(result.x)
                score = np.sum(r)
                
                # Verify validity
                valid = True
                for i in range(N):
                    if x[i] < r[i] - 1e-9 or x[i] > w - r[i] + 1e-9:
                        valid = False
                        break
                    if y[i] < r[i] - 1e-9 or y[i] > h - r[i] + 1e-9:
                        valid = False
                        break
                    if r[i] <= 0:
                        valid = False
                        break
                
                if valid:
                    for i in range(N):
                        for j in range(i+1, N):
                            d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                            if d < r[i] + r[j] - 1e-9:
                                valid = False
                                break
                        if not valid:
                            break
                
                if valid and score > best_score:
                    best_score = score
                    best_result = np.column_stack([x, y, r])
                    
        except Exception:
            continue
    
    # If optimization failed, fall back to grid
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_result
# EVOLVE-BLOCK-END
