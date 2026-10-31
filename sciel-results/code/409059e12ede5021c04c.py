import numpy as np
from scipy.optimize import minimize
import time

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    best_val = 0.0
    best_sol = None
    
    # Initial grid setup
    # 5x5 grid, pick 21 positions
    # To be symmetric, we can leave the last 4 empty in a pattern
    # Or just take the first 21 in row-major order
    idxs = []
    for i in range(5):
        for j in range(5):
            idxs.append((i, j))
            if len(idxs) == N:
                break
        if len(idxs) == N:
            break
    
    # We will try a few aspect ratios
    # w + h = 2
    # Square is w=1.0
    aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.2, 0.85, 1.15]
    
    for w in aspect_ratios:
        if time.time() > deadline:
            break
        h = 2.0 - w
        
        # Initial positions
        # Map grid indices to coordinates
        # Grid is 5x5, so cell size is w/5, h/5
        xs = []
        ys = []
        rs = []
        for i, j in idxs:
            # Center of cell (i, j)
            x = (i + 0.5) * (w / 5.0)
            y = (j + 0.5) * (h / 5.0)
            xs.append(x)
            ys.append(y)
            # Initial radius: slightly less than half the min cell dimension
            r = 0.5 * min(w/5.0, h/5.0) * 0.9
            rs.append(r)
            
        x0 = np.array(xs + ys + rs + [w]) # w is also a variable? 
        # Actually, let's fix w for this run to reduce dimensionality and speed up
        # Variables: x1..x21, y1..y21, r1..r21
        # Total 63 variables.
        
        x0 = np.array(xs + ys + rs)
        
        def objective(v):
            # v contains x, y, r for 21 circles
            xx = v[0:N]
            yy = v[N:2*N]
            rr = v[2*N:3*N]
            
            # Penalty for boundary
            pen_bound = 0.0
            for i in range(N):
                if xx[i] < rr[i]:
                    pen_bound += (rr[i] - xx[i])**2
                if xx[i] > w - rr[i]:
                    pen_bound += (xx[i] - (w - rr[i]))**2
                if yy[i] < rr[i]:
                    pen_bound += (rr[i] - yy[i])**2
                if yy[i] > h - rr[i]:
                    pen_bound += (yy[i] - (h - rr[i]))**2
            
            # Penalty for overlap
            pen_overlap = 0.0
            for i in range(N):
                for j in range(i+1, N):
                    dx = xx[i] - xx[j]
                    dy = yy[i] - yy[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    req_dist = rr[i] + rr[j]
                    if dist < req_dist:
                        pen_overlap += (req_dist - dist)**2
            
            # We want to maximize sum(r), so minimize -sum(r)
            # Weight the penalty heavily to enforce feasibility
            # If penalty is non-zero, the solution is invalid.
            # We return a large number if infeasible, or just the penalty + negative radius sum.
            # To guide the optimizer towards feasible large radii:
            # Minimize: Penalty * 100 - sum(r)
            # If Penalty is 0, it maximizes sum(r).
            # If Penalty > 0, it tries to reduce penalty.
            
            return pen_overlap * 100.0 + pen_bound * 100.0 - np.sum(rr)
            
        # Run optimization
        # Nelder-Mead is good for this
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6})
        
        # Check if time is running out
        if time.time() > deadline:
            break
            
        v_final = res.x
        xx = v_final[0:N]
        yy = v_final[N:2*N]
        rr = v_final[2*N:3*N]
        
        # Verify feasibility manually
        is_feasible = True
        for i in range(N):
            if xx[i] < rr[i] - 1e-8 or xx[i] > w - rr[i] + 1e-8:
                is_feasible = False
            if yy[i] < rr[i] - 1e-8 or yy[i] > h - rr[i] + 1e-8:
                is_feasible = False
            for j in range(i+1, N):
                dx = xx[i] - xx[j]
                dy = yy[i] - yy[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < rr[i] + rr[j] - 1e-8:
                    is_feasible = False
        
        if is_feasible:
            val = np.sum(rr)
            if val > best_val:
                best_val = val
                # Store solution
                best_sol = np.column_stack((xx, yy, rr))
                
    # If no feasible solution found, return a trivial small one?
    # But the task says "invalid result... scores 0". We must return a valid one.
    # The initial grid is valid. We should keep the best valid solution found.
    # If best_sol is None, it means optimization failed or didn't find feasible.
    # Let's construct a fallback from the initial grid for the best aspect ratio tried.
    if best_sol is None:
        # Fallback: 5x5 grid, w=1, h=1
        w = 1.0
        h = 1.0
        xs = []
        ys = []
        rs = []
        for i, j in idxs:
            x = (i + 0.5) * (w / 5.0)
            y = (j + 0.5) * (h / 5.0)
            r = 0.5 * min(w/5.0, h/5.0) * 0.9
            xs.append(x)
            ys.append(y)
            rs.append(r)
        best_sol = np.column_stack((xs, ys, rs))
        
    return best_sol
