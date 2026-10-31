"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # We want to pack N circles in a rectangle of perimeter 4.
    # Let rectangle width = w, height = h.
    # 2(w + h) = 4 => w + h = 2.
    # We can parameterize by w, then h = 2 - w.
    # To maximize sum of radii, we should likely have a roughly square aspect ratio or adjust based on packing.
    # A square of side 1 has perimeter 4.
    # Let's fix the rectangle to be a square of side 1 for simplicity first, 
    # or allow w to vary? 
    # The task says "variable rectangle of perimeter 4".
    # However, for a fixed number of circles, the optimal shape might not be square.
    # But typically, for dense packing, square or near-square is good.
    # Let's try to optimize w as well? That adds a dimension.
    # Let's start with a fixed square w=1, h=1. 
    # Actually, let's allow w to be a variable in the optimization to see if it helps.
    # Variables: x_i, y_i, r_i for i=0..N-1, and w.
    # h = 2 - w.
    # Constraints:
    # 1. 0 <= x_i <= w - r_i
    # 2. 0 <= y_i <= h - r_i
    # 3. (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
    # 4. w > 0, h > 0 => 0 < w < 2
    
    # Objective: Maximize sum(r_i)
    # Equivalent to: Minimize -sum(r_i)
    
    # This is a non-convex problem. We need a good initialization.
    # Grid initialization:
    # Place circles in a grid within the square [0,1]x[0,1].
    
    def grid_init(w=1.0, h=1.0):
        cols = int(np.ceil(np.sqrt(N * w / h))) if h > 0 else 1
        rows = int(np.ceil(N / cols))
        x = []
        y = []
        r = []
        for i in range(cols):
            for j in range(rows):
                if len(x) >= N:
                    break
                cx = (i + 0.5) * w / cols
                cy = (j + 0.5) * h / rows
                # Initial radius estimate: half the cell size minus a small gap
                cell_w = w / cols
                cell_h = h / rows
                r0 = 0.5 * min(cell_w, cell_h) * 0.9
                x.append(cx)
                y.append(cy)
                r.append(r0)
        return np.array(x), np.array(y), np.array(r)
    
    # We will use scipy.optimize.minimize with SLSQP or COBYLA.
    # SLSQP is generally better for smooth constraints.
    
    # To handle the non-smoothness of the distance constraint (min distance),
    # we can use a penalty method or soft constraints.
    # Let's try a penalty method first as it's simpler to implement robustly.
    # Penalty: sum over pairs of max(0, (r_i + r_j)^2 - (dx^2 + dy^2))^2
    # Plus boundary penalties.
    
    # Let's define the objective function for minimization.
    # Variables: [x_0..x_N-1, y_0..y_N-1, r_0..r_N-1, w]
    # Total vars: 3*N + 1
    
    def objective_and_grad(z):
        # Extract variables
        x = z[0:N]
        y = z[N:2*N]
        r = z[2*N:3*N]
        w = z[3*N]
        h = 2.0 - w
        
        # Ensure positive dimensions
        if w <= 0.1 or h <= 0.1:
            return 1e10, None
            
        obj = -np.sum(r)
        
        # Boundary penalties
        # x in [0, w-r], y in [0, h-r]
        # If x < 0, penalty. If x > w-r, penalty.
        # Use squared penalty for smoothness.
        pen = 0.0
        # Left/Right
        pen += np.sum(np.maximum(0, -x)**2)
        pen += np.sum(np.maximum(0, x - (w - r))**2)
        # Bottom/Top
        pen += np.sum(np.maximum(0, -y)**2)
        pen += np.sum(np.maximum(0, y - (h - r))**2)
        
        # Overlap penalties
        # For each pair i < j
        # d^2 = (xi-xj)^2 + (yi-yj)^2
        # min_d^2 = (ri+rj)^2
        # penalty = max(0, min_d^2 - d^2)^2
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d2 = dx*dx + dy*dy
                min_d2 = (r[i] + r[j])**2
                if d2 < min_d2:
                    diff = min_d2 - d2
                    pen += diff*diff
        
        # Also penalize if radii are too small or too large?
        # Maybe a soft limit on max radius to prevent explosion?
        # The problem doesn't state a max radius, but physical constraints apply.
        
        total_obj = obj + 100.0 * pen  # Weight for penalty
        
        # Gradient is tricky with penalties. Let's use numerical gradient or COBYLA?
        # COBYLA doesn't need gradients but is slower.
        # SLSQP with numerical gradient is okay for N=21 (64 vars).
        # Let's stick to SLSQP with finite difference gradient to avoid coding errors.
        
        return total_obj, None
    
    # Let's try a different approach: Direct SLSQP with constraints.
    # This is more robust for "hard" constraints if the solver is good.
    
    def objective(z):
        r = z[2*N:3*N]
        return -np.sum(r)
        
    def make_constraint(i, j):
        def con(z):
            x = z[0:N]
            y = z[N:2*N]
            r = z[2*N:3*N]
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            return dx*dx + dy*dy - (r[i] + r[j])**2
        return con
        
    def make_x_constraint(i):
        def con_lower(z):
            x = z[0:N]
            r = z[2*N:3*N]
            w = z[3*N]
            return x[i]
        def con_upper(z):
            x = z[0:N]
            r = z[2*N:3*N]
            w = z[3*N]
            return w - r[i] - x[i]
        return con_lower, con_upper
        
    def make_y_constraint(i):
        def con_lower(z):
            y = z[N:2*N]
            r = z[2*N:3*N]
            h = 2.0 - z[3*N]
            return y[i]
        def con_upper(z):
            y = z[N:2*N]
            r = z[2*N:3*N]
            h = 2.0 - z[3*N]
            return h - r[i] - y[i]
        return con_lower, con_upper
        
    # This creates a lot of constraints. N=21 -> ~210 pair constraints + 84 boundary + 1 width.
    # SLSQP might struggle with this many constraints.
    
    # Let's go back to the penalty method but use a more robust optimizer like L-BFGS-B on the penalized objective.
    # Or just SLSQP on penalized objective.
    
    # Let's refine the penalty method.
    # We will run multiple restarts with different initializations.
    
    best_val = -np.inf
    best_z = None
    
    # Initialization 1: Grid
    x0, y0, r0 = grid_init(1.0, 1.0)
    w0 = 1.0
    z0 = np.concatenate([x0, y0, r0, [w0]])
    
    # Run optimization
    try:
        res = minimize(objective_and_grad, z0, method='SLSQP', options={'maxiter': 200, 'ftol': 1e-6})
        if res.success:
            val = -res.fun
            # Check validity
            x = res.x[0:N]
            y = res.x[N:2*N]
            r = res.x[2*N:3*N]
            w = res.x[3*N]
            h = 2.0 - w
            
            valid = True
            if w <= 0 or h <= 0:
                valid = False
            for i in range(N):
                if x[i] < -1e-6 or x[i] > w - r[i] + 1e-6 or y[i] < -1e-6 or y[i] > h - r[i] + 1e-6:
                    valid = False
                    break
            if valid:
                for i in range(N):
                    for j in range(i+1, N):
                        if (x[i]-x[j])**2 + (y[i]-y[j])**2 < (r[i]+r[j])**2 - 1e-6:
                            valid = False
                            break
                    if not valid: break
            
            if valid:
                best_val = np.sum(r)
                best_z = res.x.copy()
    except Exception as e:
        pass
    
    # Try different aspect ratios / initializations
    # Maybe a rectangle 1.2 x 0.8?
    for w_init in [0.8, 0.9, 1.0, 1.1, 1.2]:
        if time.time() > deadline:
            break
        h_init = 2.0 - w_init
        x0, y0, r0 = grid_init(w_init, h_init)
        z0 = np.concatenate([x0, y0, r0, [w_init]])
        
        try:
            res = minimize(objective_and_grad, z0, method='SLSQP', options={'maxiter': 200, 'ftol': 1e-6})
            if res.success:
                x = res.x[0:N]
                y = res.x[N:2*N]
                r = res.x[2*N:3*N]
                w = res.x[3*N]
                h = 2.0 - w
                
                valid = True
                if w <= 0 or h <= 0:
                    valid = False
                for i in range(N):
                    if x[i] < -1e-6 or x[i] > w - r[i] + 1e-6 or y[i] < -1e-6 or y[i] > h - r[i] + 1e-6:
                        valid = False
                        break
                if valid:
                    for i in range(N):
                        for j in range(i+1, N):
                            if (x[i]-x[j])**2 + (y[i]-y[j])**2 < (r[i]+r[j])**2 - 1e-6:
                                valid = False
                                break
                        if not valid: break
                
                if valid:
                    val = np.sum(r)
                    if val > best_val:
                        best_val = val
                        best_z = res.x.copy()
        except Exception:
            pass
    
    # If no valid solution found, return a trivial one (should not happen)
    if best_z is None:
        # Fallback: small circles in a grid
        x, y, r = grid_init(1.0, 1.0)
        return np.column_stack([x, y, r])
        
    x = best_z[0:N]
    y = best_z[N:2*N]
    r = best_z[2*N:3*N]
    
    # Final check and cleanup
    # Ensure non-negative radii
    r = np.maximum(r, 0)
    
    return np.column_stack([x, y, r])

# EVOLVE-BLOCK-END
