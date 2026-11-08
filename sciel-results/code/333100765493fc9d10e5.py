import numpy as np
import time
from scipy.optimize import least_squares

N = 21

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin for 150s limit
    
    best_value = 0.0
    best_result = None
    
    # We search over aspect ratios W (width), where H = 2 - W.
    # Perimeter 2(W+H) = 4 => W+H=2.
    # We try a range of aspect ratios to find the optimal geometry.
    
    def solve_for_aspect(W_init, seed):
        """
        Solves the packing problem for a fixed aspect ratio W_init x (2-W_init)
        using scipy.optimize.least_squares.
        """
        H_init = 2.0 - W_init
        
        # 1. Generate an initial guess
        # A grid layout is a reasonable starting point.
        # Approximate number of columns
        cols = int(np.ceil(np.sqrt(N * W_init / H_init)))
        rows = int(np.ceil(N / cols))
        
        # Ensure we have enough cells
        while cols * rows < N:
            cols += 1
        
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        
        # Place centers in a grid
        for i in range(N):
            c = i % cols
            r_idx = i // cols
            # Center of each grid cell
            x[i] = (c + 0.5) * (W_init / cols)
            y[i] = (r_idx + 0.5) * (H_init / rows)
            
        # Initial radius: slightly smaller than half the minimum grid dimension to ensure feasibility
        min_dim = min(W_init / cols, H_init / rows)
        r[:] = 0.4 * min_dim
        
        # Vector of variables: [x_1...x_N, y_1...y_N, r_1...r_N]
        x0 = np.concatenate([x, y, r])
        
        # 2. Define the residual function
        # We want to minimize a penalty for constraint violations.
        # Constraints:
        # 1. Boundary: x_i >= r_i, x_i <= W - r_i, y_i >= r_i, y_i <= H - r_i
        # 2. Pairwise: dist(i,j) >= r_i + r_j
        # 3. Objective: Maximize sum(r). 
        #    We can incorporate this by adding a large constant to the residuals 
        #    or by using a weighted penalty. 
        #    A common trick in least_squares for max sum r with constraints:
        #    Minimize sum of squared violations + lambda * (r_target - r_i)^2 
        #    where r_target is a large value. 
        #    However, least_squares minimizes ||f(x)||_2.
        #    Let's define residuals as:
        #    - Boundary violations (squared, so positive)
        #    - Pairwise violations (squared, so positive)
        #    - A term encouraging larger radii: (R_max - r_i)
        #      If we just add (R_max - r_i), the solver will try to make r_i = R_max.
        #      But it will be constrained by the other residuals.
        #      To balance, we can weight the radius term.
        #      Let's try: residuals = [boundary_violations, pairwise_violations, weight * (R_max - r)]
        #      Actually, standard approach: 
        #      Minimize sum(violations^2) - sum(r_i) is not directly LSQ.
        #      LSQ minimizes sum(residuals^2).
        #      If we set residuals for radius as (r_max - r_i), minimizing their squares
        #      pushes r_i towards r_max.
        #      The conflict with boundary/pairwise residuals will settle at the optimal packing.
        
        # We need to choose weights carefully.
        # Boundary and pairwise violations should be heavily penalized to ensure feasibility.
        # Let's use a hard constraint approach via large weights on violations.
        
        def residuals(z):
            x = z[:N]
            y = z[N:2*N]
            r = z[2*N:]
            
            res = []
            
            # Boundary constraints
            # x >= r => x - r >= 0. Violation is max(0, r - x).
            # To make it smooth for LSQ, we can use soft violations or just 
            # include the signed values? 
            # LSQ handles signed residuals. 
            # If we include (r - x) as a residual, it penalizes both r>x and r<x.
            # We only want to penalize violations (r > x).
            # Standard trick: use max(0, violation) is not differentiable.
            # Better: use a smooth approximation or just include the constraint 
            # in a way that LSQ naturally handles.
            # Actually, for packing, it is often better to use 
            # scipy.optimize.minimize with method='SLSQP' or 'trust-constr' 
            # which handles inequalities directly.
            # least_squares is for equality or general nonlinear residuals.
            # Let's switch to scipy.optimize.minimize with SLSQP or trust-constr.
            # It's more appropriate for inequality constraints.
            
            raise NotImplementedError("Switching to minimize")
        
        # Let's use scipy.optimize.minimize with method='SLSQP'
        # Objective: -sum(r)
        # Constraints: 
        #   x_i - r_i >= 0
        #   W - x_i - r_i >= 0
        #   y_i - r_i >= 0
        #   H - y_i - r_i >= 0
        #   dist(i,j) - r_i - r_j >= 0
        
        from scipy.optimize import minimize
        
        def objective(z):
            r = z[2*N:]
            return -np.sum(r)
        
        # Constraints are functions returning >= 0
        # SLSQP can take a list of constraint dictionaries
        # type='ineq' means func(x) >= 0
        
        cons = []
        
        # Boundary constraints
        # x_i - r_i >= 0
        for i in range(N):
            cons.append({'type': 'ineq', 'fun': lambda z, i=i: z[i] - z[2*N+i]})
            cons.append({'type': 'ineq', 'fun': lambda z, i=i: W_init - z[i] - z[2*N+i]})
            cons.append({'type': 'ineq', 'fun': lambda z, i=i: z[N+i] - z[2*N+i]})
            cons.append({'type': 'ineq', 'fun': lambda z, i=i: H_init - z[N+i] - z[2*N+i]})
        
        # Pairwise constraints
        for i in range(N):
            for j in range(i+1, N):
                def pair_constraint(z, i=i, j=j):
                    dx = z[i] - z[j]
                    dy = z[N+i] - z[N+j]
                    dist = np.sqrt(dx**2 + dy**2)
                    return dist - z[2*N+i] - z[2*N+j]
                cons.append({'type': 'ineq', 'fun': pair_constraint})
        
        # Bounds for variables
        # x in [0, W], y in [0, H], r in [0, max_r]
        # r cannot exceed min(W, H)/2
        max_r = min(W_init, H_init) / 2
        bounds = []
        for i in range(N):
            bounds.append((0, W_init)) # x_i
        for i in range(N):
            bounds.append((0, H_init)) # y_i
        for i in range(N):
            bounds.append((0, max_r)) # r_i
            
        # Run minimize
        # SLSQP can be slow for many constraints, but N=21 is small (21*4 + 210 = 294 constraints)
        # Let's try it.
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons, 
                           options={'maxiter': 1000, 'ftol': 1e-10})
            
            if res.success or res.fun < best_value:
                # Check feasibility of the result
                z = res.x
                x = z[:N]
                y = z[N:2*N]
                r = z[2*N:]
                
                # Verify constraints manually to be safe
                feasible = True
                for i in range(N):
                    if x[i] < r[i] - 1e-9 or x[i] > W_init - r[i] + 1e-9:
                        feasible = False
                    if y[i] < r[i] - 1e-9 or y[i] > H_init - r[i] + 1e-9:
                        feasible = False
                for i in range(N):
                    for j in range(i+1, N):
                        dx = x[i] - x[j]
                        dy = y[i] - y[j]
                        dist = np.sqrt(dx**2 + dy**2)
                        if dist < r[i] + r[j] - 1e-9:
                            feasible = False
                
                if feasible:
                    value = -res.fun
                    if value > best_value:
                        best_result = np.column_stack((x, y, r))
                        best_value = value
                        return best_value, best_result
                else:
                    # If not feasible, try to repair or just ignore
                    pass
            return 0.0, None
        except Exception as e:
            return 0.0, None

    # Try multiple aspect ratios
    # W from 0.7 to 1.3
    # SLSQP is deterministic given x0, so we can use different x0s or just rely on the grid.
    # To improve robustness, we can jitter the initial grid.
    
    aspect_ratios = np.linspace(0.7, 1.3, 13)
    
    for W_init in aspect_ratios:
        if time.time() > deadline:
            break
            
        # Run with a few different random seeds for initial jitter
        for seed in range(3):
            if time.time() > deadline:
                break
                
            # Jitter the initial grid
            H_init = 2.0 - W_init
            cols = int(np.ceil(np.sqrt(N * W_init / H_init)))
            rows = int(np.ceil(N / cols))
            while cols * rows < N:
                cols += 1
                
            x = np.zeros(N)
            y = np.zeros(N)
            r = np.zeros(N)
            
            for i in range(N):
                c = i % cols
                r_idx = i // cols
                x[i] = (c + 0.5) * (W_init / cols)
                y[i] = (r_idx + 0.5) * (H_init / rows)
                
            min_dim = min(W_init / cols, H_init / rows)
            r[:] = 0.4 * min_dim
            
            # Jitter
            if seed > 0:
                np.random.seed(seed)
                x += np.random.uniform(-0.01, 0.01, N)
                y += np.random.uniform(-0.01, 0.01, N)
                # Clip to bounds
                x = np.clip(x, 0, W_init)
                y = np.clip(y, 0, H_init)
                
            x0 = np.concatenate([x, y, r])
            
            # Re-run minimize with this x0
            # We need to wrap the minimize call inside the loop or pass x0
            # Let's refactor solve_for_aspect to take x0
            
            # Inline the minimize call for speed/clarity in this loop
            from scipy.optimize import minimize
            
            def objective(z):
                r = z[2*N:]
                return -np.sum(r)
            
            cons = []
            for i in range(N):
                cons.append({'type': 'ineq', 'fun': lambda z, i=i: z[i] - z[2*N+i]})
                cons.append({'type': 'ineq', 'fun': lambda z, i=i: W_init - z[i] - z[2*N+i]})
                cons.append({'type': 'ineq', 'fun': lambda z, i=i: z[N+i] - z[2*N+i]})
                cons.append({'type': 'ineq', 'fun': lambda z, i=i: H_init - z[N+i] - z[2*N+i]})
            
            for i in range(N):
                for j in range(i+1, N):
                    def pair_constraint(z, i=i, j=j):
                        dx = z[i] - z[j]
                        dy = z[N+i] - z[N+j]
                        dist = np.sqrt(dx**2 + dy**2)
                        return dist - z[2*N+i] - z[2*N+j]
                    cons.append({'type': 'ineq', 'fun': pair_constraint})
            
            max_r = min(W_init, H_init) / 2
            bounds = []
            for i in range(N):
                bounds.append((0, W_init))
            for i in range(N):
                bounds.append((0, H_init))
            for i in range(N):
                bounds.append((0, max_r))
                
            try:
                res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons, 
                               options={'maxiter': 500, 'ftol': 1e-10})
                
                if res.success:
                    z = res.x
                    x = z[:N]
                    y = z[N:2*N]
                    r = z[2*N:]
                    
                    # Verify feasibility
                    feasible = True
                    for i in range(N):
                        if x[i] < r[i] - 1e-8 or x[i] > W_init - r[i] + 1e-8:
                            feasible = False
                        if y[i] < r[i] - 1e-8 or y[i] > H_init - r[i] + 1e-8:
                            feasible = False
                    for i in range(N):
                        for j in range(i+1, N):
                            dx = x[i] - x[j]
                            dy = y[i] - y[j]
                            dist = np.sqrt(dx**2 + dy**2)
                            if dist < r[i] + r[j] - 1e-8:
                                feasible = False
                                
                    if feasible:
                        value = -res.fun
                        if value > best_value:
                            best_value = value
                            best_result = np.column_stack((x, y, r))
                            
            except Exception:
                pass

    if best_result is None:
        # Fallback
        W = 1.0
        H = 1.0
        cols = 5
        rows = 5
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        for i in range(N):
            c = i % cols
            r_idx = i // cols
            x[i] = (c + 0.5) / cols
            y[i] = (r_idx + 0.5) / rows
            r[i] = 0.09
        best_result = np.column_stack((x, y, r))
        best_value = np.sum(r)
        
    return best_result
