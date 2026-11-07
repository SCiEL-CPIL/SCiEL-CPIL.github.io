import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Leave 10s margin for 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # Global best tracking
        best_x = np.zeros(dim)
        best_fx = float('inf')
        calls = 0
        
        # Helper to evaluate and update best
        def eval_point(x):
            nonlocal best_x, best_fx, calls
            if calls >= budget:
                return None
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            return fx
            
        # Quadratic Model Fitting
        # We store points in a list: (x, fx)
        points = []
        
        def fit_quadratic_model(points):
            """
            Fits a quadratic model to the points.
            Returns A (dim x dim), b (dim,), c (scalar).
            Uses least squares.
            """
            n_points = len(points)
            if n_points < dim + 1:
                return None, None, None
            
            X = []
            y = []
            for x, fx in points:
                # Quadratic terms: x_i x_j for i <= j
                quad_terms = []
                for i in range(dim):
                    for j in range(i, dim):
                        quad_terms.append(x[i] * x[j])
                # Linear terms
                lin_terms = list(x)
                # Constant
                const = 1.0
                
                row = quad_terms + lin_terms + [const]
                X.append(row)
                y.append(fx)
                
            X = np.array(X)
            y = np.array(y)
            
            # Solve least squares: X @ params = y
            try:
                params, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
            except:
                return None, None, None
                
            # Extract A, b, c
            n_quad = dim * (dim + 1) // 2
            quad_params = params[:n_quad]
            lin_params = params[n_quad:n_quad+dim]
            c = params[n_quad+dim]
            
            # Reconstruct A
            A = np.zeros((dim, dim))
            idx = 0
            for i in range(dim):
                for j in range(i, dim):
                    if i == j:
                        A[i, j] = quad_params[idx]
                    else:
                        A[i, j] = quad_params[idx] / 2.0
                        A[j, i] = quad_params[idx] / 2.0
                    idx += 1
                    
            b = lin_params
            return A, b, c

        def find_model_min(A, b, c, x_center, radius):
            """
            Finds the minimum of the quadratic model within a trust region of given radius around x_center.
            """
            # Solve A @ dx + b = 0 for unconstrained minimum
            try:
                # Check if A is positive definite (or close)
                # For robustness, add small regularization if needed
                A_reg = A + 1e-6 * np.eye(dim)
                dx = -np.linalg.solve(A_reg, b)
            except:
                # Fallback to steepest descent if Hessian is singular
                dx = -b * 0.1
                
            # Constrain to trust region
            norm_dx = np.linalg.norm(dx)
            if norm_dx > radius:
                dx = dx / norm_dx * radius
                
            x_new = x_center + dx
            x_new = np.clip(x_new, -5, 5)
            
            # Check if it's actually better than center in the model
            f_model_center = c + b @ x_center + 0.5 * (x_center @ A @ x_center)
            f_model_new = c + b @ x_new + 0.5 * (x_new @ A @ x_new)
            
            if f_model_new < f_model_center:
                return x_new
            else:
                return None

        # Main Loop: Multi-start with Quadratic Model
        # We will divide the budget into phases.
        # Phase 1: Initial sampling to build a model.
        # Phase 2: Iterate model updates.
        # If stagnation occurs, restart with a new center.
        
        n_restarts = 5
        budget_per_restart = budget // n_restarts
        
        for r in range(n_restarts):
            if calls >= budget:
                break
            if time.time() > deadline:
                break
                
            # Reset local points for this restart
            points = []
            
            # Initial random sampling around best_x or center
            # If r==0, sample around 0. If r>0, sample around best_x with larger spread?
            # Actually, for Bent Cigar, global exploration is needed.
            # Let's sample uniformly in [-5, 5] for the first few points of each restart.
            
            n_init_points = min(10, budget_per_restart // 2)
            for _ in range(n_init_points):
                if calls >= budget or time.time() > deadline:
                    break
                x_rand = np.random.uniform(-5, 5, dim)
                fx = eval_point(x_rand)
                if fx is not None:
                    points.append((x_rand, fx))
                    
            if not points:
                continue
                
            # Initialize trust region radius
            radius = 10.0 
            
            # Iterate until budget or time runs out for this restart
            while calls < budget and time.time() < deadline:
                if len(points) < dim + 1:
                    # Need more points to fit model
                    x_rand = np.random.uniform(-5, 5, dim)
                    fx = eval_point(x_rand)
                    if fx is not None:
                        points.append((x_rand, fx))
                    continue
                    
                A, b, c = fit_quadratic_model(points)
                if A is None:
                    # Fallback: random step
                    x_rand = np.random.uniform(-5, 5, dim)
                    fx = eval_point(x_rand)
                    if fx is not None:
                        points.append((x_rand, fx))
                    continue
                    
                # Find center for trust region: use best point so far in this restart?
                # Or global best? Let's use global best to guide exploration.
                x_center = best_x.copy()
                
                x_new = find_model_min(A, b, c, x_center, radius)
                
                if x_new is None:
                    # Model suggests no improvement from center within radius
                    # Shrink radius or move center?
                    # Let's try a random perturbation from best_x
                    x_new = best_x + np.random.normal(0, radius/10, dim)
                    x_new = np.clip(x_new, -5, 5)
                    
                fx_new = eval_point(x_new)
                if fx_new is None:
                    break
                    
                points.append((x_new, fx_new))
                
                # Update trust region
                # If improvement is good, expand radius. If poor, shrink.
                # We need to compare fx_new to the best fx in points (local best)
                local_best_fx = min([p[1] for p in points])
                if fx_new < local_best_fx:
                    radius *= 1.5
                else:
                    radius *= 0.5
                    
                # Limit radius
                radius = np.clip(radius, 0.1, 10.0)
                
                # If we've converged (radius very small), break restart loop
                if radius < 0.1:
                    break
                    
        return best_x, best_fx
