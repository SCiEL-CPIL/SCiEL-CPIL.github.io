import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        rng = np.random.default_rng(42)
        
        # Nelder-Mead parameters
        # Standard NM parameters
        alpha = 1.0  # Reflection
        gamma = 2.0  # Expansion
        rho = 0.5    # Contraction
        sigma = 0.5  # Shrinkage
        
        # Initial simplex setup
        # Start at center of the box
        center = np.zeros(dim)
        step_size = 2.0 # Initial simplex size
        
        # Create initial simplex
        simplex = np.zeros((dim + 1, dim))
        simplex[0] = center
        for i in range(1, dim + 1):
            simplex[i] = center.copy()
            simplex[i][i-1] = step_size
            
        # Evaluate initial simplex
        fs = np.array([func(x) for x in simplex])
        eval_count = dim + 1
        
        best_x = simplex[np.argmin(fs)]
        best_f = np.min(fs)
        
        # Restarts logic
        max_no_improve = 50 # If no improvement for 50 evals, restart
        no_improve_count = 0
        restart_count = 0
        max_restarts = 5
        
        while eval_count < budget:
            # Sort simplex by fitness
            order = np.argsort(fs)
            simplex = simplex[order]
            fs = fs[order]
            
            # Update best
            if fs[0] < best_f:
                best_f = fs[0]
                best_x = simplex[0].copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
                
            # Check for restart
            if no_improve_count > max_no_improve and restart_count < max_restarts:
                restart_count += 1
                # Move to a new random location
                # Use a smaller step size for subsequent restarts to focus
                new_center = rng.uniform(-4, 4, size=dim)
                new_step = step_size / (restart_count + 1)
                
                simplex = np.zeros((dim + 1, dim))
                simplex[0] = new_center
                for i in range(1, dim + 1):
                    simplex[i] = new_center.copy()
                    simplex[i][i-1] = new_step
                    
                fs = np.array([func(x) for x in simplex])
                eval_count += dim + 1
                
                if fs[0] < best_f:
                    best_f = fs[0]
                    best_x = simplex[0].copy()
                    no_improve_count = 0
                else:
                    no_improve_count = 0 # Reset counter for new basin
            
            if eval_count >= budget:
                break
                
            # Standard Nelder-Mead steps
            # Centroid of all points except the worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_ref = centroid + alpha * (centroid - simplex[-1])
            f_ref = func(x_ref)
            eval_count += 1
            
            if f_ref < fs[0]:
                # Expansion
                x_exp = centroid + gamma * (x_ref - centroid)
                f_exp = func(x_exp)
                eval_count += 1
                
                if f_exp < f_ref:
                    simplex[-1] = x_exp
                    fs[-1] = f_exp
                else:
                    simplex[-1] = x_ref
                    fs[-1] = f_ref
            elif f_ref < fs[-2]:
                # Simple reflection
                simplex[-1] = x_ref
                fs[-1] = f_ref
            else:
                # Contraction
                if f_ref < fs[-1]:
                    # Outside contraction
                    x_con = centroid + rho * (x_ref - centroid)
                    f_con = func(x_con)
                    eval_count += 1
                    
                    if f_con <= f_ref:
                        simplex[-1] = x_con
                        fs[-1] = f_con
                    else:
                        # Shrinkage
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            fs[i] = func(simplex[i])
                            eval_count += 1
                else:
                    # Inside contraction
                    x_con = centroid - rho * (centroid - simplex[-1])
                    f_con = func(x_con)
                    eval_count += 1
                    
                    if f_con < fs[-1]:
                        simplex[-1] = x_con
                        fs[-1] = f_con
                    else:
                        # Shrinkage
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            fs[i] = func(simplex[i])
                            eval_count += 1
            
            # Clip simplex to box to keep it valid? 
            # The func clips, but keeping simplex inside helps NM geometry.
            # However, standard NM often benefits from exploring outside.
            # The task says func clips. So we don't need to clip x manually.
            
            # Check for convergence (simplex size)
            simplex_size = np.max(np.std(simplex, axis=0))
            if simplex_size < 1e-10:
                # Stagnated, force restart or break
                if restart_count < max_restarts:
                     no_improve_count = max_no_improve + 1
                else:
                    break

        return best_x, best_f
