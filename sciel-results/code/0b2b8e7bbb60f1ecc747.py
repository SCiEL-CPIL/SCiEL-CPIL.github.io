import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        n = self.dim
        deadline = time.time() + 290.0
        calls_made = 0
        best_x = np.zeros(n)
        best_f = np.inf

        def eval_point(x):
            nonlocal calls_made, best_x, best_f
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: (1+1)-ES with adaptive step size
        x = np.zeros(n)
        fx = eval_point(x)
        sigma = 2.0
        success_count = 0
        total_steps = 0

        es_budget = int(self.budget * 0.6)
        while calls_made < es_budget and time.time() < deadline:
            z = rng.standard_normal(n)
            x_trial = x + sigma * z
            fx_trial = eval_point(x_trial)

            if fx_trial < fx:
                x = x_trial
                fx = fx_trial
                success_count += 1
            else:
                success_count -= 1

            total_steps += 1
            if total_steps % 10 == 0:
                # Adapt sigma based on success rate over last 10 steps
                # Simple adaptation: increase if many successes, decrease if many failures
                if success_count > 2:
                    sigma *= 1.2
                elif success_count < -2:
                    sigma *= 0.8
                sigma = np.clip(sigma, 0.01, 10.0)
                success_count = 0

        # Phase 2: Powell's method (coordinate-wise line search)
        if calls_made < self.budget and time.time() < deadline:
            # Start from best point found so far
            x = best_x.copy()
            fx = best_f
            
            # Initial directions: standard basis
            directions = [np.eye(n, dtype=float)[i] for i in range(n)]
            
            # Powell's method
            max_iterations = 50
            tol = 1e-8
            
            for iteration in range(max_iterations):
                if calls_made >= self.budget or time.time() > deadline:
                    break
                    
                old_x = x.copy()
                old_fx = fx
                
                # Search along each direction
                for i in range(n):
                    if calls_made >= self.budget or time.time() > deadline:
                        break
                        
                    d = directions[i]
                    # Find optimal step along direction d using golden section search
                    # First, find a bracket
                    step = 1.0
                    # Try to find a better point
                    x_try = x + step * d
                    f_try = eval_point(x_try)
                    
                    if f_try < fx:
                        # Move in this direction, try to expand
                        best_step = step
                        best_f_try = f_try
                        while f_try < fx and calls_made < self.budget and time.time() < deadline:
                            step *= 2.0
                            x_try = x + step * d
                            f_try = eval_point(x_try)
                            if f_try < best_f_try:
                                best_step = step
                                best_f_try = f_try
                            else:
                                break
                        
                        # Now refine with golden section between best_step/2 and best_step*2
                        a = best_step / 2.0
                        b = best_step * 2.0
                        
                        # Golden section search
                        gr = (np.sqrt(5) - 1) / 2  # ~0.618
                        for _ in range(20):
                            if calls_made >= self.budget or time.time() > deadline:
                                break
                            if b - a < 1e-10:
                                break
                            c = b - gr * (b - a)
                            d_ = a + gr * (b - a)
                            f_c = eval_point(x + c * d)
                            f_d = eval_point(x + d_ * d)
                            if f_c < f_d:
                                b = d_
                            else:
                                a = c
                        
                        # Use the midpoint
                        best_step = (a + b) / 2.0
                        x = x + best_step * d
                        fx = eval_point(x)
                    else:
                        # Try smaller step
                        step = 0.5
                        x_try = x + step * d
                        f_try = eval_point(x_try)
                        if f_try < fx:
                            best_step = step
                            best_f_try = f_try
                            while f_try < fx and calls_made < self.budget and time.time() < deadline:
                                step *= 2.0
                                if step > 10:
                                    break
                                x_try = x + step * d
                                f_try = eval_point(x_try)
                                if f_try < best_f_try:
                                    best_step = step
                                    best_f_try = f_try
                                else:
                                    break
                            
                            a = best_step / 2.0
                            b = best_step * 2.0
                            for _ in range(15):
                                if calls_made >= self.budget or time.time() > deadline:
                                    break
                                if b - a < 1e-10:
                                    break
                                c = b - gr * (b - a)
                                d_ = a + gr * (b - a)
                                f_c = eval_point(x + c * d)
                                f_d = eval_point(x + d_ * d)
                                if f_c < f_d:
                                    b = d_
                                else:
                                    a = c
                            
                            best_step = (a + b) / 2.0
                            x = x + best_step * d
                            fx = eval_point(x)
                    
                    # Update direction: new direction is old_x to new x
                    new_dir = x - old_x
                    if np.linalg.norm(new_dir) > 1e-10:
                        new_dir /= np.linalg.norm(new_dir)
                        # Replace the direction that gave the largest improvement
                        # For simplicity, just replace direction i
                        directions[i] = new_dir
                
                # Check convergence
                if abs(old_fx - fx) < tol:
                    break
                
                # Reorder directions: put the one that gave largest improvement first
                # (simplified: just continue)
        
        return best_x, best_f
