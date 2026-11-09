"""Weighted Sum Method with Multi-start Nelder-Mead for WFG4."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = 320.0  # Safety margin before 330s kill
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # WSM requires a set of weight vectors.
        # For 3 objectives, we use a grid of weights.
        # We have 200 evaluations.
        # Let's allocate evaluations to different weight vectors.
        
        # Generate weight vectors on the simplex for 3 objectives.
        # A grid of 4x4x4 is too many. 
        # Let's use a set of ~10-15 weight vectors.
        # We will run Nelder-Mead for each weight vector, starting from multiple random points.
        
        # Strategy:
        # 1. Define a set of weight vectors W.
        # 2. For each w in W, we have a budget of B_w evaluations.
        # 3. We perform multi-start NM. Each start costs 1 (initial) + iterations.
        #    NM with dim=12 takes ~dim+1 points for simplex init? No, scipy minimize 'Nelder-Mead'
        #    uses a simplex of dim+1 points. So one "run" consumes dim+1 evaluations just to start?
        #    Actually, scipy's minimize calls the function for the initial point, then for the new points.
        #    Let's check scipy behavior. It evaluates x0, then the simplex vertices.
        #    So one local optimization run consumes roughly dim+1 + k*dim evaluations.
        #    With dim=12, one run is expensive (~13+ evals).
        #    200 / 13 ~ 15 runs.
        
        # Alternative: Use a cheaper local search or reduce dim? No, dim is fixed.
        # Maybe use 'Powell' or 'BFGS' (gradient-free if jac not provided, but BFGS approximates).
        # Nelder-Mead is robust but eval-heavy.
        
        # Let's try a hybrid:
        # Use a small population of weight vectors.
        # For each weight vector, run a few short NM optimizations.
        
        # Weight vectors for 3 objectives.
        # We want to cover the Pareto front.
        # Let's create 12 weight vectors.
        # 4 on each axis (near 1,0,0), 4 in middle, 4 near (1/3, 1/3, 1/3)?
        # Better: Use a structured grid.
        
        weights = []
        # Add extreme points
        weights.append([1.0, 0.0, 0.0])
        weights.append([0.0, 1.0, 0.0])
        weights.append([0.0, 0.0, 1.0])
        
        # Add mid-points
        weights.append([0.5, 0.5, 0.0])
        weights.append([0.5, 0.0, 0.5])
        weights.append([0.0, 0.5, 0.5])
        
        # Add center
        weights.append([1/3, 1/3, 1/3])
        
        # Add some random ones to fill gaps
        for _ in range(5):
            w = np.random.rand(3)
            w = w / np.sum(w)
            weights.append(w)
            
        weights = np.array(weights)
        n_weights = len(weights)
        
        # Budget allocation
        # Total budget: 200.
        # Each NM run needs at least dim+1 = 13 evaluations to form the simplex.
        # Let's say we allow max 2 runs per weight vector.
        # 12 weights * 2 runs * 13 evals/run = 312 evals. Too many.
        # We have 200.
        # 200 / 13 = 15 runs total.
        # So we can do 1 run per weight vector for 12 weights (156 evals), leaving 44.
        # Or fewer weights, more runs.
        
        # Let's pick 10 weight vectors.
        weights = weights[:10]
        n_weights = len(weights)
        
        # 10 weights. 
        # Budget per weight: 200 / 10 = 20 evals.
        # 20 evals is enough for 1 NM run (13 evals) + a few extra steps.
        
        all_points = []
        all_fvals = []
        
        # Initial random sampling to ensure we have some points even if NM fails
        # Let's reserve 20 evals for random global search to seed the NM starts?
        # No, NM starts are random.
        
        # To test 'error_repair':
        # The problem clips inputs. 
        # If we generate a point outside bounds, problem clips it.
        # Does this "repair" help? 
        # In NM, the simplex points are always within bounds if we initialize within bounds 
        # and the step sizes are small? No, NM can step outside.
        # If NM steps outside, problem clips it. 
        # This effectively projects the point back.
        # This is a form of error repair.
        
        # Let's implement a custom NM wrapper to strictly count evaluations and ensure we don't overshoot.
        # Actually, scipy minimize is hard to control evaluation count precisely without a counter.
        
        eval_count = 0
        max_evals = self.budget
        
        # We will use a counter class
        class Evaluator:
            def __init__(self, problem, lo, hi):
                self.problem = problem
                self.lo = lo
                self.hi = hi
                self.count = 0
            
            def __call__(self, x):
                if self.count >= max_evals:
                    return 1e10 # Return large value to stop
                self.count += 1
                # Clip to bounds (Error Repair)
                x_clipped = np.clip(x, self.lo, self.hi)
                f = self.problem(x_clipped)
                # Weighted sum
                # We need to pass weights? No, Evaluator is per-weight.
                # Let's make Evaluator take weights in closure or as attribute.
                return f

        # Since weights vary, we create a new Evaluator for each weight?
        # Or just a function that computes weighted sum.
        
        results_per_weight = []
        
        for i, w in enumerate(weights):
            if eval_count >= max_evals:
                break
            
            # Determine remaining budget
            remaining = max_evals - eval_count
            # Allocate at least 13 for NM simplex
            if remaining < 13:
                break
            
            # We will run NM for this weight.
            # Objective: sum(w * f(x))
            
            # Define objective function for scipy
            def obj(x, w=w, eval_count_ref=[eval_count]):
                # We need to increment the global eval_count
                # Since Python closures capture by reference, we can use a list or nonlocal
                # But we are in a loop. Let's use a mutable container or class.
                pass

            # Let's use a class-based evaluator for strict control
            class WeightedEval:
                def __init__(self, w):
                    self.w = w
                    self.local_count = 0
                    self.global_count_ref = [eval_count]
                    
                def __call__(self, x):
                    if self.global_count_ref[0] >= max_evals:
                        return 1e10
                    self.global_count_ref[0] += 1
                    self.local_count += 1
                    x_clip = np.clip(x, lo, hi)
                    f = problem(x_clip)
                    return np.dot(self.w, f)
            
            evaluator = WeightedEval(w)
            
            # Initial point: random
            x0 = np.random.rand(self.dim) * (hi - lo) + lo
            
            # Options for Nelder-Mead
            options = {
                'maxiter': 5, # Limit iterations to stay within budget
                'xatol': 1e-4,
                'fatol': 1e-4
            }
            
            try:
                res = minimize(evaluator, x0, method='Nelder-Mead', options=options)
                # res.x is the best point found
                # res.fun is the best weighted sum
                # We should record res.x
                # Also, we might want to record the initial point if it was good?
                # The problem records ALL returned objective vectors.
                # So every call to problem() inside evaluator is recorded.
                # So we just need to run the optimization.
                
                # Update global count
                eval_count = evaluator.global_count_ref[0]
                
            except Exception:
                # If NM fails, we still have the evaluations made
                eval_count = evaluator.global_count_ref[0]
                
            # If we have spare budget for this weight, we could do a second start?
            # Let's check remaining.
            if eval_count + 13 < max_evals:
                # Do a second start
                x0_2 = np.random.rand(self.dim) * (hi - lo) + lo
                evaluator2 = WeightedEval(w)
                try:
                    res2 = minimize(evaluator2, x0_2, method='Nelder-Mead', options=options)
                    eval_count = evaluator2.global_count_ref[0]
                except Exception:
                    eval_count = evaluator2.global_count_ref[0]

        # Fill remaining budget with random sampling to ensure coverage
        while eval_count < max_evals:
            x = np.random.rand(self.dim) * (hi - lo) + lo
            f = problem(x)
            eval_count += 1
            if time.time() > deadline:
                break

        return np.array([]) # Return value ignored

# EVOLVE-BLOCK-END
