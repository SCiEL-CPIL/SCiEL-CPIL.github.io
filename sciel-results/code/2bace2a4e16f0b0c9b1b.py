import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # WFG7 specific knowledge:
        # The Pareto front of WFG7 is known. For M=3, the front is defined by a parameter t in [0,1].
        # The mapping from decision variables x to the front point y(t) is complex, 
        # but we can use a heuristic: WFG7 is separable in a way that allows us to 
        # approximate the front by optimizing a weighted sum or by using the 
        # known structure. However, we don't have the exact analytic inverse.
        
        # Strategy:
        # 1. Generate a diverse initial set of points by optimizing weighted sums 
        #    (scalarization) with different weights. This creates a spread of points 
        #    that approximates the Pareto front.
        # 2. For each of these "seeds", perform a local line search to refine 
        #    the point towards the true front.
        
        # We use a small number of seeds (e.g., 10-20) to stay within budget.
        # Budget = 200.
        # Let's use 10 seeds.
        # Each seed gets:
        #   - 1 initial eval (random or weighted sum start)
        #   - ~19 local search steps (line search with bisection or similar)
        # Total: 10 * 20 = 200.
        
        n_seeds = 10
        steps_per_seed = (budget - n_seeds) // n_seeds
        
        # Weights for scalarization to cover the front
        # For 3 objectives, we can use random weights or a grid.
        # Let's use a set of weights that cover the simplex.
        weights = []
        # Simple grid for 3 objectives
        # t1, t2, t3 such that t1+t2+t3=1
        # We'll generate a few representative weights
        for i in range(3):
            weights.append([1.0, 0.0, 0.0])
            weights.append([0.0, 1.0, 0.0])
            weights.append([0.0, 0.0, 1.0])
            weights.append([1.0/3.0, 1.0/3.0, 1.0/3.0])
        
        # Add some random weights for diversity
        rng = np.random.RandomState(42)
        for _ in range(6):
            w = rng.rand(3)
            w /= np.sum(w)
            weights.append(w)
        
        # Select n_seeds weights
        # Ensure we pick a diverse subset
        if len(weights) > n_seeds:
            # Pick evenly spaced from the list or random
            indices = np.linspace(0, len(weights)-1, n_seeds, dtype=int)
            selected_weights = [weights[i] for i in indices]
        else:
            selected_weights = weights
            
        # Pad if not enough
        while len(selected_weights) < n_seeds:
            w = rng.rand(3)
            w /= np.sum(w)
            selected_weights.append(w)
            
        selected_weights = np.array(selected_weights[:n_seeds])
        
        all_points = []
        all_objectives = []
        used = 0
        
        def get_time_left():
            return deadline - time.time()
            
        # For each seed weight, find a point on the front
        for k in range(n_seeds):
            if used >= budget or get_time_left() < 0.1:
                break
                
            w = selected_weights[k]
            
            # Initialize with a random point
            # Or better, start from the center?
            # Let's start with a random point to allow exploration
            x = lo + range_ * rng.rand(dim)
            
            # Evaluate initial point
            f = problem(x)
            used += 1
            all_points.append(x.copy())
            all_objectives.append(f.copy())
            
            if used >= budget or get_time_left() < 0.1:
                continue
                
            # Local Search: Line Search along the gradient of the scalarized objective
            # We want to minimize f_scalar = w . f(x)
            # Since we don't have gradients, we use a derivative-free line search.
            # Strategy: 
            # 1. Determine a direction d. 
            #    A good direction is towards the "ideal" point (min of each objective).
            #    But we don't know the ideal point exactly. 
            #    Alternative: Use random directions and keep the best step (Nelder-Mead style but simpler).
            #    Given the budget, a simple "random direction + bisection on step size" might be robust.
            
            # Let's try a simplified Nelder-Mead or Pattern Search.
            # Pattern Search:
            # Try moving in positive and negative directions of a basis.
            # This is expensive.
            
            # Better: Since WFG7 is continuous, we can use a finite difference gradient estimate.
            # But that costs 2*dim evaluations per step. 2*12=24. 
            # We have ~19 steps. 19*24 > 19. Too expensive.
            
            # So we must use a zeroth-order method with fewer evals.
            # Let's use "Random Direction Search" (RDS).
            # Pick a random direction v.
            # Try x + alpha * v.
            # If better, keep and try larger alpha.
            # If worse, try smaller alpha or opposite direction.
            
            # To be efficient, we will do a fixed number of "probes".
            # For each of the remaining steps, we will:
            # 1. Generate a random direction v (normalized).
            # 2. Evaluate x + step_size * v.
            # 3. If it improves the weighted sum, keep it.
            # 4. If not, discard.
            # This is basically stochastic hill climbing.
            
            # To make it more effective, we can reuse directions that worked.
            # Let's implement a simple adaptive step size.
            
            current_x = x.copy()
            current_f = f.copy()
            current_score = np.dot(w, current_f)
            step_size = 0.1 * np.max(range_)  # Initial step size
            
            for step in range(steps_per_seed):
                if used >= budget or get_time_left() < 0.1:
                    break
                    
                # Generate random direction
                v = rng.randn(dim)
                v /= np.linalg.norm(v)
                
                # Try moving in direction v
                x_new = current_x + step_size * v
                x_new = np.clip(x_new, lo, hi)
                
                f_new = problem(x_new)
                used += 1
                all_points.append(x_new.copy())
                all_objectives.append(f_new.copy())
                
                score_new = np.dot(w, f_new)
                
                if score_new < current_score:
                    # Improvement
                    current_x = x_new
                    current_f = f_new
                    current_score = score_new
                    # Increase step size slightly to explore further
                    step_size *= 1.2
                else:
                    # No improvement
                    # Decrease step size
                    step_size *= 0.7
                    # Optionally try opposite direction? Costs another eval.
                    # Let's just shrink.
                    
                # Cap step size
                if step_size > np.max(range_):
                    step_size = np.max(range_)
                if step_size < 1e-6 * np.max(range_):
                    step_size = 1e-6 * np.max(range_)

        # Final cleanup: Return best point
        if all_objectives:
            F = np.array(all_objectives)
            # Find the point with the best hypervolume contribution?
            # Or just return the one with the lowest sum?
            # The task says "return value of __call__ is ignored".
            # So we can return anything.
            best_idx = np.argmin(np.sum(F, axis=1))
            return F[best_idx]
        return np.array([1.0] * self.n_obj)
