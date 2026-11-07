# EVOLVE-BLOCK-START
import numpy as np
import time
import math

class Optimizer:
    """Multi-start (1+1)-ES optimizer for Katsuura function f23."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin before 300s limit
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf

        def clip(x):
            return np.clip(x, -5.0, 5.0)

        # Generate diverse starting points
        # 1. Center
        starts = [np.zeros(dim)]
        # 2. Corners of the box [-5, 5]^dim
        # There are 2^dim corners. For dim=5, that's 32 corners.
        # We can include all of them or a subset. Let's include all for dim=5 as it's small.
        for i in range(2 ** dim):
            x = np.zeros(dim)
            for j in range(dim):
                # Binary representation of i determines the corner
                if (i >> j) & 1:
                    x[j] = 5.0
                else:
                    x[j] = -5.0
            starts.append(x)
        # 3. Random points
        # Add 10 random points to cover intermediate regions
        for _ in range(10):
            x = rng.uniform(-5, 5, dim)
            starts.append(x)

        # Shuffle starts to ensure variety in order? 
        # Or keep deterministic order? 
        # Let's shuffle to avoid bias if some starts are consistently better/worse.
        # But reproducibility is good. We will use a fixed seed for rng, so shuffling is deterministic.
        # However, the corners are symmetric in Katsuura (mostly). 
        # Let's just iterate through them.
        
        # Strategy: 
        # We have a total budget of 500 evaluations.
        # We have ~43 starts (1 center + 32 corners + 10 random).
        # 500 / 43 is about 11-12 evaluations per start.
        # This is very low for (1+1)-ES to converge.
        
        # Better Strategy:
        # Phase 1: Quick scan of all starts with 1 evaluation each to find promising regions.
        # Phase 2: Run (1+1)-ES from the top K promising starts with the remaining budget.
        
        # Phase 1: Evaluate all starts
        evals_used = 0
        start_scores = []
        for x in starts:
            if time.time() > deadline:
                break
            if evals_used >= budget:
                break
            fx = func(x)
            start_scores.append((fx, x.copy()))
            evals_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
                
        if evals_used >= budget or time.time() > deadline:
            return best_x, best_f

        # Sort starts by fitness
        start_scores.sort(key=lambda t: t[0])
        
        # Phase 2: Run (1+1)-ES from the top K starts
        # How many top starts? 
        # Let's pick the top 5.
        # Remaining budget = budget - evals_used
        # Budget per ES run = remaining_budget // 5
        
        top_k = min(5, len(start_scores))
        remaining_budget = budget - evals_used
        if remaining_budget <= 0:
            return best_x, best_f
            
        evals_per_es = max(10, remaining_budget // top_k)
        
        for i in range(top_k):
            if time.time() > deadline:
                break
            if evals_used >= budget:
                break
                
            fx_init, x_init = start_scores[i]
            
            # Run (1+1)-ES
            x = x_init.copy()
            f = fx_init
            sigma = 1.0  # Initial step size
            success_count = 0
            total_count = 0
            window_size = 10  # Window for success rate calculation
            
            # We will run for evals_per_es evaluations
            for _ in range(evals_per_es):
                if time.time() > deadline:
                    break
                if evals_used >= budget:
                    break
                    
                # Generate offspring
                z = rng.standard_normal(dim)
                x_trial = x + sigma * z
                x_trial = clip(x_trial)
                
                f_trial = func(x_trial)
                evals_used += 1
                
                # Update global best
                if f_trial < best_f:
                    best_f = f_trial
                    best_x = x_trial.copy()
                    
                # Selection
                if f_trial < f:
                    x = x_trial
                    f = f_trial
                    success_count += 1
                else:
                    # Keep parent
                    pass
                    
                total_count += 1
                
                # Adapt step size every window_size evaluations
                if total_count >= window_size:
                    success_rate = success_count / total_count
                    # 1/5 success rule
                    if success_rate > 1.0 / 5.0:
                        sigma *= 1.5
                    elif success_rate < 1.0 / 5.0:
                        sigma /= 1.5
                    # Clamp sigma
                    sigma = np.clip(sigma, 1e-3, 10.0)
                    
                    # Reset counters
                    success_count = 0
                    total_count = 0

        return best_x, best_f
# EVOLVE-BLOCK-END
