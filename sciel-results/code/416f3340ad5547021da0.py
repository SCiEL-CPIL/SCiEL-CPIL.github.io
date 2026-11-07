import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 325.0  # Safety margin

        rng = np.random.default_rng(42)

        # State tracking
        best_x = np.zeros(dim)
        best_f = np.inf
        calls = 0

        # Parameters for the interleaved strategy
        # We will perform several "rounds". 
        # Each round consists of:
        # 1. A small batch of LHS samples (Global Exploration)
        # 2. A short (1+1)-ES run on the current best point (Local Exploitation)
        
        # Total budget is 500.
        # Let's say we do 10 rounds.
        # In each round, we spend ~10% of the budget.
        # 5 calls for LHS, 5 calls for ES.
        # This ensures we get an improvement (or at least a valid point) very early.
        
        n_rounds = 10
        budget_per_round = budget // n_rounds
        
        # In each round, split budget between global and local
        # Early rounds: more global to find the basin.
        # Later rounds: more local to refine.
        
        # We'll define the split dynamically.
        # Round 0-2: 60% global, 40% local
        # Round 3-6: 50% global, 50% local
        # Round 7-9: 30% global, 70% local
        
        # Pre-calculate LHS points for all global calls to ensure space filling
        # Total global calls = sum of global part of each round
        # Let's just generate a large pool of LHS points and consume them.
        # Total budget 500. Max global calls maybe 200.
        n_lhs_total = int(budget * 0.4) # Reserve 40% for global exploration overall
        lhs_pool = np.zeros((n_lhs_total, dim))
        for d in range(dim):
            intervals = np.linspace(-5, 5, n_lhs_total + 1)
            points = rng.uniform(intervals[:-1], intervals[1:])
            rng.shuffle(points)
            lhs_pool[:, d] = points
        
        lhs_idx = 0
        
        # Helper function for (1+1)-ES step with adaptive sigma
        def run_es(x_init, f_init, max_evals):
            nonlocal best_x, best_f, calls
            x = x_init.copy()
            f = f_init
            sigma = 0.5 # Initial step size
            
            # 1/5th rule parameters
            window = []
            window_size = 5 # Small window for fast adaptation in short runs
            target_rate = 0.2
            
            for _ in range(max_evals):
                if calls >= budget or time.time() > deadline:
                    break
                
                # Generate mutant
                mutation = rng.normal(0, sigma, size=dim)
                x_new = x + mutation
                x_new = np.clip(x_new, -5, 5)
                
                f_new = func(x_new)
                calls += 1
                
                # Update window
                if f_new < f:
                    window.append(1)
                    x = x_new
                    f = f_new
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                else:
                    window.append(0)
                
                # Adapt sigma
                if len(window) >= window_size:
                    success_rate = np.mean(window[-window_size:])
                    if success_rate > target_rate:
                        sigma *= 1.1
                    elif success_rate < target_rate:
                        sigma *= 0.9
                    sigma = np.clip(sigma, 1e-6, 10.0)
                    # Clear window to avoid bias from old data
                    window = []
                    
                # Convergence check
                if sigma < 1e-8:
                    break
            
            return x, f

        for r in range(n_rounds):
            if calls >= budget or time.time() > deadline:
                break
            
            # Determine split for this round
            if r < 3:
                global_frac = 0.6
            elif r < 7:
                global_frac = 0.5
            else:
                global_frac = 0.3
            
            n_global_this_round = max(1, int(budget_per_round * global_frac))
            n_local_this_round = max(1, budget_per_round - n_global_this_round)
            
            # 1. Global Exploration: Evaluate next LHS points
            current_best_x_for_es = best_x.copy()
            current_best_f_for_es = best_f
            
            for _ in range(n_global_this_round):
                if calls >= budget or time.time() > deadline or lhs_idx >= n_lhs_total:
                    break
                
                x_candidate = lhs_pool[lhs_idx].copy()
                lhs_idx += 1
                
                f_candidate = func(x_candidate)
                calls += 1
                
                if f_candidate < best_f:
                    best_f = f_candidate
                    best_x = x_candidate.copy()
                    # If we found a new best, update the target for ES
                    current_best_x_for_es = x_candidate.copy()
                    current_best_f_for_es = f_candidate
            
            # 2. Local Exploitation: Run ES on the best point found in this round (or global best)
            if calls < budget and time.time() < deadline:
                # If we haven't found any valid point yet (shouldn't happen), start from 0
                if not np.isfinite(best_f):
                    x_start = np.zeros(dim)
                    f_start = func(x_start)
                    calls += 1
                    best_x = x_start.copy()
                    best_f = f_start
                    current_best_x_for_es = x_start.copy()
                    current_best_f_for_es = f_start
                
                # Run ES for the allocated local budget
                run_es(current_best_x_for_es, current_best_f_for_es, n_local_this_round)
        
        # If we still have budget left, use it all for ES on the best point found
        remaining = budget - calls
        if remaining > 0 and np.isfinite(best_f) and time.time() < deadline:
            run_es(best_x.copy(), best_f, remaining)

        return best_x, best_f
