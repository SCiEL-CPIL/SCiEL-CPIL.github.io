# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES with momentum and 1/5-rule sigma adaptation for BBOB f24."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: evaluator kills after 330s. We return well before that.
        deadline = start_time + 320.0

        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls = 0

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Define number of starts
        # 10 starts seems good for 500 budget (50 evals each)
        # If we use 1/5 rule, we might need a bit more to adapt sigma.
        # Let's try 8 starts to allow ~60 evals each.
        n_starts = 8
        starts = []
        
        # Always include origin
        starts.append(np.zeros(n))
        
        # Add random points in the likely region [-4, 4]^n
        for _ in range(n_starts - 1):
            starts.append(rng.uniform(-4, 4, size=n))

        # Shuffle starts to ensure we don't always prioritize the same ones if budget is tight?
        # Actually, fixed order is fine for reproducibility, but random order might help if some starts are "bad" for specific instances?
        # The instances are fixed seeds, so fixed order is okay.
        # Let's evaluate starts and pick the best one to continue with? 
        # No, multi-start means we run local search from each.
        
        sub_budget = budget // n_starts
        # Ensure we have at least some budget
        sub_budget = max(sub_budget, 10)

        for i in range(n_starts):
            if calls >= budget:
                break
            if time.time() > deadline:
                break

            x = starts[i].copy()
            fx = func(x)
            calls += 1
            update_best(x, fx)

            # Run local (1+1)-ES from this start
            # We have remaining budget for this start
            local_budget = min(sub_budget, budget - calls)
            
            if local_budget <= 0:
                continue

            sigma = 1.0
            momentum = np.zeros(n)
            mom_coeff = 0.9
            success_count = 0
            window_size = 10
            steps_in_window = 0
            no_improve = 0
            max_no_improve = 20 # Restart local search if stuck
            
            current_x = x.copy()
            current_f = fx
            local_best_x = x.copy()
            local_best_f = fx

            while calls < budget:
                if time.time() > deadline:
                    break
                
                # Generate candidate
                z = rng.standard_normal(n)
                # Apply momentum
                # The momentum vector is in "standard normal" space or "x" space?
                # Inspiration used: candidate = x + sigma * (z + momentum)
                # This implies momentum is in the same units as z (standard normal).
                # So step = sigma * (z + momentum).
                candidate = current_x + sigma * (z + momentum)
                candidate = np.clip(candidate, -5, 5)
                
                fc = func(candidate)
                calls += 1
                
                if fc < current_f:
                    # Success
                    success_count += 1
                    no_improve = 0
                    
                    # Update momentum
                    # step in z-space is (candidate - current_x) / sigma
                    # But candidate was clipped, so this is approximate.
                    # Better: momentum = mom_coeff * momentum + (1-mom_coeff) * z
                    # Or track actual step?
                    # Let's use the z vector that generated the success.
                    momentum = mom_coeff * momentum + (1 - mom_coeff) * z
                    
                    current_x = candidate
                    current_f = fc
                    
                    if fc < local_best_f:
                        local_best_f = fc
                        local_best_x = candidate.copy()
                        
                    update_best(candidate, fc)
                else:
                    no_improve += 1
                    # Decay momentum slightly on failure? Or keep it?
                    # Keeping it might help if we are in a valley.
                    # Let's keep it but maybe decay if no_improve is high?
                    if no_improve > 5:
                        momentum *= 0.9

                steps_in_window += 1
                if steps_in_window >= window_size:
                    p = success_count / steps_in_window
                    # 1/5 rule
                    if p > 0.2:
                        sigma *= 1.1
                    elif p < 0.2:
                        sigma *= 0.9
                    
                    success_count = 0
                    steps_in_window = 0
                    
                    # Bounds for sigma
                    if sigma > 5.0:
                        sigma = 5.0
                    if sigma < 1e-6:
                        sigma = 1e-6

                # Restart logic within local search
                if no_improve >= max_no_improve:
                    # Restart from local best with larger sigma
                    current_x = local_best_x.copy()
                    current_f = local_best_f
                    sigma = min(sigma * 2.0, 5.0)
                    momentum = np.zeros(n)
                    no_improve = 0
                    success_count = 0
                    steps_in_window = 0

                if calls >= budget:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
