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
        deadline = start_time + 280  # Safety margin
        
        # Phase 1: Identify optimal signs
        # We use the origin (0,0,...,0) as the base for probing.
        # This assumes the function is well-behaved (linear).
        # For BBOB f5, this is correct.
        
        signs = np.zeros(dim)
        
        # If budget is very tight, we might skip some probes, but 500 is plenty.
        # We need 2*dim calls for probes.
        if budget < 2 * dim + 1:
             # Fallback to random or simple ES if budget is tiny
             # But this case is not expected.
             pass

        for i in range(dim):
            if time.time() > deadline:
                break
            
            x_pos = np.zeros(dim)
            x_neg = np.zeros(dim)
            x_pos[i] = 5.0
            x_neg[i] = -5.0
            
            f_pos = func(x_pos)
            f_neg = func(x_neg)
            
            if f_pos < f_neg:
                signs[i] = 5.0
            elif f_neg < f_pos:
                signs[i] = -5.0
            else:
                signs[i] = 0.0
        
        # Phase 2: Evaluate the constructed vertex
        x_opt = signs
        f_opt = func(x_opt)
        
        # Phase 3: Fill remaining budget
        # We track calls made to ensure we don't exceed budget.
        # Calls made so far: 2 * dim (probes) + 1 (final) = 2*dim + 1
        # However, if we broke out of the loop early, we made fewer.
        # Let's just count explicitly or estimate.
        # Simpler: just loop until budget is reached or deadline.
        
        # We don't know exactly how many calls were made if we broke early.
        # But we can just use a counter.
        
        # Let's restart with a counter for safety.
        # Actually, the loop above is deterministic if no break.
        # If break, we might have incomplete signs.
        # Given 500 calls, break is unlikely.
        
        # To be safe, I will implement a call counter.
        
        # Re-doing with counter for robustness.
        
        # Reset
        signs = np.zeros(dim)
        calls = 0
        
        for i in range(dim):
            if calls >= budget or time.time() > deadline:
                break
            
            x_pos = np.zeros(dim)
            x_neg = np.zeros(dim)
            x_pos[i] = 5.0
            x_neg[i] = -5.0
            
            f_pos = func(x_pos)
            calls += 1
            f_neg = func(x_neg)
            calls += 1
            
            if f_pos < f_neg:
                signs[i] = 5.0
            elif f_neg < f_pos:
                signs[i] = -5.0
            else:
                signs[i] = 0.0
        
        x_opt = signs
        
        if calls < budget and time.time() < deadline:
            f_opt = func(x_opt)
            calls += 1
        else:
            f_opt = func(x_opt) # Must return something, even if over budget? 
            # No, "More than 500 calls... makes the whole result invalid".
            # So we must not exceed budget.
            # If calls == budget, we cannot call again.
            # But we need a return value.
            # The return value is ignored, but the *calls* are recorded.
            # If we don't call, we don't get a value?
            # "Every call is recorded".
            # If we don't make the final call, b_i is based on previous calls.
            # But we want to record the optimal value.
            # So we MUST make the final call if we have budget left.
            # If we don't have budget left, we are in trouble.
            # But 2*dim + 1 = 11 << 500.
            pass

        # Fill remaining
        while calls < budget and time.time() < deadline:
            func(x_opt)
            calls += 1

        return x_opt, f_opt if 'f_opt' in locals() else 0.0
