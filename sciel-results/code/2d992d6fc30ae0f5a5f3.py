import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: The evaluator kills at 330s. We aim to finish well before.
        # Let's set a strict internal deadline to ensure we return a result.
        deadline = start_time + 295.0
        
        n = self.dim
        budget = self.budget
        
        # (1+1)-ES parameters
        # 1/5 success rule is standard and robust
        success_prob = 0.2
        # Initial step size. The domain is [-5, 5]^5.
        # A step size of 1.0 is a reasonable start for 5D.
        # If the function is scaled differently, the adaptation will adjust.
        sigma = 1.0
        
        # Start point: 0 is a good neutral start.
        # The minimizer is in [-4, 4]^5, so 0 is close.
        x = np.zeros(n)
        fx = func(x)
        calls = 1
        
        # Track history for 1/5 rule
        # We count successes in a window of (1+1)/success_prob = 6 iterations?
        # Standard 1/5 rule: after every k = (1+1)/0.2 = 6 steps, check success rate.
        # If success rate > 0.2, increase sigma; if < 0.2, decrease sigma.
        # Actually, the standard formulation updates sigma every generation based on a counter.
        
        # Let's use a simpler, more robust adaptation:
        # Keep a counter of successes in the last 10 generations?
        # Or stick to the strict 1/5 rule:
        # n_evals = 0
        # n_success = 0
        # While budget remains:
        #   Generate x_new = x + sigma * N(0, I)
        #   Evaluate
        #   If f(x_new) < f(x):
        #       x = x_new
        #       fx = f(x_new)
        #       n_success += 1
        #   n_evals += 1
        #   If n_evals == 6: # 1/0.2
        #       if n_success > 1: # > 1/5
        #           sigma *= 1.5
        #       elif n_success < 1: # < 1/5
        #           sigma /= 1.5
        #       n_success = 0
        #       n_evals = 0
        
        # However, this can be slow to react. A faster adaptation is often better for BBOB.
        # Let's use the "1/5 rule" but with a smaller window or direct multiplicative update per step?
        # No, the standard 1/5 rule is:
        # sigma_{t+1} = sigma_t * exp( (1/lambda) * (N_success - 1) )? No, that's for lambda>1.
        # For (1+1), the standard is:
        # If success, sigma *= c_inc, else sigma *= c_dec.
        # With 1/5 rule, we want P(success) = 0.2.
        # If we update every step:
        # If success: sigma *= exp(1/5 * (1 - 0.2) / 0.2)? No.
        
        # Let's use a robust variant:
        # Update sigma based on the outcome of the last 10 steps.
        # If more than 2 successes (20%), increase. If less, decrease.
        
        window_size = 10
        successes = []
        
        best_x = x.copy()
        best_f = fx
        
        while calls < budget:
            if time.time() > deadline:
                break
            
            # Generate offspring
            # Use a fixed seed for reproducibility in tests? No, random is fine.
            # But for stability, let's use a default rng.
            # Actually, the evaluator might run multiple instances.
            rng = np.random.default_rng()
            
            # Clip to box? The problem says func clips, but it's better to generate valid points
            # to avoid wasted evaluations if the function penalizes out-of-bounds (though it says it clips).
            # "func(x) clips a finite length-5 vector to the box". So we can generate anywhere.
            
            y = x + sigma * rng.standard_normal(n)
            
            # Optional: Clip y to [-5, 5] to stay within the domain explicitly?
            # The problem says the minimizer is in [-4, 4]^5 and func clips to [-5, 5].
            # It's safer to keep x within bounds to avoid numerical issues or undefined behavior
            # if the "clipping" is just a description of the evaluator's behavior and not the function's.
            # But the contract says "func(x) clips...". So we are safe.
            
            fy = func(y)
            calls += 1
            
            # Update success history
            if fy < fx:
                x = y
                fx = fy
                successes.append(1)
                
                # Update global best
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            else:
                successes.append(0)
                
            # Keep only last 'window_size' successes
            if len(successes) > window_size:
                successes.pop(0)
            
            # Adapt sigma
            if len(successes) == window_size:
                p_success = np.mean(successes)
                # 1/5 rule logic:
                # If p_success > 0.2, we are too small, increase sigma.
                # If p_success < 0.2, we are too big, decrease sigma.
                # Multiplicative factor:
                # Standard: sigma *= exp( (1/lambda) * (N - 1) ) is for lambda.
                # For (1+1), a common choice is:
                # sigma *= 1.5 if p_success > 0.2
                # sigma *= 1.5/2.5? No.
                # Let's use:
                # sigma *= exp( (1/5) * (p_success - 0.2) * 5 )?
                # Simpler:
                if p_success > 0.2:
                    sigma *= 1.5
                elif p_success < 0.2:
                    sigma /= 1.5
                # Clamp sigma
                sigma = np.clip(sigma, 1e-6, 10.0)
            
            # Stagnation detection / Restart
            # If we haven't improved best_f for a long time, restart?
            # With 500 budget, we can afford a few restarts.
            # Let's track iterations since last improvement.
            if not hasattr(self, '_iter_since_improve'):
                self._iter_since_improve = 0
            else:
                self._iter_since_improve += 1
            
            if fy < best_f:
                self._iter_since_improve = 0
            
            # If stuck for 100 iterations, restart at a new random point
            if self._iter_since_improve > 100:
                # Restart
                x = rng.uniform(-5, 5, n)
                fx = func(x)
                calls += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                sigma = 1.0
                successes = []
                self._iter_since_improve = 0

        # Final Polish: (1+1)-ES with very small sigma
        if calls < budget and time.time() < deadline:
            sigma_polish = 1e-3
            x_polish = best_x.copy()
            fx_polish = best_f
            rng_polish = np.random.default_rng(42)
            polish_calls = 0
            max_polish = min(50, budget - calls)
            
            while polish_calls < max_polish:
                if time.time() > deadline:
                    break
                y = x_polish + sigma_polish * rng_polish.standard_normal(n)
                fy = func(y)
                calls += 1
                polish_calls += 1
                if fy < fx_polish:
                    x_polish = y
                    fx_polish = fy
                    sigma_polish *= 1.1
                    if fx_polish < best_f:
                        best_f = fx_polish
                        best_x = x_polish.copy()
                else:
                    sigma_polish *= 0.9
                    if sigma_polish < 1e-8:
                        break

        return best_x, best_f
