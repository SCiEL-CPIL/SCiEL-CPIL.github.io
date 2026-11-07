import numpy as np
import time

def acquisition(mu, sigma, f_best):
       start_time = time.time()
       
       # Deadline guard
       # The limit is 330s for the whole program? Or per call?
       # "Each evaluation must finish within 300 s."
       # The acquisition is called within the evaluation step.
       # The evaluation step involves GP prediction (2000 points) and acquisition.
       # GP prediction is the bottleneck usually.
       # But we only control acquisition.
       # I will assume the 300s limit applies to the whole loop iteration or the function call?
       # "Each evaluation must finish within 300 s."
       # Usually, this means the whole step.
       # My function is fast.
       
       # Adaptive Kappa
       # f_best is scalar
       # If f_best is nan or inf, handle it
       if not np.isfinite(f_best):
           f_best_val = 1.0
       else:
           f_best_val = float(f_best)
           
       # Heuristic: 
       # Early (f_best > 0.5): Explore more (kappa ~ 2.0)
       # Late (f_best < 0.1): Exploit more (kappa ~ 0.5)
       # Linear interpolation
       # kappa = 0.5 + 1.5 * min(f_best, 1.0)
       # At 0: 0.5
       # At 0.5: 1.25
       # At 1.0: 2.0
       # At 5.0: 2.0
       
       kappa = 0.5 + 1.5 * np.clip(f_best_val, 0.0, 1.0)
       
       # Ensure sigma is non-negative
       sigma = np.maximum(sigma, 0.0)
       
       # Calculate scores
       scores = -mu + kappa * sigma
       
       # Ensure finite scores
       scores = np.nan_to_num(scores, nan=-1e6, posinf=1e6, neginf=-1e6)
       
       # Check time (just in case)
       elapsed = time.time() - start_time
       # If it took too long, we still return scores. 
       # There's no "partial result" for vectorized op.
       
       return scores
