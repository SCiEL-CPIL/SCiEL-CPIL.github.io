import numpy as np
import time

def acquisition(mu, sigma, f_best):
        # Deadline guard
        # The evaluator kills after 330s.
        # We don't know the start time, but we can assume this function is called frequently.
        # However, we can't track global time easily without a global variable or argument.
        # The prompt says "track elapsed time". This implies we might need a global or static variable.
        # But the signature is fixed.
        # We can use a module-level variable.
        
        global _start_time
        if not hasattr(acquisition, '_start_time'):
            acquisition._start_time = time.time()
            
        elapsed = time.time() - acquisition._start_time
        if elapsed > 320: # 10s margin
             # Fallback: return EI or random
             z = (f_best - mu) / (sigma + 1e-9)
             from scipy.stats import norm
             return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

        beta = 3.0
        # UCB for minimization: maximize -mu + sqrt(beta)*sigma
        scores = -mu + np.sqrt(beta) * sigma
        return scores
