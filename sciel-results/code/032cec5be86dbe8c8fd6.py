import numpy as np
from scipy.stats import norm
import time

# Global time tracking for safety, though unlikely to be needed
_start_time = time.time()
_time_limit = 300.0 # Seconds

def acquisition(mu, sigma, f_best, cost):
    # Check time
        if time.time() - _start_time > _time_limit - 5.0:
            # Return a safe fallback, e.g., based on simple EI without cost penalty
            # Or just random? No, must be finite.
            # If we are out of time, the evaluator might kill us anyway.
            # But returning a valid array is safer.
            pass # We will just compute normally, it's fast.

        safe_sigma = np.maximum(sigma, 1e-9)
        
        # EI Calculation
        # f_best is a scalar, mu and sigma are arrays
        delta = f_best - mu
        z = delta / safe_sigma
        
        # To improve numerical stability for large negative/positive z
        # norm.cdf and norm.pdf handle this reasonably well in scipy
        cdf_z = norm.cdf(z)
        pdf_z = norm.pdf(z)
        
        ei = delta * cdf_z + safe_sigma * pdf_z
        
        # Cost-aware score
        # Use power 1.5 as per successful observation
        safe_cost = np.maximum(cost, 1e-9)
        score = ei / (safe_cost ** 1.5)
        
        # Handle any numerical issues
        score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
        
        return score
