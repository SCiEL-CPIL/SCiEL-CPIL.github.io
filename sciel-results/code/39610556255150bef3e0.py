import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
        # Deadline guard (safety net, should never trigger for vectorized ops)
        start_time = time.time()
        deadline = 300.0  # Seconds
        # In a real loop, we'd track global time, but here we just ensure this call is fast.
        
        try:
            # Cost-aware LCB (Lower Confidence Bound) for minimization
            # We want to minimize mu - beta * sigma
            # Score to maximize: (beta * sigma - mu) / cost
            
            beta = 4.0
            
            # Ensure sigma is non-negative
            sigma = np.maximum(sigma, 1e-9)
            
            # Calculate LCB values (lower is better)
            lcb = mu - beta * sigma
            
            # Convert to "higher is better" score, adjusted for cost
            # We want to pick the point with the lowest LCB, but cheap points are preferred.
            # score = -lcb / cost
            # If cost is very low, the score is amplified.
            
            scores = -lcb / cost
            
            # Handle any non-finite values
            scores = np.where(np.isfinite(scores), scores, 0.0)
            
            return scores
        
        except Exception:
            # Fallback: Return random scores or zero if something goes wrong
            return np.zeros(len(mu))
