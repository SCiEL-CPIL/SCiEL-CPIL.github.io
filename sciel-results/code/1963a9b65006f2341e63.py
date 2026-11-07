import numpy as np

def acquisition(mu, sigma, f_best, cost):
        # Ensure inputs are numpy arrays
        mu = np.asarray(mu)
        sigma = np.asarray(sigma)
        cost = np.asarray(cost)
        
        # Dynamic kappa based on f_best
        # If f_best is high, explore more (higher kappa).
        # If f_best is low, exploit more (lower kappa).
        # f_best is the minimum value found so far.
        # Griewank min is 0.
        
        # Heuristic: 
        # Start with high exploration.
        # As f_best approaches 0, reduce exploration.
        
        # Let's map f_best to [0, 1] range for the schedule.
        # A typical f_best might drop from ~0.5 to ~0.001.
        # Let's use a threshold.
        f_best_val = float(f_best)
        
        # If f_best is very good, use low kappa.
        # If f_best is poor, use high kappa.
        
        # Linear decay:
        # kappa = 0.5 + 2.5 * min(1.0, f_best_val / 0.05)
        # If f_best > 0.05, kappa = 3.0
        # If f_best == 0.05, kappa = 3.0
        # If f_best == 0.0, kappa = 0.5
        
        # This might be too aggressive in dropping kappa.
        # Let's try a smoother decay.
        # kappa = 0.5 + 2.5 * np.exp(-f_best_val * 50) ?
        # If f_best = 0, kappa = 3.0. No, we want low kappa at good f_best.
        # kappa = 0.5 + 2.5 * (1 - np.exp(-f_best_val * 50)) ?
        # If f_best = 0, kappa = 0.5.
        # If f_best = 0.1, exp(-5) ~ 0.006, kappa ~ 2.98.
        # This looks good. It stays high until f_best is very small, then drops.
        
        kappa = 0.5 + 2.5 * (1.0 - np.exp(-f_best_val * 50.0))
        
        # Calculate LCB score
        # LCB = mu - kappa * sigma
        # We want to maximize this, but penalize by cost.
        # Score = LCB / cost
        
        # Avoid division by zero (cost is >= 1, so safe)
        # Avoid negative scores? The function returns scores, higher is better.
        # If mu is large (bad), LCB is large. We want small mu.
        # Wait, LCB minimization: we want the most likely low value.
        # The acquisition function usually returns a value to MAXIMIZE.
        # If we use LCB for minimization, we minimize mu - kappa*sigma.
        # So we should maximize -(mu - kappa*sigma) = -mu + kappa*sigma.
        # Or, as in the previous EI code, it returned positive values for improvement.
        # EI = (f_best - mu) * ...
        # Here, if we use LCB, the "value" of a point is how low it *might* be.
        # A point with mu=0.1 and sigma=0.01 is better than mu=0.2, sigma=0.01.
        # Score should be higher for lower mu.
        # So Score = - (mu - kappa * sigma) / cost ?
        # Or Score = ( -mu + kappa * sigma ) / cost.
        
        # Let's check the sign.
        # If mu is low (good), -mu is high (positive).
        # If sigma is high, kappa*sigma is high.
        # So this encourages low mu and high sigma.
        
        lcb = mu - kappa * sigma
        # We want to minimize lcb.
        # So we maximize -lcb.
        score = -lcb / cost
        
        # Ensure finite
        score = np.where(np.isfinite(score), score, 0.0)
        
        return score
