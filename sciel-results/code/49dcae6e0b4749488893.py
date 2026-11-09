import numpy as np

def heuristics(demands, capacity):
        n = len(demands)
        d = np.array(demands, dtype=float)
        c = float(capacity)
        
        # Sum of pairs
        pair_sums = d[:, None] + d[None, :]
        
        # Feasibility mask
        feasible = pair_sums <= c
        
        # Residual space
        residuals = c - pair_sums
        
        # Smallest item size
        w_min = np.min(d)
        
        # Avoid division by zero if w_min is 0 (though demands are 20-99, so w_min >= 20)
        if w_min == 0:
             w_min = 1.0
             
        # Modulo of residual by w_min
        # We want R % w_min to be small.
        # remainder = residuals % w_min
        # fit_score = 1 - (remainder / w_min)
        
        remainder = residuals % w_min
        fit_score = 1.0 - (remainder / w_min)
        
        # Tightness score: prefer larger sums
        # tightness = pair_sums / c
        # But if pair_sums is 0, this is 0.
        # We want to avoid 0 heuristic values if possible, or let the ACO handle them.
        # The contract says non-positive becomes 1e-9.
        
        # Combined heuristic with smooth surrogate for min-like behavior
        # Use soft-min via log-sum-exp to create a smooth approximation
        # of the minimum residual that could fit another item
        
        # Compute slack: how much space is left
        slack = residuals
        
        # For each pair, compute a smooth measure of "complementarity"
        # We want to reward pairs where the residual is close to a multiple of w_min
        # Use a Gaussian-like kernel centered at multiples of w_min
        
        # Scale factor for the Gaussian width
        sigma = w_min * 0.3
        
        # Compute distance to nearest multiple of w_min
        # This is already captured by remainder, but let's use a smoother version
        # dist_to_multiple = remainder
        # But we want a smooth peak, so use exp(-dist^2 / (2*sigma^2))
        
        # Smooth complementarity score
        complementarity = np.exp(-(remainder ** 2) / (2 * sigma ** 2))
        
        # Tightness score: prefer larger sums (smaller residuals)
        tightness = pair_sums / c
        
        # Combined heuristic: weighted combination
        # alpha for tightness, beta for complementarity
        alpha = 0.6
        beta = 0.4
        
        h = alpha * tightness + beta * complementarity
        
        # Apply feasibility
        h = np.where(feasible, h, 1e-9)
        
        # Ensure no negative values (shouldn't happen, but safe)
        h = np.maximum(h, 1e-9)
        
        return h
