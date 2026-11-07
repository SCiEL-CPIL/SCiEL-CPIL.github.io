import numpy as np

def acquisition(mu, sigma, f_best, cost):
        # Safety clamps
        sigma = np.maximum(sigma, 1e-9)
        cost = np.maximum(cost, 1e-9)
        
        # Adaptive Beta
        # Ackley max is ~29. 
        # We want beta to be high when f_best is high (early search)
        # and low when f_best is low (late search).
        f_scale = 25.0
        f_ratio = np.clip(f_best / f_scale, 0.0, 1.0)
        
        # Beta range: 
        # Early: explore more (beta ~ 3.0)
        # Late: exploit more (beta ~ 0.5)
        beta_min = 0.5
        beta_max = 3.0
        beta = beta_min + (beta_max - beta_min) * f_ratio
        
        # LCB for minimization: we want to maximize the score
        # Score = -mu + beta * sigma
        lcb_score = -mu + beta * sigma
        
        # Cost-aware: divide by cost
        scores = lcb_score / cost
        
        # Ensure finite
        scores = np.where(np.isfinite(scores), scores, 0.0)
        
        return scores
