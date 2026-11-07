import numpy as np

def acquisition(mu, sigma, f_best, cost):
        # Cost is >= 1, so safe to divide
        # UCB for minimization
        # We want to maximize the score
        # Score = (Exploration - Exploitation) / Cost
        # Exploitation: mu (lower is better) -> -mu
        # Exploration: sigma (higher is better) -> sigma
        # Combined: beta * sigma - mu
        
        # Dynamic beta? 
        # If we assume the loop runs ~25-30 times.
        # Early on, we want more exploration.
        # Without step count, we can't do it.
        # Fixed beta is the only option.
        
        beta = 2.0
        score = (beta * sigma - mu) / cost
        
        # Sanitize
        score = np.nan_to_num(score, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
        # Actually, if posinf, it's the best. If neginf, worst.
        # But np.nan_to_num with posinf=np.inf might cause issues if we take max? 
        # No, max will pick inf.
        
        return score
