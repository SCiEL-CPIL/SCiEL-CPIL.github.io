import numpy as np

def acquisition(mu, sigma, f_best):
    # Adaptive UCB strategy for minimization
    # Score = -mu + kappa * sigma
    # We want to maximize this score.
    
    # Adapt kappa based on the current best value
    # High f_best -> Early stage -> High exploration (high kappa)
    # Low f_best -> Late stage -> Low exploration (low kappa)
    
    # Base exploration level
    base_kappa = 1.0
    # Exploration bonus proportional to how far we are from the optimum
    # f_best is the square of the distance to 0, so it's a good proxy
    exploration_bonus = 0.5 * f_best
    
    kappa = base_kappa + exploration_bonus
    
    # Compute acquisition scores
    # -mu encourages low mean
    # +kappa*sigma encourages high uncertainty
    scores = -mu + kappa * sigma
    
    # Ensure finiteness
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return scores
