import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Cost-aware acquisition: combine Expected Improvement (EI) with a
    # cost-aware penalty that discourages expensive regions.
    #
    # mu: (N,) posterior mean in objective units (lower is better)
    # sigma: (N,) posterior std in objective units
    # f_best: scalar, best observed value
    # cost: (N,) evaluation cost for each candidate
    #
    # We want higher scores = more attractive to evaluate next.
    # We compute EI for minimization:
    #   EI = (f_best - mu) * Phi(z) + sigma * phi(z),  where z = (f_best - mu) / sigma
    #
    # Then we apply a cost-aware transformation. Since cost = 1 + mean(x_scaled),
    # cost is in [1, 2]. We use a multiplicative discount that is stronger
    # for higher cost, but we also want to avoid completely ignoring expensive
    # regions if they have very high EI.
    #
    # Strategy: score = EI / cost^alpha, with alpha chosen to balance
    # exploration efficiency. We also add a small exploration bonus based
    # on sigma to ensure we don't get stuck in pure exploitation.
    
    N = len(mu)
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute EI for minimization
    z = (f_best - mu) / sigma_safe
    cdf_val = norm.cdf(z)
    pdf_val = norm.pdf(z)
    ei = sigma_safe * (z * cdf_val + pdf_val)
    
    # Cost-aware scoring: divide by cost^alpha
    # alpha=1.5 gives a stronger penalty for expensive points than simple 1/cost
    # This encourages the search to find good solutions in cheaper regions first
    alpha = 1.5
    cost_safe = np.maximum(cost, 1e-9)
    cost_factor = cost_safe ** alpha
    
    # Base score: EI per unit cost^alpha
    scores = ei / cost_factor
    
    # Add a small exploration term: points with high uncertainty (sigma) get a bonus
    # This prevents the algorithm from being too greedy and missing distant minima
    # The bonus is scaled by a small constant and also discounted by cost
    exploration_weight = 0.1
    exploration_bonus = exploration_weight * sigma_safe / cost_safe
    
    scores = scores + exploration_bonus
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Shift to ensure scores are well-behaved (optional, but helps with numerical stability)
    # The relative ordering is what matters, so we can subtract the minimum
    scores = scores - np.min(scores)
    
    return scores
