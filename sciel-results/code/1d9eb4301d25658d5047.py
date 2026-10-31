import numpy as np
from scipy.stats import weibull_min

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Adversarial robustness scoring: minimize expected future regret.
    
    For each bin, after placing the item, compute the expected size of the
    largest Weibull-distributed item that could still fit in the remainder.
    Bins with remainders that can accommodate larger future items get higher scores.
    This prioritizes creating remainders robust against worst-case future demand.
    """
    # Calculate remaining space after placing the item
    remaining = bins - item
    
    # Initialize scores to -infinity to handle invalid bins
    scores = np.full(bins.shape, -np.inf, dtype=float)
    
    # Define valid bins (can fit the item)
    valid_mask = remaining >= -1e-9
    
    if np.any(valid_mask):
        r = remaining[valid_mask]
        
        # Weibull parameters for the item distribution (typical online bin packing)
        # Shape parameter k=1.5, scale lambda=30 gives mean ~30*Gamma(1+1/1.5)≈27
        # This is a reasonable assumption for Weibull sequences in bin packing
        k = 1.5
        lam = 30.0
        
        # For each remainder r, compute the expected maximum item that fits
        # P(item <= r) = F(r) = 1 - exp(-(r/lam)^k)
        # The expected value of the largest item that fits is related to the inverse CDF
        # Use a simpler proxy: score = F(r) * r, which balances probability of fitting
        # and the actual space available
        
        # Compute CDF values: probability that a random item fits in remainder r
        # F(r) = 1 - exp(-(r/lam)^k)
        cdf_vals = 1.0 - np.exp(-np.power(r / lam, k))
        
        # Adversarial regret: the expected size of the next item that WON'T fit
        # E[item | item > r] * P(item > r) = (1 - F(r)) * E[item | item > r]
        # For Weibull, E[item | item > r] = lam * Gamma(1 + 1/k) / (1 - F(r))^(1/k) approximately
        # Simpler: use survival function S(r) = exp(-(r/lam)^k)
        # Regret proxy = S(r) * lam * Gamma(1+1/k)  (expected wasted capacity if item doesn't fit)
        
        from scipy.special import gamma as _gamma
        expected_item_size = lam * _gamma(1.0 + 1.0/k)
        
        # Score = negative of expected regret + bonus for having large remaining space
        # We want to MINIMIZE regret, so score = -regret + small bonus for utilization
        survival = np.exp(-np.power(r / lam, k))
        expected_regret = survival * expected_item_size
        
        # Also add a term that rewards remainders that are "useful" 
        # (large enough to fit most items) vs "dead" (too small to be useful)
        # Useful if r > typical item size
        useful_bonus = np.where(r > 0.5 * expected_item_size, r * 0.1, 0.0)
        
        scores[valid_mask] = -expected_regret + useful_bonus
        
    return scores
