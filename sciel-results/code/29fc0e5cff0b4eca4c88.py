import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Avoid division by zero in z calculation
    sigma_safe = np.maximum(sigma, 1e-10)
    z = (f_best - mu) / sigma_safe
    
    # Calculate EI
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    ei = (f_best - mu) * cdf + sigma * pdf
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Cost safe
    cost_safe = np.maximum(cost, 1e-10)
    
    # Parametric Cost-Aware EI: Score = EI / Cost^gamma
    # Optimize gamma based on current data distribution
    # We want to maximize the contrast between high-value (high EI, low cost) and low-value points.
    # Metric: Ratio of top 1% score to median score.
    
    candidate_gammas = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
    best_gamma = 1.0
    best_metric = -np.inf
    
    # Precompute log cost for efficiency if needed, but direct power is fine for 2000 points
    for gamma in candidate_gammas:
        scores = ei / (cost_safe ** gamma)
        # Handle potential infinities/NaNs
        scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
        
        if np.max(scores) == 0:
            metric = 0
        else:
            # Use top 1% vs median to measure discriminative power
            top_1pct = np.percentile(scores, 99)
            median_score = np.median(scores)
            if median_score > 0:
                metric = top_1pct / median_score
            else:
                metric = top_1pct
        
        if metric > best_metric:
            best_metric = metric
            best_gamma = gamma
            
    # Final scores with best gamma
    final_scores = ei / (cost_safe ** best_gamma)
    final_scores = np.nan_to_num(final_scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return final_scores
