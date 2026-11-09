import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is strictly positive
    sigma = np.maximum(sigma, 1e-8)
    
    # 1. Compute Expected Improvement (EI) for all candidates
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # 2. Select top K candidates for KG approximation
    K = 50
    if len(mu) < K:
        K = len(mu)
        
    # Get indices of top K candidates by EI
    top_k_indices = np.argsort(ei)[::-1][:K]
    
    # Initialize scores with 0 for all candidates
    scores = np.zeros_like(ei)
    
    if K == 0:
        return scores
        
    # 3. For each top candidate, approximate Knowledge Gradient
    # We use a Monte Carlo estimate of the expected reduction in the best value
    # Approximation: Compare against the best *other* predicted mean in the top K set
    
    # Pre-compute the best other mean for each top candidate
    # top_mu is the mu values of the top K candidates
    top_mu = mu[top_k_indices]
    top_sigma = sigma[top_k_indices]
    
    # Number of Monte Carlo samples
    M = 50
    
    for i_idx in range(K):
        orig_idx = top_k_indices[i_idx]
        mu_i = mu[orig_idx]
        sigma_i = sigma[orig_idx]
        
        # Find the best mean among the OTHER top K candidates
        # Create a mask to exclude i_idx
        if K > 1:
            others_mask = np.ones(K, dtype=bool)
            others_mask[i_idx] = False
            best_other_mean = np.min(top_mu[others_mask])
        else:
            best_other_mean = np.inf # If only one candidate, no other mean to compare against in this set
            
        # Sample from the posterior of candidate i
        samples = np.random.normal(mu_i, sigma_i, M)
        
        # Current best if we evaluate i: min(f_best, sample)
        current_best = np.minimum(f_best, samples)
        
        # New best if we evaluate i, considering the rest of the top K set
        # We assume the other points' values are their means (mean field approximation)
        # The best of the others is best_other_mean
        new_best = np.minimum(np.minimum(f_best, samples), best_other_mean)
        
        # KG is the expected improvement of the global best
        kg_val = np.mean(current_best - new_best)
        
        scores[orig_idx] = kg_val
        
    # 4. For candidates not in top K, we can assign a small value or 0
    # To ensure we don't completely ignore exploration outside top K if EI was noisy,
    # we might add a small UCB term, but KG usually dominates.
    # Let's just return the KG scores. Candidates not in top K get 0.
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
