"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Knowledge Gradient (KG) acquisition function via Monte Carlo integration.
    Computes the expected reduction in the minimum observed value for each candidate,
    accounting for the joint distribution over all candidates to balance
    exploration and exploitation for multi-modal functions like Hartmann 3D.
    
    Parameters:
    -----------
    mu : array-like, shape (n_candidates,)
        Mean of the GP posterior at each candidate point.
    sigma : array-like, shape (n_candidates,)
        Standard deviation of the GP posterior at each candidate point.
    f_best : float
        Best (lowest) observed value so far.
    
    Returns:
    --------
    scores : ndarray, shape (n_candidates,)
        KG scores for each candidate. Higher is better.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    n_candidates = mu.shape[0]
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Monte Carlo Knowledge Gradient
    # The true KG for candidate i is:
    #   E[ max_j f_j | f_i ] - min(f_best, E[f_i])
    # where the expectation is over the joint posterior of all candidates given f_i.
    # 
    # Under the independent approximation (ignoring correlations between candidates),
    # given f_i = x, the other candidates are still N(mu_j, sigma_j^2).
    # So E[max_j f_j | f_i = x] = E[max(f_i, max_{j != i} f_j) | f_i = x]
    #                            = E[max(x, max_{j != i} f_j)]
    # 
    # We approximate this via Monte Carlo:
    # 1. Draw S samples of all candidates: f_s = (f_1^s, ..., f_n^s)
    # 2. For each candidate i, we want E[max(f_i, max_{j != i} f_j)]
    #    = E[max(f_i^s, max_{j != i} f_j^s)] over samples s
    # 
    # This can be computed as: for each sample s, compute m_s = max_j f_j^s
    # and second_max_s = second largest value in sample s.
    # Then for candidate i in sample s:
    #   if f_i^s is the max, then max(f_i^s, max_{j != i} f_j^s) = f_i^s = m_s
    #   else, max(f_i^s, max_{j != i} f_j^s) = max(f_i^s, second_max_s)
    # 
    # Actually, a cleaner way: for each sample s, let m_s = max_j f_j^s.
    # For candidate i, the contribution to E[max(f_i, max_{j != i} f_j)] in sample s is:
    #   max(f_i^s, max_{j != i} f_j^s)
    # which equals max(f_i^s, m_s) if i is NOT the argmax, or m_s if i IS the argmax.
    # Wait, if i IS the argmax, then max_{j != i} f_j^s is the second max, and
    # max(f_i^s, second_max) = f_i^s = m_s.
    # If i is NOT the argmax, then max_{j != i} f_j^s = m_s (since some other j is the max),
    # so max(f_i^s, m_s) = m_s.
    # So in ALL cases, max(f_i^s, max_{j != i} f_j^s) = m_s!
    # 
    # That means E[max(f_i, max_{j != i} f_j)] = E[m_s] for ALL i, which is independent of i.
    # That can't be right... Let me reconsider.
    #
    # Actually, the KG is:
    #   KG(i) = E_{f | f_i}[ min(f_best, min_j f_j) ] - min(f_best, E[f_i])
    # Wait, we're minimizing, so:
    #   KG(i) = min(f_best, E[f_i]) - E_{f | f_i}[ min(f_best, min_j f_j) ]
    # Hmm, let me think more carefully.
    #
    # We want to maximize the expected reduction in the best value:
    #   KG(i) = E[ f_best - min(f_best, min_j f_j) | f_i ]
    #         = f_best - E[ min(f_best, min_j f_j) | f_i ]
    #
    # Under independence: given f_i = x, min_j f_j = min(x, min_{j != i} f_j)
    # So E[min(f_best, min_j f_j) | f_i = x] = E[min(f_best, x, min_{j != i} f_j)]
    #
    # Via MC: for each sample s, we have f_j^s for all j.
    # Given f_i = f_i^s (we use the sample value as the "conditioned" value),
    # min(f_best, f_i^s, min_{j != i} f_j^s)
    # 
    # Let m_s = min_j f_j^s (the minimum in sample s)
    # Let second_min_s = second minimum in sample s
    # 
    # If i is the argmin in sample s: min(f_best, f_i^s, min_{j != i} f_j^s) = min(f_best, f_i^s, second_min_s) = min(f_best, f_i^s) since f_i^s <= second_min_s
    # If i is NOT the argmin: min(f_best, f_i^s, min_{j != i} f_j^s) = min(f_best, f_i^s, m_s) = min(f_best, m_s) since m_s <= f_i^s
    #
    # So for each sample s and candidate i:
    #   value_{i,s} = min(f_best, f_i^s) if i == argmin_s, else min(f_best, m_s)
    #
    # Then KG(i) = f_best - (1/S) * sum_s value_{i,s}
    
    n_samples = 200
    rng = np.random.default_rng(42)
    
    # Generate samples: (n_samples, n_candidates)
    samples = rng.normal(loc=mu, scale=sigma_safe, size=(n_samples, n_candidates))
    
    # Compute min and argmin for each sample
    sample_min = np.min(samples, axis=1)  # shape (n_samples,)
    min_indices = np.argmin(samples, axis=1)  # shape (n_samples,)
    
    # For each sample, compute min(f_best, sample_min)
    capped_min = np.minimum(f_best, sample_min)  # shape (n_samples,)
    
    # For each candidate i, compute the expected min(f_best, min_j f_j | f_i)
    # Initialize with capped_min for all (the "not argmin" case)
    # Then for the argmin case, replace with min(f_best, f_i^s)
    
    # Start with the "not argmin" contribution: for each sample s, all candidates get capped_min[s]
    # But for the argmin candidate in sample s, they get min(f_best, samples[s, min_indices[s]])
    
    # Create a matrix of values: (n_samples, n_candidates)
    # Default value for each (s, i) is capped_min[s]
    values = np.tile(capped_min[:, np.newaxis], (1, n_candidates))
    
    # For each sample s, the argmin candidate gets min(f_best, samples[s, min_indices[s]])
    # Instead of a loop, use advanced indexing
    argmin_vals = np.minimum(f_best, samples[np.arange(n_samples), min_indices])
    
    # Subtract the old value and add the new value for argmin positions
    values[np.arange(n_samples), min_indices] -= capped_min
    values[np.arange(n_samples), min_indices] += argmin_vals
    
    # KG(i) = f_best - mean_s(values[s, i])
    expected_min = np.mean(values, axis=0)  # shape (n_candidates,)
    scores = f_best - expected_min
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
