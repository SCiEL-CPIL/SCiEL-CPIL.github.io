import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    if time.time() - _start_time > 300:
        return np.zeros_like(mu)
    
    sigma = np.maximum(sigma, 1e-6)
    cost = np.maximum(cost, 1e-6)
    
    # Thompson Sampling: draw multiple samples from the posterior
    # For each candidate, the posterior is Normal(mu_i, sigma_i^2)
    # We want to find the candidate that minimizes f(x)/cost(x)
    # In Thompson sampling, we draw a sample from the posterior for each candidate
    # and pick the one with the lowest sample/cost
    
    # Number of Monte Carlo samples
    n_samples = 50
    
    # Draw samples from N(mu, sigma^2) for all candidates at once
    # shape: (n_samples, 2000)
    rng = np.random.default_rng()
    samples = mu[None, :] + sigma[None, :] * rng.standard_normal((n_samples, len(mu)))
    
    # Cost-normalized: for each sample and candidate, compute sample / cost
    # shape: (n_samples, 2000)
    cost_norm = samples / cost[None, :]
    
    # For each candidate, find the minimum across samples
    # This gives us the "optimistic" estimate of what we might achieve
    # Actually, in Thompson sampling, we should pick the candidate with the lowest sample
    # for a single sample. But since we're returning scores for all candidates,
    # we can use the expected minimum or the minimum of the expected values.
    
    # Better approach: for each candidate, compute the expected value of the minimum
    # over the samples. But that's not quite right either.
    
    # Standard Thompson sampling: draw ONE sample from the posterior for each candidate,
    # then pick the candidate with the lowest sample/cost.
    # Since we need to return scores for all 2000 candidates, we can use multiple samples
    # and compute a score that reflects how often each candidate would be selected.
    
    # Alternative: compute the expected minimum of f(x)/cost(x) under the posterior.
    # For a single candidate i, E[min over samples] is not straightforward.
    
    # Let's use a different approach: for each candidate, compute the probability
    # that it has the minimum cost-normalized value across all candidates,
    # averaged over multiple Monte Carlo samples.
    
    # For each sample s, find the index of the candidate with the minimum cost_norm
    # Count how often each candidate is the minimum
    min_indices = np.argmin(cost_norm, axis=1)  # shape: (n_samples,)
    
    # Count selections
    selection_counts = np.zeros(len(mu))
    np.add.at(selection_counts, min_indices, 1)
    selection_prob = selection_counts / n_samples  # shape: (2000,)
    
    # Also compute the expected minimum cost-normalized value for each candidate
    # This is E[f(x_i)/cost(x_i)] = mu_i / cost_i (since the sample is unbiased)
    # But we want to balance exploration, so let's also consider the variance.
    
    # Actually, the selection probability already captures the Thompson sampling behavior.
    # Candidates with lower mu and/or higher sigma (more uncertain) will have a higher
    # probability of being selected.
    
    # To make the scores more informative, let's also incorporate the expected value
    # of the cost-normalized objective.
    
    expected_cost_norm = mu / cost  # shape: (2000,)
    
    # Normalize expected_cost_norm to [0, 1] range (lower is better, so we invert)
    ecn_min = np.min(expected_cost_norm)
    ecn_max = np.max(expected_cost_norm)
    ecn_range = ecn_max - ecn_min + 1e-9
    ecn_scaled = 1.0 - (expected_cost_norm - ecn_min) / ecn_range  # Higher is better
    
    # Blend selection probability and expected value
    # Selection probability is in [0, 1], ecn_scaled is in [0, 1]
    w_sel = 0.7
    w_exp = 0.3
    
    scores = w_sel * selection_prob + w_exp * ecn_scaled
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
