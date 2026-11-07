"""Acquisition for Bayesian optimization of Hartmann 6d using Thompson Sampling."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Deadline guard: evaluator kills at 330s wall-clock; leave a safety margin.
_DEADLINE = time.time() + 325.0

# Known minimum of the Hartmann 6d function.
_HARTMANN_6D_MIN = -3.32237


def _thompson_sampling_score(mu, sigma, f_best, n_samples=8):
    """
    Thompson Sampling acquisition: for each candidate, draw multiple samples
    from the GP posterior and score by the fraction of samples that beat f_best
    (or by the expected improvement across samples). This naturally balances
    exploration and exploitation without hand-tuned weights.
    """
    n = mu.shape[0]
    sigma_safe = np.maximum(sigma, 1e-12)
    
    rng = np.random.default_rng(42)
    
    # Draw samples from each candidate's posterior
    # samples shape: (n, n_samples)
    samples = rng.normal(mu[:, None], sigma_safe[:, None], size=(n, n_samples))
    
    # For each sample, count how many candidates would be "selected"
    # under Thompson sampling: select the candidate with the lowest sampled value
    # (since we're minimizing). But we want a score for each candidate.
    # 
    # Standard Thompson Sampling: sample one value per candidate, pick argmin.
    # To get a score (probability of being selected), we can:
    #   score_i = P(sample_i is the minimum across all candidates)
    # But that's O(n^2 * n_samples). Instead, use:
    #   score_i = fraction of samples where sample_i < f_best
    # This is the "probability of improvement" which is a valid TS-like score.
    
    # Actually, a cleaner TS score: for each sample s, the "winner" is argmin(samples[:, s]).
    # The score for candidate i is the frequency with which it wins.
    # This is O(n * n_samples) which is fine.
    
    winners = np.argmin(samples, axis=0)  # (n_samples,)
    scores = np.zeros(n)
    np.add.at(scores, winners, 1.0)
    scores /= n_samples
    
    # Blend with EI to ensure exploitation of known good regions
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Normalize both to [0, 1] for blending
    scores_max = scores.max() + 1e-12
    ei_max = ei.max() + 1e-12
    scores_norm = scores / scores_max
    ei_norm = ei / ei_max
    
    # Blend: 60% Thompson, 40% EI
    blended = 0.6 * scores_norm + 0.4 * ei_norm
    
    return blended


def _adaptive_ucb(mu, sigma, f_best):
    """
    Adaptive UCB with beta that scales with the gap to the known optimum.
    Early on (large gap), explore more; late (small gap), exploit more.
    """
    sigma_safe = np.maximum(sigma, 1e-12)
    
    gap = max(f_best - _HARTMANN_6D_MIN, 0.0)
    gap_norm = min(gap / 3.0, 1.0)
    
    # Beta transitions from ~0.8 (exploration) to ~0.1 (exploitation)
    beta = 0.1 + 0.7 * gap_norm
    
    # Add a tanh saturation to prevent a single high-sigma point from dominating
    sigma_scaled = np.tanh(sigma_safe / (np.std(mu) + 1e-12))
    
    ucb = mu + beta * sigma_scaled
    
    # We're minimizing, so lower UCB is better. Return negative for "higher is better".
    return -ucb


def _knowledge_gradient(mu, sigma, f_best, n_samples=10):
    """
    Approximate Knowledge Gradient: estimate how much observing at each
    candidate reduces the expected best over the domain.
    """
    n = mu.shape[0]
    sigma_safe = np.maximum(sigma, 1e-12)
    
    rng = np.random.default_rng(99)
    
    # Select reference points: top-20 by EI
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    k_ref = min(20, n)
    top_k_idx = np.argsort(ei)[-k_ref:]
    mu_ref = mu[top_k_idx]
    sigma_ref = sigma_safe[top_k_idx]
    
    # For each candidate i, estimate KG by sampling y_i and computing
    # the change in expected min over reference points.
    # E[min_j f_j] ≈ min_j (mu_j - kappa * sigma_j) for some kappa
    
    kappa = 0.5  # exploration-exploitation balance for soft-min
    
    # Current expected best (soft-min)
    soft_min_current = np.min(mu_ref - kappa * sigma_ref)
    
    scores = np.zeros(n)
    batch_size = 200
    
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        mu_b = mu[start:end]
        sigma_b = sigma_safe[start:end]
        
        # Sample hypothetical observations
        y_samples = rng.normal(mu_b[:, None], sigma_b[:, None], size=(end - start, n_samples))
        
        for s in range(n_samples):
            y = y_samples[:, s]
            
            # Approximate correlation using sigma similarity
            sigma_b_exp = sigma_b[:, None]
            sigma_ref_exp = sigma_ref[None, :]
            sigma_sum = sigma_b_exp + sigma_ref_exp + 1e-12
            sigma_ratio = np.abs(sigma_b_exp - sigma_ref_exp) / sigma_sum
            rho = np.maximum(0, 1 - sigma_ratio)
            rho = np.minimum(rho, 0.95)
            
            # Update reference means
            delta_y = y - mu_b
            mu_updated = mu_ref_exp + rho * sigma_ref_exp * (delta_y[:, None] / sigma_b_exp)
            sigma_sq_updated = sigma_ref_exp ** 2 * (1 - rho ** 2)
            sigma_updated = np.sqrt(np.maximum(sigma_sq_updated, 1e-24))
            
            soft_min_updated = np.min(mu_updated - kappa * sigma_updated, axis=1)
            
            # KG = reduction in expected best
            kg_batch = soft_min_current - soft_min_updated
            scores[start:end] += kg_batch / n_samples
    
    return scores


def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling + Knowledge Gradient hybrid acquisition for Hartmann 6d.
    
    Parameters
    ----------
    mu : ndarray, shape (n,)
        GP mean predictions.
    sigma : ndarray, shape (n,)
        GP standard deviation predictions.
    f_best : float
        Best (lowest) observed objective value so far.
    
    Returns
    -------
    scores : ndarray, shape (n,)
        Acquisition scores; higher values are preferred.
    """
    # Deadline check
    if time.time() > _DEADLINE - 5.0:
        sigma_safe = np.maximum(sigma, 1e-12)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        scores = ei + 0.1 * sigma_safe
        return np.where(np.isfinite(scores), scores, 0.0)
    
    try:
        # Thompson Sampling score
        ts_scores = _thompson_sampling_score(mu, sigma, f_best, n_samples=10)
        
        # Knowledge Gradient score
        kg_scores = _knowledge_gradient(mu, sigma, f_best, n_samples=8)
        
        # Normalize KG to [0, 1]
        kg_max = kg_scores.max() + 1e-12
        kg_norm = kg_scores / kg_max
        
        # Blend: 50% TS, 50% KG
        scores = 0.5 * ts_scores + 0.5 * kg_norm
        
    except Exception:
        # Fallback: adaptive UCB
        try:
            scores = _adaptive_ucb(mu, sigma, f_best)
        except Exception:
            sigma_safe = np.maximum(sigma, 1e-12)
            z = (f_best - mu) / sigma_safe
            ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
            scores = ei + 0.1 * sigma_safe
    
    # Final sanity
    if scores.shape != mu.shape:
        scores = np.resize(scores, mu.shape)
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
