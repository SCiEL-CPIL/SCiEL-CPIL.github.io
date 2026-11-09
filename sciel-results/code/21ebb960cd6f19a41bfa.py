"""Cost-aware acquisition for rastrigin-2d BO; Thompson Sampling surrogate."""
# EVOLVE-BLOCK-START
import time
import numpy as np

# Fixed wall-clock safety margin: the evaluator kills at 330 s; return early if we
# approach that. We do NOT use this to add search; it is a hard guard only.
_DEADLINE_SECONDS = 300.0
_START_TIME = time.time()


def _safe_return(scores, fallback):
    """Ensure scores are finite and shape (2000,)."""
    if scores is None:
        return fallback
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (2000,):
        return fallback
    if not np.all(np.isfinite(scores)):
        scores = np.where(np.isfinite(scores), scores, -1e18)
    return scores


def acquisition(mu, sigma, f_best, cost):
    """
    Thompson Sampling acquisition: draw posterior samples from N(mu, sigma^2),
    score each candidate by the expected cost-normalized objective across samples.
    
    Parameters
    ----------
    mu : ndarray, shape (2000,)
        GP mean predictions in objective units.
    sigma : ndarray, shape (2000,)
        GP standard deviation in objective units.
    f_best : float
        Lowest observed value so far.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate: 1 + mean(x_scaled).
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        Higher scores are preferred for the next evaluation.
    """
    # Hard deadline guard
    if time.time() - _START_TIME > _DEADLINE_SECONDS:
        fallback = -mu / (np.maximum(cost, 1e-9) + 1e-9)
        return _safe_return(fallback, np.zeros(2000, dtype=np.float64))

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)

    # Ensure cost is positive (should be >= 1 by construction)
    cost = np.maximum(cost, 1e-9)

    # Thompson Sampling: draw K samples from the posterior
    # Each sample is a vector of K values per candidate
    # We use a moderate number of samples for efficiency
    n_samples = 64

    # Generate standard normal random numbers: shape (n_samples, 2000)
    rng = np.random.default_rng()
    z = rng.standard_normal((n_samples, 2000))

    # Posterior samples: shape (n_samples, 2000)
    # sample[i, j] = mu[j] + sigma[j] * z[i, j]
    samples = mu[None, :] + sigma[None, :] * z

    # For minimization, we want low objective values.
    # Cost-normalized objective: f / cost
    # We want to minimize f / cost, so we score by -f/cost (higher is better)
    # Expected cost-normalized objective across samples:
    # E[-f/cost] = -E[f]/cost = -mu/cost  (since E[sample] = mu)
    # But Thompson Sampling uses the sample itself, not just the mean.
    # We score each candidate by the expected value of -f/cost over the posterior samples.
    # E[-f/cost | x] = -mu/cost  (cost is deterministic given x)
    # However, the variance matters: candidates with high uncertainty have a chance
    # of being much better than the mean suggests.
    # 
    # A more principled Thompson Sampling approach: for each sample, find which candidate
    # would be selected, then score candidates by how often they are selected (weighted).
    # But that's expensive. Instead, we can use:
    # score = E[-f/cost] + lambda * E[max(0, f_best - f)/cost]  (combines mean and improvement)
    # 
    # Actually, the directive says: "score candidates by the expected minimum cost-normalized
    # objective value across samples to inherently balance exploration and cost"
    # 
    # Interpretation: for each posterior sample, compute the cost-normalized objective for
    # all candidates, then take the minimum (best) across candidates? No, that doesn't make sense.
    # 
    # Re-reading: "score candidates by the expected minimum cost-normalized objective value
    # across samples" - I think this means: for each candidate, compute the expected value
    # of the cost-normalized objective under the posterior, and use that as the score.
    # The "minimum" might refer to the fact that we're minimizing, so lower cost-normalized
    # objective is better, and we take the expected minimum in the sense of the expected
    # value of the objective function.
    #
    # Let me use a simpler, effective approach:
    # score = -mu/cost + kappa * sigma/cost
    # This is a UCB-like score with cost normalization. The kappa term provides exploration.
    #
    # But the directive specifically asks for Thompson Sampling. Let me implement it properly:
    # For each sample, compute the cost-normalized objective for all candidates.
    # Then, for each candidate, compute the expected value of -f/cost over all samples.
    # This is just -mu/cost, which doesn't use the samples.
    #
    # A better Thompson Sampling approach: 
    # 1. For each sample, find the candidate with the minimum cost-normalized objective.
    # 2. Score each candidate by the fraction of samples in which it was selected,
    #    weighted by how good it was when selected.
    #
    # This is more expensive but captures the exploration-exploitation trade-off.
    # Let me implement this with a moderate number of samples.
    
    # Compute cost-normalized objective for each sample and candidate
    # Shape: (n_samples, 2000)
    cost_norm_obj = samples / cost[None, :]
    
    # For each sample, find the candidate with the minimum cost-normalized objective
    # (since we're minimizing)
    best_idx = np.argmin(cost_norm_obj, axis=1)  # shape: (n_samples,)
    best_values = cost_norm_obj[np.arange(n_samples), best_idx]  # shape: (n_samples,)
    
    # Score each candidate by the expected value of the cost-normalized objective
    # when it is the best, weighted by how often it is the best.
    # Initialize scores
    scores = np.zeros(2000, dtype=np.float64)
    counts = np.zeros(2000, dtype=np.float64)
    
    for i in range(n_samples):
        idx = best_idx[i]
        counts[idx] += 1.0
        # When this candidate is selected, its cost-normalized objective is best_values[i]
        # We want higher scores for lower cost-normalized objective, so use negative
        scores[idx] += -best_values[i]
    
    # Average over the samples where each candidate was selected
    # For candidates never selected, use a fallback score based on -mu/cost
    mask = counts > 0
    scores[mask] /= counts[mask]
    
    # For candidates never selected in any sample, use -mu/cost as a reasonable fallback
    # This ensures they still have a chance to be considered
    scores[~mask] = -mu[~mask] / cost[~mask]
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, -1e18)
    
    return _safe_return(scores, np.zeros(2000, dtype=np.float64))
# EVOLVE-BLOCK-END
