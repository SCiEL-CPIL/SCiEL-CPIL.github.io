"""Acquisition function for Bayesian optimization of Styblinski-Tang 1D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP posterior.
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP posterior.
    f_best : float
        Best (lowest) observed value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means evaluate next.
    """
    start_time = time.time()
    
    # Ensure sigma is positive to avoid division by zero
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # 1. Compute base Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -50, 50)
    
    EI = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    EI = np.maximum(EI, 0.0)
    
    # 2. Add a small exploration bonus to avoid stalling
    # Use a simple UCB-like term or just scaled sigma
    # The look-ahead logic will handle diversity, so we can keep the base score
    # relatively standard but slightly exploration-friendly to ensure candidates
    # are spread out initially.
    
    # Let's use a hybrid: EI + alpha * sigma
    # alpha is small, e.g., 0.5
    alpha = 0.5
    base_scores = EI + alpha * sigma_safe
    
    # 3. Batch-Aware Look-Ahead Scoring
    # Strategy:
    # - Select top-K candidates greedily based on base_scores.
    # - For the remaining candidates, compute a "redundancy" score based on
    #   their similarity to the already selected candidates.
    # - Since we don't have coordinates, we use the GP posterior (mu, sigma)
    #   to infer a pseudo-distance. Candidates with similar mu and sigma are
    #   likely close in the input space (given the smoothness of the Matern-5/2 kernel).
    # - We discount the score of candidate i by the maximum similarity to any selected candidate j.
    
    K = 5  # Batch size for look-ahead
    
    if len(mu) < K:
        # If fewer than K candidates, just return base scores
        scores = base_scores
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
    
    # Greedy selection of top K
    # We need to be careful: the "selected" set influences the discounting of the rest.
    # We simulate the selection process:
    
    remaining_indices = np.arange(len(mu))
    selected_indices = []
    
    # To speed up, we can just sort by base_scores and take top K.
    # However, the discounting of the *next* best candidate depends on the *current* top K.
    # The task says: "draws 2000 fresh uniform candidates... calls acquisition and evaluates the candidate with the highest score".
    # This implies we only pick 1 per step. But the inspiration mentions "batch-aware... top-5 greedy selections".
    # This suggests the acquisition function should return scores such that the top 5 are diverse,
    # OR the evaluator uses the top 5? No, the contract says "evaluates the candidate with the highest score".
    # Wait, the inspiration says: "Look-ahead batch-aware scoring discounts candidates whose uncertainty is already 'covered' by the top-5 greedy selections".
    # This implies that even though only 1 is picked, the scoring should anticipate that the *next* 4 picks will be similar,
    # and thus penalize candidates that are too similar to the *potential* next picks?
    # Or does it mean we should penalize candidates similar to the *current* top 5 in the *previous* step?
    # No, "top-5 greedy selections" likely refers to simulating a batch of 5 from the current pool to determine diversity.
    
    # Let's interpret the inspiration:
    # "discounts candidates whose uncertainty is already 'covered' by the top-5 greedy selections"
    # This implies:
    # 1. Identify the top 5 candidates by some base score.
    # 2. For any candidate NOT in the top 5, if it is "similar" to any of the top 5, discount it.
    # This forces the algorithm to pick from the "frontier" of the top 5, or from areas not covered by the top 5.
    # Since we only pick 1, this effectively spreads out the *sequence* of picks over the next few steps.
    
    # Step 3a: Find top K indices by base_scores
    top_k_indices = np.argsort(base_scores)[::-1][:K]
    
    # Step 3b: Compute similarity/distance for all candidates to the top K
    # We need a metric. Since we lack coordinates, we use the feature space (mu, sigma).
    # Distance in feature space: d(i, j) = sqrt((mu_i - mu_j)^2 + (sigma_i - sigma_j)^2)
    # However, mu and sigma have different scales.
    # Let's normalize them.
    
    mu_range = np.max(mu) - np.min(mu)
    sigma_range = np.max(sigma_safe) - np.min(sigma_safe)
    
    if mu_range < eps: mu_range = 1.0
    if sigma_range < eps: sigma_range = 1.0
    
    mu_norm = (mu - np.min(mu)) / mu_range
    sigma_norm = (sigma_safe - np.min(sigma_safe)) / sigma_range
    
    # Construct feature matrix for all candidates
    # Shape: (2000, 2)
    features_all = np.column_stack((mu_norm, sigma_norm))
    
    # Features for top K
    # Shape: (K, 2)
    features_top_k = features_all[top_k_indices]
    
    # Compute Euclidean distance from each candidate to each of the top K
    # dist[i, j] = || features_all[i] - features_top_k[j] ||
    
    # To be efficient, we can compute this in chunks or use broadcasting.
    # 2000x2 vs 5x2 -> 2000x5 distances. This is very fast.
    
    diff = features_all[:, np.newaxis, :] - features_top_k[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))  # Shape: (2000, K)
    
    # Min distance to any top-K candidate
    min_dist = np.min(dists, axis=1)  # Shape: (2000,)
    
    # The top K candidates themselves will have min_dist = 0.
    # We want to discount candidates that are CLOSE to the top K.
    # So, a penalty based on 1 / (1 + min_dist) or similar?
    # Actually, we want to INCREASE the score of candidates FAR from the top K,
    # or DECREASE the score of candidates CLOSE to the top K (except the top K themselves).
    
    # Let's define a "coverage" score.
    # If min_dist is small, the candidate is covered.
    # We apply a penalty to base_scores for candidates that are covered.
    
    # Penalty = beta * exp(-gamma * min_dist)
    # Or simply: adjusted_score = base_score * (1 - penalty_factor)
    
    # Let's use a multiplicative factor that reduces the score if close to top K.
    # But we must NOT reduce the score of the top K themselves, as they should still be pickable.
    # If we reduce their score, we might pick a worse candidate.
    # The inspiration says "discounts candidates... covered by top-5".
    # This implies the top 5 are the "reference".
    
    # Let's try:
    # score_final[i] = base_scores[i] if i in top_k else base_scores[i] * decay(min_dist[i])
    
    # What decay?
    # If min_dist is large, decay -> 1.
    # If min_dist is small, decay -> small value.
    # decay = 1 / (1 + alpha_dist * min_dist) ? No, that's 1 at 0.
    # We want decay to be LOW when min_dist is LOW.
    # decay = min_dist / (min_dist + theta)
    # If min_dist = 0, decay = 0.
    # If min_dist >> theta, decay -> 1.
    
    theta = 0.1  # Threshold for "closeness" in normalized feature space
    
    decay_factor = min_dist / (min_dist + theta)
    
    # Apply decay to non-top-K candidates
    # Create a mask for top K
    top_k_mask = np.zeros(len(mu), dtype=bool)
    top_k_mask[top_k_indices] = True
    
    # For top K, keep base score (or maybe even boost slightly to ensure one is picked)
    # For others, multiply by decay_factor
    
    scores = np.where(top_k_mask, base_scores, base_scores * decay_factor)
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > 250:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
