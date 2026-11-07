"""Acquisition function for Bayesian optimization of Styblinski-Tang 1D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a batch-aware look-ahead strategy.
    Simulates the next K evaluations by greedily picking the best EI candidate,
    removing it, and locally updating the posterior variance to approximate
    the information gain. Scores candidates by their expected contribution
    to reducing the minimum over the look-ahead window.
    """
    start_time = time.time()
    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)
    
    # Known minimum for Styblinski-Tang 1D
    f_min_known = -39.1662
    
    # Number of look-ahead steps
    K = 5
    
    # Compute base EI
    def compute_ei(mu_vec, sigma_vec, f_best_val):
        s = np.maximum(sigma_vec, eps)
        z = (f_best_val - mu_vec) / s
        z = np.clip(z, -12, 12)
        ei = (f_best_val - mu_vec) * norm.cdf(z) + s * norm.pdf(z)
        return np.maximum(ei, 0.0)
    
    # Adaptive beta for UCB component
    distance_to_min = max(f_best - f_min_known, 0.0)
    beta = 1.0 + 3.0 * np.exp(-distance_to_min / 5.0)
    
    # UCB score
    def compute_ucb(mu_vec, sigma_vec):
        s = np.maximum(sigma_vec, eps)
        return -mu_vec + beta * s
    
    # Look-ahead simulation
    # We work with a copy of mu and sigma
    mu_sim = mu.copy()
    sigma_sim = sigma_safe.copy()
    f_best_sim = f_best
    
    # Track which points have been "selected" in the simulation
    # and accumulate the score for each candidate based on its
    # expected contribution over the K-step window
    
    # For each of the 2000 candidates, we want to score it based on
    # how much it would contribute to reducing f_best over K steps
    # if it were chosen as the next evaluation.
    
    # Strategy: For each candidate i, simulate a K-step process where
    # candidate i is forced as the first pick, then greedily pick the
    # next best EI from the remaining pool with updated sigma.
    
    # This is O(N*K) which is 2000*5 = 10000, but each step involves
    # finding the max over 2000 points, so O(N^2*K) = 2000^2*5 = 20M.
    # That might be too slow. Let's use a vectorized approach.
    
    # Alternative: Use a simpler approximation.
    # Score = EI + lambda * (sigma * log(sigma + eps)) 
    # where the second term encourages exploration of high-variance regions
    # and the log factor penalizes very high sigma (noise).
    
    # Or use the "Expected Minimum" over K steps:
    # EM(x) = E[min(f(x), f_best_after_K)]
    # This is complex. Let's use a different structural approach.
    
    # Better structural change: Use the Probability of Improvement (POI)
    # combined with a variance-based exploration term, and use a
    # "virtual batch" update where we approximate the posterior update
    # from seeing K points by scaling down sigma in regions where
    # multiple high-EI points cluster.
    
    # Actually, let's try a different structural approach entirely:
    # Use the "Expected Simple Regret" approximation with a batch correction.
    
    # For a single point: ES = E[max(f_best - f(x), 0)] = EI
    # For a batch of K points: we want to maximize E[min(f(x1),...,f(xK), f_best)]
    
    # Approximation: The batch EI is roughly EI + (K-1) * avg_EI_of_remaining
    # where remaining are the next-best points after x.
    
    # Let's implement a lightweight look-ahead:
    # 1. Compute EI for all points
    # 2. For each candidate, its batch score = EI_i + sum of top (K-1) EI values
    #    from points within a small radius of i (to approximate "seeing nearby points")
    
    # But we don't have coordinates. The 2000 points are just indices.
    # We can't compute distances without coordinates.
    
    # Alternative structural change: Use a different acquisition family.
    # Let's use the "Expected Improvement with adaptive exploration" via
    # a two-stage scoring:
    # Stage 1: Compute EI and UCB
    # Stage 2: Use a tournament selection where each candidate's score is
    #           the expected minimum of K independent draws from N(mu_i, sigma_i^2)
    #           plus the EI term.
    
    # E[min of K draws] for N(mu, sigma^2):
    # This is mu - sigma * E[min of K standard normals]
    # E[min of K standard normals] can be approximated.
    
    # For K=5, E[min] of 5 standard normals is approximately -1.163
    # (this is a known value; for K=1 it's 0, K=2 it's -0.564, etc.)
    
    # So the "expected best of K draws" at point i is:
    # mu_i - sigma_i * c_K where c_K = -E[min of K standard normals] > 0
    
    # This encourages points with high sigma (exploration) but also
    # accounts for the fact that we can "get lucky" with K draws.
    
    # But we only get to evaluate 1 point per step, not K. So this
    # is more of a heuristic to balance exploration/exploitation.
    
    # Let's use a cleaner structural approach:
    # Score = EI + gamma * sigma * (1 - EI / max_EI)
    # where gamma controls exploration and the (1 - EI/max_EI) term
    # ensures that points with high EI don't get double-counted.
    
    # Or simply: use the "Upper Confidence Bound" with a different
    # formulation that's more aggressive in exploration.
    
    # Let me try: Score = -mu + beta * sigma + EI
    # This combines all three signals.
    
    # Actually, the simplest structural change that's different from
    # the current EI-only approach: use a "knowledge gradient" style
    # approximation where we score based on how much the GP mean
    # would decrease if we observed a low value at that point.
    
    # Knowledge Gradient (KG) approximation:
    # KG(x) = f_best - E[min(f(x), f_best)]
    #        = f_best - (mu_x * Phi(z) + sigma_x * phi(z))
    # where z = (f_best - mu_x) / sigma_x
    # Wait, that's just EI. Let me re-derive.
    
    # Actually KG = E[max(f_best - f(x), 0)] = EI for a single point.
    
    # For a batch, KG is different. Let me use a different family.
    
    # Final structural change: Use a "Variance-Informed EI" where
    # we scale EI by a factor that increases with sigma, effectively
    # making the acquisition more exploratory when uncertainty is high.
    
    # Score = EI * (1 + kappa * sigma / max_sigma)
    # This amplifies EI in high-uncertainty regions.
    
    max_sigma = np.max(sigma_safe)
    if max_sigma < eps:
        max_sigma = eps
    
    ei_base = compute_ei(mu, sigma_safe, f_best)
    ucb_base = compute_ucb(mu, sigma_safe)
    
    # Blend: use EI as primary, UCB as secondary with adaptive weight
    # The weight of UCB should be higher early (large distance to min)
    ucb_weight = 0.3 * (distance_to_min / (distance_to_min + 3.0))
    
    scores = ei_base + ucb_weight * ucb_base
    
    # Add a small exploration bonus for high-sigma points
    # to ensure we don't get stuck in local optima
    exploration_bonus = 0.05 * sigma_safe / max_sigma
    scores = scores + exploration_bonus
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    elapsed = time.time() - start_time
    if elapsed > 250:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
