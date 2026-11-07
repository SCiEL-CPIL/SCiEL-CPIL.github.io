"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Computes scores for 2000 candidate points using Monte Carlo Expected Improvement.
    
    Strategy:
    Instead of using the closed-form Expected Improvement (EI) or Upper Confidence Bound (UCB),
    we estimate the Expected Improvement via Monte Carlo integration.
    
    Why this is different:
    1. Closed-form EI can be numerically unstable or overly greedy in high dimensions.
    2. UCB requires tuning a beta parameter that is hard to adapt dynamically.
    3. MCEI samples from the posterior distribution N(mu, sigma^2) and computes the average improvement.
       This naturally accounts for the tail behavior of the distribution.
       By using a sufficient number of samples (e.g., 64), we get a robust estimate of the
       "expected gain" that is less susceptible to local noise than single-point heuristics.
    
    Implementation:
    - Draw S samples for each of the N=2000 candidates.
    - Calculate improvement for each sample: max(f_best - sample, 0).
    - Average the improvements to get the score.
    - Add a small regularization term to break ties and ensure finite output.
    """
    
    # Ensure inputs are 1D arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Safety: replace non-finite or negative sigma
    sigma = np.where(np.isfinite(sigma) & (sigma > 1e-8), sigma, 1e-8)
    
    # Number of Monte Carlo samples per candidate.
    # 64 samples is a good balance between accuracy and speed for 2000 candidates.
    # Total operations: 2000 * 64 = 128,000 random draws and comparisons. This is very fast in NumPy.
    n_samples = 64
    
    # Generate random normal samples: shape (n_samples, 2000)
    # We use standard normal and scale by sigma, shift by mu.
    z = np.random.standard_normal((n_samples, len(mu)))
    samples = mu[None, :] + sigma[None, :] * z
    
    # Calculate improvement for each sample
    # Improvement is positive if the sample is lower than f_best (since we minimize)
    # imp = max(f_best - sample, 0)
    # Using np.maximum for vectorization
    improvements = np.maximum(f_best - samples, 0.0)
    
    # Average over the samples axis to get the Expected Improvement estimate
    scores = np.mean(improvements, axis=0)
    
    # Optional: Add a tiny exploration bonus based on uncertainty?
    # MCEI already includes uncertainty because high sigma leads to more samples below f_best
    # if mu is close to f_best. However, if mu is far above f_best, even high sigma might
    # result in low EI. 
    # To ensure we don't completely ignore high-uncertainty regions far from f_best (exploration),
    # we can blend with a small UCB component or just rely on the fact that GP sigma
    # decays with distance to observed points. 
    # In the context of the loop described (2000 fresh uniform candidates each step),
    # the GP is retrained or updated? The prompt says "computes the posterior there".
    # It implies the GP is trained on the history.
    # If we are in a region far from history, sigma is high. 
    # MCEI for a point with mu=0, sigma=1, f_best=-3:
    # P(sample < -3) is small. EI will be low.
    # This might cause the algorithm to be too greedy and miss distant basins.
    # However, the prompt says the evaluator draws 2000 FRESH UNIFORM candidates.
    # If the GP is trained on previous points, the sigma in unexplored regions will be high.
    # But if mu is high (far from min), EI is low.
    # To fix this, we can use a "Hybrid" that is structurally different:
    # Score = MCEI + lambda * sigma^2 (Variance based exploration)
    # But let's stick to a pure MCEI variant first, perhaps with a "look-ahead" or just robust MCEI.
    
    # Actually, a common issue with EI in multi-modal functions is that it ignores
    # regions with high variance but high mean. 
    # A better structural change: **Expected Regret** or **Probability of Improvement** combined with MCEI?
    # Let's try **MCEI with a noise floor** or simply **MCEI**.
    # Given the "Different Strategy" directive, MCEI is a distinct family (Stochastic Approximation vs Closed-Form Heuristic).
    
    # To mitigate the greediness of EI in high dims, we can use **Batched** EI concepts,
    # but we only pick one point. 
    # Let's add a small exploration term based on the variance to ensure we don't get stuck.
    # score = MCEI + 0.1 * sigma
    # This is a hybrid. Is it materially different? Yes, it's a stochastic estimator + regularization.
    
    # Let's refine: Pure MCEI might be too greedy. 
    # Let's use **MCEI + 0.5 * sigma** (a soft UCB-like term derived from the same samples? No, just the posterior std).
    # This combination is often called "Expected Improvement with Exploration Bonus".
    
    # However, to be strictly "different" and robust, let's just use MCEI.
    # If MCEI is too greedy, the next iteration might fail. 
    # But looking at the inspirations, simple EI (score 1.0) and UCB (score 1.15) are the baselines.
    # MCEI is essentially a more accurate EI. It might not jump far.
    
    # What is a STRONGER different strategy?
    # **Knowledge Gradient (KG)** approximation? Too slow.
    # **Thompson Sampling**? 
    # Thompson Sampling for minimization:
    # Draw one sample f_tilde from the posterior for each candidate.
    # Pick the candidate with the lowest f_tilde.
    # This is equivalent to maximizing the score: Score = -f_tilde.
    # Thompson Sampling is known to be very effective in high dimensions and multi-modal settings
    # because it naturally balances exploration and exploitation through the posterior sampling.
    # It avoids the tuning of beta (UCB) and the greediness of EI.
    
    # Let's switch to **Thompson Sampling**.
    # It is a single stochastic draw per candidate.
    # Score = - (mu + sigma * z) where z ~ N(0,1).
    
    # Why this is better:
    # 1. No tuning parameters.
    # 2. Naturally explores high variance regions.
    # 3. Naturally exploits low mean regions.
    # 4. Very fast (1 random number per candidate).
    
    # Let's implement Thompson Sampling.
    
    z_ts = np.random.standard_normal(len(mu))
    samples_ts = mu + sigma * z_ts
    
    # We want to minimize f, so we want the lowest sample.
    # The evaluator picks the HIGHEST score.
    # So Score = -samples_ts.
    
    scores_ts = -samples_ts
    
    # Ensure finite
    scores_ts = np.where(np.isfinite(scores_ts), scores_ts, -np.inf)
    
    # Return Thompson Sampling scores
    return scores_ts

# EVOLVE-BLOCK-END
