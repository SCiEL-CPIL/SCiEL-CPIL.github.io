"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Beale 2D.
    
    Strategy: Batched Greedy EIGR (Expected Improvement Greedy Reduction) approximation.
    Instead of a simple pointwise EI or UCB, we approximate the Expected Information Gain 
    or use a greedy selection that considers the interaction between the current best 
    and the uncertainty. 
    
    However, true EIGR requires knowing the other points in the batch or the full posterior.
    Since we only get mu and sigma for 2000 candidates and must return scores for all,
    we can't do a full batched optimization inside this single call efficiently without 
    iterating.
    
    A better structural change: **Thompson Sampling with a "Look-Ahead" or "Batch" heuristic**.
    The previous best was UCB (beta=2). The inspiration suggests multi-sample TS or LCB.
    
    Let's try a **Greedy Batched EI** approximation.
    Actually, a very effective strategy for small N (2000 candidates) is to compute 
    **Expected Improvement (EI)** but with a **correction for the "best of the batch"** effect.
    When selecting 1 point from 2000, the expected improvement of the *batch* is higher 
    if the points are diverse. But we pick the max score.
    
    Let's look at the "Open Question": Does operator 'diff' change the chance...
    This is meta. We need a strategy jump.
    
    Current: UCB (beta=2).
    Inspiration 1: Multi-sample optimistic LCB (score 2.7).
    Inspiration 3: Thompson Sampling (score 1.7).
    
    Let's implement **Fast Thompson Sampling with multiple samples and a "min-max" selection**.
    For each of K samples, we draw x ~ N(mu, sigma). We find the index of the minimum value 
    in each sample. We then aggregate these indices. The score for a point is the frequency 
    with which it was the minimum across the K samples.
    
    This is a structural change from a deterministic score (like EI or UCB) to a 
    stochastic frequency-based score. It naturally handles the "batch" aspect by 
    implicitly voting for the most likely global minimum among the candidates.
    
    K should be small enough to be fast but large enough to be stable.
    2000 candidates. K=10 samples. 20,000 operations. Very fast.
    
    The score returned should be higher for points more likely to be the minimum.
    So, score[i] = count of times i was argmin among the K samples.
    
    To break ties or add nuance, we can add a small term based on the actual minimum value 
    found in the sample where it was the winner, or just use the count.
    Using just the count is a valid "score" where higher means "evaluate next".
    
    Let's refine: 
    1. Generate K posterior samples: X (K, 2000).
    2. For each sample k, find argmin X[k].
    3. Count occurrences for each index.
    4. Return the counts as scores.
    
    This is **Thompson Sampling with Batch Voting**.
    It is structurally different from pointwise EI/UCB because it considers the 
    relative ranking of all 2000 points in each sample, not just their individual 
    expected improvement. It is a "best-of-K" selection process.
    """
    start_time = time.time()
    deadline = start_time + 250.0  # Safety margin
    
    n_candidates = len(mu)
    
    # Number of Thompson samples
    # We want to be fast. 2000 * K operations.
    # K=20 is ~40k ops, negligible.
    # Let's use K=50 to be more robust.
    K = 50
    
    # Draw samples: shape (K, n_candidates)
    # X = mu + sigma * Z
    # Z is standard normal
    # We need to be careful with memory and speed.
    # Generating (50, 2000) randoms is fast.
    
    # Check time
    if time.time() > deadline:
        # Fallback to simple UCB if we are already too slow (unlikely)
        sigma_safe = np.maximum(sigma, 1e-9)
        return -mu + 2.0 * sigma_safe

    Z = np.random.randn(K, n_candidates)
    X = mu[None, :] + sigma[None, :] * Z
    
    # Find argmin for each sample
    # argmin returns index of minimum in each row
    mins = np.argmin(X, axis=1)
    
    # Count occurrences
    scores = np.zeros(n_candidates, dtype=np.float64)
    np.add.at(scores, mins, 1.0)
    
    # The scores are counts. Higher count = more likely to be the min.
    # This is a valid acquisition score.
    
    # Optional: Add a small exploitation term to break ties or favor lower mu 
    # among points with equal counts? 
    # If we just return counts, points with count=0 have score 0.
    # Points with count>0 have score >0.
    # The evaluator picks argmax. If multiple have max count, it picks the first one.
    # This might be suboptimal if the first one has a higher mu than another with same count.
    # Let's refine the score to be: Count * 1000 - mu (scaled)
    # But mu can be large. Let's normalize mu? No, we don't know the scale.
    # Actually, if two points are both the min in some samples, the one that was 
    # "more" min (lower value) in those samples is better.
    
    # Let's compute a weighted score:
    # For each sample k, let i_k be the argmin.
    # We can add the negative of the value X[k, i_k] to the score of i_k?
    # No, that mixes scales.
    
    # Simpler: Just use the count. The "first max" behavior of argmax is a minor issue.
    # To be safer, we can add a tiny bit of -mu to the score to break ties.
    # scores = counts * 10.0 - mu * 1e-5
    # This ensures that among points with the same vote count, the one with lower mu is preferred.
    
    # Let's check the scale of mu. Beale function values can be large.
    # If mu is around 100, 1e-5 * 100 = 0.001.
    # Counts are integers 0..K.
    # So 10.0 * count dominates.
    
    final_scores = scores * 10.0 - mu * 1e-5
    
    # Ensure finite
    final_scores = np.nan_to_num(final_scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return final_scores
# EVOLVE-BLOCK-END
