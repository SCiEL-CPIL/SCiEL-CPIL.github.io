"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization (minimization).
    Uses a hybrid strategy combining Expected Improvement (EI) with
    Lower Confidence Bound (LCB) to balance exploitation and exploration.
    """
    start_time = time.time()
    deadline = 330.0
    safety_margin = 5.0

    n = len(mu)
    
    # Handle sigma ~ 0 cases to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Compute EI (Expected Improvement)
    # For minimization: EI = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Compute LCB (Lower Confidence Bound) score for minimization
    # We want to maximize the score, so we use -LCB = -(mu - kappa*sigma) = -mu + kappa*sigma
    # A larger kappa encourages exploration.
    # Typical values for kappa in LCB are around 1-3.
    # Let's try a moderate value.
    kappa = 1.5
    lcb_score = -mu + kappa * sigma_safe
    
    # Normalize EI and LCB scores to be comparable?
    # EI is in objective units. LCB score is also in objective units (roughly).
    # However, their scales might differ.
    # A common approach is to just sum them or take a weighted average.
    # Since EI can be 0 where LCB is high (exploration), and vice versa,
    # summing them creates a balance.
    
    # Let's try a weighted sum.
    # If EI dominates, it's exploitative. If LCB dominates, it's explorative.
    # For Weierstrass, which has multi-scale features, a balance is good.
    
    # Let's try: score = ei + 0.5 * lcb_score
    # Or maybe just EI is enough? The prompt says EI got 1.0, TS got 3.47.
    # Wait, the "inspiration" says EI got 1.0. The current TS got 2.96 (in the parent, though the text says 3.47 for a different variant).
    # The current program IS the TS one that scored 2.9598.
    # The inspiration says "EI: 1.0".
    # So TS (2.96) > EI (1.0).
    
    # Why is TS better?
    # TS samples from the posterior. If the posterior is multimodal (due to Weierstrass),
    # TS can jump between modes. EI is local.
    
    # Can we get the best of both?
    # Maybe use EI but with a larger exploration term?
    # Or use a "Lookahead" EI? No, too expensive.
    
    # Let's try to improve the TS.
    # The current TS uses a fixed seed derived from sum(mu).
    # This makes the random noise pattern deterministic relative to the input.
    # This is good for reproducibility but might be bad if the pattern is unlucky.
    
    # What if we use a different acquisition that is also stochastic but more robust?
    # Or what if we use EI but with a "diversity" penalty?
    # The directive suggests: "batch-aware, diversity-penalized acquisition that computes the maximum pairwise correlation among the top-K candidates based on the Matern-5/2 kernel and subtracts a penalty proportional to this correlation".
    
    # Let's try to implement a simple version of this.
    # 1. Compute EI scores.
    # 2. Select top K candidates (e.g., K=10).
    # 3. Compute pairwise correlations between these K candidates using the Matern-5/2 kernel.
    #    But we don't have the coordinates!
    #    "No coordinates are passed."
    #    So we cannot compute the kernel directly.
    #    However, we have mu and sigma.
    #    Can we estimate correlation from mu and sigma?
    #    Not directly.
    #    
    #    Wait, the directive says "based on the Matern-5/2 kernel".
    #    If we don't have coordinates, how can we compute the kernel?
    #    Maybe the "correlation" refers to something else?
    #    Or maybe we can approximate the diversity using the sigma values?
    #    Points with high sigma are likely far from observed data (exploratory).
    #    Points with low sigma are likely near observed data (exploitative).
    #    
    #    Alternative interpretation:
    #    Maybe we can use the indices? No, indices are arbitrary.
    #    
    #    Let's re-read the directive carefully.
    #    "computes the maximum pairwise correlation among the top-K candidates based on the Matern-5/2 kernel"
    #    This implies we need coordinates. But the contract says "No coordinates are passed."
    #    This is a contradiction unless the "correlation" is computed from the GP posterior parameters in some indirect way, or the directive is a generic suggestion that doesn't apply perfectly here.
    #    
    #    However, we can approximate "redundancy" by looking at the similarity of the posterior predictions.
    #    If two points have very similar mu and sigma, they might be redundant?
    #    Not necessarily.
    #    
    #    Let's stick to a robust analytic method.
    #    Since TS (2.96) > EI (1.0), let's try to refine TS.
    #    
    #    Idea: Use EI for the base score, but add a "noise" term that encourages exploration?
    #    score = ei + beta * sigma
    #    This is similar to UCB but with EI as the exploitation term.
    #    
    #    Let's try: score = ei + 0.5 * sigma
    #    This should be better than pure EI (1.0) and maybe competitive with TS (2.96).
    
    # Let's try a blend of EI and UCB (on -f).
    # UCB_score = -mu + kappa * sigma
    # EI_score = ei
    # 
    # Let's try: score = ei + 0.2 * (-mu + 2.0 * sigma)
    # This adds a small amount of exploration.
    
    # Let's try multiple values of kappa and weights and pick the best?
    # We can't test multiple values.
    
    # Let's try a different angle.
    # The Weierstrass function is multi-modal.
    # EI is good for local refinement.
    # TS is good for global jumps.
    # 
    # What if we use a "Probabilistic" EI?
    # Or "Expected Improvement with Diversity"?
    # 
    # Let's try the "EI + sigma" blend.
    # score = ei + alpha * sigma
    # If alpha is small, it's mostly EI.
    # If alpha is large, it's mostly exploration.
    
    # Let's try alpha = 0.5.
    # score = ei + 0.5 * sigma
    
    # Let's also try a pure UCB with a tuned kappa.
    # UCB with kappa=2.0: score = -mu + 2.0 * sigma
    # UCB with kappa=1.0: score = -mu + 1.0 * sigma
    
    # Given TS (2.96) is the best so far, and TS is essentially sampling from N(mu, sigma^2),
    # the expected value of the TS score is -mu.
    # But the variance is sigma^2.
    # The argmax of a Gaussian process sample is biased towards high sigma regions.
    # This is a known property.
    
    # To make TS more robust, we can use "Thompson Sampling with Diversity".
    # But without coordinates, we can't do it properly.
    
    # Let's try a simple improvement:
    # Use EI, but add a small random perturbation to encourage exploration?
    # No, that's just TS.
    
    # Let's try the blend: score = ei + 0.5 * sigma
    # This is deterministic and fast.
    
    # Let's compare:
    # EI: 1.0
    # TS: 2.96
    # 
    # Maybe the TS seed is the key?
    # The current TS uses seed = int(abs(sum_mu) * 12345) % (2**32).
    # This seed changes as mu changes.
    # 
    # Let's try a different seed strategy.
    # Or just use a fixed seed?
    # If we use a fixed seed, the noise pattern is always the same.
    # This might be bad if the noise pattern is unlucky for the specific problem structure.
    # 
    # Let's try using the global random state?
    # No, risky.
    # 
    # Let's try a deterministic acquisition that mimics TS.
    # score = -mu + z * sigma, where z is a fixed value?
    # If z=0, it's LCB (mu).
    # If z is large, it's UCB.
    # 
    # Let's try z=1.0.
    # score = -mu + 1.0 * sigma
    # This is LCB with kappa=1.
    
    # Let's try a combination of EI and LCB.
    # score = ei + 0.5 * (-mu + 1.0 * sigma)
    
    # I will try this blend.
    
    score = ei + 0.5 * (-mu + 1.0 * sigma_safe)
    
    # Ensure finite
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Check time
    if time.time() > deadline - safety_margin:
        return score
    
    return score

# EVOLVE-BLOCK-END
