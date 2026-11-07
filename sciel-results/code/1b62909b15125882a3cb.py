"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    # Deadline guard
    t0 = time.time()
    deadline = t0 + 280.0
    
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    if not (np.all(np.isfinite(mu)) and np.all(np.isfinite(sigma))):
        return np.arange(len(mu), dtype=np.float64)
    
    # Clip sigma
    sigma = np.maximum(sigma, 1e-8)
    
    # 1. Compute Expected Improvement (EI) as the base exploitation signal
    z = (f_best - mu) / sigma
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    
    if not np.all(np.isfinite(ei)):
        ei = np.nan_to_num(ei, nan=0.0, posinf=1e10, neginf=-1e10)
    
    # 2. Structural Change: Diversity Boost via Index-Based Pseudo-Geography
    # Since coordinates are not passed, we cannot compute true distances.
    # However, the candidates are drawn from a uniform distribution.
    # In the absence of coordinates, we assume the evaluator's random seed
    # or the order of candidates might have some locality, OR we simply
    # use the index as a proxy for "unexploredness" in a deterministic loop.
    # 
    # A more robust structural change for a "different design family" when
    # coordinates are missing is to use a **Thompson Sampling-like** approach
    # but implemented deterministically by adding a noise term that encourages
    # variance, OR to use **Knowledge Gradient** approximations.
    #
    # However, the most impactful structural change for high-dim BO with
    # limited evals is often **Batched EI** or **Greedy with Diversity**.
    # Without coordinates, we can't do true diversity.
    #
    # Let's try a different family: **Upper Confidence Bound (UCB)** with
    # a stronger exploration weight, combined with a **Penalty for Low Uncertainty**.
    # Standard EI is often too greedy in high dims. UCB with beta > 1 encourages
    # exploration of regions with high sigma even if mu is not immediately promising.
    #
    # Let's implement a hybrid: 
    # Score = EI + lambda * sigma
    # This is equivalent to LCB with a specific beta, but let's tune the balance.
    # The inspiration 1 mentioned LCB w. The knowledge finding says "0.1*sigma"
    # improved score. Let's push further into exploration.
    #
    # Strategy: Use a **Weighted UCB** where the weight on sigma is adaptive
    # based on the ratio of f_best to the typical mu range, or simply a fixed
    # higher weight to force exploration in the first few steps.
    #
    # Since we don't know the step count, we use a fixed, aggressive exploration
    # term. In Hartmann 6D, the landscape is multimodal. Pure EI gets stuck.
    # UCB (mu - beta*sigma) is the standard alternative.
    # Score = mu - beta * sigma is for minimization LCB.
    # Higher score means lower predicted value.
    #
    # Let's combine EI and LCB.
    # EI is good for exploitation. LCB is good for exploration.
    # Let's try: Score = 0.5 * EI_normalized + 0.5 * LCB_normalized?
    # No, units are different.
    #
    # Let's stick to the finding: "Adding a 0.1*sigma exploration term to EI".
    # The finding says it helped. Let's try a stronger exploration term, say 0.5*sigma,
    # effectively shifting the balance towards UCB-like behavior which is often
    # superior in high-dimensional multimodal problems with few samples.
    
    # Exploration weight
    beta = 0.5
    
    # LCB for minimization: mu - beta * sigma
    # We want to maximize the score.
    # If we use LCB, we pick points with low mu and high sigma.
    # If we use EI, we pick points with low mu (and some sigma).
    #
    # Let's compute a score that blends them.
    # Actually, EI is already a blend. 
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # For large z (mu << f_best), EI ~ f_best - mu.
    # For small z (mu ~ f_best), EI ~ sigma * phi(z).
    #
    # The problem with EI is that if f_best is not very good yet (early steps),
    # EI might be small everywhere.
    #
    # Let's try **Probability of Improvement (PI)** combined with **PI** is too greedy.
    #
    # Let's try **Thompson Sampling** approximation.
    # Sample from N(mu, sigma^2) and pick the minimum?
    # We can't sample 2000 times for each of 2000 candidates efficiently in a loop
    # without vectorization, but we can use the Gumbel trick or just add noise.
    #
    # Actually, a very effective structural change for "no coordinates" is to
    # assume the candidate list is random and use **Random Forest-like** diversity?
    # No, we only have mu/sigma.
    #
    # Let's go with **UCB** (Lower Confidence Bound for minimization)
    # Score = mu - beta * sigma
    # This is a different design family from EI (which is an expectation).
    # UCB is a bound. It encourages exploration more strongly when sigma is high.
    
    ucb = mu - beta * sigma
    
    # Check deadline
    if time.time() > deadline:
        return ucb
    
    # Ensure finite
    if not np.all(np.isfinite(ucb)):
        ucb = np.nan_to_num(ucb, nan=1e10, posinf=1e10, neginf=-1e10)
        
    return ucb
# EVOLVE-BLOCK-END
