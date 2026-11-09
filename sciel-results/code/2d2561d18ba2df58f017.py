"""Acquisition function for Bayesian optimization of branin 2d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Branin 2D.
    
    Strategy: Combine Expected Improvement (EI) for exploitation with UCB for exploration.
    The current program (score 27.7) used a normalized mix of UCB and POI, which likely
    failed due to the non-linearity of normalization distorting the acquisition landscape
    or the specific weighting. Inspiration 1 (pure UCB, beta=1.0) scored 1.42, indicating
    that a simple, strong exploration/exploitation signal works well.
    
    We will use a hybrid: 
    1. Compute Standard EI.
    2. Compute UCB (beta=1.0).
    3. Combine them. Since EI is in objective units and UCB is in objective units,
       they are directly comparable without min-max normalization which can be brittle.
       However, EI tends to be smaller in magnitude than the full range of UCB.
       
       A robust approach for multi-modal functions like Branin is to use a weighted sum
       where the weights are fixed, or to use a "soft" maximum.
       
       Let's look at the scores:
       - Pure UCB (beta=1.0): 1.42
       - Pure EI (with inflated sigma): 1.0
       - Current (Norm UCB + Norm POI): 27.7 (Very bad, likely due to normalization issues or bad balance)
       
       The "seed" program mentioned in the contract likely refers to a baseline.
       The goal is to beat 1.0 (the seed/island best mentioned in inspiration 2's context, 
       though inspiration 2 itself is the EI one).
       
       Actually, looking at the data:
       - Inspiration 1 (Pure UCB beta=1.0) -> 1.42
       - Inspiration 2 (EI + sigma inflation) -> 1.0
       - Current (Norm UCB + POI) -> 27.7
       
       It seems that simpler, non-normalized approaches work better. Normalization
       across 2000 random points can be noisy and might not preserve the global structure
       of the acquisition function effectively if the candidate set is uniform random
       rather than optimized.
       
       Let's try a direct weighted sum of EI and UCB without min-max normalization.
       EI = (f_best - mu) * Phi(z) + sigma * phi(z)
       UCB = -mu + beta * sigma
       
       If we take score = EI + 0.5 * UCB, we get a strong exploitation signal from EI
       and a strong exploration signal from UCB.
       
       Alternatively, the "Knowledge Gradient" is often better, but harder to compute.
       
       Let's stick to the proven UCB (1.42) but try to enhance it with EI.
       Or, perhaps the issue with the current program is that it returns scores in [0,1]
       while the evaluator might just pick the argmax. The scale doesn't matter for argmax,
       but the *shape* does.
       
       Why did the current program fail? 
       ucb_norm = (ucb - min) / range. 
       poi = cdf(z).
       score = 0.5 * ucb_norm + 0.5 * poi.
       
       If the GP has very high uncertainty everywhere (early on), UCB is dominated by sigma.
       If the GP has low uncertainty (later on), UCB is dominated by -mu.
       
       Let's try a different combination: 
       Use UCB with a slightly higher beta to encourage exploration in the multi-modal
       Branin function, but add a small EI term to guide it towards known good regions.
       
       Actually, Inspiration 1 (Pure UCB beta=1.0) is the highest scoring inspiration (1.42).
       The current program is 27.7. The "seed" is 1.0.
       
       I will revert to a strategy close to Inspiration 1 but refine it.
       Maybe beta=1.0 is too low? Or maybe it's optimal?
       
       Let's try a hybrid: 
       score = -mu + beta * sigma + gamma * EI
       
       With beta=1.0 and gamma=0.5?
       
       Or, simply, use UCB with beta=1.5?
       
       Let's look at the "Lessons": None.
       
       Let's combine the structural idea of Inspiration 1 (UCB) with the refinement
       of adding EI (from Inspiration 2/3) but without the problematic normalization.
       
       Proposed Score:
       z = (f_best - mu) / sigma_safe
       ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
       ucb = -mu + 1.0 * sigma_safe
       
       score = ei + 0.5 * ucb
       
       This ensures that if EI is high (likely improvement), we pick it.
       If UCB is high (high uncertainty), we pick it.
       
       Let's verify numerical stability.
       """
    t_start = time.time()
    
    # Safety margin for sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standardized improvement
    z = (f_best - mu) / sigma_safe
    
    # Calculate Expected Improvement
    # Use vectorized operations
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Calculate UCB (beta = 1.0, as per Inspiration 1 which scored 1.42)
    beta = 1.0
    ucb = -mu + beta * sigma_safe
    
    # Combine them.
    # EI is generally smaller in magnitude than UCB in the early stages where sigma is large.
    # In later stages, sigma is small, UCB ~ -mu, EI ~ (f_best-mu) if mu < f_best.
    # Let's weight them equally to balance exploration and exploitation.
    # score = 0.5 * ei + 0.5 * ucb
    # Or just sum them?
    # Let's try summing them, as they are both "scores" where higher is better.
    score = ei + 0.5 * ucb
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, -np.inf)
    
    # Deadline guard
    # The computation is O(N) and very fast for N=2000.
    # No need for complex deadline handling, but we check just in case.
    t_end = time.time()
    if t_end - t_start > 290:
        # Return the best available (which is just the computed score)
        pass
        
    return score
# EVOLVE-BLOCK-END
