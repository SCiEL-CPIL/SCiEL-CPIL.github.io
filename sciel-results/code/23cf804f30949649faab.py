"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function based on Expected Improvement (EI) 
    normalized by cost, with a safety floor on cost to avoid division by 
    near-zero values which can cause numerical instability or overly 
    aggressive exploitation of the exact origin if cost is very small.
    
    The strategy uses EI to balance exploration (via sigma) and exploitation 
    (via f_best - mu). Dividing by cost accounts for the varying evaluation 
    cost in the domain.
    """
    # Ensure sigma is strictly positive to avoid division by zero in z
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate the z-score for expected improvement
    z = (f_best - mu) / sigma_safe
    
    # Standard EI calculation
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # We want to maximize EI per unit cost.
    # However, simple EI/cost can be unstable if cost is very small.
    # The cost is 1 + mean(x_scaled). min cost is 1.0.
    # So cost is always >= 1.0. Division is safe.
    
    # To improve upon the baseline (which is EI/cost), we can adjust the 
    # exploration-exploitation balance. 
    # The baseline score was 1.0. 
    # The inspiration suggests that a "calibrated exploitation term" plus 
    # "relative uncertainty" helped (score 1.546 in another context).
    # 
    # Let's try a hybrid: 
    # Score = EI / cost
    # But to be materially different and potentially better, let's look at 
    # the "Probability of Improvement" (PI) or a weighted combination.
    # 
    # Actually, the directive says: "Model the acquisition score as a pure 
    # 'probability of improvement' (PI) weighted by inverse cost... strictly 
    # prioritize the highest-confidence exploitation".
    # 
    # Let's follow the directive's specific suggestion for this step:
    # Score = norm.cdf((f_best - mu) / sigma) / cost
    # This is PI / cost.
    #
    # Why this might be different:
    # EI includes the *magnitude* of the expected improvement. 
    # PI only includes the *probability* that an improvement exists.
    # In smooth functions like Griewank, once you are close to the minimum, 
    # the expected improvement magnitude becomes small, but the probability 
    # of finding a *slightly* better point might still be useful for 
    # fine-tuning, whereas EI might drop off too fast.
    # However, PI ignores the potential gain. 
    #
    # Let's try a blend that emphasizes exploitation more strongly than 
    # standard EI, but retains some exploration signal.
    # 
    # Alternative: Use a "Weighted PI" where we scale by the potential 
    # improvement but dampen the exploration term.
    #
    # Let's stick to the directive's specific "Pure PI weighted by inverse cost" 
    # as a distinct strategy to test.
    
    pi = norm.cdf(z)
    
    # Avoid division by zero in cost (though cost >= 1)
    cost_safe = np.maximum(cost, 1e-9)
    
    scores = pi / cost_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
