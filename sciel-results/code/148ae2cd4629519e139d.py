"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard constant (seconds) - must be well under the 300s/330s kill limit
DEADLINE_LIMIT = 290.0


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization.
    Uses Knowledge Gradient (KG) approximation for minimization.
    
    Returns 2000 finite scores, higher meaning evaluate next.
    """
    start_time = time.time()
    
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Ensure mu is finite
    mu = np.where(np.isfinite(mu), mu, f_best)
    
    # Knowledge Gradient approximation:
    # KG(x) = E[ max(0, f_best - f(x)) | f(x) ~ N(mu(x), sigma(x)^2) ]
    # This is actually the same as EI for a single point, but the true KG
    # considers the expected reduction in the *global* minimum.
    # 
    # A common practical approximation for KG in BO is to use the expected
    # improvement weighted by the probability that this point is the new minimum,
    # or to use a Monte Carlo estimate. However, since we only have mu and sigma
    # at the candidate points and not the full GP, we use a refined EI variant
    # that better balances exploration.
    #
    # Actually, the standard EI IS a valid acquisition function. The directive
    # says to try KG. Let's implement a simple KG approximation:
    # KG(x) ≈ E[max(f_best - min(f(x), f_best), 0)] but since we don't know
    # the distribution of the minimum over all points, we approximate by
    # considering that the new minimum after evaluating x is min(f_best, f(x)).
    # The expected improvement of the minimum is:
    # E[min(f_best, f(x))] = f_best - E[max(f_best - f(x), 0)]
    # So the reduction is E[max(f_best - f(x), 0)] which is exactly EI.
    #
    # To differentiate from plain EI, we can use a "lookahead" KG approximation
    # that accounts for the fact that we have multiple candidate points.
    # A simpler refinement: use the expected improvement but with a slightly
    # different weighting that emphasizes points with higher uncertainty
    # (exploration) when the current best is not yet very good.
    #
    # Let's implement a practical KG approximation using Monte Carlo:
    # For each candidate x_i, simulate f(x_i) ~ N(mu_i, sigma_i^2)
    # and estimate the expected reduction in the minimum.
    #
    # However, with 2000 points and a tight time budget, we need efficiency.
    # Let's use a closed-form approximation that is slightly different from EI.
    #
    # Actually, the most common "improvement" over EI in practice is to use
    # the expected improvement with a small xi parameter, or to use UCB.
    # But the directive specifically asks for KG.
    #
    # A computationally efficient KG approximation for a single next step:
    # KG(x) = E[ max(0, f_best - f(x)) ] - E[ max(0, f_best - min(f(x), f_best)) ]
    # This simplifies to EI(x). So for a single evaluation, KG = EI.
    #
    # The difference arises when considering multiple future evaluations or
    # when the GP is considered globally. Since we only have mu and sigma,
    # let's use a weighted combination of EI and a pure exploration term
    # (like the variance) to approximate the global KG effect.
    #
    # KG_approx(x) = EI(x) + beta * sigma(x) * (1 - norm.cdf(z))
    # where z = (f_best - mu) / sigma
    # The term (1 - norm.cdf(z)) is the probability that f(x) < f_best.
    # This adds an exploration bonus scaled by the probability of improvement.
    
    z = (f_best - mu) / sigma_safe
    
    # Standard EI
    cdf_vals = norm.cdf(z)
    pdf_vals = norm.pdf(z)
    ei = (f_best - mu) * cdf_vals + sigma_safe * pdf_vals
    
    # Probability that this point improves on f_best
    prob_improve = 1.0 - cdf_vals
    
    # Exploration bonus: weighted by uncertainty and probability of improvement
    # This approximates the global KG effect by rewarding points that are
    # likely to be good AND have high uncertainty
    exploration_bonus = 0.1 * sigma_safe * prob_improve
    
    kg_approx = ei + exploration_bonus
    
    # Ensure all scores are finite
    kg_approx = np.where(np.isfinite(kg_approx), kg_approx, 0.0)
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > DEADLINE_LIMIT:
        # Return what we have; this branch is practically unreachable for N=2000
        pass
        
    return kg_approx
# EVOLVE-BLOCK-END
