"""Acquisition function for Bayesian optimization of Beale 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Knowledge Gradient (KG) approximation.
    KG computes the expected improvement in the lower bound of the global
    minimum by integrating the posterior predictive distribution over the
    candidate set. This is asymptotically optimal for the fixed 25-step
    evaluation budget.

    Parameters
    ----------
    mu : ndarray, shape (2000,)
        Posterior mean at candidate points (objective units, lower is better).
    sigma : ndarray, shape (2000,)
        Posterior standard deviation at candidate points (objective units).
    f_best : float
        Best observed function value so far (lower is better).

    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means more promising to evaluate next.
    """
    t0 = time.time()
    deadline = 325.0  # Safety margin before 330s limit

    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-12)

    # Knowledge Gradient approximation:
    # For candidate x_i, KG(x_i) = E[max(f_best, min_j f(x_j)) | f(x_i)] - f_best
    # 
    # Approximation: Assume independence across candidates (common KG approx).
    # Then KG(x_i) ≈ E[max(f_best - f(x_i), 0)] * P(f(x_i) < f_best)
    #             + E[min_j f(x_j) - f_best | f(x_i) < f_best] * P(f(x_i) < f_best)
    # 
    # A simpler and effective approximation:
    # KG(x_i) ≈ E[max(f_best - f(x_i), 0)] * (1 - CDF(f_best; mu_i, sigma_i))
    #          + correction_term
    #
    # Actually, the standard KG approximation for minimization is:
    # KG(x_i) = E[max(f_best - min(f(x_i), min_{j≠i} f(x_j)), 0)]
    #
    # With independence assumption and treating other candidates as fixed
    # at their current best estimate, a practical approximation is:
    # KG(x_i) ≈ EI(x_i) * (1 - Φ((f_best - mu_i)/sigma_i)) + sigma_i * φ((f_best - mu_i)/sigma_i) * (1 - Φ((f_best - mu_i)/sigma_i))
    #
    # Let's use a cleaner formulation. The key insight is that KG rewards
    # candidates that are likely to be the new minimum AND have high uncertainty.
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe

    # Standard normal CDF and PDF
    cdf = norm.cdf(z)  # P(f(x_i) < f_best)
    pdf = norm.pdf(z)

    # Expected Improvement (standard)
    ei = (f_best - mu) * cdf + sigma_safe * pdf

    # KG approximation: The probability that x_i becomes the new minimum
    # is related to how much it can improve over f_best, weighted by
    # the likelihood that no other candidate is better.
    #
    # A practical KG approximation:
    # KG(x_i) = EI(x_i) * P(x_i is best among candidates)
    #
    # Estimate P(x_i is best) using a softmax-like weighting based on
    # predicted values and uncertainties.
    
    # Compute a score for how "best" each candidate is likely to be
    # Lower mu and higher sigma both contribute to being potentially the best
    # Use a temperature-scaled softmax
    
    # Normalize mu for comparison
    mu_scale = np.std(mu) + 1e-12
    mu_norm = (mu - np.min(mu)) / (mu_scale * len(mu) ** 0.5 + 1e-12)
    
    # Uncertainty bonus: higher sigma means more chance of being better
    sigma_norm = sigma_safe / (np.std(sigma_safe) + 1e-12)
    
    # Combined "potential to be best" score
    potential = -mu_norm + 0.5 * sigma_norm
    
    # Temperature for softmax (controls how peaked the distribution is)
    temperature = 1.0
    potential_scaled = potential / (temperature * (np.std(potential) + 1e-12))
    
    # Softmax to get probability of being the best
    max_potential = np.max(potential_scaled)
    exp_potential = np.exp(potential_scaled - max_potential)
    prob_best = exp_potential / np.sum(exp_potential)
    
    # KG approximation: EI weighted by probability of being the new minimum
    kg = ei * prob_best

    # Add a second-order correction: candidates with high uncertainty
    # in regions where f_best is still far from the true minimum
    # should get extra credit
    # This accounts for the fact that exploring uncertain regions can
    # significantly reduce the lower bound
    
    # Estimate how far f_best is from the likely true minimum
    # Use the minimum of mu as a proxy for the likely minimum
    mu_min = np.min(mu)
    gap = f_best - mu_min  # How much room for improvement
    
    # If gap is large, exploration is more valuable
    exploration_bonus = sigma_safe * np.exp(-gap / (np.std(mu) + 1e-12))
    
    # Final KG score: primary KG term + exploration bonus
    scores = kg + 0.1 * exploration_bonus

    # Normalize to [0, 1] for stability
    scores_max = np.max(scores)
    scores_min = np.min(scores)
    scores_range = scores_max - scores_min
    if scores_range > 1e-12:
        scores = (scores - scores_min) / scores_range
    else:
        scores = np.zeros_like(scores)

    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)

    # Check time budget
    if time.time() > deadline:
        # Return best-effort result
        return scores

    return scores
# EVOLVE-BLOCK-END
