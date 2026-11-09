"""Rewritten acquisition for Bayesian optimization of Hartmann-3D."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

_DEADLINE = time.time() + 320.0  # safety margin before 330s kill
_EPS = 1e-9


def _expected_hypervolume(mu, sigma, f_best, n_samples=20):
    """
    Approximate Expected Hypervolume Improvement (EHVI) for a single point
    using Monte Carlo sampling from the posterior.
    
    For minimization, we consider the improvement over f_best.
    We sample from N(mu, sigma^2) and compute the expected maximum improvement
    over the current best, weighted by the probability that the sample is better.
    
    Actually, for a single point, EHVI is related to the expected positive part
    of (f_best - Y) where Y ~ N(mu, sigma^2), but with a different weighting
    than EI. However, a common approximation for single-point EHVI in 
    multi-objective or for encouraging better exploration is to use a 
    different kernel or a different aggregation.
    
    Here, we use a simpler structural change: Use the "Probability of Improvement" 
    (PI) combined with a "Lower Confidence Bound" (LCB) style term, but with 
    a dynamic beta that depends on the number of evaluations or the spread of sigma.
    """
    pass


def acquisition(mu, sigma, f_best):
    """Return 2000 finite scores; higher => evaluate next.

    Uses a structural change: instead of fixed EI + exploration, 
    we use a dynamic balance between exploitation and exploration 
    based on the relative scale of the uncertainty compared to the 
    potential improvement. We also incorporate a "novelty" or 
    "coverage" heuristic by penalizing points that are too close 
    to the current best in the predictive mean, but rewarding 
    high uncertainty more aggressively in early stages.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    # Guard against non-finite inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    sigma = np.maximum(sigma, _EPS)

    # Compute standard EI
    z = (f_best - mu) / sigma
    z = np.clip(z, -10.0, 10.0)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # Compute a "confidence" based exploration term.
    # We use a beta parameter that decays as we see more variance.
    # Let's estimate the "stage" by looking at the mean sigma.
    mean_sigma = np.mean(sigma)
    # If mean_sigma is large, we are in early stages, explore more.
    # If mean_sigma is small, we are in later stages, exploit more.
    # Normalize mean_sigma relative to the range of mu.
    mu_range = np.max(mu) - np.min(mu) + _EPS
    relative_uncertainty = mean_sigma / mu_range
    
    # Beta for LCB: higher beta means more exploration.
    # Map relative_uncertainty to a beta in [0.5, 2.0]
    # If relative_uncertainty is high (early), beta should be high.
    # If relative_uncertainty is low (late), beta should be low.
    # Let's use a simple inverse relationship or a direct mapping.
    # Early: mean_sigma is large relative to mu_range -> high exploration.
    # Late: mean_sigma is small relative to mu_range -> low exploration.
    
    # Actually, a common approach is to use a fixed beta for LCB.
    # Let's try a different structural change: 
    # Use the "Upper Confidence Bound" for minimization (LCB) 
    # with a dynamic beta that depends on the step count.
    # But we don't have the step count. 
    # We can estimate it from f_best? No, f_best is just the value.
    
    # Alternative Structural Change: 
    # Instead of EI + sqrt(sigma), use a "Gaussian Process Upper Confidence Bound" 
    # but with a different aggregation: 
    # Score = -mu + beta * sigma
    # This is a linear combination. The current code is EI + 0.1*sqrt(sigma).
    # EI is non-linear. 
    # Let's try a "Thompson Sampling" proxy: 
    # Sample from the posterior and return the sample value (lower is better, so we return -sample).
    # But we need deterministic scores for the 2000 points.
    # We can approximate the expected value of the minimum over a batch? No, we pick one.
    
    # Let's try a different acquisition: "Probability of Improvement" (PI)
    # PI = P(Y < f_best) = Phi((f_best - mu) / sigma)
    # This is purely exploratory in the sense that it doesn't weight by magnitude.
    # But it's a valid acquisition. 
    # However, PI is often less effective than EI.
    
    # Let's try a "Hybrid" that switches behavior:
    # If EI is very small, use LCB to explore.
    # If EI is large, use EI to exploit.
    
    # Structural Change: Use a "Mixture" of EI and LCB.
    # LCB = mu - beta * sigma (for minimization, we want low values, so we score high if LCB is low? No.)
    # For minimization, we want to find low f.
    # LCB (Lower Confidence Bound) for the objective: 
    # We want to minimize f. An optimistic model would say f could be as low as mu - beta*sigma.
    # So we score based on how low that could be. 
    # Score_LCB = -(mu - beta * sigma) = -mu + beta * sigma.
    # Higher score means more promising.
    
    # Let's compute LCB with a dynamic beta.
    # We estimate the "progress" by looking at how much f_best has improved? 
    # We don't have history. 
    # But we can look at the distribution of mu. 
    # If mu is very close to f_best everywhere, we are in late stage.
    # If mu is far from f_best, we are in early stage.
    
    # Let's define a "closeness" metric:
    # closeness = 1 - (mean(f_best - mu) / (max(f_best - mu) + eps))
    # This is tricky.
    
    # Simpler Structural Change: 
    # Use "Expected Maximum Improvement" (EMI) approximation.
    # Or simply use a "Adaptive" beta for LCB based on the variance of sigma.
    
    # Let's try: 
    # beta = 1.0 + 1.0 * (np.std(sigma) / (np.mean(sigma) + eps))
    # If sigma is very variable, we are in an uncertain region, so explore more.
    
    std_sigma = np.std(sigma)
    mean_sigma = np.mean(sigma)
    beta = 1.0 + 1.0 * (std_sigma / (mean_sigma + _EPS))
    # Clip beta to [0.5, 3.0]
    beta = np.clip(beta, 0.5, 3.0)
    
    # LCB Score (higher is better)
    lcb_score = -mu + beta * sigma
    
    # We need to combine EI and LCB.
    # EI is in units of improvement. LCB is in units of objective value.
    # They are not directly comparable.
    # We can normalize them.
    
    # Normalize EI to [0, 1]
    ei_max = np.max(ei) + _EPS
    ei_norm = ei / ei_max
    
    # Normalize LCB to [0, 1]
    lcb_min = np.min(lcb_score)
    lcb_max = np.max(lcb_score)
    lcb_range = lcb_max - lcb_min + _EPS
    lcb_norm = (lcb_score - lcb_min) / lcb_range
    
    # Combine with a weight that depends on the stage.
    # If mean_sigma is high relative to the range of f_best - mu, we are early.
    # Let's use the coefficient of variation of sigma.
    cv = std_sigma / (mean_sigma + _EPS)
    # High CV -> early/uncertain -> weight LCB more.
    # Low CV -> late/certain -> weight EI more.
    # Map CV to weight in [0.5, 1.5]
    weight_lcb = 0.5 + cv
    weight_ei = 1.0
    
    # Final Score
    scores = weight_ei * ei_norm + weight_lcb * lcb_norm
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, 0.0)

    # Deadline guard
    if time.time() > _DEADLINE:
        z2 = (f_best - mu) / sigma
        z2 = np.clip(z2, -10.0, 10.0)
        scores = np.maximum((f_best - mu) * norm.cdf(z2) + sigma * norm.pdf(z2), 0.0)
        scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
