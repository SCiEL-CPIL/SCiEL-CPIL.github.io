"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function: a blend of Expected Improvement (EI) and Upper Confidence Bound (UCB).
    
    Parameters
    ----------
    mu : np.ndarray, shape (2000,)
        Posterior mean at candidate points.
    sigma : np.ndarray, shape (2000,)
        Posterior standard deviation at candidate points.
    f_best : float
        Best (lowest) observed function value so far.
    
    Returns
    -------
    np.ndarray, shape (2000,)
        Acquisition scores; higher values indicate more promising candidates.
    """
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    sigma = np.maximum(sigma, 1e-9)  # avoid division by zero
    
    # Expected Improvement
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    # For points where mu > f_best, EI is small/negative; we want to encourage exploration
    # but EI naturally handles this via the CDF/PDF terms.
    
    # UCB term: balance exploitation and exploration
    # We use a time-aware or fixed exploration parameter.
    # Since we don't have step info, we use a fixed gamma.
    # A common choice is gamma ~ 1.0 to 3.0. Let's use 2.0.
    gamma = 2.0
    ucb = mu + gamma * sigma  # Note: we are MINIMIZING, so we want LOW mu and HIGH sigma
    
    # Since we are minimizing, the UCB score should be NEGATIVE of the upper bound
    # or we can define it as -mu - gamma * sigma? No.
    # Standard UCB for minimization: score = -mu - kappa * sigma is wrong.
    # We want to maximize the acquisition score.
    # If we define the "regret" of a point as mu - f_best (if mu < f_best, this is negative, good).
    # Let's stick to EI which is well-defined for minimization.
    # EI is already a good acquisition function.
    # However, pure EI can be too greedy.
    # Let's combine EI with a bonus for high uncertainty (sigma).
    
    # Normalize EI and sigma to combine them?
    # Or simply add a term like -sigma? No, we want to explore where sigma is high.
    # But if mu is very high, we shouldn't explore.
    # EI already accounts for this: if mu >> f_best, z is very negative, cdf(z) ~ 0, pdf(z) ~ 0, so EI ~ 0.
    # If mu << f_best, z is very positive, cdf(z) ~ 1, pdf(z) ~ 0, so EI ~ f_best - mu (large positive).
    # So EI is a solid baseline.
    
    # To improve upon EI, we can use a "hypervolume" or "Pareto" front idea, but that's complex.
    # Another option: Expected Improvement with a noise parameter (EI-noise).
    # Or, use a UCB-like score but corrected for minimization.
    # Let's try a simple blend: score = EI - alpha * sigma? No, that penalizes exploration.
    # score = EI + beta * sigma? This encourages exploration even when EI is low.
    # But if mu is far from f_best, EI is 0, so score = beta * sigma. This might lead to exploring random areas.
    # This is effectively UCB.
    
    # Let's use a weighted sum of EI and a UCB-like term.
    # UCB for minimization: we want to select points with low mu and high sigma.
    # So, score_ucb = -mu + gamma * sigma.
    # But mu and sigma are in different scales.
    
    # Let's stick to EI but with a slightly modified z to be more exploratory.
    # Or, use "Expected Regret" which is similar.
    
    # Actually, a very effective acquisition function for BO is "Expected Improvement" itself.
    # The seed program uses EI. To beat it, we need a better acquisition function.
    # One popular alternative is "Probability of Improvement" (PI), but it's more greedy.
    # Another is "Expected Hypervolume Improvement" (EHVI), but that's for multi-objective.
    # For single-objective, EI is standard.
    
    # Let's try "Upper Confidence Bound" (UCB) for minimization:
    # score = -mu - kappa * sigma  (if we want to minimize -score)
    # But we return higher scores for better candidates.
    # So, for minimization, a good candidate has low mu.
    # So, score = -mu. But this ignores uncertainty.
    # UCB: score = -mu + kappa * sigma.
    
    # Let's blend EI and UCB.
    # First, normalize both to have zero mean and unit variance over the candidates.
    ei_mean = np.mean(ei)
    ei_std = np.std(ei) + 1e-9
    ei_norm = (ei - ei_mean) / ei_std
    
    ucb_score = -mu + gamma * sigma
    ucb_mean = np.mean(ucb_score)
    ucb_std = np.std(ucb_score) + 1e-9
    ucb_norm = (ucb_score - ucb_mean) / ucb_std
    
    # Weighted sum
    weight_ei = 0.5
    weight_ucb = 0.5
    score = weight_ei * ei_norm + weight_ucb * ucb_norm
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
