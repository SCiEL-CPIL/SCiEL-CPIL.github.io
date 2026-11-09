<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a normalized weighted combination
    of Expected Improvement (EI) and Probability of Improvement (PI).
    
    The score is designed to balance exploitation (finding points better than f_best)
    and exploration (reducing uncertainty) while penalizing high-cost evaluations.
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Handle edge cases where sigma might be zero or negative (numerical stability)
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Probability of Improvement (PI)
    # PI = Phi(z)
    pi = norm.cdf(z)
    
    # Normalize EI and PI to have mean 1 (or similar scale) to make them comparable
    # Adding epsilon to avoid division by zero if all improvements are 0
    mean_ei = np.mean(ei) + 1e-9
    mean_pi = np.mean(pi) + 1e-9
    
    ei_norm = ei / mean_ei
    pi_norm = pi / mean_pi
    
    # Normalize Cost to have mean 1
    mean_cost = np.mean(cost) + 1e-9
    cost_norm = cost / mean_cost
    
    # Weighted combination of normalized EI and PI
    # We use equal weights initially, but the normalization makes them comparable.
    # The insight mentioned "adaptive exploitation/exploration weighting".
    # A simple adaptive weight could be based on the average uncertainty.
    # If average sigma is high relative to f_best, we are exploring.
    # However, without history, a fixed 0.5/0.5 blend of normalized metrics is a strong baseline.
    # Let's try a slight bias towards EI as it captures magnitude of improvement, 
    # while PI captures likelihood.
    
    w_ei = 0.6
    w_pi = 0.4
    
    combined_score = w_ei * ei_norm + w_pi * pi_norm
    
    # Final score: Combined improvement potential per unit of normalized cost
    scores = combined_score / cost_norm
    
    # Ensure scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def _fast_norm_cdf(x):
    """Vectorized normal CDF using the error function."""
    return 0.5 * (1.0 + np.erf(x / np.sqrt(2.0)))


def _fast_norm_pdf(x):
    """Vectorized normal PDF."""
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Cost-Adjusted Expected Improvement (EI/Cost).
    
    This structural change removes the adaptive UCB/PI blend and uses strict
    EI divided by cost, which aligns with measured improvements in tight budget regimes.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score for EI
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Use fast vectorized implementations
    cdf_z = _fast_norm_cdf(z)
    pdf_z = _fast_norm_pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score: EI / cost
    # Cost is given as 1 + mean(x_scaled), so higher cost = more expensive
    # We want to maximize EI per unit cost
    cost_safe = np.maximum(cost, 1e-9)
    scores = ei / cost_safe
    
    # Ensure scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
BET: {mechanism: "Switching from a normalized blend of EI and PI with cost normalization to strict Cost-Adjusted EI (EI/Cost) removes the exploration bias from PI and the double-normalization artifact, directly optimizing the efficiency metric that measured improvements favor in tight budgets.", q: 0.65, delta: [0.05, 0.25]}
