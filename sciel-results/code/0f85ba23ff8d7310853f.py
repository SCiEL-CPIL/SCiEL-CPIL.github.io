"""Cost-aware acquisition function for Bayesian optimization (shekel 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global state for deadline tracking
_START_TIME = None

def _init_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def _time_remaining():
    global _START_TIME
    if _START_TIME is None:
        return float('inf')
    # 330s hard limit, use 300s as safety margin for the evaluator's kill switch
    return 300.0 - (time.time() - _START_TIME)

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Stochastic Expected Improvement (SEI).
    Models cost as log-normal random variable for probabilistic integration.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best (lowest) observed function value.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    _init_time()
    
    # If we are close to the deadline, return a simple, fast score to avoid timeout
    if _time_remaining() < 5.0:
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        cost_safe = np.maximum(cost, 1e-9)
        scores = ei / cost_safe
        return np.where(np.isfinite(scores), scores, 0.0)

    # Stochastic Expected Improvement (SEI) formulation
    # Model cost as log-normal: C ~ LogNormal(log(c), sigma_c)
    # where sigma_c captures uncertainty in cost estimation
    
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute z-scores for EI
    z = (f_best - mu) / sigma_safe
    
    # Base Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Model cost uncertainty: assume ~15% relative uncertainty in cost
    # This gives a log-normal distribution with moderate spread
    cost_uncertainty = 0.15
    
    # For SEI, we want E[max(f_best - Y, 0) / C] where Y ~ N(mu, sigma^2) and C ~ LogNormal
    # Approximate via numerical quadrature over log-normal cost samples
    
    # Use Gauss-Hermite quadrature for efficiency
    # Transform to standard normal: C = exp(log(c) + sigma_c * Z) where Z ~ N(0,1)
    # E[1/C] = E[exp(-log(c) - sigma_c * Z)] = exp(-log(c) + 0.5*sigma_c^2)
    # But we need E[EI / C] which is more complex
    
    # Alternative: use importance sampling with a few quadrature points
    # For 2000 points, we can afford ~10 quadrature points
    
    n_quad = 10
    # Gauss-Hermite nodes and weights
    x_nodes, w_nodes = np.polynomial.hermite.hermgauss(n_quad)
    # Scale to standard normal: z = sqrt(2) * x
    z_quad = np.sqrt(2.0) * x_nodes
    w_quad = w_nodes / np.sqrt(np.pi)
    
    # Initialize SEI scores
    sei_scores = np.zeros_like(mu)
    
    for i in range(n_quad):
        # Sample cost at this quadrature point
        # C_i = exp(log(cost) + cost_uncertainty * z_quad[i])
        log_cost = np.log(cost_safe)
        c_sample = np.exp(log_cost + cost_uncertainty * z_quad[i])
        c_sample = np.maximum(c_sample, 1e-9)
        
        # Compute EI / C for this cost sample
        ei_over_c = ei / c_sample
        
        # Accumulate weighted sum
        sei_scores += w_quad[i] * ei_over_c
    
    # Add exploration term: encourage high uncertainty regions
    # Weighted by inverse expected cost
    expected_cost_inv = np.exp(-np.log(cost_safe) + 0.5 * cost_uncertainty**2)
    exploration = sigma_safe * expected_cost_inv
    
    # Combine SEI with exploration
    # Tune the balance between exploitation and exploration
    scores = sei_scores + 0.05 * exploration
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
