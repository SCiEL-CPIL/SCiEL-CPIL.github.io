import numpy as np
import time
from scipy.stats import norm

_START_TIME = time.time()
_TIME_LIMIT = 295.0  # Safety margin for 300s limit

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    if time.time() - _START_TIME > _TIME_LIMIT:
        # Fallback: Greedy exploitation
        cost_safe = np.maximum(cost, 1e-9)
        return -mu / cost_safe

    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Ensure sigma is non-negative and cost is positive
    sigma = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)

    # Strategy: Adaptive blend of PI+UCB (exploration) and -mu/cost (exploitation)
    # Early on (high f_best), we want more exploration.
    # Later (low f_best), we want more exploitation.
    
    # Normalize components to prevent scale dominance
    # Component 1: Exploitation score (higher is better)
    # We want to minimize mu, so -mu is the score. Normalize by cost for cost-awareness.
    exp_score = -mu / cost_safe
    
    # Component 2: Exploration score (PI + UCB)
    # Probability of Improvement (PI): P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    # UCB: mu + kappa * sigma
    # We combine them into a single exploration metric.
    # Let's use a normalized blend.
    
    # Compute PI
    z_pi = (f_best - mu) / sigma
    pi = norm.cdf(z_pi)
    
    # Compute UCB (uncertainty weighted)
    # Adaptive kappa: higher when f_best is high (early), lower when f_best is low (late)
    # f_best=100 -> kappa=4, f_best=0 -> kappa=0.5
    kappa = 0.5 + 3.5 * (f_best / 100.0)
    kappa = np.clip(kappa, 0.5, 4.0)
    
    ucb = mu + kappa * sigma
    
    # We want to maximize exploration. 
    # A good exploration score is high when PI is high and UCB is low (uncertainty high but likely good).
    # Let's create a composite exploration score.
    # Score_exploration = w1 * PI + w2 * (-UCB / cost)
    # But we need to blend with exploitation.
    
    # Let's try a simpler structural change:
    # Use Expected Improvement (EI) as the base, which is a standard and robust metric.
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Then normalize by cost.
    
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # EI is non-negative. Higher EI means higher expected improvement.
    # Normalize EI by cost to make it cost-aware.
    ei_score = ei / cost_safe
    
    # Now blend EI (exploration/exploitation balance) with pure exploitation (-mu/cost).
    # We want to shift from EI to pure exploitation as f_best gets low.
    # When f_best is low, EI becomes small, and we want to exploit the best known.
    # Let's use a weight that depends on f_best.
    
    # If f_best is very low, we trust exploitation.
    # If f_best is high, we trust EI (which includes exploration).
    
    # Normalize scores to [0, 1] range for blending?
    # Or just use a linear combination with adaptive weights.
    
    # Weight for exploitation: increases as f_best decreases.
    # w_exp = 1 / (1 + f_best / 10)
    # w_ei = 1 - w_exp
    
    # But we need to be careful with scales.
    # Let's try a different structural approach:
    # Use a hybrid that explicitly models the trade-off.
    
    # Actually, the inspiration mentioned "Adaptive blend shifts from exploration (PI+UCB) to exploitation (-mu/cost)".
    # Let's implement that directly.
    
    # Exploration score: PI * (1 + kappa * sigma / (f_best + 1))
    # This rewards high probability of improvement and high uncertainty.
    # But we need to make it comparable to exploitation.
    
    # Let's stick to the EI-based approach but with a twist:
    # Use a "cost-aware EI" that also considers the cost of evaluating.
    
    # Final decision: Use a blend of EI and UCB, weighted by f_best.
    # When f_best is high, use more UCB (exploration).
    # When f_best is low, use more EI (exploitation).
    
    # Compute UCB score (higher is better, so -UCB)
    ucb_score = -ucb / cost_safe
    
    # EI score is already computed
    ei_score = ei / cost_safe
    
    # Blend
    # Weight for EI: w_ei = 1 / (1 + f_best / 20)
    # Weight for UCB: w_ucb = 1 - w_ei
    
    w_ei = 1.0 / (1.0 + f_best / 20.0)
    w_ucb = 1.0 - w_ei
    
    # Ensure weights are in [0, 1]
    w_ei = np.clip(w_ei, 0.0, 1.0)
    w_ucb = np.clip(w_ucb, 0.0, 1.0)
    
    scores = w_ei * ei_score + w_ucb * ucb_score
    
    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
