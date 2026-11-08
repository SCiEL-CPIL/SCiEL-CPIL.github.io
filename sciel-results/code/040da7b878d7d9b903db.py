"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global cache for kernel factorization
# Key: hash of the cost vector (proxy for instance/candidate set)
_kernel_cache = {}

def _get_kernel_factor(cost, n_samples=2000):
    """
    Compute and cache the Cholesky factor of the Matern-5/2 kernel matrix.
    The candidates are uniform random, so the cost vector (1 + mean(x)) is unique per instance.
    We assume the evaluator passes the same 2000 candidates every iteration for a given instance.
    """
    # Use the cost vector as a fingerprint. Since x is uniform, cost is a proxy for x.
    # However, cost is 1D, x is 3D. We need x to compute the kernel.
    # Wait, the prompt says: "mu, sigma and cost have shape (2000,). Return 2000 finite scores... No coordinates are passed."
    # This is a critical constraint. We do NOT have x.
    # We only have mu, sigma, cost.
    # Therefore, we CANNOT compute the GP kernel matrix from scratch because we don't have x.
    #
    # RE-READING TASK:
    # "GP: Matern-5/2, length scale 0.2 on coordinates scaled to [0, 1]... mu and sigma ... are in objective units."
    # "mu, sigma and cost have shape (2000,). ... No coordinates are passed."
    #
    # This implies the GP has ALREADY been fitted, and mu/sigma are the predictions at the 2000 candidate points.
    # We do not have the training data, nor the candidate coordinates x.
    # Thus, we CANNOT perform Thompson Sampling which requires sampling f(x) ~ N(mu, sigma) jointly.
    # We only have marginal distributions N(mu_i, sigma_i) for each candidate i.
    #
    # So, Batch Thompson Sampling is NOT possible without the covariance matrix K(x, x).
    # We are stuck with marginal distributions.
    #
    # Therefore, the strategy must be based on the marginal mu, sigma, and cost.
    # The current strategy is EI / cost.
    #
    # Alternative Strategy: **Probability of Improvement (PI) with Cost-Adjusted Exploration**
    # Or: **Knowledge Gradient (KG)** approximation?
    # Or: **Entropy Search**?
    #
    # Since we only have marginals, we can't do proper joint sampling.
    # However, we can use a different acquisition function that behaves differently from EI.
    # EI is known to be too exploitative early on.
    # A different family: **UCB (Upper Confidence Bound)** for minimization.
    # UCB = mu - kappa * sigma.
    # Score = (f_best - (mu - kappa * sigma)) / cost.
    #
    # Another option: **Probability of Improvement (PI)**
    # PI = P(f(x) < f_best) = Phi((f_best - mu) / sigma).
    # Score = PI / cost.
    #
    # Let's try a **Cost-Weighted PI with an exploration bonus**.
    # Or better, let's look at the "Inspiration 5" which scored 1.2657. It used `_calibrate_params`.
    # This suggests that adapting kappa or beta dynamically might be key.
    #
    # Let's try **Dynamic UCB**.
    # Start with high exploration (large kappa) and decay it as cost increases or f_best improves.
    # Since we don't know the iteration count, we can use `f_best` or the magnitude of `sigma` to adapt.
    #
    # Strategy: **Adaptive UCB**
    # kappa = base_kappa * (f_best - min(mu)) / (max(sigma) + eps)
    # This scales exploration based on the potential improvement relative to uncertainty.
    
    # Actually, a very robust strategy for minimization with cost is:
    # **Expected Improvement (EI)** is standard.
    # **Probability of Improvement (PI)** is simpler.
    # **UCB** is deterministic.
    #
    # Let's try **PI / cost** but with a twist: **Weighted PI**.
    # Or, let's try **Lower Confidence Bound (LCB)** which is equivalent to UCB for minimization.
    # LCB = mu - beta * sigma.
    # We want to minimize LCB.
    # Score = -LCB / cost = (beta * sigma - mu) / cost.
    # This is different from EI. EI integrates the tail. LCB uses a point estimate.
    # LCB is often more robust to noise and less prone to the "greedy trap" of EI if beta is chosen well.
    
    # Let's implement an **Adaptive LCB**.
    # beta = 0.5 + 0.5 * (f_best - min(mu)) / (max(sigma) + 1e-9)
    # This increases exploration when there is a large gap between current best and predicted best,
    # or when uncertainty is high.
    
    pass # Placeholder, will implement in the function below

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Adaptive Lower Confidence Bound (LCB).
    
    LCB for minimization: LCB(x) = mu(x) - beta * sigma(x)
    We want to maximize the "promise" of improvement per cost.
    Score = (f_best - LCB(x)) / cost(x)
           = (f_best - mu(x) + beta * sigma(x)) / cost(x)
    
    Adaptive beta:
    We scale beta based on the relative uncertainty and potential improvement.
    beta = beta_min + (beta_max - beta_min) * uncertainty_factor
    
    uncertainty_factor = mean(sigma) / (std(mu) + mean(sigma) + eps)
    This balances exploitation (low sigma) and exploration (high sigma).
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety check for finite values
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(sigma)) or not np.all(np.isfinite(cost)):
        return np.zeros(len(mu))
    
    # Avoid division by zero in cost
    cost_safe = np.maximum(cost, 1e-9)
    
    # Calculate adaptive beta
    # Heuristic: Higher beta when sigma is large relative to the spread of mu
    mu_spread = np.std(mu)
    mean_sigma = np.mean(sigma)
    
    # Normalize uncertainty: 0 if no uncertainty, 1 if uncertainty dominates
    # If mu_spread is 0, we rely entirely on sigma
    denom = mu_spread + mean_sigma + 1e-9
    uncertainty_ratio = mean_sigma / denom
    
    # Beta ranges from 0.5 (exploitative) to 1.5 (exploratory)
    beta_min = 0.5
    beta_max = 1.5
    beta = beta_min + (beta_max - beta_min) * uncertainty_ratio
    
    # LCB Score: We want to maximize (f_best - LCB) / cost
    # LCB = mu - beta * sigma
    # Improvement potential = f_best - LCB = f_best - mu + beta * sigma
    
    improvement_potential = f_best - mu + beta * sigma
    
    # Final score
    score = improvement_potential / cost_safe
    
    # Ensure finite output
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
# EVOLVE-BLOCK-END
