"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Thompson Sampling with cost-aware weighting.
    Sample from GP posterior, weight by inverse cost, return expected improvement
    of the sampled minimum relative to f_best.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: handle non-finite inputs
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 1e-6), cost, 1.0)
    
    sigma_safe = np.maximum(sigma, 1e-9)
    
    n = len(mu)
    
    # Use multiple Thompson samples to get a robust score
    # The score for each candidate is the expected improvement of its posterior sample
    # weighted by cost (higher score = more likely to be evaluated next)
    
    # Cost-aware Thompson: sample f ~ N(mu, sigma), then score = max(0, f_best - f) / cost
    # This naturally balances exploration (high sigma -> wide samples) and cost
    
    # Use a fixed number of samples for stability
    n_samples = 16
    rng = np.random.default_rng(42)
    
    # Sample from posterior
    samples = rng.normal(0, 1, size=(n_samples, n))
    f_samples = mu[None, :] + sigma_safe[None, :] * samples  # (n_samples, n)
    
    # For each candidate, compute the expected improvement across samples
    # EI_i = E[max(f_best - f_i, 0)] where f_i ~ N(mu_i, sigma_i)
    # This is the same as standard EI, but we compute it via Monte Carlo for consistency
    
    # Actually, let's use a hybrid: compute EI analytically but add a Thompson-inspired
    # exploration bonus that's adaptive based on the distribution of samples
    
    # Analytic EI for minimization
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -50, 50)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Thompson-inspired exploration: measure how often each candidate is the argmin
    # across samples. This gives a natural exploration signal.
    argmin_indices = np.argmin(f_samples, axis=1)
    argmin_counts = np.bincount(argmin_indices, minlength=n).astype(np.float64)
    argmin_freq = argmin_counts / n_samples
    
    # Cost-aware: weight by inverse cost
    inv_cost = 1.0 / cost
    
    # Combine: EI provides exploitation, argmin_freq provides exploration
    # Scale argmin_freq to be comparable to EI
    ei_scale = np.std(ei) + 1e-9
    freq_scaled = argmin_freq * ei_scale
    
    score = (ei + freq_scaled) * inv_cost
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
