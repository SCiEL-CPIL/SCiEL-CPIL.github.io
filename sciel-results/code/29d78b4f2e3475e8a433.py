import numpy as np
import time
from scipy.stats import norm

_start_time = None

def acquisition(mu, sigma, f_best):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
        
    elapsed = time.time() - _start_time
    # Safety margin: 290s
    if elapsed > 290.0:
        # Fallback to greedy exploitation
        mu = np.asarray(mu, dtype=float)
        mu = np.where(np.isfinite(mu), mu, f_best + 1e6)
        return -mu

    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    mu = np.where(np.isfinite(mu), mu, f_best + 1e6)
    
    # Adaptive EI with Gamma
    # Gamma scales with f_best to balance exploration/exploitation
    # High f_best -> High Gamma -> More Exploration
    # Low f_best -> Low Gamma -> More Exploitation
    gamma = 0.5 * f_best
    
    # z = (f_best - mu + gamma) / sigma
    z = (f_best - mu + gamma) / sigma
    
    # Clip z to prevent numerical issues
    z = np.clip(z, -10, 10)
    
    # EI = (f_best - mu + gamma) * Phi(z) + sigma * phi(z)
    # Note: This is EI for the shifted target (f_best - gamma)
    # Actually, standard EI is E[max(f_best - f, 0)].
    # If we use (f_best - mu + gamma), we are effectively using a target of f_best - gamma.
    # Let's verify the sign.
    # Improvement = f_best - f.
    # We want E[max(f_best - f, 0)].
    # If we shift the target to f_target = f_best - gamma.
    # Then we calculate E[max(f_target - f, 0)]?
    # No, the formula $(f_{best} - \mu + \gamma)\Phi(z) + \sigma\phi(z)$ with $z=(f_{best}-\mu+\gamma)/\sigma$
    # corresponds to $E[\max(f_{best} - \gamma - f, 0)]$?
    # Let $Y \sim N(\mu, \sigma^2)$.
    # $E[\max(K - Y, 0)] = (K - \mu)\Phi(\frac{K-\mu}{\sigma}) + \sigma\phi(\frac{K-\mu}{\sigma})$.
    # Here $K = f_{best} - \gamma$.
    # So $K - \mu = f_{best} - \gamma - \mu$.
    # My formula has $f_{best} - \mu + \gamma$.
    # So $K = f_{best} + \gamma$.
    # This means we are calculating $E[\max(f_{best} + \gamma - f, 0)]$.
    # This is improvement over a *worse* target ($f_{best} + \gamma$).
    # This **increases** the EI for points with $\mu > f_{best}$.
    # So it **increases** exploration.
    # This is what we want when f_best is high.
    
    ei = (f_best - mu + gamma) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure finite
    ei = np.where(np.isfinite(ei), ei, -np.inf)
    
    return ei
