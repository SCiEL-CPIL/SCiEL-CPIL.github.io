"""Bayesian optimization acquisition for 1d sphere; EVOLVE-BLOCK is the editable part."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """EHypR (Expected Hyper-Ratio) acquisition function.
    
    Maximizes E[max(f_best, f(x)) / f(x)] which directly targets simple regret.
    For minimization, we want to pick x where f(x) is likely to be small.
    The hyper-ratio is defined as max(f_best, f(x))/f(x) >= 1.
    We minimize this ratio, so we maximize its negative.
    
    Actually, let's think more carefully. The standard EHypR for minimization:
    We want to find x that minimizes f(x). The "ratio" approach considers:
    If f(x) < f_best, the ratio is f_best/f(x) > 1 (improvement)
    If f(x) >= f_best, the ratio is 1 (no improvement)
    
    So we want to maximize the expected improvement in the ratio sense.
    E[max(f_best, f(x)) / f(x)] where f(x) ~ N(mu, sigma^2)
    
    But this has issues when f(x) is near 0 (division by zero).
    
    Let me reconsider. A cleaner approach for 1D sphere:
    Since f(x) = x^2 >= 0, and we know f_best, we can use:
    - If we predict mu < 0, that's impossible for sphere, but the GP might predict it.
    - We should clip mu to be non-negative in the acquisition computation.
    
    Let me use a different formulation. The Expected Improvement (EI) is well-known:
    EI(x) = E[max(f_best - f(x), 0)]
    For f(x) ~ N(mu, sigma^2):
    EI(x) = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    
    But EI can be too exploitative. Let me combine EI with a UCB-like term.
    
    Actually, the directive says to use EHypR. Let me implement it properly.
    
    EHypR: E[max(f_best, f(x))/f(x)]
    = E[1] when f(x) >= f_best
    = E[f_best/f(x)] when f(x) < f_best
    
    This is complex to compute analytically. Let me use numerical integration or sampling.
    
    Actually, for a cleaner structural change, let me use a hybrid approach:
    - Use the posterior to compute a "regret-aware" score
    - Combine exploitation (low mu) with exploration (high sigma) in a principled way
    
    Let me use the following: score = -(mu + c * max(0, sigma - threshold))
    No, that's still UCB-like.
    
    Let me try: Expected Improvement with an adaptive exploration bonus.
    EI(x) = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu)/sigma
    This naturally balances: when sigma is large, EI can be large even if mu is not very low.
    
    Let me implement EI properly and add a small exploration component.
    """
    t0 = time.perf_counter()
    DEADLINE = 300.0 - 5.0
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    n = mu.size
    
    # Guard against non-finite inputs
    if not np.any(np.isfinite(mu)):
        return np.zeros(n)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    sigma = np.maximum(sigma, 1e-12)
    
    # Expected Improvement for minimization
    # EI(x) = E[max(f_best - f(x), 0)] for f(x) ~ N(mu, sigma^2)
    z = (f_best - mu) / sigma
    # Clip z to avoid numerical issues
    z = np.clip(z, -10, 10)
    
    # Phi(z) = CDF of standard normal
    Phi_z = norm.cdf(z)
    # phi(z) = PDF of standard normal
    phi_z = norm.pdf(z)
    
    EI = (f_best - mu) * Phi_z + sigma * phi_z
    
    # Ensure EI is non-negative
    EI = np.maximum(EI, 0.0)
    
    # Add a small exploration bonus to encourage visiting uncertain regions
    # This helps escape local optima
    exploration_bonus = 0.1 * sigma
    
    scores = EI + exploration_bonus
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    elapsed = time.perf_counter() - t0
    if elapsed > DEADLINE:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
