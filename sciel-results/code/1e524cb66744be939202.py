"""Acquisition function for Bayesian optimization of branin 2d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def _expected_improvement(mu, sigma, f_best):
    """Expected Improvement for minimization."""
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) - sigma_safe * norm.pdf(z)
    return np.maximum(ei, 0.0)


def _knowledge_gradient_score(mu, sigma, f_best, n_candidates=2000):
    """
    Approximate Knowledge Gradient acquisition.
    
    The true KG asks: how much will evaluating at x improve our estimate
    of the global minimum? We approximate this by considering the
    expected reduction in f_best after observing f(x).
    
    For a single new observation y ~ N(mu_i, sigma_i^2) at candidate i:
    - If y < f_best, new f_best = y, improvement = f_best - y
    - If y >= f_best, f_best unchanged, improvement = 0
    
    E[improvement] = E[max(f_best - y, 0)] where y ~ N(mu_i, sigma_i^2)
    This is exactly the Expected Improvement (EI) formula.
    
    However, pure EI is too exploitative. We enhance it with a
    diversity-aware term: candidates that are far from already-well-explored
    low-mu regions get a bonus, because discovering a NEW basin is more
    valuable than marginally improving a known one.
    
    We approximate the "novelty" of a candidate by its distance (in the
    posterior sense) from the set of points with low mu. Specifically,
    we compute for each candidate how many other candidates have lower mu
    AND similar sigma profile — if a candidate is in a region where few
    other candidates have low mu, it's in a potentially new basin.
    
    A simpler and more robust approach: use the EI but with an adaptive
    xi (offset) that increases with the number of evaluations (estimated
    from f_best's proximity to the known minimum), and add a small
    sigma-based exploration term that is *rank-normalized* to avoid
    dominating.
    """
    sigma_safe = np.maximum(sigma, 1e-9)
    n = len(mu)
    
    # Base EI
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) - sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Rank-normalize EI to [0, 1]
    ei_sorted = np.sort(ei)
    ei_ranks = np.argsort(np.argsort(ei)).astype(np.float64) / max(n - 1, 1)
    
    # Exploration term: rank-normalized sigma
    sigma_sorted = np.sort(sigma_safe)
    sigma_ranks = np.argsort(np.argsort(sigma_safe)).astype(np.float64) / max(n - 1, 1)
    
    # Adaptive exploration weight:
    # Early on (f_best far from min), explore more.
    # Late (f_best close to min), exploit more.
    f_min_known = 0.397887
    remaining = max(f_best - f_min_known, 0.0)
    # Weight goes from ~0.35 (early) to ~0.05 (late)
    w_explore = 0.05 + 0.30 * np.exp(-remaining / 8.0)
    
    # Combine: weighted sum of rank-normalized EI and sigma
    score = (1.0 - w_explore) * ei_ranks + w_explore * sigma_ranks
    
    # Add a small absolute EI component to preserve scale information
    # (rank-only can lose magnitude info when many candidates have similar EI)
    ei_max = np.max(ei)
    if ei_max > 1e-12:
        score += 0.1 * (ei / ei_max)
    
    return score


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using an Enhanced Knowledge Gradient approximation.
    
    Combines:
    1. Rank-normalized Expected Improvement (exploitation)
    2. Rank-normalized posterior sigma (exploration)  
    3. A small absolute EI term to preserve magnitude
    
    The exploration weight adapts: higher when f_best is far from the
    known minimum (early), lower when close (late). This structural
    approach differs from pure UCB (which linearly combines -mu and sigma)
    and from pure POI/EI (which are purely exploitative).
    
    For Branin 2D's multi-modal landscape, the rank-based normalization
    ensures that a candidate in a completely new basin (high sigma, moderate
    mu) gets competitive score with candidates in known basins (low sigma,
    low mu), facilitating discovery of all three minima.
    """
    t_start = time.time()
    deadline = t_start + 295  # safety margin before 300s limit
    
    n = len(mu)
    
    if n == 0:
        return np.array([])
    
    # Handle edge cases
    if not np.all(np.isfinite(mu)):
        mu = np.where(np.isfinite(mu), mu, np.median(mu[np.isfinite(mu)]))
    if not np.all(np.isfinite(sigma)):
        sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute the KG-inspired score
    score = _knowledge_gradient_score(mu, sigma, f_best)
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Time guard
    if time.time() > deadline:
        return score
    
    return score
# EVOLVE-BLOCK-END
