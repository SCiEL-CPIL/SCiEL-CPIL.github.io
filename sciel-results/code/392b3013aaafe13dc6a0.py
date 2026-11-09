"""Cost-aware acquisition function for Bayesian optimization (shekel 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
from scipy.integrate import quad


def _expected_improvement(mu, sigma, f_best, xi=0.0):
    """Compute expected improvement with exploration parameter xi."""
    sigma_safe = np.maximum(sigma, 1e-8)
    z = (f_best - mu - xi) / sigma_safe
    ei = (f_best - mu - xi) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    return np.maximum(ei, 0.0)


def _ucb(mu, sigma, beta=2.0):
    """Upper Confidence Bound."""
    return mu + beta * sigma


def _pi(mu, sigma, f_best, xi=0.0):
    """Probability of improvement."""
    sigma_safe = np.maximum(sigma, 1e-8)
    z = (f_best - mu - xi) / sigma_safe
    return norm.cdf(z)


def _kde_density(x, samples, bandwidth_factor=1.0):
    """Gaussian KDE density estimate for 1D points."""
    if len(samples) == 0:
        return np.ones_like(x) * 1e-10
    # Silverman's rule of thumb for bandwidth
    std = np.std(samples)
    if std < 1e-10:
        std = 1e-10
    n = len(samples)
    h = bandwidth_factor * (1.06 * std * n ** (-1.0 / 5.0))
    h = max(h, 1e-4)
    # Vectorized KDE
    diff = x[:, None] - samples[None, :]
    density = np.exp(-0.5 * (diff / h) ** 2) / (h * np.sqrt(2 * np.pi))
    return density.mean(axis=1)


def _cost_aware_ei(mu, sigma, f_best, cost, xi=0.0):
    """Expected improvement per unit cost."""
    ei = _expected_improvement(mu, sigma, f_best, xi)
    # cost is the total cost so far; use it as a penalty
    # Normalize cost to [0,1] range for better scaling
    cost_norm = cost / (cost.max() + 1e-10)
    return ei / (1.0 + cost_norm)


def _cost_aware_ucb(mu, sigma, cost, beta=2.0):
    """UCB per unit cost."""
    ucb = _ucb(mu, sigma, beta)
    cost_norm = cost / (cost.max() + 1e-10)
    return ucb / (1.0 + cost_norm)


def _adaptive_weights(mu, sigma, f_best, cost, n_evals):
    """
    Adaptive combination of EI, UCB, and PI based on exploration progress.
    Early: more exploration (UCB). Late: more exploitation (EI).
    """
    # Progress ratio: 0 early, 1 late
    progress = min(n_evals / 30.0, 1.0)
    
    # Weights that shift from exploration to exploitation
    w_ucb = 0.5 * (1.0 - progress) + 0.1
    w_ei = 0.5 * progress + 0.1
    w_pi = 0.2 * (1.0 - progress) + 0.05
    
    # Normalize weights
    w_sum = w_ucb + w_ei + w_pi
    w_ucb /= w_sum
    w_ei /= w_sum
    w_pi /= w_sum
    
    # Compute each component
    ucb_score = _ucb(mu, sigma, beta=2.0)
    ei_score = _expected_improvement(mu, sigma, f_best, xi=0.01)
    pi_score = _pi(mu, sigma, f_best, xi=0.0)
    
    # Combine
    combined = w_ucb * ucb_score + w_ei * ei_score + w_pi * pi_score
    return combined


def _thompson_sampling(mu, sigma, f_best, cost, rng):
    """Thompson sampling with cost awareness."""
    # Sample from posterior
    samples = mu + sigma * rng.standard_normal(len(mu))
    # Select the one with best (lowest) sampled value, weighted by cost
    cost_norm = cost / (cost.max() + 1e-10)
    # We want low values, so score = -sampled_value / cost_factor
    score = -samples / (1.0 + cost_norm)
    return score


def _entropy_based(mu, sigma, f_best, cost):
    """
    Acquisition based on reducing uncertainty in the region near f_best.
    Uses information gain proxy: high sigma where mu is close to f_best.
    """
    sigma_safe = np.maximum(sigma, 1e-8)
    # Distance from f_best in standard deviation units
    dist = np.abs(f_best - mu) / sigma_safe
    # High uncertainty near f_best is valuable
    info_gain = sigma_safe * np.exp(-0.5 * dist ** 2)
    # Cost awareness
    cost_norm = cost / (cost.max() + 1e-10)
    return info_gain / (1.0 + cost_norm)


def _hybrid_acquisition(mu, sigma, f_best, cost, n_evals):
    """
    Hybrid acquisition combining multiple strategies with adaptive weighting.
    """
    n = len(mu)
    rng = np.random.RandomState(42)
    
    # Compute progress
    progress = min(n_evals / 30.0, 1.0)
    
    # Strategy 1: Cost-aware EI
    score_ei = _cost_aware_ei(mu, sigma, f_best, cost, xi=0.01)
    
    # Strategy 2: Cost-aware UCB
    score_ucb = _cost_aware_ucb(mu, sigma, cost, beta=2.5)
    
    # Strategy 3: Entropy/information gain
    score_entropy = _entropy_based(mu, sigma, f_best, cost)
    
    # Strategy 4: PI with cost
    pi_score = _pi(mu, sigma, f_best, xi=0.0)
    cost_norm = cost / (cost.max() + 1e-10)
    score_pi = pi_score / (1.0 + cost_norm)
    
    # Adaptive weights based on progress
    # Early: explore more (UCB, entropy)
    # Late: exploit more (EI, PI)
    w_ucb = 0.3 * (1.0 - progress) + 0.1
    w_ei = 0.4 * progress + 0.2
    w_entropy = 0.2 * (1.0 - progress) + 0.05
    w_pi = 0.1 * progress + 0.05
    
    w_sum = w_ucb + w_ei + w_entropy + w_pi
    w_ucb /= w_sum
    w_ei /= w_sum
    w_entropy /= w_sum
    w_pi /= w_sum
    
    # Normalize scores to [0, 1] for fair combination
    def normalize(s):
        s_min = s.min()
        s_max = s.max()
        if s_max - s_min < 1e-10:
            return np.ones_like(s) * 0.5
        return (s - s_min) / (s_max - s_min)
    
    score_ei_n = normalize(score_ei)
    score_ucb_n = normalize(score_ucb)
    score_entropy_n = normalize(score_entropy)
    score_pi_n = normalize(score_pi)
    
    # Combine
    final_score = (w_ucb * score_ucb_n + w_ei * score_ei_n + 
                   w_entropy * score_entropy_n + w_pi * score_pi_n)
    
    return final_score


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    - mu: array of shape (2000,) - GP mean predictions
    - sigma: array of shape (2000,) - GP standard deviations
    - f_best: scalar - best (lowest) observed value
    - cost: array of shape (2000,) - total cost so far for each candidate
    
    Returns:
    - array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    n = len(mu)
    
    # Estimate number of evaluations from cost
    # Initial 5 points cost roughly 5 * (1 + 0.5) = 7.5
    # Each subsequent eval costs ~1 + mean(x) where x in [0,10]^4 scaled to [0,1]
    # So mean cost per eval is ~1.5
    n_evals = int((cost.mean() - 7.5) / 1.5) + 5 if cost.mean() > 7.5 else 5
    n_evals = max(n_evals, 5)
    
    # Use hybrid acquisition
    scores = _hybrid_acquisition(mu, sigma, f_best, cost, n_evals)
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
