# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Module-level state for time tracking
_start_time = None
_DEADLINE_MARGIN = 10.0  # seconds before the 330s limit

def _get_elapsed():
    """Return elapsed time since module load, initializing on first call."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time

def _is_deadline_exceeded():
    """Check if we are close to the 330s wall-clock limit."""
    return _get_elapsed() > (330.0 - _DEADLINE_MARGIN)

def acquisition(mu, sigma, f_best):
    """
    Knowledge Gradient (KG) acquisition function for Bayesian optimization.
    
    KG estimates the expected reduction in the best-known objective value
    by considering the probability that each candidate point becomes the
    new global best. This provides a more global view than local improvement
    heuristics like EI or UCB.
    
    For minimization:
    - f_best is the current best (lowest) observed value
    - mu[i], sigma[i] are the GP posterior mean and std at candidate i
    - We want to maximize the expected improvement over f_best
    
    The exact KG for a single evaluation is:
    KG(i) = E[max(0, f_best - y_i)] where y_i ~ N(mu_i, sigma_i^2)
    which is actually just EI. But the full KG considers the joint distribution
    and the fact that evaluating point i updates our belief about the global minimum.
    
    A common approximation for KG in practice (especially with limited evaluations)
    is to use a weighted combination or to approximate the full KG. Here we use
    a practical approximation that captures the global nature of KG:
    
    For a single new evaluation, the KG value can be approximated by considering
    that the new point will be the best if it beats f_best, and the expected
    reduction is the expected amount by which it beats f_best.
    
    However, a more principled KG approximation considers that after evaluating
    point i, the updated best is min(f_best, y_i), and the expected reduction
    in the best is E[f_best - min(f_best, y_i)] = E[max(0, f_best - y_i)] = EI(i).
    
    But the true KG accounts for the fact that the GP posterior at other points
    also changes. A simplified KG that accounts for this is to use a "lookahead"
    or to use the fact that the optimal next point maximizes the expected
    reduction in the best, which for a single evaluation is indeed EI.
    
    To make this structurally different from pure EI and capture more global
    behavior, we use a KG-inspired score that incorporates the probability
    that the candidate is among the top-k best points, weighted by the expected
    improvement. This is an approximation that better reflects the global
    nature of the KG criterion.
    
    We use: score(i) = EI(i) * (1 + beta * P(y_i < f_best))
    where P(y_i < f_best) is the probability that candidate i beats f_best.
    This weights points that are likely to be new bests more heavily, which
    is a KG-like behavior.
    
    Actually, let's use a cleaner KG approximation. The exact KG for selecting
    the next point to evaluate, given that we have current data D and f_best,
    is:
    KG(i) = E_D'[f_best - min(f_best, y_i)] where D' is the updated data
    But this is complex. A practical KG approximation used in literature is:
    KG(i) ≈ E[max(0, f_best - y_i)] + gamma * sigma_i^2
    where gamma is a small constant that encourages exploration.
    
    Let's use a different structural approach: we'll compute a KG-inspired score
    that considers the expected number of points that would be improved upon.
    
    For clarity and effectiveness, let's use the following KG approximation:
    score(i) = EI(i) + lambda * sigma_i * P(y_i < f_best - margin)
    where margin is a small threshold and lambda is a tuning parameter.
    
    Actually, the simplest effective KG-like modification is to use the
    expected maximum improvement, which for a single evaluation is EI, but
    we can enhance it by considering the second-best point's potential.
    
    Let's implement a practical KG approximation that is computationally
    efficient and structurally different from pure EI:
    
    We'll use: score(i) = EI(i) + w * (sigma_i * Phi((f_best - mu_i) / sigma_i))
    where w is a small weight. The second term rewards points with high
    uncertainty that are likely to beat f_best, which is a KG-like signal.
    
    For the deadline guard: if we are close to the limit, we return a simple
    EI-based score to ensure we finish in time.
    """
    # Deadline guard: if we're close to the limit, use a fast fallback
    if _is_deadline_exceeded():
        eps = 1e-9
        sigma_safe = np.maximum(sigma, eps)
        z = (f_best - mu) / sigma_safe
        cdf_z = norm.cdf(z)
        pdf_z = norm.pdf(z)
        ei_scores = (f_best - mu) * cdf_z + sigma_safe * pdf_z
        ei_scores = np.where(np.isfinite(ei_scores), ei_scores, -np.inf)
        if not np.any(np.isfinite(ei_scores)):
            return np.zeros_like(mu)
        return ei_scores
    
    # Ensure sigma is positive and avoid division by zero
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Calculate z-scores
    z = (f_best - mu) / sigma_safe
    
    # Standard normal CDF and PDF
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    # Base EI score
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # KG-inspired enhancement:
    # The KG criterion values points that have a high probability of being
    # the new global best AND a high expected improvement.
    # P(y_i < f_best) = Phi(z)
    # We add a term that rewards points with high probability of beating
    # f_best, weighted by their uncertainty.
    
    # Probability that candidate i is a new best
    prob_new_best = cdf_z  # P(y_i < f_best)
    
    # KG-like score: EI plus a bonus for likely new bests with high uncertainty
    # The bonus is scaled by sigma to reward exploration of uncertain regions
    # that are likely to be good
    kg_bonus = 0.1 * sigma_safe * prob_new_best
    
    scores = ei + kg_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    # If all are -inf, return zeros to avoid invalidity
    if not np.any(np.isfinite(scores)):
        return np.zeros_like(mu)
    
    return scores
# EVOLVE-BLOCK-END
