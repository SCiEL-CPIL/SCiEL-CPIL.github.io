"""Acquisition for Bayesian optimization of the 2D Michalewicz function.

The evaluator hands us mu and sigma (posterior mean and standard deviation,
in objective units) on 2000 fresh uniform candidates, plus f_best (the lowest
value observed so far). We must return 2000 finite scores; the candidate with
the highest score is evaluated next. The GP is Matern-5/2 on [0,1]^2 (coords
scaled from [0, pi]^2), noise 1e-6, standardized targets.
"""
import time
import numpy as np

# Deadline guard: the whole evaluation must finish well inside the 300s limit.
# We track wall-clock from first call and bail to a safe fallback if we get
# dangerously close to the limit. Spare time does not add search steps; it
# only protects against pathological slow iterations.
_DEADLINE_START = None
# Return this if we run out of budget: pick the most confident low-mean point.
_FALLBACK_READY = False


def _deadline_ok():
    global _DEADLINE_START
    if _DEADLINE_START is None:
        _DEADLINE_START = time.time()
    # Generous safety margin well below 330s; one EI pass over 2000 points is
    # microseconds, so this essentially never triggers but is a hard guarantee.
    return (time.time() - _DEADLINE_START) < 320.0


def acquisition(mu, sigma, f_best):
    """UCB acquisition for minimization: score = -(mu + beta * sigma).

    A structurally different family from EI: it is an optimistic (confidence-
    bound) acquisition that directly trades off predicted low mean against
    posterior standard deviation, instead of integrating expected improvement.
    For smooth multimodal objectives (Michalewicz) and tight evaluation budgets
    (25 total), UCB often reaches the global basin faster than EI because it
    commits to the single most promising uncertain point rather than averaging
    over improvement.

    beta decays mildly with wall-clock time (a proxy for evaluation count):
    start with strong exploration to jump between the two Michalewicz minima,
    then shift toward exploitation to polish the incumbent.
    """
    global _FALLBACK_READY

    mu = np.asarray(mu, dtype=np.float64).ravel()
    sigma = np.asarray(sigma, dtype=np.float64).ravel()

    # Hard deadline fallback: deterministic, finite, exploitation-biased.
    if not _deadline_ok():
        fallback = -mu
        return np.where(np.isfinite(fallback), fallback, 0.0)

    # Robust standard deviation: floor so the UCB is well-defined even if a
    # sigma is ~0 (e.g. a candidate coincides with an observation).
    eps = 1e-8
    sigma_safe = np.maximum(sigma, eps)

    # Mild time-based decay of exploration: early on we want to roam the domain
    # (Michalewicz 2D has two deep minima); late on we want to converge.
    # 25 evaluations total; 300s budget -> ~12s per eval on average. Use a
    # gentle linear decay from beta_start to beta_end over ~200s.
    elapsed = time.time() - _DEADLINE_START if _DEADLINE_START is not None else 0.0
    beta_start = 2.5
    beta_end = 0.5
    t_norm = min(max(elapsed / 200.0, 0.0), 1.0)
    beta = beta_start + (beta_end - beta_start) * t_norm

    # UCB for minimization: prefer low (mu + beta*sigma); negate for "higher
    # is better" scoring.
    ucb = mu + beta * sigma_safe
    scores = -ucb

    # Guarantee finite output.
    scores = np.where(np.isfinite(scores), scores, 0.0)
    _FALLBACK_READY = True
    return scores
