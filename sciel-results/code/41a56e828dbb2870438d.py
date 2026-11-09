import numpy as np
import time

# Module-level deadline guard state
_START = None
_DEADLINE_S = 300.0  # safety margin well under the 330s kill

def _elapsed():
    global _START
    if _START is None:
        _START = time.time()
    return time.time() - _START

def _deadline_ok():
    return _elapsed() < _DEADLINE_S

def acquisition(mu, sigma, f_best):
    global _START
    if _START is None:
        _START = time.time()

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    # Safety: ensure finite inputs
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e9, neginf=-1e9)
    sigma = np.nan_to_num(sigma, nan=0.0, posinf=1e6, neginf=0.0)
    sigma = np.maximum(sigma, 1e-12)

    # Deadline guard: if we are close to the limit, return a safe fallback
    if not _deadline_ok():
        # Fallback: pure exploitation (lowest mu)
        return np.nan_to_num(-mu, nan=0.0, posinf=1e6, neginf=-1e6)

    # Adaptive uncertainty scale: use a robust measure of the current sigma spread
    # so the exploration term is calibrated to the GP's current uncertainty level.
    # This lets the acquisition adapt as the GP tightens (fewer evaluations -> high sigma,
    # more evaluations -> low sigma).
    sigma_scale = np.std(sigma)
    if sigma_scale < 1e-9:
        sigma_scale = 1e-9

    # Normalize sigma to [0, 1] range for the PI-weighted exploration term
    s_min = np.min(sigma)
    s_max = np.max(sigma)
    s_range = s_max - s_min
    if s_range < 1e-9:
        s_range = 1e-9
    sigma_norm = (sigma - s_min) / s_range

    # Expected Improvement (minimization):
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # We use scipy's norm for Phi and phi.
    try:
        from scipy.stats import norm
        z = (f_best - mu) / sigma
        phi = norm.pdf(z)
        Phi = norm.cdf(z)
        ei = (f_best - mu) * Phi + sigma * phi
    except Exception:
        # Fallback: simple approximation without scipy
        z = (f_best - mu) / sigma
        # Approximate Phi and phi with simple formulas
        Phi = 0.5 * (1.0 + np.tanh(z / np.sqrt(2.0)))
        phi = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
        ei = (f_best - mu) * Phi + sigma * phi

    # EI is higher when mu is low and sigma is high.
    # We want higher score = evaluate next, so EI itself is a good base.
    scores = ei.copy()

    # Add a small PI-sigma exploration blend:
    # PI = Phi((f_best - mu)/sigma)  (probability that f(x) < f_best)
    # Weight by normalized uncertainty to explore where GP is uncertain AND plausibly better.
    try:
        from scipy.stats import norm
        pi_prob = norm.cdf((f_best - mu) / sigma)
    except Exception:
        z = (f_best - mu) / sigma
        pi_prob = 0.5 * (1.0 + np.tanh(z / np.sqrt(2.0)))

    # Exploration term: small weight, scaled by uncertainty
    exploration_weight = 0.3
    exploration = exploration_weight * pi_prob * sigma_norm

    scores = scores + exploration

    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)

    return scores
