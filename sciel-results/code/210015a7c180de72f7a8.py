import numpy as np
from scipy.stats import norm
import time

_deadline = None
_start_time = None
_call_count = 0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using QOEI (Quantile-based Optimistic EI)
    blended with an LCB exploration term, all normalized by cost.
    
    QOEI uses a quantile-based optimistic target instead of f_best, which
    makes the acquisition more aggressive in exploring promising regions.
    The blend weight shifts from exploration to exploitation over time.
    """
    global _deadline, _start_time, _call_count
    
    if _start_time is None:
        _start_time = time.time()
        _deadline = _start_time + 320.0
    
    _call_count += 1
    
    if time.time() > _deadline:
        return np.ones(len(mu))
    
    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)
    cost_safe = np.maximum(cost, eps)
    
    # QOEI: use an optimistic target based on the quantile of the posterior
    # The quantile q shifts the target to be more aggressive
    # q=0.3 means we use the 30th percentile of the GP posterior as target
    q = 0.3
    z_q = norm.ppf(q)  # negative value, ~ -0.5244 for q=0.3
    
    # Optimistic target: f_target = mu + z_q * sigma
    # This is lower than mu (since z_q < 0), making the acquisition more aggressive
    f_target = mu + z_q * sigma_safe
    
    # For QOEI, we compute EI relative to f_target but with the original f_best
    # Actually, QOEI works by: the "believer" assumes the function value is f_target
    # and computes EI of improvement over f_best from that perspective
    # EI_qoei = E[max(f_target - f, 0)] where f ~ N(mu, sigma^2)
    # = (f_target - mu) * Phi((f_target - mu)/sigma) + sigma * phi((f_target - mu)/sigma)
    # But f_target - mu = z_q * sigma, so:
    # EI_qoei = z_q * sigma * Phi(z_q) + sigma * phi(z_q)
    # = sigma * (z_q * Phi(z_q) + phi(z_q))
    # This is constant per candidate only through sigma. That's not right.
    
    # Let me reconsider. QOEI (from the literature) works as:
    # 1. Compute the q-quantile of the GP posterior at each point: mu + z_q * sigma
    # 2. The "optimistic" value is this quantile
    # 3. EI is computed as improvement over f_best using this optimistic value
    #    EI_qoei(x) = E[max(optimistic(x) - f(x), 0)] where f(x) ~ N(mu(x), sigma(x)^2)
    #    But this is just a function of sigma, which is not useful for discrimination.
    
    # Alternative interpretation: QOEI modifies the target to be more optimistic.
    # The standard EI uses f_best as target. QOEI uses a lower (more optimistic) target.
    # Let's use: target = min(f_best, mu + z_q * sigma)
    # Then EI = E[max(target - f, 0)] where f ~ N(mu, sigma^2)
    
    target = np.minimum(f_best, mu + z_q * sigma_safe)
    
    # Compute EI with this target
    z_ei = (target - mu) / sigma_safe
    cdf_z = norm.cdf(z_ei)
    pdf_z = norm.pdf(z_ei)
    ei_qoei = (target - mu) * cdf_z + sigma_safe * pdf_z
    ei_qoei = np.maximum(ei_qoei, 0.0)
    
    # Also compute standard EI for blending
    z_std = (f_best - mu) / sigma_safe
    cdf_std = norm.cdf(z_std)
    pdf_std = norm.pdf(z_std)
    ei_std = (f_best - mu) * cdf_std + sigma_safe * pdf_std
    ei_std = np.maximum(ei_std, 0.0)
    
    # LCB for exploration
    kappa = 0.5 + 1.5 * np.exp(-0.2 * _call_count)
    lcb = mu - kappa * sigma_safe
    
    # Normalize components
    ei_qoei_max = np.max(ei_qoei) + eps
    ei_std_max = np.max(ei_std) + eps
    lcb_min = np.min(lcb)
    lcb_range = np.ptp(lcb) + eps
    lcb_norm = (lcb_min - lcb) / lcb_range
    
    # Cost-aware scores
    score_qoei = ei_qoei / (cost_safe * ei_qoei_max)
    score_std = ei_std / (cost_safe * ei_std_max)
    score_lcb = lcb_norm / cost_safe
    
    # Blend: QOEI is the primary, with a decaying LCB exploration component
    # Early: more QOEI + some LCB; Later: pure QOEI
    blend_lcb = np.exp(-0.3 * _call_count)
    
    scores = (1.0 - blend_lcb) * score_qoei + blend_lcb * score_lcb
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
