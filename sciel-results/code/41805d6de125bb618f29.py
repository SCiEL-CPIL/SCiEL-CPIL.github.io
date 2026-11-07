import numpy as np
from scipy.stats import norm
import time

_deadline = None
_start_time = None
_call_count = 0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Kriging Believer acquisition function for Bayesian optimization.
    
    Core idea (Kriging Believer / EI-Believer):
    1. Compute Expected Improvement (EI) for each candidate.
    2. Identify the candidate with the highest EI.
    3. The "believer" selects the point that maximizes the GP mean,
       but constrained to be near the high-EI region. A practical
       single-pass approximation: score = EI/cost, but re-weighted
       by a softmax over EI to concentrate mass on the top EI points,
       then blended with a pure exploitation term (low mu / cost)
       that grows over time.
    
    This differs from plain EI/cost by explicitly concentrating
    probability mass on the top-k EI candidates (softmax temperature
    decays over time), mimicking the believer's "pick the best EI
       point, then trust the GP mean there" logic without a two-pass
       selection.
    
    Parameters:
    mu: (2000,) array of mean predictions
    sigma: (2000,) array of standard deviations
    f_best: float, current best observed value
    cost: (2000,) array of evaluation costs for each candidate
    
    Returns:
    (2000,) array of scores (higher is better)
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
    n = len(mu)
    
    # --- Step 1: Compute Expected Improvement for minimization ---
    z = (f_best - mu) / sigma_safe
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    ei = np.maximum(ei, 0.0)
    
    # --- Step 2: Cost-normalized EI ---
    score_ei_cost = ei / (cost + eps)
    
    # --- Step 3: Kriging Believer concentration via softmax over EI ---
    # Temperature decays: early on, broad exploration; later, concentrate
    # on the top EI candidates (mimicking the believer picking the argmax).
    # Use a numerically stable softmax.
    temperature = 0.1 + 0.9 * np.exp(-0.15 * _call_count)  # 1.0 early -> 0.1 late
    
    # Softmax over score_ei_cost to get a probability-like weight
    # This concentrates mass on high-EI/cost candidates
    max_score = np.max(score_ei_cost)
    # Clip to avoid overflow
    shifted = (score_ei_cost - max_score) / (temperature + eps)
    shifted = np.clip(shifted, -50, 50)
    weights = np.exp(shifted)
    weight_sum = np.sum(weights) + eps
    believer_weights = weights / weight_sum  # sums to 1
    
    # --- Step 4: Exploitation term: low predicted value / cost ---
    # As optimization progresses, shift toward exploiting low-mu points
    # Normalize mu to [0,1] range for scoring
    mu_min = np.min(mu)
    mu_max = np.max(mu)
    mu_range = (mu_max - mu_min) + eps
    mu_norm = (mu_max - mu) / mu_range  # 1 = lowest mu (best), 0 = highest mu
    
    score_exploit = mu_norm / (cost + eps)
    
    # --- Step 5: Blend ---
    # Early: rely on EI/cost (exploration via uncertainty)
    # Late: blend in exploitation (low mu) with believer concentration
    # The believer_weights act as a multiplicative boost on the EI score,
    # concentrating selection on the top EI/cost candidates.
    
    # Adaptive blend weight: starts at 0 (pure EI), grows toward 0.5 (half exploit)
    exploit_weight = 0.5 * (1.0 - np.exp(-0.2 * _call_count))
    
    # Base score: EI/cost boosted by believer concentration
    # Multiply by (1 + concentration_boost) where concentration_boost
    # amplifies top candidates relative to the rest
    concentration_boost = 2.0 * (1.0 - np.exp(-0.1 * _call_count))  # 0 -> 2
    boosted_ei = score_ei_cost * (1.0 + concentration_boost * believer_weights * n)
    
    # Normalize boosted_ei to [0, 1]
    be_max = np.max(boosted_ei) + eps
    be_min = np.min(boosted_ei)
    be_range = (be_max - be_min) + eps
    boosted_ei_norm = (boosted_ei - be_min) / be_range
    
    # Normalize exploit score to [0, 1]
    se_max = np.max(score_exploit) + eps
    se_min = np.min(score_exploit)
    se_range = (se_max - se_min) + eps
    score_exploit_norm = (score_exploit - se_min) / se_range
    
    # Final blend
    scores = (1.0 - exploit_weight) * boosted_ei_norm + exploit_weight * score_exploit_norm
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
