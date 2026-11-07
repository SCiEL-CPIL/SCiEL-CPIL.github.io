import numpy as np
from scipy.stats import norm
import time

_acq_state = {'start_time': None, 'call_count': 0}

def acquisition(mu, sigma, f_best, cost):
    if _acq_state['start_time'] is None:
        _acq_state['start_time'] = time.time()
        _acq_state['call_count'] = 0
    _acq_state['call_count'] += 1
    
    elapsed = time.time() - _acq_state['start_time']
    if elapsed > 325.0:
        return np.zeros(len(mu), dtype=np.float64)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.5)
    
    sigma = np.maximum(sigma, 1e-8)
    cost = np.maximum(cost, 1e-8)
    
    # Phase
    call_idx = min(_acq_state['call_count'], 50)
    phase = call_idx / 50.0
    
    # EI
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    ei = (f_best - mu) * Phi_z + sigma * phi_z
    ei = np.maximum(ei, 0.0)
    
    # LCB (for minimization, lower is better, so we want high score for low mu - beta*sigma)
    # Score contribution: -(mu - beta*sigma) = -mu + beta*sigma
    beta = 2.0 - 1.0 * phase  # 2.0 -> 1.0
    
    lcb_score = -mu + beta * sigma
    
    # Cost scaling
    # Use a slightly sharper cost penalty to avoid corners
    # cost is in [1.1, 1.9]
    # 1.5/cost is linear inverse.
    # Let's try exp penalty: exp(-1.0 * (cost - 1.1))
    # At 1.1: 1.0
    # At 1.9: exp(-0.8) = 0.45
    # This is a stronger penalty for high cost.
    cost_penalty = np.exp(-1.0 * (cost - 1.1))
    
    # Combine
    # Weight for EI vs LCB
    # Early: more LCB (exploration)
    # Late: more EI (exploitation)
    w_ei = 0.5 + 0.5 * phase  # 0.5 -> 1.0
    w_lcb = 1.0 - w_ei        # 0.5 -> 0.0
    
    # Scale EI and LCB by cost penalty
    # EI is in objective units. LCB is in objective units.
    # We divide by cost to normalize.
    
    score_ei = ei * cost_penalty
    score_lcb = lcb_score * cost_penalty
    
    scores = w_ei * score_ei + w_lcb * score_lcb
    
    # Add a small exploration bonus for high sigma, also cost weighted
    # This ensures we don't completely ignore uncertain regions if they are cheap
    exploration_bonus = 0.1 * sigma * cost_penalty * (1 - phase)
    scores = scores + exploration_bonus
    
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
