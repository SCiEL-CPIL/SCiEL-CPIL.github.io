# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global state to persist across calls within the same seed run
_acq_state = {
    'call_count': 0,
    'last_f_best': None,
    'stagnation_count': 0,
    'rng': np.random.default_rng(42)
}

def acquisition(mu, sigma, f_best):
    t_start = time.time()
    deadline = t_start + 295.0  # Safety margin for 300s limit
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    f_best = float(f_best)
    n = len(mu)
    
    if n == 0:
        return np.array([], dtype=np.float64)
    
    # Ensure sigma is safe for division
    sigma_safe = np.where(sigma > 1e-10, sigma, 1e-10)
    
    # Update state
    _acq_state['call_count'] += 1
    calls = _acq_state['call_count']
    
    prev_f_best = _acq_state.get('last_f_best', None)
    improved = False
    if prev_f_best is not None:
        if f_best < prev_f_best - 1e-6:
            improved = True
            
    if improved:
        _acq_state['stagnation_count'] = 0
    else:
        _acq_state['stagnation_count'] += 1
        
    _acq_state['last_f_best'] = f_best
    stagnation = _acq_state['stagnation_count']
    
    # GPDR Strategy:
    # We model the posterior probability that f(x) < f_best.
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    # This is the "Success Probability".
    
    # To create a density ratio, we compare the likelihood of observing a value 
    # close to f_best (or better) under the posterior vs a baseline.
    # A common formulation for GPDR (Gaussian Process Density Ratio) in BO
    # uses the ratio of the posterior density to the prior (or a reference) 
    # evaluated at the threshold.
    
    # However, a more direct and robust "classification" view is to use 
    # the log-likelihood of the class "Improvement".
    # Let's use a variant that balances the probability of improvement 
    # with the "sharpness" of the posterior (uncertainty).
    
    # Standard EI is: (f_best - mu) * Phi(z) + sigma * phi(z)
    # This is E[max(0, f_best - f)].
    
    # GPDR-inspired score:
    # We want to maximize the probability that x is the argmin, but we don't have 
    # the full joint. Instead, we approximate by maximizing the likelihood 
    # that f(x) is significantly better than f_best.
    
    # Let's use the "Probability of Improvement" (PI) as a base, but modulated 
    # by the density of the posterior at the threshold.
    # Score = P(f < f_best) * pdf(f_best; mu, sigma)
    # This peaks when the posterior is centered near f_best AND has low variance.
    # This encourages exploitation of regions where we are confident we can beat f_best.
    
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement
    pi = norm.cdf(z)
    
    # Density at the threshold f_best
    # This term penalizes high uncertainty (low density) and rewards 
    # means that are close to f_best (high density at the threshold).
    # Actually, pdf(f_best) is max when mu = f_best.
    # If mu < f_best (already better), pdf might be lower if sigma is large, 
    # but pi is high.
    
    pdf_at_threshold = norm.pdf(z)
    
    # The product pi * pdf_at_threshold is a measure of "how likely is it that 
    # the true value is just below f_best?"
    # This is different from EI which integrates the tail.
    # This "Peak" strategy is very exploitative.
    
    # To allow exploration, we add a small exploration term based on entropy 
    # or simply sigma, but weighted by stagnation.
    
    # Base GPDR Score
    score_gpd = pi * pdf_at_threshold
    
    # Exploration Score: High uncertainty
    score_explore = sigma_safe
    
    # Adaptive Weighting:
    # If we are improving (stagnation == 0), we trust the GPDR exploitation.
    # If we are stagnating, we need to explore.
    
    if stagnation == 0:
        w_exploit = 0.9
    elif stagnation == 1:
        w_exploit = 0.7
    elif stagnation == 2:
        w_exploit = 0.4
    else:
        w_exploit = 0.1
        
    w_explore = 1.0 - w_exploit
    
    # Normalize scores to [0, 1] to make them comparable
    def normalize(s):
        s_min = np.min(s)
        s_max = np.max(s)
        if s_max - s_min < 1e-9:
            return np.zeros_like(s)
        return (s - s_min) / (s_max - s_min)
    
    s_gpd_norm = normalize(score_gpd)
    s_explore_norm = normalize(score_explore)
    
    final_scores = w_exploit * s_gpd_norm + w_explore * s_explore_norm
    
    # Ensure finite
    final_scores = np.nan_to_num(final_scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    if time.time() > deadline:
        pass
        
    return final_scores

# EVOLVE-BLOCK-END
