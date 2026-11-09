import numpy as np
from scipy.stats import norm, t as student_t
import time

_start_time = None
DEADLINE = 300.0
SAFETY = 20.0

def _elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.perf_counter()
    return time.perf_counter() - _start_time

def _ei_gaussian(mu, s, f_best, xi=0.0):
    z = (f_best - mu - xi) / s
    return (f_best - mu - xi) * norm.cdf(z) + s * norm.pdf(z)

def _ei_student_t(mu, s, f_best, df=5.0, xi=0.0):
    """Student-t based EI for heavier-tailed exploration."""
    s_safe = np.where(s > 1e-9, s, 1e-9)
    # Scale sigma to match t-distribution variance
    # t(df) has variance df/(df-2) for df>2
    scale_factor = np.sqrt(df / (df - 2.0)) if df > 2 else 1.0
    s_t = s_safe * scale_factor
    
    z = (f_best - mu - xi) / s_t
    cdf = student_t.cdf(z, df)
    pdf = student_t.pdf(z, df)
    
    # t-EI formula
    ei = (f_best - mu - xi) * cdf + s_t * pdf
    return ei

def acquisition(mu, sigma, f_best):
    global _start_time
    if _start_time is None:
        _start_time = time.perf_counter()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    n = mu.shape[0]
    
    s = np.where(sigma > 1e-9, sigma, 1e-9)
    
    # Deadline guard: return simple EI if close to deadline
    if _elapsed() > DEADLINE - SAFETY:
        ei0 = _ei_gaussian(mu, s, f_best, 0.0)
        ei0 = np.where(np.isfinite(ei0), ei0, 0.0)
        return ei0
    
    # Compute base EI (Gaussian) for reference
    ei_gauss = _ei_gaussian(mu, s, f_best, 0.0)
    ei_gauss = np.where(np.isfinite(ei_gauss), ei_gauss, 0.0)
    
    # Student-t based acquisition with different degrees of freedom
    # Lower df = heavier tails = more exploration
    df_candidates = [3.0, 5.0, 10.0]
    
    best_scores = None
    best_obj = -np.inf
    
    for df in df_candidates:
        # Check deadline
        if _elapsed() > DEADLINE - SAFETY:
            break
        
        # Compute t-EI
        ei_t = _ei_student_t(mu, s, f_best, df=df, xi=0.0)
        ei_t = np.where(np.isfinite(ei_t), ei_t, 0.0)
        
        # Normalize to comparable scale with Gaussian EI
        t_range = float(np.max(ei_t) - np.min(ei_t)) + 1e-9
        g_range = float(np.max(ei_gauss) - np.min(ei_gauss)) + 1e-9
        
        # Blend: use t-EI as primary, scaled to match Gaussian EI range
        # This preserves the relative ordering while ensuring comparable magnitudes
        scores = ei_t * (g_range / t_range)
        
        # Add a small exploration bonus for high uncertainty points
        # that are likely to beat f_best (KG-inspired bonus)
        # P(new_best) approximated by CDF of standardized improvement
        z_prob = (f_best - mu) / s
        p_new_best = norm.cdf(z_prob)
        bonus = 0.1 * s * p_new_best
        
        scores = scores + bonus
        
        scores = np.where(np.isfinite(scores), scores, 0.0)
        
        # Evaluate: use the point with highest score, measure by Gaussian EI
        idx = int(np.argmax(scores))
        obj = ei_gauss[idx]
        
        if obj > best_obj:
            best_obj = obj
            best_scores = scores
    
    if best_scores is None:
        best_scores = ei_gauss
    
    return best_scores
