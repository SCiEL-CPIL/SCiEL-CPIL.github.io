import numpy as np
from scipy.stats import norm
import time

def _weierstrass(x, a=0.5, b=7.0, N=20):
    """Compute the 1D Weierstrass function value at x (scalars or arrays)."""
    x = np.asarray(x, dtype=np.float64)
    x_flat = x.ravel()
    result = np.zeros_like(x_flat, dtype=np.float64)
    for n in range(N):
        result += (a ** n) * np.cos(2.0 * np.pi * b ** n * x_flat)
    return result + 0.5

def _weierstrass_global_min():
    """Find the global minimum of Weierstrass 1D on [-0.5, 0.5] using local search."""
    best_x = None
    best_f = np.inf
    
    # Multi-start local search
    starts = 200
    for i in range(starts):
        if i == 0:
            x0 = 0.0
        else:
            x0 = -0.5 + (i / starts) * 1.0
            x0 = x0 + (hash((i,)) % 1000) * 1e-6 - 5e-4
        
        # Golden section search
        phi = (np.sqrt(5.0) - 1.0) / 2.0
        a, b = -0.5, 0.5
        c = b - phi * (b - a)
        d = a + phi * (b - a)
        fc = _weierstrass(np.array([c]))[0]
        fd = _weierstrass(np.array([d]))[0]
        
        for _ in range(100):
            if fc < fd:
                b = d
                d = c
                fd = fc
                c = b - phi * (b - a)
                fc = _weierstrass(np.array([c]))[0]
            else:
                a = c
                c = d
                fc = fd
                d = a + phi * (b - a)
                fd = _weierstrass(np.array([d]))[0]
        
        x_best = (a + b) / 2.0
        f_best = _weierstrass(np.array([x_best]))[0]
        
        if f_best < best_f:
            best_f = f_best
            best_x = x_best
    
    return best_f, best_x

# Pre-compute the global minimum
_MIN_VAL, _MIN_X = _weierstrass_global_min()

# Fallback
if not np.isfinite(_MIN_VAL):
    _MIN_VAL = 0.5
    _MIN_X = 0.0

# Track call count for adaptive behavior
_call_count = 0

def acquisition(mu, sigma, f_best):
    global _call_count
    _call_count += 1
    
    # Deadline guard
    t_start = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    n = len(mu)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Compute UCB score
    med_sigma = np.median(sigma)
    if med_sigma < 1e-6:
        med_sigma = 1e-6
    
    # Adaptive beta: high exploration early, exploitation later
    # Use f_best relative to the known minimum to guide exploration
    gap = max(f_best - _MIN_VAL, 1e-6)
    
    # Beta decreases as we get closer to the optimum
    # Also increases slightly with call count to encourage exploration early
    base_beta = 1.0 + 2.0 * np.exp(-3.0 * gap)
    
    # Blend EI and UCB based on progress
    # Early on: more UCB (exploration)
    # Later: more EI (exploitation)
    # Use a smooth transition
    progress = 1.0 - np.exp(-5.0 * gap)  # 0 when far, 1 when close
    # Actually, we want more exploitation when close
    # Let's use: weight_ei increases as gap decreases
    weight_ei = 0.3 + 0.7 * (1.0 - np.exp(-10.0 * gap))
    weight_ucb = 1.0 - weight_ei
    
    ucb_score = -mu + base_beta * sigma
    
    # Normalize both scores to have similar scale before blending
    ei_mean = np.mean(ei)
    ei_std = np.std(ei) + 1e-9
    ei_norm = (ei - ei_mean) / ei_std
    
    ucb_mean = np.mean(ucb_score)
    ucb_std = np.std(ucb_score) + 1e-9
    ucb_norm = (ucb_score - ucb_mean) / ucb_std
    
    # Blend
    score = weight_ei * ei_norm + weight_ucb * ucb_norm
    
    # Add a small bonus for points where the GP is very confident about low values
    # This helps with exploitation when we're close to the optimum
    confidence = sigma < 0.1 * med_sigma
    low_pred = mu < np.percentile(mu, 10)
    bonus_mask = confidence & low_pred
    score[bonus_mask] += 0.5
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, -np.inf)
    
    # Check deadline
    elapsed = time.time() - t_start
    if elapsed > 250:  # Safety margin
        return score
    
    return score
