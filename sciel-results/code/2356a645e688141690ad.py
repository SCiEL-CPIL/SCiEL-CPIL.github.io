"""Erdos minimum overlap: piecewise-constant with a single sigmoidal transition."""
import numpy as np
from scipy.optimize import minimize_scalar

N = 100

def _max_overlap(h):
    """Compute the maximum overlap of h with shifted 1-h over all shifts."""
    one_minus_h = 1.0 - h
    max_ov = 0.0
    for s in range(N):
        shifted = np.roll(one_minus_h, s)
        ov = np.dot(h, shifted)
        if ov > max_ov:
            max_ov = ov
    return max_ov

def _construct_h(tau, center):
    """
    Construct h[i] = 0.5 + 0.5 * tanh((i - center) / tau)
    """
    i = np.arange(N, dtype=float)
    if tau < 1e-6:
        tau = 1e-6
    h = 0.5 + 0.5 * np.tanh((i - center) / tau)
    return h

def _solve_center(tau):
    """Find center such that sum(h) = N/2 via binary search."""
    target = N / 2.0
    lo, hi = -10.0, float(N) + 10.0
    
    for _ in range(60):
        mid = (lo + hi) / 2.0
        s = np.sum(_construct_h(tau, mid))
        if s > target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0

def _objective(log_tau):
    """Objective for 1D optimization over log(tau)."""
    tau = np.exp(log_tau)
    if tau < 1e-3:
        tau = 1e-3
    if tau > 500.0:
        tau = 500.0
    
    center = _solve_center(tau)
    h = _construct_h(tau, center)
    
    # Correct sum to exactly N/2
    diff = N / 2.0 - np.sum(h)
    if abs(diff) > 1e-10:
        h = h + diff / N
        h = np.clip(h, 0.0, 1.0)
        # Re-fix if clipping changed sum
        for _ in range(10):
            diff = N / 2.0 - np.sum(h)
            if abs(diff) < 1e-12:
                break
            if diff > 0:
                mask = h < 1.0
                if not np.any(mask):
                    break
                h[mask] = np.minimum(1.0, h[mask] + diff / np.sum(mask))
            else:
                mask = h > 0.0
                if not np.any(mask):
                    break
                h[mask] = np.maximum(0.0, h[mask] + diff / np.sum(mask))
    
    return _max_overlap(h)

def construct():
    """
    Construct the sequence of step heights.
    """
    # Coarse grid search over log(tau)
    log_tau_min = np.log(0.01)
    log_tau_max = np.log(500.0)
    
    # First pass: coarse grid
    log_taus_coarse = np.linspace(log_tau_min, log_tau_max, 200)
    best_val = float('inf')
    best_log_tau = None
    best_h = None
    
    for lt in log_taus_coarse:
        val = _objective(lt)
        if val < best_val:
            best_val = val
            best_log_tau = lt
            best_h = None  # Will recompute at end
    
    # Second pass: fine grid around best
    if best_log_tau is not None:
        lower = max(log_tau_min, best_log_tau - 0.5)
        upper = min(log_tau_max, best_log_tau + 0.5)
        log_taus_fine = np.linspace(lower, upper, 300)
        
        for lt in log_taus_fine:
            val = _objective(lt)
            if val < best_val:
                best_val = val
                best_log_tau = lt
    
    # Third pass: ultra-fine grid
    if best_log_tau is not None:
        lower = max(log_tau_min, best_log_tau - 0.1)
        upper = min(log_tau_max, best_log_tau + 0.1)
        log_taus_ultra = np.linspace(lower, upper, 100)
        
        for lt in log_taus_ultra:
            val = _objective(lt)
            if val < best_val:
                best_val = val
                best_log_tau = lt
    
    # Final construction
    tau = np.exp(best_log_tau)
    center = _solve_center(tau)
    h = _construct_h(tau, center)
    
    # Correct sum to exactly N/2
    diff = N / 2.0 - np.sum(h)
    if abs(diff) > 1e-10:
        h = h + diff / N
        h = np.clip(h, 0.0, 1.0)
        for _ in range(20):
            diff = N / 2.0 - np.sum(h)
            if abs(diff) < 1e-12:
                break
            if diff > 0:
                mask = h < 1.0
                if not np.any(mask):
                    break
                h[mask] = np.minimum(1.0, h[mask] + diff / np.sum(mask))
            else:
                mask = h > 0.0
                if not np.any(mask):
                    break
                h[mask] = np.maximum(0.0, h[mask] + diff / np.sum(mask))
    
    return h
