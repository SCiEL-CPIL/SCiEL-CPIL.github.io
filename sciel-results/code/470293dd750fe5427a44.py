"""Rewritten program: optimize step heights to minimize max overlap of h with shifted 1-h."""
import numpy as np
from scipy.optimize import minimize
import time

N = 100
DEADLINE = 140  # seconds, safety margin before 150s kill


def _max_overlap(h):
    """Compute the maximum overlap of h with shifted (1-h) over all shifts."""
    h = np.asarray(h, dtype=np.float64)
    one_minus_h = 1.0 - h
    max_ov = 0.0
    for k in range(N):
        shifted = np.roll(h, k)
        ov = np.dot(shifted, one_minus_h)
        if ov > max_ov:
            max_ov = ov
    return max_ov


def _objective(h):
    """Objective: maximize the minimum slack (i.e., minimize max overlap).
    We minimize max_overlap directly."""
    return _max_overlap(h)


def _objective_smooth(h):
    """Smooth approximation using log-sum-exp for gradient-based methods."""
    h = np.asarray(h, dtype=np.float64)
    one_minus_h = 1.0 - h
    overlaps = np.zeros(N)
    for k in range(N):
        shifted = np.roll(h, k)
        overlaps[k] = np.dot(shifted, one_minus_h)
    # log-sum-exp for smooth max
    m = np.max(overlaps)
    if m < 1e-10:
        return 0.0
    return m + np.log(np.sum(np.exp(overlaps - m))) / N


def construct():
    t0 = time.time()
    
    # Start with uniform
    h0 = np.full(N, 0.5)
    best_h = h0.copy()
    best_val = _max_overlap(h0)
    
    # Try SLSQP to minimize max overlap
    try:
        # Use the smooth objective for gradient-based optimization
        res = minimize(
            _objective_smooth,
            h0,
            method='L-BFGS-B',
            bounds=[(0.0, 1.0)] * N,
            constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - N / 2.0,
                          'jac': lambda x: np.ones(N)}],
            options={'maxiter': 500, 'ftol': 1e-12}
        )
        elapsed = time.time() - t0
        if elapsed < DEADLINE:
            h_candidate = res.x
            # Project to valid range and fix sum
            h_candidate = np.clip(h_candidate, 0.0, 1.0)
            # Adjust to ensure sum is exactly N/2
            diff = N / 2.0 - np.sum(h_candidate)
            if abs(diff) > 1e-10:
                # Distribute adjustment
                h_candidate += diff / N
                h_candidate = np.clip(h_candidate, 0.0, 1.0)
                # Re-fix sum
                diff2 = N / 2.0 - np.sum(h_candidate)
                h_candidate += diff2 / N
            
            val = _max_overlap(h_candidate)
            if val < best_val:
                best_h = h_candidate.copy()
                best_val = val
    except Exception:
        pass
    
    elapsed = time.time() - t0
    if elapsed < DEADLINE - 5:
        # Try SLSQP with direct objective
        try:
            res2 = minimize(
                _objective,
                h0,
                method='SLSQP',
                bounds=[(0.0, 1.0)] * N,
                constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - N / 2.0,
                              'jac': lambda x: np.ones(N)}],
                options={'maxiter': 200, 'ftol': 1e-10}
            )
            h_candidate = res2.x
            h_candidate = np.clip(h_candidate, 0.0, 1.0)
            diff = N / 2.0 - np.sum(h_candidate)
            if abs(diff) > 1e-10:
                h_candidate += diff / N
                h_candidate = np.clip(h_candidate, 0.0, 1.0)
                diff2 = N / 2.0 - np.sum(h_candidate)
                h_candidate += diff2 / N
            
            val = _max_overlap(h_candidate)
            if val < best_val:
                best_h = h_candidate.copy()
                best_val = val
        except Exception:
            pass
    
    # Ensure exact sum
    final_h = best_h.copy()
    final_h = np.clip(final_h, 0.0, 1.0)
    diff = N / 2.0 - np.sum(final_h)
    if abs(diff) > 1e-12:
        # Add to elements with room
        for i in range(N):
            if diff > 0 and final_h[i] < 1.0:
                add = min(diff, 1.0 - final_h[i])
                final_h[i] += add
                diff -= add
            elif diff < 0 and final_h[i] > 0.0:
                sub = min(-diff, final_h[i])
                final_h[i] -= sub
                diff += sub
            if abs(diff) < 1e-12:
                break
    
    return final_h
