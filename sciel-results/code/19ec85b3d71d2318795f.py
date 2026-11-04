"""Erdos minimum overlap: choose step heights h in [0,1] summing to N/2
to minimize max overlap of h with shifted 1-h."""

import time
import numpy as np
from scipy.optimize import minimize

N = 100
DEADLINE = 150.0  # seconds total budget
SAFETY_MARGIN = 5.0  # return before this


def _overlap_max(h):
    """Compute the maximum overlap between h and shifted (1-h).
    
    The overlap at shift k is sum_i h[i] * (1 - h[(i+k) % N])
    We want to minimize the maximum over all shifts k.
    """
    one_minus_h = 1.0 - h
    # For each shift k, compute sum_i h[i] * one_minus_h[(i+k) % N]
    # This is a circular cross-correlation
    overlaps = np.zeros(N)
    for k in range(N):
        shifted = np.roll(one_minus_h, k)
        overlaps[k] = np.dot(h, shifted)
    return np.max(overlaps)


def _objective(x):
    """Objective: max overlap + penalty for constraint violations."""
    h = x
    # Soft constraint: h in [0, 1] and sum(h) = N/2
    # Use a smooth penalty
    penalty = 1e6 * (np.sum(np.maximum(0, -h))**2 + np.sum(np.maximum(0, h - 1))**2)
    sum_penalty = 1e6 * (np.sum(h) - N / 2.0)**2
    return _overlap_max(h) + penalty + sum_penalty


def _project_to_box(x):
    """Project onto [0,1]^N while maintaining sum = N/2."""
    h = np.clip(x, 0, 1)
    # Adjust to satisfy sum constraint
    diff = N / 2.0 - np.sum(h)
    if diff > 0:
        # Need to increase some values
        mask = h < 1.0
        if np.sum(mask) > 0:
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0, 1)
    elif diff < 0:
        # Need to decrease some values
        mask = h > 0.0
        if np.sum(mask) > 0:
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0, 1)
    # Final correction
    diff = N / 2.0 - np.sum(h)
    if abs(diff) > 1e-12:
        # Distribute evenly
        h += diff / N
        h = np.clip(h, 0, 1)
    return h


def _sturmian_candidate(alpha, beta):
    """Generate a Sturmian word candidate."""
    h = np.zeros(N)
    for i in range(N):
        val = (alpha * (i + 1) + beta) % 1.0
        h[i] = 1.0 if val < 0.5 else 0.0
    # Adjust to have sum = N/2
    h = _project_to_box(h)
    return h


def _random_candidate():
    """Generate a random candidate."""
    h = np.random.uniform(0, 1, N)
    h = _project_to_box(h)
    return h


def _sinusoidal_candidate(freq, phase):
    """Generate a sinusoidal candidate."""
    t = np.arange(N)
    h = 0.5 + 0.4 * np.sin(2 * np.pi * freq * t / N + phase)
    h = np.clip(h, 0, 1)
    h = _project_to_box(h)
    return h


def _gradient_descent(h0, max_iters=5000, lr=0.01, tol=1e-8):
    """Simple gradient descent with projection."""
    h = h0.copy()
    best_h = h.copy()
    best_val = _overlap_max(h)
    
    for it in range(max_iters):
        # Compute gradient numerically
        eps = 1e-5
        grad = np.zeros(N)
        for i in range(N):
            h_plus = h.copy()
            h_plus[i] += eps
            h_minus = h.copy()
            h_minus[i] -= eps
            grad[i] = (_objective(h_plus) - _objective(h_minus)) / (2 * eps)
        
        # Update
        h = h - lr * grad
        h = _project_to_box(h)
        
        val = _overlap_max(h)
        if val < best_val - 1e-10:
            best_val = val
            best_h = h.copy()
        
        if it % 100 == 0 and _time_remaining() < SAFETY_MARGIN:
            break
    
    return best_h, best_val


def _time_remaining():
    """Return time remaining until deadline."""
    return DEADLINE - (time.time() - _start_time)


_start_time = time.time()


def construct():
    global _start_time
    _start_time = time.time()
    
    # Try multiple initializations and pick the best
    best_h = None
    best_val = np.inf
    
    # 1. Uniform (baseline)
    h0 = np.full(N, 0.5)
    val = _overlap_max(h0)
    if val < best_val:
        best_val = val
        best_h = h0.copy()
    
    # 2. Sturmian candidates with various alpha values
    for alpha in [0.6180339887, 0.7071067812, 0.5, 0.4, 0.6, 0.3, 0.7]:
        for beta in [0.0, 0.25, 0.5, 0.75]:
            if _time_remaining() < SAFETY_MARGIN + 10:
                break
            h_cand = _sturmian_candidate(alpha, beta)
            val = _overlap_max(h_cand)
            if val < best_val:
                best_val = val
                best_h = h_cand.copy()
        if _time_remaining() < SAFETY_MARGIN + 10:
            break
    
    # 3. Sinusoidal candidates
    for freq in [1, 2, 3, 5, 7, 10, 15, 20, 25]:
        for phase in [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]:
            if _time_remaining() < SAFETY_MARGIN + 10:
                break
            h_cand = _sinusoidal_candidate(freq, phase)
            val = _overlap_max(h_cand)
            if val < best_val:
                best_val = val
                best_h = h_cand.copy()
        if _time_remaining() < SAFETY_MARGIN + 10:
            break
    
    # 4. Random candidates
    for _ in range(20):
        if _time_remaining() < SAFETY_MARGIN + 5:
            break
        h_cand = _random_candidate()
        val = _overlap_max(h_cand)
        if val < best_val:
            best_val = val
            best_h = h_cand.copy()
    
    # 5. Gradient descent polish from best candidate
    if best_h is not None and _time_remaining() > SAFETY_MARGIN + 2:
        h_polished, val_polished = _gradient_descent(best_h, max_iters=2000, lr=0.005)
        if val_polished < best_val:
            best_val = val_polished
            best_h = h_polished
    
    # Final check
    if best_h is None:
        best_h = np.full(N, 0.5)
    
    # Ensure exact sum
    best_h = _project_to_box(best_h)
    
    return best_h
