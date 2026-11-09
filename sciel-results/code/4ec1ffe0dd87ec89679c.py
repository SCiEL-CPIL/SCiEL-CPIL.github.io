"""Rewritten program for Erdos minimum overlap task.
Uses a spectral/FFT-based approach combined with local search to minimize
the maximum circular cross-correlation of h with 1-h."""

import numpy as np
import time

N = 100

def _circular_cross_correlation(h, one_minus_h):
    """Compute circular cross-correlation of h with one_minus_h.
    Returns array of length N where result[k] = sum_i h[i] * one_minus_h[(i+k) % N]
    """
    # Use FFT for efficiency
    H = np.fft.fft(h)
    O = np.fft.fft(one_minus_h)
    # Cross-correlation via FFT: corr[k] = sum_i h[i] * one_minus_h[(i+k) % N]
    # This is equivalent to circular convolution of h with reversed one_minus_h
    # Let's use the direct definition via FFT
    # corr[k] = sum_i h[i] * one_minus_h[(i+k) % N]
    # = sum_i h[i] * one_minus_h[(i+k) mod N]
    
    # Using FFT: if we define g[j] = one_minus_h[(-j) % N], then
    # conv[k] = sum_i h[i] * g[k-i] = sum_i h[i] * one_minus_h[(-(k-i)) % N]
    #         = sum_i h[i] * one_minus_h[(i-k) % N]
    # So corr[k] = conv[-k] = conv[(N-k) % N]
    
    # Simpler: just compute directly for N=100 (it's fast enough)
    corr = np.zeros(N)
    for k in range(N):
        corr[k] = np.sum(h * np.roll(one_minus_h, -k))
    return corr

def _max_overlap(h):
    """Compute the maximum overlap: max_k sum_i h[i] * (1-h)[(i+k)%N]"""
    one_minus_h = 1.0 - h
    corr = _circular_cross_correlation(h, one_minus_h)
    return np.max(corr)

def _sum_constraint(h):
    """Check that sum is exactly N/2"""
    return np.sum(h)

def _project_to_simplex(h, target_sum):
    """Project h onto the set {x in [0,1]^N : sum(x) = target_sum}"""
    h = np.clip(h, 0.0, 1.0)
    current_sum = np.sum(h)
    diff = target_sum - current_sum
    
    # If already close, just adjust
    if abs(diff) < 1e-10:
        return h
    
    # Distribute the difference
    if diff > 0:
        # Need to increase sum; add to elements that are < 1
        mask = h < 1.0
        if np.sum(mask) == 0:
            return h
        n_mask = np.sum(mask)
        per_element = diff / n_mask
        h[mask] = np.clip(h[mask] + per_element, 0.0, 1.0)
    else:
        # Need to decrease sum; subtract from elements that are > 0
        mask = h > 0.0
        if np.sum(mask) == 0:
            return h
        n_mask = np.sum(mask)
        per_element = -diff / n_mask
        h[mask] = np.clip(h[mask] - per_element, 0.0, 1.0)
    
    # Fine-tune to get exact sum
    residual = target_sum - np.sum(h)
    if abs(residual) > 1e-12:
        # Add residual to one element that has room
        if residual > 0:
            idx = np.argmin(h)
            h[idx] = min(1.0, h[idx] + residual)
        else:
            idx = np.argmax(h)
            h[idx] = max(0.0, h[idx] + residual)
    
    return h

def construct():
    """Construct a sequence of N step heights in [0,1] with sum N/2
    that minimizes the maximum circular cross-correlation of h with 1-h."""
    
    deadline = time.time() + 120.0
    target_sum = N / 2.0
    
    # Start with a good initial guess: use a pattern that alternates
    # to minimize correlation. A 0-1 pattern with equal zeros and ones
    # arranged to minimize cross-correlation.
    
    # Initial: try several starting points
    best_h = np.full(N, 0.5)
    best_score = _max_overlap(best_h)
    
    # Try a binary initial pattern
    # Pattern: 1,0,1,0,... gives sum = N/2
    h_binary = np.zeros(N)
    h_binary[::2] = 1.0
    h_binary[1::2] = 0.0
    score_binary = _max_overlap(h_binary)
    if score_binary < best_score:
        best_h = h_binary.copy()
        best_score = score_binary
    
    # Try another pattern: 1,1,0,0,1,1,0,0...
    h_pattern2 = np.zeros(N)
    for i in range(N):
        if (i % 4) < 2:
            h_pattern2[i] = 1.0
    score_p2 = _max_overlap(h_pattern2)
    if score_p2 < best_score:
        best_h = h_pattern2.copy()
        best_score = score_p2
    
    # Try pattern: 1,0,0,1,0,0,1,0,0... (period 3, but N=100 not divisible by 3)
    # Let's try a few more patterns
    
    # Use local search (gradient-free, coordinate descent)
    h = best_h.copy()
    h = _project_to_simplex(h, target_sum)
    current_score = _max_overlap(h)
    
    # Coordinate descent / local search
    step_size = 0.1
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() < deadline - 5.0:
        improved = False
        
        for i in range(N):
            if time.time() > deadline - 5.0:
                break
            
            # Try increasing h[i]
            for delta in [step_size, step_size * 0.5, step_size * 0.1]:
                h_new = h.copy()
                # To maintain sum, decrease another element
                # Find element j != i with h[j] > 0 to decrease
                # Or just project after modification
                h_new[i] = min(1.0, h_new[i] + delta)
                h_new = _project_to_simplex(h_new, target_sum)
                new_score = _max_overlap(h_new)
                if new_score < current_score - 1e-10:
                    h = h_new
                    current_score = new_score
                    if current_score < best_score:
                        best_h = h.copy()
                        best_score = current_score
                    improved = True
                    break
            
            if time.time() > deadline - 5.0:
                break
            
            # Try decreasing h[i]
            for delta in [step_size, step_size * 0.5, step_size * 0.1]:
                h_new = h.copy()
                h_new[i] = max(0.0, h_new[i] - delta)
                h_new = _project_to_simplex(h_new, target_sum)
                new_score = _max_overlap(h_new)
                if new_score < current_score - 1e-10:
                    h = h_new
                    current_score = new_score
                    if current_score < best_score:
                        best_h = h.copy()
                        best_score = current_score
                    improved = True
                    break
        
        if not improved:
            no_improve_count += 1
            # Reduce step size
            step_size *= 0.7
            if step_size < 1e-6 or no_improve_count > max_no_improve:
                break
        else:
            no_improve_count = 0
    
    # Final projection to ensure exact sum
    best_h = _project_to_simplex(best_h, target_sum)
    
    return best_h
