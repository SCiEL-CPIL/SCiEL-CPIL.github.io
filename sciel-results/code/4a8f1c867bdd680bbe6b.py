"""Rewritten program for Erdos minimum overlap task."""
import numpy as np
import time

N = 100

# EVOLVE-BLOCK-START

def _overlap_value(h, shifts):
    """Compute maximum overlap of h with shifted 1-h over given shifts.
    
    Overlap for shift s is sum_i h[i] * (1 - h[(i+s) % N])
    We want to minimize the maximum over shifts.
    """
    h = np.asarray(h)
    n = len(h)
    # 1-h
    one_minus_h = 1.0 - h
    max_overlap = 0.0
    for s in shifts:
        shifted = np.roll(one_minus_h, s)
        overlap = np.sum(h * shifted)
        if overlap > max_overlap:
            max_overlap = overlap
    return max_overlap


def _chirp_heights(sweep_rate, N):
    """Generate step heights from squared magnitudes of a complex chirp.
    
    h[i] = |exp(i * pi * sweep_rate * i^2)|^2 ... that's always 1.
    Instead, use a linear chirp in amplitude: 
    Actually, let's think differently. 
    
    A linear chirp signal: x[i] = exp(i * 2*pi * (f0*i + 0.5*rate*i^2))
    |x[i]|^2 = 1 always. So that doesn't work.
    
    Let's use: h[i] = |a_i|^2 where a_i = exp(i * phase_i) * amp_i
    with amp_i chosen so that sum of amp_i^2 = N/2.
    
    Better approach: Use the chirp to define a periodic-like pattern.
    Let h[i] = 0.5 + A * cos(2*pi * (rate * i + 0.5 * rate2 * i^2))
    then normalize so sum = N/2 and clip to [0,1].
    
    Actually, let's use a simpler parametric family:
    h[i] = 0.5 + A * cos(2*pi * rate * i * (i+1) / (2*N))
    This is a quadratic phase (chirp-like) in the cosine argument.
    """
    i = np.arange(N, dtype=float)
    # Quadratic phase: phase = 2*pi * rate * i^2 / (2*N) = pi * rate * i^2 / N
    phase = np.pi * sweep_rate * i * i / N
    # Amplitude modulation to keep in [0,1] after normalization
    raw = 0.5 + np.cos(phase)
    # Normalize so sum = N/2
    s = np.sum(raw)
    if s < 1e-12:
        return np.full(N, 0.5)
    h = raw / s * (N / 2.0)
    # Clip to [0, 1]
    h = np.clip(h, 0.0, 1.0)
    # Re-normalize to ensure exact sum
    s2 = np.sum(h)
    if s2 > 1e-12:
        h = h / s2 * (N / 2.0)
    return h


def _optimize_single_param(sweep_rate, N):
    """Try a given sweep_rate and return the overlap value."""
    h = _chirp_heights(sweep_rate, N)
    # Use a subset of shifts for speed during optimization
    shifts = list(range(1, N))
    return _overlap_value(h, shifts)


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5 second safety margin
    
    best_h = np.full(N, 0.5)
    best_val = _overlap_value(best_h, list(range(1, N)))
    
    # Grid search over sweep_rate
    # The chirp rate determines the frequency of oscillation
    # Try a range of rates
    rates = np.linspace(0.1, 10.0, 200)
    
    for rate in rates:
        if time.time() > deadline:
            break
        h = _chirp_heights(rate, N)
        val = _overlap_value(h, list(range(1, N)))
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    # Refine around the best rate with finer grid
    if time.time() < deadline - 10:
        # Find index of best rate
        best_idx = np.argmin([_overlap_value(_chirp_heights(r, N), list(range(1, N))) for r in rates])
        r_lo = rates[max(0, best_idx - 5)]
        r_hi = rates[min(len(rates) - 1, best_idx + 5)]
        fine_rates = np.linspace(r_lo, r_hi, 100)
        for rate in fine_rates:
            if time.time() > deadline:
                break
            h = _chirp_heights(rate, N)
            val = _overlap_value(h, list(range(1, N)))
            if val < best_val:
                best_val = val
                best_h = h.copy()
    
    # Final polish: try to locally adjust individual elements to reduce max overlap
    if time.time() < deadline - 5:
        # Identify the shift that gives max overlap
        shifts = list(range(1, N))
        h = best_h.copy()
        for _ in range(50):
            if time.time() > deadline:
                break
            # Compute overlap for each shift
            overlaps = np.array([_overlap_value(h, [s]) for s in shifts])
            max_shift = shifts[np.argmax(overlaps)]
            # Try small adjustments to reduce overlap at max_shift
            # overlap(s) = sum_i h[i] * (1 - h[(i+s)%N])
            # To reduce this, we can try to decrease h[i] where h[(i+s)%N] is large, or vice versa
            # Simple approach: gradient-like step
            s = max_shift
            shifted_one_minus_h = np.roll(1.0 - h, s)
            # Gradient of overlap w.r.t. h[i] is (1 - h[(i+s)%N]) + h[(i-s)%N] ... 
            # Actually d/dh[i] of sum_j h[j](1-h[(j+s)%N]) 
            # = (1 - h[(i+s)%N]) - h[(i-s)%N] (from the term where j=i and j such that (j+s)%N=i)
            grad = (1.0 - np.roll(h, s)) - np.roll(h, -s)
            # Move h in direction -grad * step_size, but keep in [0,1] and sum constraint
            step_size = 0.01
            h_new = h - step_size * grad
            h_new = np.clip(h_new, 0.0, 1.0)
            # Renormalize
            s2 = np.sum(h_new)
            if s2 > 1e-12:
                h_new = h_new / s2 * (N / 2.0)
            h_new = np.clip(h_new, 0.0, 1.0)
            val_new = _overlap_value(h_new, shifts)
            if val_new < best_val:
                best_val = val_new
                best_h = h_new.copy()
                h = h_new.copy()
            else:
                # Reduce step size
                step_size *= 0.5
                if step_size < 1e-6:
                    break
    
    # Ensure final result is valid
    h_final = best_h.copy()
    h_final = np.clip(h_final, 0.0, 1.0)
    s_final = np.sum(h_final)
    if s_final > 1e-12:
        h_final = h_final / s_final * (N / 2.0)
    h_final = np.clip(h_final, 0.0, 1.0)
    
    return h_final

# EVOLVE-BLOCK-END
