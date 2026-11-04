"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np

def _compute_overlap_value(h, N):
    """
    Computes the raw score: max over circular shifts t of the integral of h(x)(1-h(x+t)).
    Discretized as (2/N) * sum_i h[i] * (1-h[(i+t)%N]).
    """
    h = np.asarray(h, dtype=np.float64)
    comp = 1.0 - h
    # Circular cross-correlation: sum_j h[j] * comp[j+t]
    H = np.fft.fft(h)
    C = np.fft.fft(comp)
    # ifft(conj(H) * C) gives sum_j h[j] * comp[j+t]
    overlap_sums = np.real(np.fft.ifft(np.conj(H) * C))
    # The integral value for shift t is (2/N) * overlap_sums[t]
    # We want the maximum of these values
    max_sum = np.max(overlap_sums)
    return (2.0 / N) * max_sum

def construct():
    N = 100
    # We search over a parametric family of functions.
    # h(x) = 0.5 + A * cos(pi * x) + B * cos(2 * pi * x)
    # Domain x in [0, 2].
    # We sample N points.
    x = np.linspace(0, 2, N, endpoint=False)
    
    best_h = np.full(N, 0.5)
    best_val = _compute_overlap_value(best_h, N)
    
    # Grid search over A and B
    # A controls the fundamental mode, B the second harmonic.
    # Constraints: h in [0, 1] and sum(h) = N/2.
    # sum(h) = N * 0.5 + A * sum(cos) + B * sum(cos2)
    # sum(cos(pi * x_i)) over N points in [0,2) is approx 0.
    # sum(cos(2*pi * x_i)) is approx 0.
    # So sum is approx N/2. We will project to ensure exact sum.
    
    def project(h):
        h = np.clip(h, 0.0, 1.0)
        # Adjust sum to N/2
        # We need to be careful not to break [0,1] too much.
        diff = N / 2.0 - np.sum(h)
        # Add diff/N to all elements
        h += diff / N
        # Re-clip
        h = np.clip(h, 0.0, 1.0)
        # Iterate a few times
        for _ in range(10):
            diff = N / 2.0 - np.sum(h)
            h += diff / N
            h = np.clip(h, 0.0, 1.0)
        return h

    # Search space
    A_vals = np.linspace(-0.4, 0.4, 17)
    B_vals = np.linspace(-0.2, 0.2, 9)
    
    for A in A_vals:
        for B in B_vals:
            h_raw = 0.5 + A * np.cos(np.pi * x) + B * np.cos(2 * np.pi * x)
            h_proj = project(h_raw)
            val = _compute_overlap_value(h_proj, N)
            if val < best_val:
                best_val = val
                best_h = h_proj

    return best_h
# EVOLVE-BLOCK-END
