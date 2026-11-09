"""Spectral optimization approach for Erdos minimum overlap."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    # Represent step heights as baseline + sinusoidal perturbation
    # h_i = 0.5 + A * cos(omega * i + phase)
    # where A = 0.5 to stay within [0, 1]
    # Choose omega to minimize the maximum overlap
    
    A = 0.5
    
    # The overlap at shift k is:
    # M(k) = sum_i h_i * h_{i+k}
    # With h_i = 0.5 + A*cos(omega*i), the cross term involves:
    # sum_i [0.5 + A*cos(omega*i)] * [0.5 + A*cos(omega*(i+k))]
    # = N*0.25 + 0.5*A*sum_i[cos(omega*i) + cos(omega*(i+k))] + A^2*sum_i[cos(omega*i)*cos(omega*(i+k))]
    
    # Using cos(a)*cos(b) = 0.5*[cos(a-b) + cos(a+b)]:
    # sum_i cos(omega*i)*cos(omega*(i+k)) = 0.5*sum_i[cos(omega*k) + cos(omega*(2i+k))]
    # The second term oscillates and averages out for large N
    
    # So M(k) ≈ N*0.25 + 0.5*A*A*cos(omega*k) + (oscillatory terms that average to 0)
    # = N/4 + A^2/2 * cos(omega*k)
    
    # Wait, let me recalculate. The mean overlap should be N/4 for constant 0.5.
    # For h_i = 0.5 + A*cos(omega*i):
    # E[h_i] = 0.5, so E[h_i * h_{i+k}] ≈ 0.25 + A^2/2 * cos(omega*k)
    
    # The maximum of cos(omega*k) is 1, so max M(k) ≈ N/4 + A^2/2
    # The minimum of cos(omega*k) is -1, so min M(k) ≈ N/4 - A^2/2
    
    # To minimize the MAXIMUM, we want to make the oscillation as small as possible
    # but we're forced to have A = 0.5 for the sum constraint
    
    # Actually, the sum constraint is sum h_i = N/2, which is satisfied since E[h_i] = 0.5
    
    # The key insight: we want to choose omega such that the autocorrelation
    # has its peak minimized. For a single sinusoid, the autocorrelation is
    # proportional to cos(omega*k), so the peak is always at k=0 with value A^2/2
    
    # But wait - the problem asks for maximum OVERLAP, which might mean the maximum
    # of the actual sum, not just the correlation coefficient
    
    # Let me reconsider: if we use a high-frequency sinusoid, the cross terms
    # sum_i cos(omega*i) and sum_i cos(omega*(i+k)) will be small for k not near 0
    
    # For a single period in N samples: omega = 2*pi/N
    # Then sum_i cos(2*pi*i/N) ≈ 0 (geometric series)
    
    # So M(k) ≈ N/4 + A^2/2 * cos(2*pi*k/N)
    # Max at k=0: N/4 + A^2/2 = 25 + 0.125 = 25.125
    # This is much better than the baseline N/2 = 50!
    
    omega = 2.0 * np.pi / N
    i = np.arange(N)
    h = 0.5 + A * np.cos(omega * i)
    
    # Ensure h is in [0, 1] and sum is exactly N/2
    # Due to floating point, adjust to make sum exactly N/2
    current_sum = np.sum(h)
    h += (N / 2.0 - current_sum) / N
    
    # Clip to [0, 1] to ensure validity
    h = np.clip(h, 0.0, 1.0)
    
    # Final adjustment to ensure exact sum
    current_sum = np.sum(h)
    diff = N / 2.0 - current_sum
    if abs(diff) > 1e-15:
        # Distribute the difference
        h += diff / N
        h = np.clip(h, 0.0, 1.0)
    
    return h
# EVOLVE-BLOCK-END
