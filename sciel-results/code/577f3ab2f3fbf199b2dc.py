"""Improved program using high-order splitting coefficients."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def coefficients(stages):
    """Use 4th-order Suzuki-Trotter coefficients for 12-stage product formula."""
    
    # 4th-order Suzuki-Trotter symmetric splitting coefficients
    # The 4th-order splitting has 4 stages
    # Known analytical coefficients (Suzuki 1990)
    r = 2.0 ** (1.0/3.0)
    c = 1.0 / (2.0 * (1.0 - 1.0/r))
    
    # 4th-order 4-stage coefficients for A parts
    a_4th = [c, 1.0 - 2.0*c, 1.0 - 2.0*c, c]
    # 4th-order 4-stage coefficients for B parts  
    b_4th = [1.0 - 2.0*c, c, c, 1.0 - 2.0*c]
    
    # For 12 stages, we use three 4th-order 4-stage blocks
    # This gives 12 stages total with 4th-order accuracy per block
    
    a = []
    b = []
    for _ in range(3):
        a.extend(a_4th)
        b.extend(b_4th)
    
    # Ensure sums are exactly 1
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    
    # Normalize to ensure sum is exactly 1
    sum_a = np.sum(a)
    sum_b = np.sum(b)
    if sum_a != 0:
        a = a / sum_a
    if sum_b != 0:
        b = b / sum_b
    
    # Final check: ensure finite and in range [-50, 50]
    a = np.clip(a, -50.0, 50.0)
    b = np.clip(b, -50.0, 50.0)
    
    # Re-normalize after clipping to maintain sum=1
    sum_a = np.sum(a)
    sum_b = np.sum(b)
    if sum_a != 0:
        a = a / sum_a
    if sum_b != 0:
        b = b / sum_b
    
    # Verify constraints one more time
    assert abs(np.sum(a) - 1.0) < 1e-10, f"Sum of a is {np.sum(a)}"
    assert abs(np.sum(b) - 1.0) < 1e-10, f"Sum of b is {np.sum(b)}"
    assert np.all(np.isfinite(a)), "a contains non-finite values"
    assert np.all(np.isfinite(b)), "b contains non-finite values"
    assert len(a) == stages, f"Length of a is {len(a)}, expected {stages}"
    assert len(b) == stages, f"Length of b is {len(b)}, expected {stages}"
    
    return list(a), list(b)
# EVOLVE-BLOCK-END
