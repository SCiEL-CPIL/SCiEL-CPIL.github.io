"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def coefficients(stages):
    """
    12-stage symmetric high-order product formula for Heisenberg-chain time evolution.
    
    Uses nested 4th-order Suzuki-Trotter coefficients to minimize leading-order
    error terms via commutator algebra. The 4th-order 3-stage coefficients are
    composed 4 times to yield 12 stages total, preserving palindromic symmetry.
    
    The 4th-order coefficients (for exp(t(A+B)) ≈ prod exp(x_k B) exp(x_k A)):
      x1 = 1 / (2 - 2^(1/3))
      x2 = -2^(1/3) / (2 - 2^(1/3))
      x3 = 1 - x1 - x2
    
    Each block is scaled by 1/4 in time, so the full product has sum(a)=sum(b)=1.
    """
    deadline = time.time() + 50  # safety margin before 60s CPU budget
    
    if stages != 12:
        # Fallback to simple uniform splitting for non-12 stage counts
        substeps = max(1, stages // 2)
        a = [0.5 / substeps] * (2 * substeps)
        b = []
        for _ in range(substeps):
            b += [1.0 / substeps, 0.0]
        while len(a) < stages:
            a.append(0.0)
            b.append(0.0)
        return a, b
    
    # 4th-order 3-stage symmetric coefficients
    two_thirds = 2.0 ** (1.0 / 3.0)
    denom = 2.0 - two_thirds
    x1 = 1.0 / denom
    x2 = -two_thirds / denom
    x3 = 1.0 - x1 - x2
    
    # 4th-order block: (b_k, a_k) pairs for k=1,2,3
    block_b = [x1, x2, x3]
    block_a = [x1, x2, x3]
    
    # Compose 4 blocks, each scaled by 1/4
    scale = 1.0 / 4.0
    a = []
    b = []
    for _ in range(4):
        for k in range(3):
            a.append(block_a[k] * scale)
            b.append(block_b[k] * scale)
    
    # Verify sums
    assert abs(sum(a) - 1.0) < 1e-12, f"sum(a) = {sum(a)}"
    assert abs(sum(b) - 1.0) < 1e-12, f"sum(b) = {sum(b)}"
    assert len(a) == 12 and len(b) == 12
    
    # Check all values are finite and within bounds
    for val in a + b:
        assert np.isfinite(val), f"Non-finite value: {val}"
        assert -50.0 <= val <= 50.0, f"Value out of bounds: {val}"
    
    return a, b
# EVOLVE-BLOCK-END
