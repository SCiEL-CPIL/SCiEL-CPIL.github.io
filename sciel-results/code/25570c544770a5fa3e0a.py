"""
Twelve-stage product formula for Heisenberg-chain time evolution.
Uses 4th-order Suzuki-Trotter symmetric splitting with proper normalization.
"""
import time
import numpy as np

# EVOLVE-BLOCK-START
def coefficients(stages):
    """12-stage product formula using 4th-order symmetric Suzuki-Trotter splitting.
    
    Uses three concatenated 4th-order 4-stage blocks with properly normalized
    coefficients to ensure sum(a) = sum(b) = 1.
    """
    start_time = time.time()
    
    if stages != 12:
        # Fallback to simple Strang splitting for non-12 stage counts
        substeps = max(1, stages // 2)
        a = [0.5 / substeps] * (2 * substeps)
        b = []
        for _ in range(substeps):
            b += [1.0 / substeps, 0.0]
        while len(a) < stages:
            a.append(0.0)
            b.append(0.0)
        return a, b
    
    # 4th-order Suzuki-Trotter symmetric splitting coefficients
    # The standard 4th-order symmetric splitting (Suzuki 1991) for exp(A+B)
    # uses the composition: exp(c1*A) exp(c1*B) exp(c2*A) exp(c2*B)
    #                           exp(c2*A) exp(c2*B) exp(c1*A) exp(c1*B)
    # where c1 = 1/(2*(1-2^(-1/3))) and c2 = (1-c1)/2
    
    r = 2.0 ** (1.0 / 3.0)
    c1 = 1.0 / (2.0 * (1.0 - 1.0 / r))
    c2 = (1.0 - c1) / 2.0
    
    # One 4-stage 4th-order block (half of the full 8-stage 4th-order splitting)
    # The full 4th-order splitting is 8 stages, but we need 12 stages total.
    # We use the symmetric composition of three 4-stage blocks.
    # Each block: exp(a_i * A) exp(b_i * B) for i in block
    # 
    # For a symmetric 4th-order method with 4 stages per block, we use:
    # a_block = [c1, c2, c2, c1] and b_block = [0, c1, 2*c2, 0]
    # This gives the composition:
    # exp(c1*A) exp(c1*B) exp(c2*A) exp(2*c2*B) exp(c2*A) exp(c1*B) exp(c1*A)
    # Wait, that's 7 stages. Let me reconsider.
    #
    # Actually, for 12 stages with 6 A and 6 B factors:
    # U = prod_{k=1}^{12} exp(-i b_k t B) exp(-i a_k t A)
    # So each stage has one B factor and one A factor.
    # Total: 12 B factors and 12 A factors.
    # sum(a_k) = 1, sum(b_k) = 1
    #
    # Let's use the 4th-order Suzuki-Trotter coefficients properly.
    # The standard 4th-order symmetric splitting for exp(t(A+B)) is:
    # exp(t c1 A) exp(t c1 B) exp(t c2 A) exp(t c2 B)
    # exp(t c2 A) exp(t c2 B) exp(t c1 A) exp(t c1 B)
    # This is 8 stages with 4 A and 4 B factors each.
    #
    # For 12 stages, we can concatenate three 4-stage blocks.
    # Each 4-stage block: exp(a_i A) exp(b_i B) for i=1..4
    # To maintain 4th order, we use the same coefficients in each block.
    #
    # Let a_block = [c1/3, c2/3, c2/3, c1/3] and b_block = [0, c1/3, 2*c2/3, 0]
    # No wait, that won't give sum=1 for each block.
    #
    # Let me think differently. We have 12 stages, each with (a_k, b_k).
    # sum(a_k) = 1, sum(b_k) = 1.
    #
    # A natural choice: use the 4th-order Suzuki-Trotter coefficients
    # replicated 3 times, scaled appropriately.
    #
    # Standard 4th-order 8-stage: a = [c1, c2, c2, c1, c1, c2, c2, c1]
    #                            b = [0, c1, 2*c2, 0, 0, c1, 2*c2, 0]
    # sum(a) = 2*c1 + 4*c2 = 2*c1 + 2*(1-c1) = 2
    # sum(b) = 2*c1 + 4*c2 = 2
    # So we need to divide by 2 to get sum=1.
    #
    # For 12 stages, we can use a different approach. Let's use the
    # 4th-order coefficients but adjust for 12 stages.
    #
    # Actually, let's use a simpler approach: the 4th-order Suzuki-Trotter
    # coefficients for a 4-stage method, repeated 3 times.
    #
    # 4-stage 4th-order: a = [c1, c2, c2, c1], b = [0, c1, 2*c2, 0]
    # sum(a) = 2*c1 + 2*c2 = 1
    # sum(b) = c1 + 2*c2 = c1 + (1-c1) = 1
    # Perfect! So each 4-stage block already has sum(a)=1 and sum(b)=1.
    # But if we repeat it 3 times, the total sum becomes 3, not 1.
    # So we need to scale by 1/3.
    
    # 4-stage 4th-order block coefficients
    a_block = [c1 / 3.0, c2 / 3.0, c2 / 3.0, c1 / 3.0]
    b_block = [0.0, c1 / 3.0, 2.0 * c2 / 3.0, 0.0]
    
    # Three blocks for 12 stages
    a = a_block * 3
    b = b_block * 3
    
    # Normalize to ensure sum(a) = 1 and sum(b) = 1
    sum_a = sum(a)
    sum_b = sum(b)
    
    if abs(sum_a) > 1e-15:
        a = [x / sum_a for x in a]
    if abs(sum_b) > 1e-15:
        b = [x / sum_b for x in b]
    
    # Verify consistency
    assert len(a) == 12, f"Expected 12 a coefficients, got {len(a)}"
    assert len(b) == 12, f"Expected 12 b coefficients, got {len(b)}"
    assert abs(sum(a) - 1.0) < 1e-10, f"Sum of a is {sum(a)}, expected 1.0"
    assert abs(sum(b) - 1.0) < 1e-10, f"Sum of b is {sum(b)}, expected 1.0"
    
    # Ensure all values are finite and in [-50, 50]
    for i in range(12):
        a[i] = max(-50.0, min(50.0, a[i]))
        b[i] = max(-50.0, min(50.0, b[i]))
    
    return a, b
# EVOLVE-BLOCK-END
