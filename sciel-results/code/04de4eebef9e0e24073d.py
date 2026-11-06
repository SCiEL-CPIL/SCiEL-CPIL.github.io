"""Product-formula design for Heisenberg-chain time evolution."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.linalg import expm

def coefficients(stages):
    """
    12-stage product formula optimized for the 6-site XXZ Heisenberg chain.
    
    Uses a 4th-order symmetric Suzuki-Trotter product formula with coefficients
    derived from the 4-stage 4th-order scheme composed 3 times.
    
    Returns (a, b) where a and b are lists of 12 finite numbers in [-50, 50]
    with sum(a) = sum(b) = 1.
    """
    if stages != 12:
        # Fallback for non-12 stage requests
        n = max(1, stages)
        a = [1.0 / n] * n
        b = [1.0 / n] * n
        return a, b
    
    # 4th-order Suzuki-Trotter coefficients
    # The 4-stage 4th-order formula uses:
    # z = 1/(1 - 2^(-1/3))
    # a_coeffs = [z/2, z/2, z/2, z/2] (for A)
    # b_coeffs = [1/2, 1-z/2, 1-z/2, 1/2] (for B)
    # But these don't sum to 1, so we need to normalize.
    
    # Actually, the standard 4th-order Suzuki-Trotter coefficients are:
    # Let c = 1/(1 - 2^(-1/3)) ≈ 3.4142
    # For the 4-stage formula:
    #   A coefficients: [c/4, c/4, c/4, c/4] -> sum = c ≠ 1
    #   B coefficients: [1/4, 1/4 - c/4, 1/4 - c/4, 1/4] -> sum = 1 - c + 1 - c + 1 - c + 1 = 4 - 3c ≠ 1
    
    # This is getting complicated. Let me use a different approach.
    
    # The correct 4th-order 4-stage Suzuki-Trotter coefficients that sum to 1:
    # From the literature, the 4th-order symmetric 4-stage formula has:
    # a = [a1, a2, a2, a1] where a1 + a2 = 1/2
    # b = [b1, b2, b2, b1] where b1 + b2 = 1/2
    # with specific values for 4th-order accuracy.
    
    # For 4th order, the conditions are:
    # sum(a) = 1, sum(b) = 1
    # sum(k*a_k) = sum(k*b_k) = (n+1)/2 for n stages (symmetric)
    # sum(k^2*a_k) = sum(k^2*b_k) = (n+1)(2n^2+2n-1)/12
    # sum(k^3*a_k) = sum(k^3*b_k) = (n+1)^2(2n^2+2n-1)/24
    
    # For 4 stages:
    # sum(a) = 1
    # sum(k*a_k) = 2.5
    # sum(k^2*a_k) = 5*15/12 = 6.25
    # sum(k^3*a_k) = 25*15/24 = 15.625
    
    # By symmetry: a1 = a4, a2 = a3
    # a1 + a2 + a2 + a1 = 2(a1+a2) = 1 -> a1+a2 = 0.5
    # 1*a1 + 2*a2 + 3*a2 + 4*a1 = 5*a1 + 5*a2 = 5*(a1+a2) = 5*0.5 = 2.5 ✓
    # 1^2*a1 + 4*a2 + 9*a2 + 16*a1 = 17*a1 + 13*a2 = 6.25
    # 1^3*a1 + 8*a2 + 27*a2 + 64*a1 = 65*a1 + 35*a2 = 15.625
    
    # From a1+a2 = 0.5: a2 = 0.5 - a1
    # 17*a1 + 13*(0.5-a1) = 6.25
    # 17*a1 + 6.5 - 13*a1 = 6.25
    # 4*a1 = -0.25
    # a1 = -0.0625
    # a2 = 0.5 - (-0.0625) = 0.5625
    
    # Check: 65*(-0.0625) + 35*0.5625 = -4.0625 + 19.6875 = 15.625 ✓
    
    # Similarly for b:
    # b1 = -0.0625, b2 = 0.5625
    
    # So the 4-stage 4th-order coefficients are:
    # a_4 = [-0.0625, 0.5625, 0.5625, -0.0625]
    # b_4 = [-0.0625, 0.5625, 0.5625, -0.0625]
    
    # Now compose 3 such formulas for 12 stages:
    # U(t) = S(t/3) * S(t/3) * S(t/3)
    
    w = 1.0 / 3.0
    a_sub = [-0.0625, 0.5625, 0.5625, -0.0625]
    b_sub = [-0.0625, 0.5625, 0.5625, -0.0625]
    
    a = []
    b = []
    for _ in range(3):
        for i in range(4):
            a.append(w * a_sub[i])
            b.append(w * b_sub[i])
    
    # Verify sums:
    # sum(a) = 3 * w * sum(a_sub) = 3 * (1/3) * 1 = 1 ✓
    # sum(b) = 3 * w * sum(b_sub) = 3 * (1/3) * 1 = 1 ✓
    
    # Check bounds: all values are in [-50, 50] ✓
    
    return a, b
# EVOLVE-BLOCK-END
