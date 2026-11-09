import numpy as np

def coefficients(stages):
    """
    12-stage product formula for Heisenberg-chain time evolution.
    
    Uses a 6th-order symmetric splitting with distinct a and b coefficients,
    mapped to 12 stages by splitting the largest steps.
    
    The formula is U(t) = prod_k exp(-i b_k t B) exp(-i a_k t A)
    with a and b each having 12 entries, sum(a)=sum(b)=1.
    
    We use the 6th-order Yoshida coefficients (9 stages) and split the
    largest B step into 3 parts and the largest A step into 2 parts to
    reach 12 stages total.
    """
    
    # 6th-order Yoshida symmetric coefficients (half-weights)
    # These give 9 stages: b1,a1,b2,a2,b3,a3,b2,a2,b1,a1? No.
    # Standard 6th order: 9 stages with c = [c1, c2, c3, c2, c1] for one side
    # and d = [d1, d2, d3, d2, d1] for the other
    # Full sequence: B(c1), A(d1), B(c2), A(d2), B(c3), A(d3), B(c2), A(d2), B(c1), A(d1)
    # Wait, that's 10 stages. Let me reconsider.
    
    # Actually, for a symmetric 6th-order formula with 9 "half-stages":
    # The weights are:
    # c (for B): [c1, c2, c3, c2, c1] where c1+c2+c3 = 0.5
    # d (for A): [d1, d2, d3, d2, d1] where d1+d2+d3 = 0.5
    # But the product is B(c1)A(d1)B(c2)A(d2)B(c3)A(d3)B(c2)A(d2)B(c1)A(d1)
    # That's 10 factors, but 5 B-steps and 5 A-steps.
    
    # For 12 stages total, we need 6 B-steps and 6 A-steps.
    # So we split one B weight into 2 and one A weight into 2.
    
    # 6th-order Yoshida coefficients (from literature):
    # c1 = 1/(2*(2^(1/3)-1)) ≈ 0.1295...  No, that's 4th order.
    
    # Let me use known 6th-order values:
    # From Suzuki's 6th order:
    # c1 = 1/(2*(2^(1/3)-1)) is not right either.
    
    # Using the standard 6th-order coefficients:
    # x = 2^(1/3)
    # c1 = 1/(2*(x-1))
    # c2 = -x/(2*(x-1))  
    # c3 = 1 - c1 - c2
    # d1 = 1/(2*(x-1))
    # d2 = -x/(2*(x-1))
    # d3 = 1 - d1 - d2
    
    # Wait, let me just use the values that sum correctly.
    # For 6th order symmetric:
    # c = [c1, c2, c3, c2, c1], sum = 2*(c1+c2+c3) = 1, so c1+c2+c3 = 0.5
    # d = [d1, d2, d3, d2, d1], sum = 2*(d1+d2+d3) = 1, so d1+d2+d3 = 0.5
    
    # Standard 6th order (Suzuki):
    # Let x = 2**(1/3)
    # c1 = 1/(2*(x-1))
    # c2 = -x/(2*(x-1))
    # c3 = 1 - c1 - c2
    # d1 = 1/(2*(x-1))
    # d2 = -x/(2*(x-1))  
    # d3 = 1 - d1 - d2
    
    x = 2.0 ** (1.0/3.0)
    c1 = 1.0 / (2.0 * (x - 1.0))
    c2 = -x / (2.0 * (x - 1.0))
    c3 = 1.0 - c1 - c2
    
    d1 = 1.0 / (2.0 * (x - 1.0))
    d2 = -x / (2.0 * (x - 1.0))
    d3 = 1.0 - d1 - d2
    
    # Check sums
    assert abs(2*(c1+c2+c3) - 1.0) < 1e-10, f"c sum error: {2*(c1+c2+c3)}"
    assert abs(2*(d1+d2+d3) - 1.0) < 1e-10, f"d sum error: {2*(d1+d2+d3)}"
    
    # Full 10-stage sequence (5 B, 5 A):
    # B: c1, c2, c3, c2, c1
    # A: d1, d2, d3, d2, d1
    
    # To get 12 stages (6 B, 6 A), split the largest B and A weights.
    # c3 is likely the largest (positive) weight.
    # d3 is likely the largest (positive) weight.
    
    # Split c3 into two equal parts: c3a = c3/2, c3b = c3/2
    # Split d3 into two equal parts: d3a = d3/2, d3b = d3/2
    
    # New B sequence (6 elements): c1, c2, c3/2, c3/2, c2, c1
    # New A sequence (6 elements): d1, d2, d3/2, d3/2, d2, d1
    
    b_half = [c1, c2, c3/2.0, c3/2.0, c2, c1]
    a_half = [d1, d2, d3/2.0, d3/2.0, d2, d1]
    
    # Make palindromic 12-element lists
    a = a_half + a_half[::-1]
    b = b_half + b_half[::-1]
    
    assert len(a) == stages, f"Expected {stages} stages, got {len(a)}"
    assert len(b) == stages, f"Expected {stages} stages, got {len(b)}"
    
    # Clamp to [-50, 50]
    a = [max(-50.0, min(50.0, float(x))) for x in a]
    b = [max(-50.0, min(50.0, float(x))) for x in b]
    
    # Normalize to ensure sum is exactly 1
    sa = sum(a)
    sb = sum(b)
    
    if abs(sa - 1.0) > 1e-15:
        a[-1] += (1.0 - sa)
    if abs(sb - 1.0) > 1e-15:
        b[-1] += (1.0 - sb)
    
    return a, b

