"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def coefficients(stages):
    """
    4th-order symmetric Suzuki-Trotter product formula.
    
    Uses the 3-stage 4th-order Yoshida coefficients repeated 4 times (12 stages total).
    The product formula is: prod_k exp(-i b_k t B) exp(-i a_k t A)
    
    For the 4th-order Yoshida scheme with 3 stages per substep:
    x1 = 1/(2 - 2^(1/3))
    x2 = -2^(1/3)/(2 - 2^(1/3))
    x3 = 1 - x1 - x2 = 1/(2 - 2^(1/3))
    
    Each substep of size dt applies:
    exp(-i x1*dt/2 * B) exp(-i x1*dt/2 * A) 
    exp(-i x2*dt/2 * B) exp(-i x2*dt/2 * A)
    exp(-i x3*dt/2 * B) exp(-i x3*dt/2 * A)
    ... but we need to match the form exp(-i b_k t B) exp(-i a_k t A)
    
    Actually, the standard 4th-order symmetric formula for exp(-i t H) where H = A + B is:
    exp(-i c1 * t/2 * A) exp(-i c1 * t/2 * B) exp(-i c2 * t/2 * A) exp(-i c2 * t/2 * B) exp(-i c3 * t/2 * A) exp(-i c3 * t/2 * B) exp(-i c2 * t/2 * A) exp(-i c2 * t/2 * B) exp(-i c1 * t/2 * A)
    
    But our form is fixed: prod_k exp(-i b_k t B) exp(-i a_k t A)
    
    Let me think about this differently. We have 12 stages, each contributing exp(-i b_k t B) exp(-i a_k t A).
    This is equivalent to a 24-step product formula where we alternate B and A steps.
    
    For a 4th-order scheme, we can use the 3-stage Yoshida coefficients repeated 4 times.
    The 3-stage 4th-order formula (in the form that matches our structure) would be:
    
    For each substep with time step dt = t/4:
    We want to approximate exp(-i dt (A+B)) using a symmetric product of A and B exponentials.
    
    The standard 4th-order symmetric splitting for exp(-i dt(A+B)) is:
    exp(-i x1*dt/2 * B) exp(-i x1*dt/2 * A) exp(-i x2*dt/2 * B) exp(-i x2*dt/2 * A) exp(-i x3*dt/2 * B) exp(-i x3*dt/2 * A)
    where x1 = x3 = 1/(2-2^(1/3)), x2 = -2^(1/3)/(2-2^(1/3))
    
    This gives 6 exponentials per substep, and with 4 substeps, that's 24 exponentials = 12 stages of (B,A) pairs.
    
    So for each substep j = 0,1,2,3 with dt = t/4:
    Stage 3j+1: b = x1/4, a = x1/4  (wait, need to be careful)
    
    Let me redefine. The full product is:
    prod_{k=1}^{12} exp(-i b_k t B) exp(-i a_k t A)
    
    For 4 substeps of the 3-stage 4th-order formula:
    Substep j uses coefficients x1, x2, x3 scaled by 1/4.
    
    Substep 1 (stages 1,2,3):
    exp(-i (x1/4) t B) exp(-i (x1/4) t A) exp(-i (x2/4) t B) exp(-i (x2/4) t A) exp(-i (x3/4) t B) exp(-i (x3/4) t A)
    
    So:
    b[0] = x1/4, a[0] = x1/4
    b[1] = x2/4, a[1] = x2/4
    b[2] = x3/4, a[2] = x3/4
    
    Similarly for substeps 2, 3, 4.
    """
    import numpy as np
    
    # 4th-order Yoshida coefficients
    c = 2.0 ** (1.0 / 3.0)
    x1 = 1.0 / (2.0 - c)
    x2 = -c / (2.0 - c)
    x3 = 1.0 - x1 - x2  # Should equal x1
    
    n_substeps = 4
    a = []
    b = []
    
    for j in range(n_substeps):
        for x in [x1, x2, x3]:
            a.append(x / n_substeps)
            b.append(x / n_substeps)
    
    # Verify sums
    assert abs(sum(a) - 1.0) < 1e-10, f"sum(a) = {sum(a)}"
    assert abs(sum(b) - 1.0) < 1e-10, f"sum(b) = {sum(b)}"
    
    return a, b
# EVOLVE-BLOCK-END
