"""4th-order Yoshida product formula for Heisenberg-chain time evolution."""
# EVOLVE-BLOCK-START
def coefficients(stages):
    """
    4th-order symmetric product formula using Yoshida coefficients.
    
    The 4th-order Yoshida decomposition uses 3 stages with coefficients:
    x1 = 1/(2 - 2^(1/3))
    x2 = -2^(1/3)/(2 - 2^(1/3))
    x3 = 1 - x1 - x2 = 1/(2 - 2^(1/3))
    
    For a 12-stage formula, we use 4 substeps of the 3-stage 4th-order formula.
    Each substep has the structure:
      exp(-i*x1*t/4 * A) exp(-i*x1*t/4 * B) exp(-i*x2*t/4 * A) exp(-i*x2*t/4 * B) exp(-i*x3*t/4 * A) exp(-i*x3*t/4 * B)
    
    But the task specifies the form: prod_k exp(-i b_k t B) exp(-i a_k t A)
    So we need to rearrange. Let's use the symmetric form properly.
    
    Actually, for the 4th-order Suzuki-Trotter (which is equivalent to Yoshida for 3 stages):
    S_4(t) = S_2(c1*t) * S_2(c2*t) * S_2(c1*t)
    where S_2(t) = exp(-i t A/2) exp(-i t B) exp(-i t A/2)
    
    c1 = 1/(2 - 2^(1/3)), c2 = -2^(1/3)/(2 - 2^(1/3))
    
    Expanding S_4(t):
    = exp(-i c1*t*A/2) exp(-i c1*t*B) exp(-i c1*t*A/2)
      * exp(-i c2*t*A/2) exp(-i c2*t*B) exp(-i c2*t*A/2)
      * exp(-i c1*t*A/2) exp(-i c1*t*B) exp(-i c1*t*A/2)
    
    Combining adjacent A terms:
    = exp(-i c1*t*A/2) exp(-i c1*t*B) exp(-i (c1/2 + c2/2)*t*A) exp(-i c2*t*B) exp(-i (c2/2 + c1/2)*t*A) exp(-i c1*t*B) exp(-i c1*t*A/2)
    
    = exp(-i c1/2 * t * A) exp(-i c1 * t * B) exp(-i (c1+c2)/2 * t * A) exp(-i c2 * t * B) exp(-i (c1+c2)/2 * t * A) exp(-i c1 * t * B) exp(-i c1/2 * t * A)
    
    This gives 3 B-stages and 4 A-stages per 4th-order step.
    
    For 12 stages total with the pattern (B, A) repeated 12 times, we need to map this.
    Let's use 2 substeps of the 4th-order formula, each contributing 6 stages (3 B + 3 A in alternating form).
    
    Actually, let me reconsider. The 12-stage formula with the form prod_k exp(-i b_k t B) exp(-i a_k t A)
    can be viewed as 12 pairs of (B, A) operations.
    
    A simpler approach: use the 4th-order coefficients directly in a 12-stage symmetric formula.
    
    For 4th order with 12 stages, we can use 4 substeps of a 3-stage 4th-order formula.
    Each 3-stage 4th-order formula (Yoshida) has coefficients x1, x2, x3 where:
    x1 = x3 = 1/(2 - 2^(1/3))
    x2 = -2^(1/3)/(2 - 2^(1/3))
    and x1 + x2 + x3 = 1
    
    The 3-stage formula is: exp(-i x1 * t * B) exp(-i x1 * t * A) exp(-i x2 * t * B) exp(-i x2 * t * A) exp(-i x3 * t * B) exp(-i x3 * t * A)
    
    Wait, that's not right either. Let me think more carefully.
    
    The standard 4th-order Suzuki-Trotter for exp(t(H1+H2)) is:
    exp(t/2 * H1) exp(t * H2) exp(t/2 * H1)  [this is 2nd order]
    
    4th order: 
    c1 = 1/(2 - 2^(1/3)), c2 = -2^(1/3)/(2 - 2^(1/3))
    S_4(t) = S_2(c1*t) * S_2(c2*t) * S_2(c1*t)
    where S_2(t) = exp(t/2 * H1) exp(t * H2) exp(t/2 * H1)
    
    Expanding:
    = exp(c1*t/2 * H1) exp(c1*t * H2) exp(c1*t/2 * H1)
      * exp(c2*t/2 * H1) exp(c2*t * H2) exp(c2*t/2 * H1)
      * exp(c1*t/2 * H1) exp(c1*t * H2) exp(c1*t/2 * H1)
    
    = exp(c1/2 * t * H1) exp(c1 * t * H2) exp((c1/2 + c2/2) * t * H1) exp(c2 * t * H2) exp((c2/2 + c1/2) * t * H1) exp(c1 * t * H2) exp(c1/2 * t * H1)
    
    Let H1 = A, H2 = B. Then:
    = exp(-i c1/2 * t * A) exp(-i c1 * t * B) exp(-i (c1+c2)/2 * t * A) exp(-i c2 * t * B) exp(-i (c1+c2)/2 * t * A) exp(-i c1 * t * B) exp(-i c1/2 * t * A)
    
    This has 3 B-exponentials and 4 A-exponentials.
    
    To fit into the form prod_k exp(-i b_k t B) exp(-i a_k t A), we need to pair them up.
    We can write this as:
    [exp(-i c1/2 * t * A)] * [exp(-i c1 * t * B) * exp(-i (c1+c2)/2 * t * A)] * [exp(-i c2 * t * B) * exp(-i (c1+c2)/2 * t * A)] * [exp(-i c1 * t * B) * exp(-i c1/2 * t * A)]
    
    But the first A is standalone. We can absorb it by setting b_1 = 0:
    [exp(-i 0 * t * B) * exp(-i c1/2 * t * A)] * [exp(-i c1 * t * B) * exp(-i (c1+c2)/2 * t * A)] * [exp(-i c2 * t * B) * exp(-i (c1+c2)/2 * t * A)] * [exp(-i c1 * t * B) * exp(-i c1/2 * t * A)]
    
    This gives 4 stages. For 12 stages, we can use 3 substeps.
    
    Let's use 3 substeps of the 4-stage 4th-order formula.
    """
    # 4th-order Yoshida coefficients
    c1 = 1.0 / (2.0 - 2.0 ** (1.0/3.0))
    c2 = -2.0 ** (1/3.0) / (2.0 - 2.0 ** (1/3.0))
    c3 = 1.0 - c1 - c2  # should equal c1
    
    # Per 4th-order substep (4 stages):
    # b = [0, c1, c2, c1]
    # a = [c1/2, (c1+c2)/2, (c1+c2)/2, c1/2]
    
    n_substeps = stages // 4  # 3 substeps for 12 stages
    
    a = []
    b = []
    for _ in range(n_substeps):
        b += [0.0, c1, c2, c1]
        a += [c1/2.0, (c1+c2)/2.0, (c1+c2)/2.0, c1/2.0]
    
    # Scale by 1/n_substeps since each substep handles t/n_substeps
    a = [x / n_substeps for x in a]
    b = [x / n_substeps for x in b]
    
    # Pad to exactly 'stages' elements if needed
    while len(a) < stages:
        a.append(0.0)
        b.append(0.0)
    
    return a[:stages], b[:stages]
# EVOLVE-BLOCK-END
