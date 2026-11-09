import numpy as np

def coefficients(stages):
    """
    12-stage 4th-order product formula.
    Uses the 4-stage 4th-order Ruth splitting coefficients repeated 3 times.
    """
    if stages != 12:
        # Fallback to Strang if stages is not 12, though task implies 12
        substeps = stages // 2
        a = [0.5 / substeps] * (2 * substeps)
        b = []
        for _ in range(substeps):
            b += [1.0 / substeps, 0.0]
        while len(a) < stages:
            a.append(0.0)
            b.append(0.0)
        return a, b

    # 4-stage 4th order coefficients
    mu = 2.0 ** (1.0/3.0)
    c1 = 1.0 / (2.0 * (1.0 - mu))
    c2 = -mu / (2.0 * (1.0 - mu))
    
    # c1 + c2 = 0.5
    # Sequence of A coefficients in one block: c1, c2, c2, c1
    # Sequence of B coefficients in one block: c2, c1, c1, c2
    # Note: The product formula is exp(-i b B) exp(-i a A).
    # Standard splitting e^{t(A+B)} = e^{t a1 A} e^{t b1 B} ...
    # We need to ensure the order matches the task's U(t) = prod exp(-i b_k t B) exp(-i a_k t A).
    # This is equivalent to e^{-i t b1 B} e^{-i t a1 A} e^{-i t b2 B} e^{-i t a2 A} ...
    # Let H_A = -i A_ham, H_B = -i B_ham.
    # U = e^{t b1 H_B} e^{t a1 H_A} ...
    # This is a splitting of e^{t(H_A+H_B)}.
    
    a_block = [c1, c2, c2, c1]
    b_block = [c2, c1, c1, c2]
    
    # Repeat 3 times to fill 12 stages
    a = []
    b = []
    for _ in range(3):
        a.extend([x/3.0 for x in a_block])
        b.extend([x/3.0 for x in b_block])
        
    return a, b
