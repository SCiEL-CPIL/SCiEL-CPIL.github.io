"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

def coefficients(stages):
    """
    4th-order symmetric Suzuki-Trotter splitting mapped to 12 stages.
    
    The 4th-order scheme S_4(t) = S_2(c1*t) S_2(c2*t) S_2(c3*t) 
    expands to a sequence of exponentials. We map these to the first 7 
    (b_k, a_k) pairs and pad the rest with zeros.
    
    S_2(tau) = exp(-i tau/2 B) exp(-i tau A) exp(-i tau/2 B)
    
    Sequence:
    1. exp(-i c1/2 t B)  -> b1 = c1/2, a1 = 0
    2. exp(-i c1 t A)    -> b2 = 0, a2 = c1
    3. exp(-i (c1+c2)/2 t B) -> b3 = (c1+c2)/2, a3 = 0
    4. exp(-i c2 t A)    -> b4 = 0, a4 = c2
    5. exp(-i (c2+c3)/2 t B) -> b5 = (c2+c3)/2, a5 = 0
    6. exp(-i c3 t A)    -> b6 = 0, a6 = c3
    7. exp(-i c3/2 t B)  -> b7 = c3/2, a7 = 0
    8-12. zeros
    """
    if stages != 12:
        # Fallback to seed behavior if stages != 12, though task implies 12
        substeps = max(1, stages // 2)
        a = [0.5 / substeps] * (2 * substeps)
        b = []
        for _ in range(substeps):
            b += [1.0 / substeps, 0.0]
        while len(a) < stages:
            a.append(0.0)
            b.append(0.0)
        return a, b

    # 4th order Suzuki coefficients
    c1 = 1.0 / (2.0 - 2.0**(1.0/3.0))
    c2 = -2.0**(1.0/3.0) / (2.0 - 2.0**(1.0/3.0))
    c3 = c1

    a = [0.0] * stages
    b = [0.0] * stages

    # Map the 7 non-zero exponentials to the 12 stages
    # The product is prod_k exp(-i b_k t B) exp(-i a_k t A)
    
    # k=1: exp(-i b_1 t B) exp(-i a_1 t A) corresponds to first term exp(-i c1/2 t B)
    b[0] = c1 / 2.0
    a[0] = 0.0
    
    # k=2: corresponds to second term exp(-i c1 t A)
    b[1] = 0.0
    a[1] = c1
    
    # k=3: corresponds to third term exp(-i (c1+c2)/2 t B)
    b[2] = (c1 + c2) / 2.0
    a[2] = 0.0
    
    # k=4: corresponds to fourth term exp(-i c2 t A)
    b[3] = 0.0
    a[3] = c2
    
    # k=5: corresponds to fifth term exp(-i (c2+c3)/2 t B)
    b[4] = (c2 + c3) / 2.0
    a[4] = 0.0
    
    # k=6: corresponds to sixth term exp(-i c3 t A)
    b[5] = 0.0
    a[5] = c3
    
    # k=7: corresponds to seventh term exp(-i c3/2 t B)
    b[6] = c3 / 2.0
    a[6] = 0.0
    
    # k=8..12 are already 0.0
    
    return a, b
# EVOLVE-BLOCK-END
