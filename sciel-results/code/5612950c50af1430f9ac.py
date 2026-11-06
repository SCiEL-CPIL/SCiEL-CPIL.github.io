import numpy as np

def coefficients(stages):
    """
    Implements a 4th-order symmetric product formula using 12 stages.
    The 4th-order Suzuki-Trotter scheme is composed of 3 sub-steps with weights
    alpha_1, alpha_2, alpha_3. Each sub-step is a 2nd-order Strang step.
    To utilize 12 stages, each of the 3 main sub-steps is split into two
    equal 2nd-order Strang sub-sub-steps. This results in 6 Strang steps
    total, mapped to 12 stages (2 stages per Strang step).
    """
    if stages != 12:
        # Fallback to uniform 2nd order if stages mismatch, though task implies 12
        a = [1.0 / stages] * stages
        b = [1.0 / stages] * stages
        return a, b

    # 4th order Suzuki-Trotter coefficients
    x = 2.0 ** (1.0 / 3.0)
    alpha1 = 1.0 / (2.0 - x)
    alpha2 = -x / (2.0 - x)
    alpha3 = alpha1

    # We have 3 main steps, each split into 2 equal parts
    # Total 6 Strang steps
    # Sizes of the 6 Strang steps
    deltas = [
        alpha1 / 2.0,
        alpha1 / 2.0,
        alpha2 / 2.0,
        alpha2 / 2.0,
        alpha3 / 2.0,
        alpha3 / 2.0
    ]

    a = [0.0] * stages
    b = [0.0] * stages

    for j in range(6):
        idx1 = 2 * j
        idx2 = 2 * j + 1
        delta = deltas[j]
        
        # Stage 1 of this Strang step: B(delta/2) A(delta)
        b[idx1] = delta / 2.0
        a[idx1] = delta
        
        # Stage 2 of this Strang step: B(delta/2) A(0)
        b[idx2] = delta / 2.0
        a[idx2] = 0.0

    return a, b
