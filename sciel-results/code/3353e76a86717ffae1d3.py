"""Optimized 12-stage product formula for Heisenberg chain time evolution.

Uses the 4th-order Yoshida decomposition embedded into a 12-stage (B, A)
product-formula structure.  The 4th-order symmetric splitting
    U(t) = exp(-i t B/2) exp(-i t c1 A) exp(-i t c2 B) exp(-i t c1 A) exp(-i t B/2)
with c = 2^(1/3), c1 = 1/(2-c), c2 = -c/(2-c) is embedded by placing its
five non-zero operator factors into the first five stages and setting the
remaining seven stages to zero.  Coefficients are normalized so that
sum(a) = sum(b) = 1 and clamped to [-50, 50].
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def coefficients(stages):
    """
    Returns (a, b): two lists of 12 finite numbers in [-50, 50]
    with sum(a) = sum(b) = 1.

    The product formula is
        U(t) = prod_{k=1}^{12} exp(-i b_k t B) exp(-i a_k t A)

    We embed the 4th-order symmetric Suzuki-Trotter (Yoshida) decomposition
    into the first five (B, A) pairs and zero out the rest.
    """
    if stages != 12:
        a = [1.0 / stages] * stages
        b = [1.0 / stages] * stages
        return a, b

    # 4th-order Suzuki-Trotter / Yoshida coefficients
    c = 2.0 ** (1.0 / 3.0)
    c1 = 1.0 / (2.0 - c)
    c2 = -c / (2.0 - c)

    # Operator sequence (in order):
    #   1: B * 0.5
    #   2: A * c1
    #   3: B * c2
    #   4: A * c1
    #   5: B * 0.5
    #
    # Each stage k contributes exp(-i b_k t B) exp(-i a_k t A).
    # We assign:
    #   k=1: b1 = 0.5, a1 = 0
    #   k=2: b2 = 0,   a2 = c1
    #   k=3: b3 = c2,  a3 = 0
    #   k=4: b4 = 0,   a4 = c1
    #   k=5: b5 = 0.5, a5 = 0
    #   k=6..12: zeros

    a_raw = [0.0, c1, 0.0, c1, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    b_raw = [0.5, 0.0, c2, 0.0, 0.5,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Normalize so sums equal 1
    sa = sum(a_raw)
    sb = sum(b_raw)

    if abs(sa) < 1e-15 or abs(sb) < 1e-15:
        return [1.0 / 12.0] * 12, [1.0 / 12.0] * 12

    a = [x / sa for x in a_raw]
    b = [x / sb for x in b_raw]

    # Clamp to [-50, 50]
    a = [max(-50.0, min(50.0, x)) for x in a]
    b = [max(-50.0, min(50.0, x)) for x in b]

    return a, b
# EVOLVE-BLOCK-END
