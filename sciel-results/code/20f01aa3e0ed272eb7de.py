import numpy as np

N = 600

def construct():
    # The theoretical minimum for the normalized autoconvolution peak
    # of a non-negative function on a fixed domain of length L is 1/L * (Integral)^2 ?
    # No, Score = max(f*f) / (int f)^2.
    # max(f*f) = int f^2 (for non-negative f, max is at 0).
    # By Cauchy-Schwarz, (int f)^2 <= (int 1^2 * L) * (int f^2) = L * int f^2.
    # So int f^2 / (int f)^2 >= 1/L.
    # Here L = 0.5. So lower bound is 1/0.5 = 2.
    # The uniform distribution achieves this bound.
    
    # Return uniform heights
    return [1.0] * N
