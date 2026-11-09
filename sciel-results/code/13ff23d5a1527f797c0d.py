import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = x.flatten()
    y = y.flatten()

    # Chebyshev polynomial basis of total degree up to 4 (10 terms)
    # This provides a well-conditioned, linear-in-parameters basis
    # that can approximate smooth bivariate functions on [-1, 1]^2.

    # Chebyshev polynomials of the first kind
    def T0(t):
        return np.ones_like(t)

    def T1(t):
        return t

    def T2(t):
        return 2.0 * t**2 - 1.0

    def T3(t):
        return 4.0 * t**3 - 3.0 * t

    def T4(t):
        return 8.0 * t**4 - 8.0 * t**2 + 1.0

    # Build tensor-product Chebyshev basis with total degree <= 4
    # Terms: T_i(x) * T_j(y) where i + j <= 4
    # i=0: j=0,1,2,3,4 -> 5 terms
    # i=1: j=0,1,2,3   -> 4 terms
    # i=2: j=0,1        -> 2 terms... wait that's 11 terms
    # Let me recount: i+j <= 4 gives:
    # i=0: j=0..4 (5)
    # i=1: j=0..3 (4)
    # i=2: j=0..2 (3)
    # i=3: j=0..1 (2)
    # i=4: j=0     (1)
    # Total = 5+4+3+2+1 = 15 terms. Too many.
    
    # Use total degree <= 3: 
    # i=0: j=0..3 (4)
    # i=1: j=0..2 (3)
    # i=2: j=0..1 (2)
    # i=3: j=0    (1)
    # Total = 4+3+2+1 = 10 terms. Perfect.

    tx = [T0(x), T1(x), T2(x), T3(x)]
    ty = [T0(y), T1(y), T2(y), T3(y)]

    # Build the 10 basis functions
    basis = []
    for i in range(4):
        for j in range(4 - i):
            basis.append(tx[i] * ty[j])

    # Stack into (N, 10) matrix
    features = np.column_stack(basis)

    return features @ params
