"""Matrix multiplication 4x4x4 rank-minimal decomposition."""
DIMENSIONS = [4, 4, 4]
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-49 decomposition for 4x4 matrix multiplication
    via the tensor (Kronecker) product of the rank-7 Strassen scheme.
    Uses numpy kron for the expansion, which is faster than manual loops.
    """
    start = time.time()
    limit = start + 140

    n, m, p = DIMENSIONS

    # Strassen 2x2x2 factor matrices (4,7)
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)

    # M0: (A00+A11)(B00+B11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    # M1: (A10+A11)B00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[1, 1] = 1; W2[3, 1] = -1
    # M2: A00(B01-B11)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[2, 2] = 1; W2[3, 2] = 1
    # M3: A11(B10-B00)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[1, 3] = 1
    # M4: (A00+A01)B11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[2, 4] = 1
    # M5: (A10-A00)(B00+B01)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    # M6: (A01-A11)(B10+B11)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1

    # Kronecker product expansion: rank = 7*7 = 49
    U = np.kron(U2, U2).reshape(n * m, 49)
    V = np.kron(V2, V2).reshape(m * p, 49)
    W = np.kron(W2, W2).reshape(p * n, 49)

    if time.time() > limit:
        raise TimeoutError()

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
