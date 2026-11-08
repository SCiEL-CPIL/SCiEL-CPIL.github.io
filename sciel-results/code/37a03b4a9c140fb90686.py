"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

_DIMENSIONS = [2, 3, 6]
DIMENSIONS = _DIMENSIONS  # Keep name for compatibility

# EVOLVE-BLOCK-START
def construct():
    _t0 = time.time()
    _deadline = _t0 + 110
    n, m, p = DIMENSIONS

    # Block decomposition: split p=6 into 3 blocks of size 2.
    # Each block is a 2x3x2 multiplication with rank 11 (Strassen 7 + 4 scalar).
    # Total rank = 33.

    R = 33
    U = np.zeros((n * m, R), dtype=float)
    V = np.zeros((m * p, R), dtype=float)
    W = np.zeros((p * n, R), dtype=float)

    # Local indices for a 2x3x2 block:
    #   A (2x3): a[i,j], i in {0,1}, j in {0,1,2}  -> flat = i*3+j
    #   B (3x2): b[j,k], j in {0,1,2}, k in {0,1}  -> flat = j*2+k
    #   C (2x2): c[i,k], i in {0,1}, k in {0,1}    -> flat = i*2+k
    #
    # Strassen on A[:,0:2] (2x2) * B[0:2,:] (2x2):
    #   A00=A[0,0]=0, A01=A[0,1]=1, A10=A[1,0]=3, A11=A[1,1]=4
    #   B00=B[0,0]=0, B01=B[0,1]=1, B10=B[1,0]=2, B11=B[1,1]=3
    #
    #   M1=(A00+A11)(B00+B11) -> C00+=M1, C11+=M1
    #   M2=(A10+A11)B00       -> C10+=M2, C11-=M2
    #   M3=A00(B01-B11)       -> C01+=M3, C11+=M3
    #   M4=A11(B10-B00)       -> C00+=M4, C10+=M4
    #   M5=(A00+A01)B11       -> C00-=M5, C01+=M5
    #   M6=(A10-A00)(B00+B01) -> C11+=M6
    #   M7=(A01-A11)(B10+B11) -> C00+=M7
    #
    # Scalar terms for column j=2 of A times row j=2 of B:
    #   K1=A02*B20 -> C00, K2=A02*B21 -> C01, K3=A12*B20 -> C10, K4=A12*B21 -> C11

    # Each entry: (u_dict, v_dict, w_dict)
    terms = [
        ({0: 1, 4: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({3: 1, 4: 1}, {0: 1},         {2: 1, 3: -1}),
        ({0: 1},       {1: 1, 3: -1},  {1: 1, 3: 1}),
        ({4: 1},       {2: 1, 0: -1},  {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1},         {0: -1, 1: 1}),
        ({3: 1, 0: -1},{0: 1, 1: 1},   {3: 1}),
        ({1: 1, 4: -1},{2: 1, 3: 1},   {0: 1}),
        ({2: 1},       {4: 1},         {0: 1}),
        ({2: 1},       {5: 1},         {1: 1}),
        ({5: 1},       {4: 1},         {2: 1}),
        ({5: 1},       {5: 1},         {3: 1}),
    ]

    for blk in range(3):
        if time.time() > _deadline:
            break
        base = blk * 11
        for t, (ud, vd, wd) in enumerate(terms):
            col = base + t
            for a, val in ud.items():
                U[a, col] = val
            for b, val in vd.items():
                j = b >> 1
                k = b & 1
                V[j * p + blk * 2 + k, col] = val
            for c, val in wd.items():
                i = c >> 1
                k = c & 1
                W[(blk * 2 + k) * n + i, col] = val

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
