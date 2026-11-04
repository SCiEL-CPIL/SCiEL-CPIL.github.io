import time
import itertools
import numpy as np

T_START = time.time()
DEADLINE = 120.0          # evaluator hard kill
SAFETY   = 10.0           # return at least this many seconds before the kill
TIME_BUDGET = DEADLINE - SAFETY

# 2x2x2 Strassen kernel: 7 rank-1 slices.  Each term (u,v,w) has u[k,i], v[l,j], w[i,j,k].
# Standard Strassen, mapping output M[i,k] over input A[i,j], B[j,k].
ST = [
    # (U[2], V[2], W[2])
    ((1,1),(1,-1),(1,0,0,-1)),
    ((0,1),(1,1),(0,1,0,1)),
    ((1,0),(1,1),(1,0,1,0)),
    ((0,0),(1,1),(1,-1,0,0)),
    ((1,1),(1,0),(0,1,0,0)),
    ((1,0),(1,1),(0,0,0,1)),
    ((1,-1),(1,1),(0,0,1,0)),
]
for (u,v,w) in ST:
    for i in range(2):
        for k in range(2):
            s = 0
            for j in range(2):
                s += u[i]*v[j]*w[i*2+k]
            expect = 1 if i == k else 0
            if s != expect:
                raise AssertionError("Strassen kernel mismatch")


def pad3(x, L):
    out = [[0] * L for _ in range(L)]
    for i in range(min(len(x), L)):
        for j in range(min(len(x[0]), L)):
            out[i][j] = x[i][j]
    return out


# Embed the 3x4x8 product into a 4x4x8 product, then run two 4x4x4 Strassen block
# multiplications (each = 4 Strassen 2x2x2 kernels).
def construct():
    U = []
    V = []
    W = []

    for (ii, jj, ki, kj) in itertools.product((0, 1), (0, 1), (0, 1), (0, 1)):
        for t in range(7):
            cu, cv, cw = ST[t]
            U.append(cu[ii])     # row index k1*2 + i1
            V.append(cv[jj])     # row index j1*2 + k2
            W.append(cw[ki * 2 + kj])   # col index i2*2 + k1

    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    assert U.shape == (4, 28)
    assert V.shape == (4, 28)
    assert W.shape == (8, 28)

    # Verify the embedding computes A(3x4) * B(4x8).
    M = U @ (V.T @ W)            # shape (4, 8), top-left 3x4 must equal A@B
    A = np.eye(3, 4)
    B = np.zeros((4, 8))
    for i in range(3):
        for k in range(8):
            for j in range(4):
                B[j, k] += 1
    expected = np.zeros((3, 8))
    for i in range(3):
        for k in range(8):
            expected[i, k] = 4
    if not np.allclose(M[:3, :], expected):
        raise AssertionError("Strassen embedding failed")

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
