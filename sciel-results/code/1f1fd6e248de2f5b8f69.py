"""Strassen-based rank-35 decomposition for 2x2x10 matrix multiplication."""
import time

_DIMENSIONS = [2, 2, 10]
_DEADLINE = 150.0
_START = time.time()


def _elapsed():
    return time.time() - _START


def construct():
    """Return a rank-35 exact tensor decomposition for 2x2x10 matrix multiplication."""
    n, m, p = _DIMENSIONS
    # 5 independent 2x2x2 blocks, each with Strassen rank 7
    num_blocks = p // 2  # = 5
    rank = 7 * num_blocks  # = 35

    U = [[0.0] * rank for _ in range(n * m)]  # 4 x 35
    V = [[0.0] * rank for _ in range(m * p)]  # 20 x 35
    W = [[0.0] * rank for _ in range(p * n)]  # 20 x 35

    # Strassen multipliers for 2x2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    #
    # Output:
    # C00 = M1 - M5 + M3 + M4
    # C01 = M5 + M7
    # C10 = M2 + M4
    # C11 = M1 - M2 + M6 + M7

    # For each block b (k = 2b, 2b+1):
    # A indices: A[i,j] for i,j in {0,1}
    # B indices: B[j, k] for j in {0,1}, k in {2b, 2b+1}
    # C indices: C[i, k] for i in {0,1}, k in {2b, 2b+1}

    for b in range(num_blocks):
        k0 = 2 * b
        k1 = 2 * b + 1
        base = 7 * b

        # M1 = (A00+A11)(B0k0+B1k0) ... wait, I need to be careful.
        # For a 2x2x2 block with k-indices k0, k1:
        # The "B" matrix is 2x2 with B[j, k0-k0] = B[j, k0] and B[j, k1-k0] = B[j, k1].
        # So B00 = B[0, k0], B01 = B[0, k1], B10 = B[1, k0], B11 = B[1, k1].

        # M1 = (A00+A11)(B00+B11) = (A00+A11)(B[0,k0]+B[1,k1])
        # M2 = (A10+A11)B00 = (A10+A11)B[0,k0]
        # M3 = A00(B10-B11) = A00(B[1,k0]-B[1,k1])
        # M4 = A11(B01-B10) = A11(B[0,k1]-B[1,k0])
        # M5 = (A00+A01)B11 = (A00+A01)B[1,k1]
        # M6 = (A10-A00)(B00+B10) = (A10-A00)(B[0,k0]+B[1,k0])
        # M7 = (A01-A11)(B01+B11) = (A01-A11)(B[0,k1]+B[1,k1])

        # Output:
        # C0k0 = M1 - M5 + M3 + M4
        # C0k1 = M5 + M7
        # C1k0 = M2 + M4
        # C1k1 = M1 - M2 + M6 + M7

        # A coefficients (shared across blocks):
        # A00 is at U[0*2+0, :] = U[0, :]
        # A01 is at U[0*2+1, :] = U[1, :]
        # A10 is at U[1*2+0, :] = U[2, :]
        # A11 is at U[1*2+1, :] = U[3, :]

        # M1: A00+A11, so U[0, base+0]=1, U[3, base+0]=1
        U[0][base + 0] += 1.0
        U[3][base + 0] += 1.0
        # M2: A10+A11, so U[2, base+1]=1, U[3, base+1]=1
        U[2][base + 1] += 1.0
        U[3][base + 1] += 1.0
        # M3: A00, so U[0, base+2]=1
        U[0][base + 2] += 1.0
        # M4: A11, so U[3, base+3]=1
        U[3][base + 3] += 1.0
        # M5: A00+A01, so U[0, base+4]=1, U[1, base+4]=1
        U[0][base + 4] += 1.0
        U[1][base + 4] += 1.0
        # M6: A10-A00, so U[2, base+5]=1, U[0, base+5]=-1
        U[2][base + 5] += 1.0
        U[0][base + 5] += -1.0
        # M7: A01-A11, so U[1, base+6]=1, U[3, base+6]=-1
        U[1][base + 6] += 1.0
        U[3][base + 6] += -1.0

        # B coefficients:
        # B[j, k] is at V[j*p + k, :] = V[j*10 + k, :]
        # M1: B[0,k0]+B[1,k1], so V[0*10+k0, base+0]=1, V[1*10+k1, base+0]=1
        V[0 * 10 + k0][base + 0] += 1.0
        V[1 * 10 + k1][base + 0] += 1.0
        # M2: B[0,k0], so V[0*10+k0, base+1]=1
        V[0 * 10 + k0][base + 1] += 1.0
        # M3: B[1,k0]-B[1,k1], so V[1*10+k0, base+2]=1, V[1*10+k1, base+2]=-1
        V[1 * 10 + k0][base + 2] += 1.0
        V[1 * 10 + k1][base + 2] += -1.0
        # M4: B[0,k1]-B[1,k0], so V[0*10+k1, base+3]=1, V[1*10+k0, base+3]=-1
        V[0 * 10 + k1][base + 3] += 1.0
        V[1 * 10 + k0][base + 3] += -1.0
        # M5: B[1,k1], so V[1*10+k1, base+4]=1
        V[1 * 10 + k1][base + 4] += 1.0
        # M6: B[0,k0]+B[1,k0], so V[0*10+k0, base+5]=1, V[1*10+k0, base+5]=1
        V[0 * 10 + k0][base + 5] += 1.0
        V[1 * 10 + k0][base + 5] += 1.0
        # M7: B[0,k1]+B[1,k1], so V[0*10+k1, base+6]=1, V[1*10+k1, base+6]=1
        V[0 * 10 + k1][base + 6] += 1.0
        V[1 * 10 + k1][base + 6] += 1.0

        # W coefficients:
        # C[i, k] is at W[k*n + i, :] = W[k*2 + i, :]
        # C0k0 = M1 - M5 + M3 + M4
        W[k0 * 2 + 0][base + 0] += 1.0
        W[k0 * 2 + 0][base + 4] += -1.0
        W[k0 * 2 + 0][base + 2] += 1.0
        W[k0 * 2 + 0][base + 3] += 1.0
        # C0k1 = M5 + M7
        W[k1 * 2 + 0][base + 4] += 1.0
        W[k1 * 2 + 0][base + 6] += 1.0
        # C1k0 = M2 + M4
        W[k0 * 2 + 1][base + 1] += 1.0
        W[k0 * 2 + 1][base + 3] += 1.0
        # C1k1 = M1 - M2 + M6 + M7
        W[k1 * 2 + 1][base + 0] += 1.0
        W[k1 * 2 + 1][base + 1] += -1.0
        W[k1 * 2 + 1][base + 5] += 1.0
        W[k1 * 2 + 1][base + 6] += 1.0

    return {"U": U, "V": V, "W": W}
