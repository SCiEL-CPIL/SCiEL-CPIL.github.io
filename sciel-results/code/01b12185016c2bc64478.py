import time
import numpy as np

DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def _strassen_2x2():
    # Strassen rank-7 decomposition of 2x2 * 2x2 multiplication.
    # Returns U (4x7), V (4x7), W (4x7).
    # Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c].
    U = [
        [1, 0, 0, 0, 0, 0, 0],  # a00
        [1, 1, 0, 0, 0, 0, 0],  # a01
        [0, 1, 1, 0, 0, 0, 0],  # a10
        [0, 0, 0, 1, 1, 1, 1],  # a11
    ]
    V = [
        [1, 0, 0, 0, 0, 0, 0],  # b00
        [0, 1, 0, 0, 0, 0, 0],  # b01
        [0, 0, 1, 1, 0, 0, 0],  # b10
        [0, 0, 0, 0, 1, 1, 1],  # b11
    ]
    W = [
        [1, 0, 0, 0, 0, 0, 0],  # c00
        [0, 1, 1, 0, 0, 0, 0],  # c01
        [0, 0, 0, 1, 0, 0, 0],  # c10
        [1, 0, 0, 0, 1, 0, 0],  # c11
    ]
    return U, V, W


def _verify(U, V, W, n, m, p):
    # Verify the decomposition reproduces the matrix multiplication tensor.
    # C = A (n x m) @ B (m x p).
    # A_flat[i*m+j], B_flat[j*p+k], C_flat[k*p+i]  (trace convention).
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    # T[i*m+j, j*p+k, k*p+i] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*p+i,c]
    # Build the tensor and compare to the schoolbook tensor.
    nm = n * m
    mp = m * p
    pn = p * n
    # Compute the candidate tensor via einsum over c
    # T[a, b, cidx] where a in [0,nm), b in [0,mp), cidx in [0,pn)
    # a = i*m+j, b = j'*p+k, cidx = k'*n+i'
    # The valid terms require j == j' and k == k' and i == i'
    # so the tensor should be nonzero only on the "diagonal".
    T = np.einsum('ac,bc,dc->abd', Ua, Va, Wa)
    # Build reference
    Tref = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                cidx = k * n + i
                Tref[a, b, cidx] = 1
    return np.allclose(T, Tref, atol=1e-9)


def _construct_block():
    n, m, p = DIMENSIONS
    # p must be even to split into p/2 blocks of size 2.
    assert p % 2 == 0, "p must be even for the block-diagonal Strassen approach"
    nblocks = p // 2
    blk = 2  # each block is 2x2
    rank = nblocks * 7  # 5 * 7 = 35

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    sU, sV, sW = _strassen_2x2()
    for b in range(nblocks):
        base = b * 7
        # A rows: i in [0,2), j in [0,2)  -> a = i*m + j = i*2 + j (m=2)
        for i in range(2):
            for j in range(2):
                a = i * m + j
                for c in range(7):
                    U[a][base + c] = sU[i * 2 + j][c]
        # B rows: j in [0,2), k in [0,2)  -> b = j*p + (b*2 + k)
        for j in range(2):
            for k in range(2):
                bb = j * p + (b * blk + k)
                for c in range(7):
                    V[bb][base + c] = sV[j * 2 + k][c]
        # C rows: k in [0,2), i in [0,2)  -> cidx = (b*2 + k)*n + i
        for k in range(2):
            for i in range(2):
                cc = (b * blk + k) * n + i
                for c in range(7):
                    W[cc][base + c] = sW[k * 2 + i][c]
    return U, V, W


def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin before 150s wall-clock limit

    U, V, W = _construct_block()

    # Verify correctness; if it fails, fall back to schoolbook.
    if time.time() < deadline:
        if _verify(U, V, W, 2, 2, 10):
            return {"U": U, "V": V, "W": W}
        # Fallback to schoolbook (rank 40) to guarantee validity.
        n, m, p = DIMENSIONS
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    # Over deadline: return best found (the block result).
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
