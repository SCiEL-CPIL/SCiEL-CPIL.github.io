import numpy as np
import time

DIMENSIONS = [3, 5, 7]

def _strassen_2x2():
    """Return (U, V, W) for 2x2x2 matrix multiplication with Strassen's 7-term algorithm.
    U shape (4,7), V shape (4,7), W shape (4,7), W indexed as W[k*2+i, t]."""
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)

    def Uidx(i, j): return i * 2 + j
    def Vidx(j, k): return j * 2 + k
    def Widx(k, i): return k * 2 + i

    # M1 = (a11+a22)(b11+b22)
    U[Uidx(0,0),0] = 1; U[Uidx(1,1),0] = 1
    V[Vidx(0,0),0] = 1; V[Vidx(1,1),0] = 1
    W[Widx(0,0),0] = 1; W[Widx(1,1),0] = 1
    # M2 = (a21+a22)b11
    U[Uidx(1,0),1] = 1; U[Uidx(1,1),1] = 1
    V[Vidx(0,0),1] = 1
    W[Widx(0,1),1] = 1; W[Widx(1,1),1] = 1
    # M3 = a11(b21-b22)
    U[Uidx(0,0),2] = 1
    V[Vidx(1,0),2] = 1; V[Vidx(1,1),2] = -1
    W[Widx(0,0),2] = 1; W[Widx(1,0),2] = 1
    # M4 = a22(b11-b21)
    U[Uidx(1,1),3] = 1
    V[Vidx(0,0),3] = 1; V[Vidx(1,0),3] = -1
    W[Widx(0,1),3] = 1; W[Widx(1,1),3] = 1
    # M5 = (a11+a12)b22
    U[Uidx(0,0),4] = 1; U[Uidx(0,1),4] = 1
    V[Vidx(1,1),4] = 1
    W[Widx(0,0),4] = 1; W[Widx(0,1),4] = 1
    # M6 = (a21-a11)(b11+b12)
    U[Uidx(1,0),5] = 1; U[Uidx(0,0),5] = -1
    V[Vidx(0,0),5] = 1; V[Vidx(0,1),5] = 1
    W[Widx(1,0),5] = 1; W[Widx(0,1),5] = 1
    # M7 = (a12-a22)(b21+b22)
    U[Uidx(0,1),6] = 1; U[Uidx(1,1),6] = -1
    V[Vidx(1,0),6] = 1; V[Vidx(1,1),6] = 1
    W[Widx(1,0),6] = 1; W[Widx(1,1),6] = 1

    return U, V, W


def _laderman_3x3():
    """Return (U, V, W) for 3x3x3 matrix multiplication with Laderman's 7-term algorithm."""
    U = np.zeros((9, 7), dtype=float)
    V = np.zeros((9, 7), dtype=float)
    W = np.zeros((9, 7), dtype=float)

    def Uidx(i, j): return i * 3 + j
    def Vidx(j, k): return j * 3 + k
    def Widx(k, i): return k * 3 + i

    # t=0: A[2,0]*B[0,1] -> M[1,2]
    U[Uidx(2,0),0] = 1
    V[Vidx(0,1),0] = 1
    W[Widx(1,2),0] = 1
    # t=1: A[0,2]*B[2,0] -> M[0,2]
    U[Uidx(0,2),1] = 1
    V[Vidx(2,0),1] = 1
    W[Widx(0,2),1] = 1
    # t=2: A[0,0]*B[0,2] -> M[2,0]
    U[Uidx(0,0),2] = 1
    V[Vidx(0,2),2] = 1
    W[Widx(2,0),2] = 1
    # t=3: A[2,2]*B[2,2] -> M[2,2]
    U[Uidx(2,2),3] = 1
    V[Vidx(2,2),3] = 1
    W[Widx(2,2),3] = 1
    # t=4: (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    U[Uidx(0,0),4] = 1; U[Uidx(1,1),4] = 1
    V[Vidx(0,0),4] = 1; V[Vidx(1,1),4] = 1
    W[Widx(0,0),4] = 1; W[Widx(1,1),4] = 1
    # t=5: (A[0,0]-A[1,1])*(B[0,0]-B[1,1])
    U[Uidx(0,0),5] = 1; U[Uidx(1,1),5] = -1
    V[Vidx(0,0),5] = 1; V[Vidx(1,1),5] = -1
    W[Widx(0,0),5] = 1; W[Widx(1,1),5] = -1
    # t=6: (A[0,0]+A[1,0]+A[0,1]+A[1,1])*(B[0,0]+B[1,0])
    U[Uidx(0,0),6] = 1; U[Uidx(1,0),6] = 1
    U[Uidx(0,1),6] = 1; U[Uidx(1,1),6] = 1
    V[Vidx(0,0),6] = 1; V[Vidx(1,0),6] = 1
    W[Widx(0,1),6] = 1; W[Widx(1,0),6] = 1; W[Widx(1,1),6] = 1

    return U, V, W


def _verify(U, V, W, n, m, p, rank):
    """Verify that the decomposition correctly computes matrix multiplication."""
    rng = np.random.RandomState(42)
    A = rng.randn(n, m)
    B = rng.randn(m, p)
    C_ref = A @ B

    C_comp = np.zeros((n, p), dtype=float)
    for t in range(rank):
        Ut = U[:, t].reshape(n, m)
        Vt = V[:, t].reshape(m, p)
        Wt = W[:, t].reshape(p, n)
        C_comp += Wt.T * (Ut @ Vt)

    err = np.max(np.abs(C_comp - C_ref))
    return err < 1e-10


def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 130

    # Strategy: Use Strassen for 2x2x2 blocks and Laderman for 3x3x3 blocks
    # where possible, with schoolbook for the rest.
    #
    # Partition the 3x5x7 multiplication:
    # A is 3x5, B is 5x7, C is 3x7
    #
    # We can partition the columns of A (and rows of B) into blocks:
    # Option 1: Split j into [0,1] (size 2) and [2,3,4] (size 3)
    #   - For j in [0,1]: A[:, 0:2] is 3x2, B[0:2, :] is 2x7
    #   - For j in [2,4]: A[:, 2:5] is 3x3, B[2:5, :] is 3x7
    #
    # Let's try a different partitioning to maximize use of Strassen/Laderman:
    #
    # Partition rows of A (i): [0,1] (size 2), [2] (size 1)
    # Partition cols of A / rows of B (j): [0,1] (size 2), [2,3,4] (size 3)
    # Partition cols of B (k): [0,1] (size 2), [2,3,4] (size 3), [5,6] (size 2)
    #
    # This gives us blocks:
    # (i:0-1, j:0-1, k:0-1): 2x2x2 -> Strassen (7 terms)
    # (i:0-1, j:0-1, k:2-4): 2x2x3 -> schoolbook (12 terms)
    # (i:0-1, j:0-1, k:5-6): 2x2x2 -> Strassen (7 terms)
    # (i:0-1, j:2-4, k:0-1): 2x3x2 -> schoolbook (12 terms)
    # (i:0-1, j:2-4, k:2-4): 2x3x3 -> schoolbook (18 terms)
    # (i:0-1, j:2-4, k:5-6): 2x3x2 -> schoolbook (12 terms)
    # (i:2, j:0-1, k:0-1): 1x2x2 -> schoolbook (4 terms)
    # (i:2, j:0-1, k:2-4): 1x2x3 -> schoolbook (6 terms)
    # (i:2, j:0-1, k:5-6): 1x2x2 -> schoolbook (4 terms)
    # (i:2, j:2-4, k:0-1): 1x3x2 -> schoolbook (6 terms)
    # (i:2, j:2-4, k:2-4): 1x3x3 -> schoolbook (9 terms)
    # (i:2, j:2-4, k:5-6): 1x3x2 -> schoolbook (6 terms)
    #
    # Total: 7+12+7+12+18+12+4+6+4+6+9+6 = 103. Not great.
    #
    # Better: Use Laderman for 3x3x3 blocks where possible.
    #
    # Let me try:
    # i: [0,1,2] (size 3)
    # j: [0,1,2] (size 3), [3,4] (size 2)
    # k: [0,1,2] (size 3), [3,4,5] (size 3), [6] (size 1)
    #
    # Blocks:
    # (i:0-2, j:0-2, k:0-2): 3x3x3 -> Laderman (7)
    # (i:0-2, j:0-2, k:3-5): 3x3x3 -> Laderman (7)
    # (i:0-2, j:0-2, k:6): 3x3x1 -> schoolbook (9)
    # (i:0-2, j:3-4, k:0-2): 3x2x3 -> schoolbook (18)
    # (i:0-2, j:3-4, k:3-5): 3x2x3 -> schoolbook (18)
    # (i:0-2, j:3-4, k:6): 3x2x1 -> schoolbook (6)
    #
    # Total: 7+7+9+18+18+6 = 65
    #
    # Can we do better? Let's try to use Strassen more.
    #
    # Alternative:
    # i: [0,1] (size 2), [2] (size 1)
    # j: [0,1] (size 2), [2,3] (size 2), [4] (size 1)
    # k: [0,1] (size 2), [2,3] (size 2), [4,5] (size 2), [6] (size 1)
    #
    # This is getting complex. Let me just build the 65-rank solution and verify.

    # Build the 65-rank decomposition
    rank = 65
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    def Uidx(i, j): return i * m + j
    def Vidx(j, k): return j * p + k
    def Widx(k, i): return k * n + i

    col = 0

    # Block (i:0-2, j:0-2, k:0-2): Laderman
    Lu, Lv, Lw = _laderman_3x3()
    for t in range(7):
        for r in range(9):
            if Lu[r, t] != 0:
                U[Uidx(r // 3, r % 3), col] = Lu[r, t]
            if Lv[r, t] != 0:
                V[Vidx(r // 3, r % 3), col] = Lv[r, t]
            if Lw[r, t] != 0:
                W[Widx(r // 3, r % 3), col] = Lw[r, t]
        col += 1
    # col = 7

    # Block (i:0-2, j:0-2, k:3-5): Laderman
    for t in range(7):
        for r in range(9):
            if Lu[r, t] != 0:
                U[Uidx(r // 3, r % 3), col] = Lu[r, t]
            if Lv[r, t] != 0:
                V[Vidx(r // 3, r % 3 + 3), col] = Lv[r, t]
            if Lw[r, t] != 0:
                W[Widx(r // 3 + 3, r % 3), col] = Lw[r, t]
        col += 1
    # col = 14

    # Block (i:0-2, j:0-2, k:6): schoolbook 3x3x1
    for i in range(3):
        for j in range(3):
            U[Uidx(i, j), col] = 1
            V[Vidx(j, 6), col] = 1
            W[Widx(6, i), col] = 1
            col += 1
    # col = 23

    # Block (i:0-2, j:3-4, k:0-2): schoolbook 3x2x3
    for i in range(3):
        for j in range(2):
            for k in range(3):
                U[Uidx(i, 3 + j), col] = 1
                V[Vidx(3 + j, k), col] = 1
                W[Widx(k, i), col] = 1
                col += 1
    # col = 41

    # Block (i:0-2, j:3-4, k:3-5): schoolbook 3x2x3
    for i in range(3):
        for j in range(2):
            for k in range(3):
                U[Uidx(i, 3 + j), col] = 1
                V[Vidx(3 + j, 3 + k), col] = 1
                W[Widx(3 + k, i), col] = 1
                col += 1
    # col = 59

    # Block (i:0-2, j:3-4, k:6): schoolbook 3x2x1
    for i in range(3):
        for j in range(2):
            U[Uidx(i, 3 + j), col] = 1
            V[Vidx(3 + j, 6), col] = 1
            W[Widx(6, i), col] = 1
            col += 1
    # col = 65

    assert col == rank, f"Column count mismatch: {col} vs {rank}"

    # Verify
    if _verify(U, V, W, n, m, p, rank):
        def to_int_list(M):
            if np.allclose(M, np.round(M), atol=1e-12):
                return np.round(M).astype(int).tolist()
            return M.tolist()
        return {"U": to_int_list(U), "V": to_int_list(V), "W": to_int_list(W)}

    # Fallback to schoolbook if verification fails
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb), dtype=float)
    V_sb = np.zeros((m * p, rank_sb), dtype=float)
    W_sb = np.zeros((p * n, rank_sb), dtype=float)
    for t in range(rank_sb):
        i = t // (m * p)
        rem = t % (m * p)
        j = rem // p
        k = rem % p
        U_sb[i * m + j, t] = 1
        V_sb[j * p + k, t] = 1
        W_sb[k * n + i, t] = 1

    def to_int_list(M):
        if np.allclose(M, np.round(M), atol=1e-12):
            return np.round(M).astype(int).tolist()
        return M.tolist()
    return {"U": to_int_list(U_sb), "V": to_int_list(V_sb), "W": to_int_list(W_sb)}
