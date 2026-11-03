import time
import numpy as np

DIMENSIONS = [3, 3, 3]

def construct():
    start_time = time.time()
    deadline = 110.0  # safety margin before 120s limit
    n, m, p = DIMENSIONS

    def iu(i, j): return i * m + j
    def iv(j, k): return j * p + k
    def iw(k, i): return k * n + i

    # ============================================================
    # Attempt rank-23 decomposition (Alon-Yuster-Zwick style)
    # 7 Strassen terms for 2x2 core + 16 border terms
    # ============================================================
    R23 = 23
    U = np.zeros((n * m, R23))
    V = np.zeros((m * p, R23))
    W = np.zeros((p * n, R23))

    # Strassen's 7 multiplications for 2x2 core (i,j,k in {0,1})
    # m1 = (a00+a11)(b00+b11) -> contributes to C00, C11
    r = 0
    U[iu(0,0), r] = 1; U[iu(1,1), r] = 1
    V[iv(0,0), r] = 1; V[iv(1,1), r] = 1
    W[iw(0,0), r] = 1; W[iw(1,1), r] = 1

    # m2 = (a10+a11)b00 -> contributes to C01, C11
    r = 1
    U[iu(1,0), r] = 1; U[iu(1,1), r] = 1
    V[iv(0,0), r] = 1
    W[iw(0,1), r] = 1; W[iw(1,1), r] = -1

    # m3 = a00(b01-b11) -> contributes to C10, C11
    r = 2
    U[iu(0,0), r] = 1
    V[iv(0,1), r] = 1; V[iv(1,1), r] = -1
    W[iw(1,0), r] = 1; W[iw(1,1), r] = 1

    # m4 = a11(b10-b00) -> contributes to C00, C01
    r = 3
    U[iu(1,1), r] = 1
    V[iv(1,0), r] = 1; V[iv(0,0), r] = -1
    W[iw(0,0), r] = 1; W[iw(0,1), r] = 1

    # m5 = (a00+a01)b11 -> contributes to C10, C00
    r = 4
    U[iu(0,0), r] = 1; U[iu(0,1), r] = 1
    V[iv(1,1), r] = 1
    W[iw(1,0), r] = 1; W[iw(0,0), r] = -1

    # m6 = (a10-a00)(b00+b01) -> contributes to C11
    r = 5
    U[iu(1,0), r] = 1; U[iu(0,0), r] = -1
    V[iv(0,0), r] = 1; V[iv(0,1), r] = 1
    W[iw(1,1), r] = 1

    # m7 = (a01-a11)(b10+b11) -> contributes to C00
    r = 6
    U[iu(0,1), r] = 1; U[iu(1,1), r] = -1
    V[iv(1,0), r] = 1; V[iv(1,1), r] = 1
    W[iw(0,0), r] = 1

    # 16 border terms for entries where at least one index is 2
    # These cover all (i,j,k) with i=2 or j=2 or k=2 (19 total entries)
    # using 16 terms by cleverly combining some border multiplications
    #
    # Strategy: Use direct terms for most, but merge where possible
    # Border entries: (i,j,k) where i>=2 or j>=2 or k>=2
    # Count: total 27 - core 8 = 19 entries
    #
    # Terms 7-15: 9 terms covering specific border patterns
    # Terms 16-22: 7 more terms
    #
    # Let's enumerate the 19 border entries:
    # i=0: (0,0,2),(0,1,2),(0,2,0),(0,2,1),(0,2,2) -> 5
    # i=1: (1,0,2),(1,1,2),(1,2,0),(1,2,1),(1,2,2) -> 5
    # i=2: (2,0,0),(2,0,1),(2,0,2),(2,1,0),(2,1,1),(2,1,2),(2,2,0),(2,2,1),(2,2,2) -> 9
    # Total: 19
    #
    # We need 16 terms. Let's try to group:
    # For k=2 column (j=0,1; i=0,1): 4 entries -> 4 terms
    # For j=2 column (i=0,1; k=0,1): 4 entries -> 4 terms  
    # For i=2 row (j=0,1,2; k=0,1,2): 9 entries -> 8 terms (one can be merged)
    # That's 16 terms. Let's construct them.

    col = 7
    # k=2, i in {0,1}, j in {0,1}: C[i,j,2] = A[i,j]*B[j,2]
    for i in range(2):
        for j in range(2):
            U[iu(i,j), col] = 1
            V[iv(j,2), col] = 1
            W[iw(2,i), col] = 1
            col += 1

    # j=2, i in {0,1}, k in {0,1}: C[i,2,k] = A[i,2]*B[2,k]
    for i in range(2):
        for k in range(2):
            U[iu(i,2), col] = 1
            V[iv(2,k), col] = 1
            W[iw(k,i), col] = 1
            col += 1

    # i=2, j in {0,1,2}, k in {0,1,2}: C[2,j,k] = A[2,j]*B[j,k]
    # 9 entries, need 8 terms (merge C[2,0,0] with something or use separate)
    # Actually let's just use 9 direct terms and see if we're at rank 26
    # No wait, 7+4+4+9 = 24. Still > 23.
    #
    # Let me reconsider. The known rank-23 uses a different structure.
    # Let me try a different grouping:
    #
    # Actually, let's just go with 26 (verified) and skip the risky 23.
    # The 19 border terms give rank 26 which is verified.

    # Reset and use the verified rank-26
    R26 = 26
    U26 = np.zeros((n * m, R26))
    V26 = np.zeros((m * p, R26))
    W26 = np.zeros((p * n, R26))

    # Copy Strassen terms 0-6
    for r in range(7):
        U26[:, r] = U[:, r]
        V26[:, r] = V[:, r]
        W26[:, r] = W[:, r]

    # 19 border terms
    col = 7
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    continue
                U26[iu(i,j), col] = 1
                V26[iv(j,k), col] = 1
                W26[iw(k,i), col] = 1
                col += 1

    # Verify rank-26
    U26_reshaped = U26.reshape((n, m, R26))
    V26_reshaped = V26.reshape((m, p, R26))
    W26_reshaped = W26.reshape((p, n, R26))
    T_recon = np.einsum('ijr,jkr,kir->ijk', U26_reshaped, V26_reshaped, W26_reshaped)
    T_target = np.ones((n, m, p))

    if np.allclose(T_recon, T_target, atol=1e-10):
        return {"U": U26.tolist(), "V": V26.tolist(), "W": W26.tolist()}

    # Fallback: Schoolbook 27
    R27 = n * m * p
    U27 = np.zeros((n * m, R27))
    V27 = np.zeros((m * p, R27))
    W27 = np.zeros((p * n, R27))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U27[iu(i,j), col] = 1
                V27[iv(j,k), col] = 1
                W27[iw(k,i), col] = 1
                col += 1

    return {"U": U27.tolist(), "V": V27.tolist(), "W": W27.tolist()}
