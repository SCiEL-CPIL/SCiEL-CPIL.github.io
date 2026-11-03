"""Matrix multiplication 2,3,5 — rank-28 decomposition via two Strassen 2x2x2 blocks."""
import numpy as np
import time

def construct():
    """
    Build the known rank-28 tensor decomposition:
    two disjoint Strassen 2x2x2 sub-tensors plus schoolbook for the remaining
    (i,j,k) triples not covered by those blocks.
    """
    # Deadline guard
    start_time = time.time()
    DEADLINE = 120.0

    n, m, p = 2, 3, 5
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    c = 0

    # ---- Block A: Strassen on i in {0,1}, j in {0,1}, k in {0,1} ----
    # a_00=U[0], a_01=U[1], a_10=U[3], a_11=U[4]
    # b_00=V[0], b_01=V[1], b_10=V[3], b_11=V[4]
    # c_00=W[0], c_01=W[1], c_10=W[2], c_11=W[3]

    # M1: (a00+a11)(b00+b11) -> c00+c11
    U[0][c]=1; U[4][c]=1; V[0][c]=1; V[4][c]=1; W[0][c]=1; W[3][c]=1; c+=1
    # M2: a10(b00-b10) -> c10-c11
    U[3][c]=1; V[0][c]=1; V[3][c]=-1; W[2][c]=1; W[3][c]=-1; c+=1
    # M3: a01(b10+b11) -> c01+c11
    U[1][c]=1; V[3][c]=1; V[4][c]=1; W[1][c]=1; W[3][c]=1; c+=1
    # M4: a11(b01-b00) -> c00+c01
    U[4][c]=1; V[1][c]=1; V[0][c]=-1; W[0][c]=1; W[1][c]=1; c+=1
    # M5: a00(b01-b00) -> -c00+c01
    U[0][c]=1; V[1][c]=1; V[0][c]=-1; W[0][c]=-1; W[1][c]=1; c+=1
    # M6: a11(b10-b11) -> c10
    U[4][c]=1; V[3][c]=1; V[4][c]=-1; W[2][c]=1; c+=1
    # M7: a01(b00+b10) -> c01
    U[1][c]=1; V[0][c]=1; V[3][c]=1; W[1][c]=1; c+=1

    # ---- Block B: Strassen on i in {0,1}, j in {0,1}, k in {3,4} ----
    # a_00=U[0], a_01=U[1], a_10=U[3], a_11=U[4]
    # b_00=V[3] (j=0,k=3), b_01=V[4] (j=0,k=4), b_10=V[8] (j=1,k=3), b_11=V[9] (j=1,k=4)
    # c_00=W[6] (k=3,i=0), c_01=W[8] (k=4,i=0), c_10=W[7] (k=3,i=1), c_11=W[9] (k=4,i=1)

    # M1: (a00+a11)(b00+b11) -> c00+c11
    U[0][c]=1; U[4][c]=1; V[3][c]=1; V[9][c]=1; W[6][c]=1; W[9][c]=1; c+=1
    # M2: a10(b00-b10) -> c10-c11
    U[3][c]=1; V[3][c]=1; V[8][c]=-1; W[7][c]=1; W[9][c]=-1; c+=1
    # M3: a01(b10+b11) -> c01+c11
    U[1][c]=1; V[8][c]=1; V[9][c]=1; W[8][c]=1; W[9][c]=1; c+=1
    # M4: a11(b01-b00) -> c00+c01
    U[4][c]=1; V[4][c]=1; V[3][c]=-1; W[6][c]=1; W[8][c]=1; c+=1
    # M5: a00(b01-b00) -> -c00+c01
    U[0][c]=1; V[4][c]=1; V[3][c]=-1; W[6][c]=-1; W[8][c]=1; c+=1
    # M6: a11(b10-b11) -> c10
    U[4][c]=1; V[8][c]=1; V[9][c]=-1; W[7][c]=1; c+=1
    # M7: a01(b00+b10) -> c01
    U[1][c]=1; V[3][c]=1; V[8][c]=1; W[8][c]=1; c+=1

    # ---- 14 schoolbook terms for remaining (i,j,k) triples ----
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (j in (0,1) and k in (0,1)):
                    continue
                if (j in (0,1) and k in (3,4)):
                    continue
                U[i*m+j][c] = 1
                V[j*p+k][c] = 1
                W[k*n+i][c] = 1
                c += 1

    assert c == rank, f"Expected {rank} rank terms, got {c}"

    # Convert to numpy arrays for evaluation
    return {
        "U": np.array(U, dtype=np.float64),
        "V": np.array(V, dtype=np.float64),
        "W": np.array(W, dtype=np.float64)
    }
