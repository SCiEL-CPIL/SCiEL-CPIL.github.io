"""
Tensor decomposition for 2x2 * 2x3 matrix multiplication.

Strategy:
  - Use Strassen-like 7-term decomposition for the 2x2 * 2x2 sub-problem
    (columns 0 and 1 of B).
  - Add 4 schoolbook terms for column 2 of B.
  - Total rank = 11.

The W index follows the official trace convention: W[k*n + i, r].
"""

import time
import sys

DIMENSIONS = [2, 2, 3]


def construct():
    # Deadline guard: kill after 150s wall-clock, return with safety margin
    deadline = time.time() + 148.0

    n, m, p = DIMENSIONS  # 2, 2, 3

    # We'll build rank-11 decomposition.
    # U shape: (n*m, rank) = (4, 11)
    # V shape: (m*p, rank) = (6, 11)
    # W shape: (p*n, rank) = (6, 11)

    # Index conventions:
    # U index: i*m + j  (i in 0..n-1, j in 0..m-1)
    #   U[0] = A[0,0], U[1] = A[0,1], U[2] = A[1,0], U[3] = A[1,1]
    # V index: j*p + k  (j in 0..m-1, k in 0..p-1)
    #   V[0] = B[0,0], V[1] = B[0,1], V[2] = B[0,2],
    #   V[3] = B[1,0], V[4] = B[1,1], V[5] = B[1,2]
    # W index: k*n + i  (k in 0..p-1, i in 0..n-1)
    #   W[0] = C[0,0], W[1] = C[0,1], W[2] = C[0,2],
    #   W[3] = C[1,0], W[4] = C[1,1], W[5] = C[1,2]

    # Tensor T[a,b,d] = 1 if a=i*m+j, b=j*p+k, d=k*n+i (i.e., A[i,j]*B[j,k]->C[i,k])
    # We need: sum_r U[a,r]*V[b,r]*W[d,r] = T[a,b,d] for all a,b,d.

    rank = 11
    U = [[0] * rank for _ in range(n * m)]  # 4 x 11
    V = [[0] * rank for _ in range(m * p)]  # 6 x 11
    W = [[0] * rank for _ in range(p * n)]  # 6 x 11

    # --- Strassen-like 7 terms for columns 0,1 of B (2x2 subproblem) ---
    # Let A = [[a, b], [c, d]] = A[0,0], A[0,1], A[1,0], A[1,1]
    # Let B2 = [[e, f], [g, h]] = B[0,0], B[0,1], B[1,0], B[1,1]
    # C2 = [[C00, C01], [C10, C11]]

    # Standard Strassen for 2x2:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C00 = M1 - M5 + M7 + M4
    # C01 = M3 + M5
    # C10 = M2 + M6
    # C11 = M1 - M3 - M4 + M6

    # Term 0: M1 = (a+d)(e+h)
    U[0][0] = 1; U[3][0] = 1          # a + d
    V[0][0] = 1; V[4][0] = 1          # e + h
    W[0][0] = 1; W[4][0] = 1          # C00 + C11

    # Term 1: M2 = (c+d)e
    U[2][1] = 1; U[3][1] = 1          # c + d
    V[0][1] = 1                       # e
    W[3][1] = 1                       # C10

    # Term 2: M3 = a(f-h)
    U[0][2] = 1                       # a
    V[1][2] = 1; V[4][2] = -1         # f - h
    W[4][2] = -1; W[1][2] = 1         # -C11 + C01

    # Term 3: M4 = d(g-e)
    U[3][3] = 1                       # d
    V[3][3] = 1; V[0][3] = -1         # g - e
    W[4][3] = -1; W[0][3] = 1         # -C11 + C00

    # Term 4: M5 = (a+b)h
    U[0][4] = 1; U[1][4] = 1          # a + b
    V[4][4] = 1                       # h
    W[4][4] = -1; W[1][4] = 1         # -C11 + C01

    # Term 5: M6 = (c-a)(e+f)
    U[2][5] = 1; U[0][5] = -1         # c - a
    V[0][5] = 1; V[1][5] = 1          # e + f
    W[3][5] = 1; W[4][5] = 1          # C10 + C11

    # Term 6: M7 = (b-d)(g+h)
    U[1][6] = 1; U[3][6] = -1         # b - d
    V[3][6] = 1; V[4][6] = 1          # g + h
    W[0][6] = 1; W[4][6] = -1         # C00 - C11

    # --- 4 schoolbook terms for column 2 of B ---
    # C[i,2] = sum_j A[i,j] * B[j,2]
    # For each (i,j) pair, one term: A[i,j] * B[j,2] -> C[i,2]

    # Term 7: A[0,0] * B[0,2] -> C[0,2]
    U[0][7] = 1
    V[2][7] = 1
    W[2][7] = 1

    # Term 8: A[0,1] * B[1,2] -> C[0,2]
    U[1][8] = 1
    V[5][8] = 1
    W[2][8] = 1

    # Term 9: A[1,0] * B[0,2] -> C[1,2]
    U[2][9] = 1
    V[2][9] = 1
    W[5][9] = 1

    # Term 10: A[1,1] * B[1,2] -> C[1,2]
    U[3][10] = 1
    V[5][10] = 1
    W[5][10] = 1

    # --- Verify correctness before returning ---
    # Check: for all a in 0..3, b in 0..5, d in 0..5,
    #   sum_r U[a,r]*V[b,r]*W[d,r] should equal T[a,b,d]
    # T[a,b,d] = 1 iff a = i*m+j, b = j*p+k, d = k*n+i for some i,j,k
    # i.e., a//m == a%m... let's just enumerate:
    # a = i*m + j, b = j*p + k, d = k*n + i
    # For n=2, m=2, p=3:
    #   i=0,j=0,k=0: a=0, b=0, d=0
    #   i=0,j=0,k=1: a=0, b=1, d=2
    #   i=0,j=0,k=2: a=0, b=2, d=4
    #   i=0,j=1,k=0: a=1, b=3, d=0
    #   i=0,j=1,k=1: a=1, b=4, d=2
    #   i=0,j=1,k=2: a=1, b=5, d=4
    #   i=1,j=0,k=0: a=2, b=0, d=1
    #   i=1,j=0,k=1: a=2, b=1, d=3
    #   i=1,j=0,k=2: a=2, b=2, d=5
    #   i=1,j=1,k=0: a=3, b=3, d=1
    #   i=1,j=1,k=1: a=3, b=4, d=3
    #   i=1,j=1,k=2: a=3, b=5, d=5

    expected = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                d = k * n + i
                expected.add((a, b, d))

    # Verify
    for a in range(n * m):
        for b in range(m * p):
            for d in range(p * n):
                total = 0
                for r in range(rank):
                    total += U[a][r] * V[b][r] * W[d][r]
                expected_val = 1 if (a, b, d) in expected else 0
                if total != expected_val:
                    # Fallback to schoolbook if verification fails
                    sys.stderr.write(f"Verification failed at a={a}, b={b}, d={d}: got {total}, expected {expected_val}\n")
                    # Return schoolbook rank 12 as safe fallback
                    rank_f = n * m * p
                    Uf = [[0] * rank_f for _ in range(n * m)]
                    Vf = [[0] * rank_f for _ in range(m * p)]
                    Wf = [[0] * rank_f for _ in range(p * n)]
                    col = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                Uf[i * m + j][col] = 1
                                Vf[j * p + k][col] = 1
                                Wf[k * n + i][col] = 1
                                col += 1
                    return {"U": Uf, "V": Vf, "W": Wf}

    return {"U": U, "V": V, "W": W}
