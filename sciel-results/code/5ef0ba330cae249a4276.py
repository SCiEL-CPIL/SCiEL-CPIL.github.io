"""Improved program for matrix multiplication tensor decomposition (2,2,3)."""
import time

DIMENSIONS = [2, 2, 3]


def _build_rank11():
    """
    Hybrid decomposition:
    - 7 terms from Strassen for the 2x2 block (k=0,1)
    - 4 terms from schoolbook for the remaining column (k=2)
    Total rank = 11.
    """
    n, m, p = 2, 2, 3
    rank = 11
    U = [[0] * rank for _ in range(n * m)]  # 4 x 11
    V = [[0] * rank for _ in range(m * p)]  # 6 x 11
    W = [[0] * rank for _ in range(p * n)]  # 6 x 11

    # Strassen decomposition for 2x2 matrix multiplication
    # For 2x2 x 2x2 -> 2x2
    # Standard Strassen formulas:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22) B11
    # M3 = A11 (B12 - B22)
    # M4 = A22 (B21 - B11)
    # M5 = (A11 + A12) B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6

    # In our convention:
    # U[i*m+j, c] -> U[(i,j), c]
    # V[j*p+k, c] -> V[(j,k), c]
    # W[k*n+i, c] -> W[(k,i), c]

    # For the 2x2 block (k=0,1), Strassen terms:
    # c=0: M1
    U[0][0] = 1   # A11 (i=0,j=0)
    U[3][0] = 1   # A22 (i=1,j=1)
    V[0][0] = 1   # B11 (j=0,k=0)
    V[3][0] = 1   # B22 (j=1,k=1)
    W[0][0] = 1   # C11 (k=0,i=0)
    W[2][0] = 1   # C22 (k=1,i=1)
    W[1][0] = 1   # C12 (k=0,i=1) -> wait, let me re-derive

    # Let me be more careful. The output W[k*n+i] means:
    # k=0,i=0 -> W[0] -> C[0][0] = C11
    # k=0,i=1 -> W[1] -> C[0][1] = C12
    # k=1,i=0 -> W[2] -> C[1][0] = C21
    # k=1,i=1 -> W[3] -> C[1][1] = C22

    # Strassen:
    # M1 = (A11+A22)(B11+B22) -> contributes to C11 and C22
    # M2 = (A21+A22)B11 -> contributes to C21 and C22
    # M3 = A11(B12-B22) -> contributes to C12 and C22
    # M4 = A22(B21-B11) -> contributes to C11 and C21
    # M5 = (A11+A12)B22 -> contributes to C12 and C11
    # M6 = (A21-A11)(B11+B12) -> contributes to C22
    # M7 = (A12-A22)(B21+B22) -> contributes to C11

    # Let me reset and build carefully.

    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(6)]
    W = [[0] * rank for _ in range(6)]

    # c=0: M1 = (A11+A22)(B11+B22)
    # U: A11(0,0)=1, A22(1,1)=1 -> U[0][0]=1, U[3][0]=1
    # V: B11(0,0)=1, B22(1,1)=1 -> V[0][0]=1, V[3][0]=1
    # W: C11(0,0)=1, C22(1,1)=1 -> W[0][0]=1, W[3][0]=1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1

    # c=1: M2 = (A21+A22)B11
    # U: A21(1,0)=1, A22(1,1)=1 -> U[2][1]=1, U[3][1]=1
    # V: B11(0,0)=1 -> V[0][1]=1
    # W: C21(1,0)=1, C22(1,1)=1 -> W[2][1]=1, W[3][1]=1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = 1

    # c=2: M3 = A11(B12-B22)
    # U: A11(0,0)=1 -> U[0][2]=1
    # V: B12(0,1)=1, B22(1,1)=-1 -> V[1][2]=1, V[3][2]=-1
    # W: C12(0,1)=1, C22(1,1)=1 -> W[1][2]=1, W[3][2]=1
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1

    # c=3: M4 = A22(B21-B11)
    # U: A22(1,1)=1 -> U[3][3]=1
    # V: B21(1,0)=1, B11(0,0)=-1 -> V[2][3]=1, V[0][3]=-1
    # W: C11(0,0)=1, C21(1,0)=1 -> W[0][3]=1, W[2][3]=1
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1

    # c=4: M5 = (A11+A12)B22
    # U: A11(0,0)=1, A12(0,1)=1 -> U[0][4]=1, U[1][4]=1
    # V: B22(1,1)=1 -> V[3][4]=1
    # W: C11(0,0)=-1, C12(0,1)=1 -> W[0][4]=-1, W[1][4]=1
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1

    # c=5: M6 = (A21-A11)(B11+B12)
    # U: A21(1,0)=1, A11(0,0)=-1 -> U[2][5]=1, U[0][5]=-1
    # V: B11(0,0)=1, B12(0,1)=1 -> V[0][5]=1, V[1][5]=1
    # W: C22(1,1)=1 -> W[3][5]=1
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1

    # c=6: M7 = (A12-A22)(B21+B22)
    # U: A12(0,1)=1, A22(1,1)=-1 -> U[1][6]=1, U[3][6]=-1
    # V: B21(1,0)=1, B22(1,1)=1 -> V[2][6]=1, V[3][6]=1
    # W: C11(0,0)=1 -> W[0][6]=1
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1

    # Schoolbook for column 2 (k=2), 4 terms
    # c=7: A11*B12 -> C12
    U[0][7] = 1   # A11
    V[2][7] = 1   # B12 (j=0,k=2)
    W[1][7] = 1   # C12 (k=2,i=0? No, C[0][2])

    # Wait, let me recheck the indexing.
    # Output C is 2x3, so C[i][k] for i=0,1 and k=0,1,2
    # W[k*n+i] = W[k*2+i]
    # C[0][0] = W[0], C[0][1] = W[1], C[0][2] = W[2]
    # C[1][0] = W[3], C[1][1] = W[4], C[1][2] = W[5]

    # Let me redo the W assignments for Strassen with correct indexing:
    # C11 = C[0][0] -> W[0]
    # C12 = C[0][1] -> W[1]
    # C21 = C[1][0] -> W[3]  (not W[2]!)
    # C22 = C[1][1] -> W[4]  (not W[3]!)

    # I made an error. Let me rebuild.
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(6)]
    W = [[0] * rank for _ in range(6)]

    # c=0: M1 = (A11+A22)(B11+B22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1   # C[0][0]
    W[4][0] = 1   # C[1][1]

    # c=1: M2 = (A21+A22)B11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[3][1] = 1   # C[1][0]
    W[4][1] = 1   # C[1][1]

    # c=2: M3 = A11(B12-B22)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1   # C[0][1]
    W[4][2] = 1   # C[1][1]

    # c=3: M4 = A22(B21-B11)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1   # C[0][0]
    W[3][3] = 1   # C[1][0]

    # c=4: M5 = (A11+A12)B22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1  # C[0][0]
    W[1][4] = 1   # C[0][1]

    # c=5: M6 = (A21-A11)(B11+B12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[4][5] = 1   # C[1][1]

    # c=6: M7 = (A12-A22)(B21+B22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1   # C[0][0]

    # Schoolbook for k=2 column: 4 terms
    # c=7: A11*B12 -> C[0][2]
    U[0][7] = 1
    V[2][7] = 1   # B[0][2]
    W[2][7] = 1   # C[0][2]

    # c=8: A12*B22 -> C[0][2]
    U[1][8] = 1
    V[5][8] = 1   # B[1][2]
    W[2][8] = 1   # C[0][2]

    # c=9: A21*B12 -> C[1][2]
    U[2][9] = 1
    V[2][9] = 1   # B[0][2]
    W[5][9] = 1   # C[1][2]

    # c=10: A22*B22 -> C[1][2]
    U[3][10] = 1
    V[5][10] = 1  # B[1][2]
    W[5][10] = 1  # C[1][2]

    return {"U": U, "V": V, "W": W}


def construct():
    """
    Return a rank-11 decomposition using Strassen (7 terms for 2x2x2)
    plus schoolbook (4 terms for the third column).
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    # Direct construction is O(1), no search needed
    result = _build_rank11()

    return result
