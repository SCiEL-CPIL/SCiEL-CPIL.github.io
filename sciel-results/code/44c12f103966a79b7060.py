import time
import numpy as np

DIMENSIONS = [4, 5, 5]
START_TIME = time.time()
TIME_LIMIT = 115

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Return schoolbook decomposition for n x m times m x p -> n x p."""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return U, V, W, rank


def _strassen_2x2():
    """
    Strassen decomposition for 2x2 * 2x2 -> 2x2.
    Rank 7.
    A indices: 00,01,10,11 -> 0,1,2,3
    B indices: 00,01,10,11 -> 0,1,2,3
    W indices: k*n+i -> 0,1,2,3 (n=2)
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    # m1 = (a+d)(e+h)
    U[0][0] = 1.0; U[3][0] = 1.0
    V[0][0] = 1.0; V[3][0] = 1.0
    W[0][0] = 1.0  # C00

    # m2 = (c+d)(e+f)
    U[2][1] = 1.0; U[3][1] = 1.0
    V[0][1] = 1.0; V[1][1] = 1.0
    W[2][1] = 1.0  # C10
    W[3][1] = -1.0 # C11

    # m3 = a(f-h)
    U[0][2] = 1.0
    V[1][2] = 1.0; V[3][2] = -1.0
    W[1][2] = 1.0  # C01
    W[3][2] = 1.0  # C11

    # m4 = d(g-e)
    U[3][3] = 1.0
    V[2][3] = 1.0; V[0][3] = -1.0
    W[0][3] = 1.0  # C00
    W[2][3] = 1.0  # C10

    # m5 = (a+b)h
    U[0][4] = 1.0; U[1][4] = 1.0
    V[3][4] = 1.0
    W[0][4] = -1.0 # C00
    W[1][4] = 1.0  # C01

    # m6 = (c-d)e
    U[2][5] = 1.0; U[3][5] = -1.0
    V[0][5] = 1.0
    W[3][5] = 1.0  # C11

    # m7 = (b-d)(f+g)
    U[1][6] = 1.0; U[3][6] = -1.0
    V[1][6] = 1.0; V[2][6] = 1.0
    W[0][6] = 1.0  # C00

    return U, V, W, rank


def _strassen_3x3():
    """
    Decomposition for 3x3 * 3x3 -> 3x3 using block Strassen.
    Split into 2x2 + 1x1 blocks.
    C11 (2x2) = A11*B11 + A12*B21
    C12 (2x1) = A11*B12 + A12*B22
    C21 (1x2) = A21*B11 + A22*B21
    C22 (1x1) = A21*B12 + A22*B22
    
    Sub-problems and their ranks:
    1. 2x2 * 2x2 -> 2x2: Strassen rank 7
    2. 2x1 * 1x2 -> 2x2: outer product, rank 4
    3. 2x1 * 1x1 -> 2x1: rank 2
    4. 1x2 * 2x2 -> 1x2: rank 4
    5. 1x1 * 1x2 -> 1x2: rank 2
    6. 1x2 * 2x1 -> 1x1: dot product, rank 2
    7. 1x1 * 1x1 -> 1x1: rank 1
    Total: 7+4+2+4+2+2+1 = 22
    """
    n, m, p = 3, 3, 3
    rank = 22
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    
    # 1. A11*B11 -> C11 (2x2 * 2x2 -> 2x2), Strassen rank 7
    U1, V1, W1, r1 = _strassen_2x2()
    for i in range(2):
        for j in range(2):
            for c in range(r1):
                U[i * 3 + j][col + c] = U1[i * 2 + j][c]
    for j in range(2):
        for k in range(2):
            for c in range(r1):
                V[j * 3 + k][col + c] = V1[j * 2 + k][c]
    for k in range(2):
        for i in range(2):
            for c in range(r1):
                W[k * 3 + i][col + c] = W1[k * 2 + i][c]
    col += r1
    
    # 2. A12*B21 -> C11 (2x1 * 1x2 -> 2x2), rank 4
    U2, V2, W2, r2 = _schoolbook(2, 1, 2)
    for i in range(2):
        for c in range(r2):
            U[i * 3 + 2][col + c] = U2[i * 1][c]
    for j in range(1):
        for k in range(2):
            for c in range(r2):
                V[(2 + j) * 3 + k][col + c] = V2[j * 2 + k][c]
    for k in range(2):
        for i in range(2):
            for c in range(r2):
                W[k * 3 + i][col + c] += W2[k * 2 + i][c]
    col += r2
    
    # 3. A11*B12 -> C12 (2x2 * 2x1 -> 2x1), rank 4
    U3, V3, W3, r3 = _schoolbook(2, 2, 1)
    for i in range(2):
        for j in range(2):
            for c in range(r3):
                U[i * 3 + j][col + c] = U3[i * 2 + j][c]
    for j in range(2):
        for k in range(1):
            for c in range(r3):
                V[j * 3 + (2 + k)][col + c] = V3[j * 1 + k][c]
    for k in range(1):
        for i in range(2):
            for c in range(r3):
                W[(2 + k) * 3 + i][col + c] = W3[k * 2 + i][c]
    col += r3
    
    # 4. A12*B22 -> C12 (2x1 * 1x1 -> 2x1), rank 2
    U4, V4, W4, r4 = _schoolbook(2, 1, 1)
    for i in range(2):
        for c in range(r4):
            U[i * 3 + 2][col + c] = U4[i * 1][c]
    for j in range(1):
        for k in range(1):
            for c in range(r4):
                V[(2 + j) * 3 + (2 + k)][col + c] = V4[j * 1 + k][c]
    for k in range(1):
        for i in range(2):
            for c in range(r4):
                W[(2 + k) * 3 + i][col + c] += W4[k * 2 + i][c]
    col += r4
    
    # 5. A21*B11 -> C21 (1x2 * 2x2 -> 1x2), rank 4
    U5, V5, W5, r5 = _schoolbook(1, 2, 2)
    for i in range(1):
        for j in range(2):
            for c in range(r5):
                U[(2 + i) * 3 + j][col + c] = U5[i * 2 + j][c]
    for j in range(2):
        for k in range(2):
            for c in range(r5):
                V[j * 3 + k][col + c] = V5[j * 2 + k][c]
    for k in range(2):
        for i in range(1):
            for c in range(r5):
                W[k * 3 + (2 + i)][col + c] = W5[k * 1 + i][c]
    col += r5
    
    # 6. A22*B21 -> C21 (1x1 * 1x2 -> 1x2), rank 2
    U6, V6, W6, r6 = _schoolbook(1, 1, 2)
    for i in range(1):
        for c in range(r6):
            U[(2 + i) * 3 + 2][col + c] = U6[i * 1][c]
    for j in range(1):
        for k in range(2):
            for c in range(r6):
                V[(2 + j) * 3 + k][col + c] = V6[j * 2 + k][c]
    for k in range(2):
        for i in range(1):
            for c in range(r6):
                W[k * 3 + (2 + i)][col + c] += W6[k * 1 + i][c]
    col += r6
    
    # 7. A21*B12 -> C22 (1x2 * 2x1 -> 1x1), rank 2
    U7, V7, W7, r7 = _schoolbook(1, 2, 1)
    for i in range(1):
        for j in range(2):
            for c in range(r7):
                U[(2 + i) * 3 + j][col + c] = U7[i * 2 + j][c]
    for j in range(2):
        for k in range(1):
            for c in range(r7):
                V[j * 3 + (2 + k)][col + c] = V7[j * 1 + k][c]
    for k in range(1):
        for i in range(1):
            for c in range(r7):
                W[(2 + k) * 3 + (2 + i)][col + c] = W7[k * 1 + i][c]
    col += r7
    
    # 8. A22*B22 -> C22 (1x1 * 1x1 -> 1x1), rank 1
    U8, V8, W8, r8 = _schoolbook(1, 1, 1)
    for i in range(1):
        for c in range(r8):
            U[(2 + i) * 3 + 2][col + c] = U8[i * 1][c]
    for j in range(1):
        for k in range(1):
            for c in range(r8):
                V[(2 + j) * 3 + (2 + k)][col + c] = V8[j * 1 + k][c]
    for k in range(1):
        for i in range(1):
            for c in range(r8):
                W[(2 + k) * 3 + (2 + i)][col + c] += W8[k * 1 + i][c]
    col += r8
    
    return U, V, W, col


def _verify_decomposition(U, V, W, n, m, p):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    # Use numpy for speed
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # T[i,j,k,r] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+r, c]
    # We want T[i,j,k,r] = delta_{i,r} delta_{j,k}
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for r in range(n):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+r][c]
                    expected = 1.0 if (i == r and j == k) else 0.0
                    if abs(s - expected) > 1e-9:
                        return False
    return True


def _combine_decompositions(U1, V1, W1, U2, V2, W2, n, m, p):
    """
    Combine two decompositions that target the same output tensor.
    This is only valid if the decompositions are for disjoint parts of the tensor
    or if we are summing them (which is not standard for rank minimization unless we can cancel).
    Actually, to combine, we just concatenate the columns.
    The resulting decomposition is the sum of the two tensors.
    If T1 + T2 = T_target, this works.
    """
    r1 = len(W1[0])
    r2 = len(W2[0])
    r = r1 + r2
    U = []
    for row in range(n * m):
        U.append(U1[row] + U2[row])
    V = []
    for row in range(m * p):
        V.append(V1[row] + V2[row])
    W = []
    for row in range(p * n):
        W.append(W1[row] + W2[row])
    return U, V, W, r


def _decompose_2x3x3():
    """
    Decompose 2x3 * 3x3 -> 2x3.
    Split m=3 into 2+1, p=3 into 2+1.
    Sub-problems:
    1. 2x2 * 2x2 -> 2x2: Strassen rank 7
    2. 2x1 * 1x2 -> 2x2: rank 4
    3. 2x2 * 2x1 -> 2x1: rank 4
    4. 2x1 * 1x1 -> 2x1: rank 2
    Total: 7+4+4+2 = 17 (vs schoolbook 18)
    """
    n, m, p = 2, 3, 3
    rank = 17
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    
    # 1. A11*B11 -> C11 (2x2 * 2x2 -> 2x2), Strassen
    U1, V1, W1, r1 = _strassen_2x2()
    for i in range(2):
        for j in range(2):
            for c in range(r1):
                U[i * 3 + j][col + c] = U1[i * 2 + j][c]
    for j in range(2):
        for k in range(2):
            for c in range(r1):
                V[j * 3 + k][col + c] = V1[j * 2 + k][c]
    for k in range(2):
        for i in range(2):
            for c in range(r1):
                W[k * 2 + i][col + c] = W1[k * 2 + i][c]
    col += r1
    
    # 2. A12*B21 -> C11 (2x1 * 1x2 -> 2x2), rank 4
    U2, V2, W2, r2 = _schoolbook(2, 1, 2)
    for i in range(2):
        for c in range(r2):
            U[i * 3 + 2][col + c] = U2[i * 1][c]
    for j in range(1):
        for k in range(2):
            for c in range(r2):
                V[(2 + j) * 3 + k][col + c] = V2[j * 2 + k][c]
    for k in range(2):
        for i in range(2):
            for c in range(r2):
                W[k * 2 + i][col + c] += W2[k * 2 + i][c]
    col += r2
    
    # 3. A11*B12 -> C12 (2x2 * 2x1 -> 2x1), rank 4
    U3, V3, W3, r3 = _schoolbook(2, 2, 1)
    for i in range(2):
        for j in range(2):
            for c in range(r3):
                U[i * 3 + j][col + c] = U3[i * 2 + j][c]
    for j in range(2):
        for k in range(1):
            for c in range(r3):
                V[j * 3 + (2 + k)][col + c] = V3[j * 1 + k][c]
    for k in range(1):
        for i in range(2):
            for c in range(r3):
                W[(2 + k) * 2 + i][col + c] = W3[k * 2 + i][c]
    col += r3
    
    # 4. A12*B22 -> C12 (2x1 * 1x1 -> 2x1), rank 2
    U4, V4, W4, r4 = _schoolbook(2, 1, 1)
    for i in range(2):
        for c in range(r4):
            U[i * 3 + 2][col + c] = U4[i * 1][c]
    for j in range(1):
        for k in range(1):
            for c in range(r4):
                V[(2 + j) * 3 + (2 + k)][col + c] = V4[j * 1 + k][c]
    for k in range(1):
        for i in range(2):
            for c in range(r4):
                W[(2 + k) * 2 + i][col + c] += W4[k * 2 + i][c]
    col += r4
    
    return U, V, W, col


def _decompose_4x5x5():
    """
    Decompose 4x5 * 5x5 -> 4x5.
    Strategy: Split n=4 into two 2x blocks.
    For each 2x5 * 5x5 -> 2x5 block:
    Split m=5 into 2+3, p=5 into 2+3.
    Sub-problems:
    1. 2x2 * 2x2 -> 2x2: Strassen rank 7
    2. 2x3 * 3x2 -> 2x2: rank 12 (schoolbook)
    3. 2x2 * 2x3 -> 2x3: rank 12 (schoolbook)
    4. 2x3 * 3x3 -> 2x3: rank 17 (improved from 18)
    Total per block: 7+12+12+17 = 48
    Total: 48*2 = 96
    """
    n, m, p = 4, 5, 5
    
    total_rank = 96
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    for block in range(2):
        i_offset = block * 2
        
        # Sub 1: 2x2 * 2x2 -> 2x2 (Strassen)
        U1, V1, W1, r1 = _strassen_2x2()
        for i in range(2):
            for j in range(2):
                for c in range(r1):
                    U[(i_offset + i) * m + j][col + c] = U1[i * 2 + j][c]
        for j in range(2):
            for k in range(2):
                for c in range(r1):
                    V[j * p + k][col + c] = V1[j * 2 + k][c]
        for k in range(2):
            for i in range(2):
                for c in range(r1):
                    W[k * n + (i_offset + i)][col + c] = W1[k * 2 + i][c]
        col += r1
        
        # Sub 2: 2x3 * 3x2 -> 2x2 (Schoolbook)
        U2, V2, W2, r2 = _schoolbook(2, 3, 2)
        for i in range(2):
            for j in range(3):
                for c in range(r2):
                    U[(i_offset + i) * m + (2 + j)][col + c] = U2[i * 3 + j][c]
        for j in range(3):
            for k in range(2):
                for c in range(r2):
                    V[(2 + j) * p + k][col + c] = V2[j * 2 + k][c]
        for k in range(2):
            for i in range(2):
                for c in range(r2):
                    W[k * n + (i_offset + i)][col + c] += W2[k * 2 + i][c]
        col += r2
        
        # Sub 3: 2x2 * 2x3 -> 2x3 (Schoolbook)
        U3, V3, W3, r3 = _schoolbook(2, 2, 3)
        for i in range(2):
            for j in range(2):
                for c in range(r3):
                    U[(i_offset + i) * m + j][col + c] = U3[i * 2 + j][c]
        for j in range(2):
            for k in range(3):
                for c in range(r3):
                    V[j * p + (2 + k)][col + c] = V3[j * 3 + k][c]
        for k in range(3):
            for i in range(2):
                for c in range(r3):
                    W[(2 + k) * n + (i_offset + i)][col + c] = W3[k * 2 + i][c]
        col += r3
        
        # Sub 4: 2x3 * 3x3 -> 2x3 (Improved, rank 17)
        U4, V4, W4, r4 = _decompose_2x3x3()
        for i in range(2):
            for j in range(3):
                for c in range(r4):
                    U[(i_offset + i) * m + (2 + j)][col + c] = U4[i * 3 + j][c]
        for j in range(3):
            for k in range(3):
                for c in range(r4):
                    V[(2 + j) * p + (2 + k)][col + c] = V4[j * 3 + k][c]
        for k in range(3):
            for i in range(2):
                for c in range(r4):
                    W[(2 + k) * n + (i_offset + i)][col + c] += W4[k * 2 + i][c]
        col += r4
    
    return U, V, W, col


def construct():
    n, m, p = DIMENSIONS
    # Try the 96-rank decomposition
    try:
        U, V, W, rank = _decompose_4x5x5()
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Fallback to schoolbook
    U, V, W, rank = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
