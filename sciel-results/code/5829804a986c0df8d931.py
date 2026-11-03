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


def _try_sat_based_search(n, m, p, target_rank, time_limit=10):
    """
    Try to find a decomposition of rank target_rank using SAT-based support search.
    This is a heuristic: we try to find a sparse support pattern that can be lifted to rationals.
    """
    import itertools
    
    # For small ranks, we can try to solve the linear system directly
    # The tensor equation is: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+r, c] = delta_{i,r} delta_{j,k}
    
    # This is a bilinear system in U, V, W. We can fix one matrix and solve for the others.
    # Let's fix W to be the identity-like matrix (or a sparse matrix) and solve for U, V.
    
    # Actually, let's try a simpler approach: use the known structure of matrix multiplication
    # and try to find a low-rank approximation.
    
    # For now, return None to indicate failure
    return None


def construct():
    n, m, p = DIMENSIONS
    
    # Try the 96-rank decomposition first
    try:
        U, V, W, rank = _decompose_4x5x5()
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Try to find a lower rank decomposition using SAT-based search
    # This is a placeholder for future work
    for target_rank in range(80, 96):
        if time.time() - START_TIME > TIME_LIMIT - 5:
            break
        result = _try_sat_based_search(n, m, p, target_rank, time_limit=5)
        if result is not None:
            U, V, W, r = result
            if _verify_decomposition(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    U, V, W, rank = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
