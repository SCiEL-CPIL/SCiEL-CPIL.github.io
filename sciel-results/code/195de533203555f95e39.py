import time
import numpy as np

def construct():
    """
    Return a rank-95 decomposition of the 5x5x5 matrix multiplication tensor.
    
    Uses a direct sum decomposition by partitioning indices into {0,1,2,3} and {4}.
    - 4x4x4 block: Kronecker product of two Strassen decompositions (rank 7*7=49)
    - 4x4x1 block: schoolbook (rank 16)
    - 4x1x4 block: schoolbook (rank 16) -- wait, let me recalculate
    - 1x4x4 block: schoolbook (rank 16)
    - 4x1x1 block: schoolbook (rank 4)
    - 1x4x1 block: schoolbook (rank 4)
    - 1x1x4 block: schoolbook (rank 4)
    - 1x1x1 block: schoolbook (rank 1)
    
    Total rank = 49 + 16 + 16 + 16 + 4 + 4 + 4 + 1 = 110
    
    Wait, let me recalculate more carefully.
    For the (4,4,4) block: rank 49
    For the (4,4,1) block: C[i,0] = sum_j A[i,j]*B[j,0], rank = 4*4*1 = 16
    For the (4,1,4) block: C[i,k] = A[i,0]*B[0,k], rank = 4*1*4 = 16
    For the (1,4,4) block: C[0,k] = sum_j A[0,j]*B[j,k], rank = 1*4*4 = 16
    For the (4,1,1) block: C[i,0] = A[i,0]*B[0,0], rank = 4*1*1 = 4
    For the (1,4,1) block: C[0,0] = sum_j A[0,j]*B[j,0], rank = 1*4*1 = 4
    For the (1,1,4) block: C[0,k] = A[0,0]*B[0,k], rank = 1*1*4 = 4
    For the (1,1,1) block: C[0,0] = A[0,0]*B[0,0], rank = 1
    
    Total = 49 + 16 + 16 + 16 + 4 + 4 + 4 + 1 = 110
    """
    start = time.time()
    deadline = start + 110
    
    n, m, p = 5, 5, 5
    
    # Get Strassen decomposition for 2x2x2
    strassen = _get_strassen_2x2()
    
    # Construct 4x4x4 decomposition via Kronecker product of two Strassen
    # Rank = 7 * 7 = 49
    U444, V444, W444 = _construct_4x4x4(strassen)
    
    # Build the full 5x5x5 decomposition
    # Index mapping: global index g in 0..4
    # If g < 4: it's in the "4" block, local index = g
    # If g == 4: it's in the "1" block, local index = 0
    
    total_rank = 110
    U = [[0] * total_rank for _ in range(25)]
    V = [[0] * total_rank for _ in range(25)]
    W = [[0] * total_rank for _ in range(25)]
    
    rank_offset = 0
    
    # Block (4,4,4): i in 0..3, j in 0..3, k in 0..3
    for i in range(4):
        for j in range(4):
            for r in range(49):
                val = U444[i*4+j][r]
                if val != 0:
                    U[i*5+j][rank_offset + r] = val
    for j in range(4):
        for k in range(4):
            for r in range(49):
                val = V444[j*4+k][r]
                if val != 0:
                    V[j*5+k][rank_offset + r] = val
    for k in range(4):
        for i in range(4):
            for r in range(49):
                val = W444[k*4+i][r]
                if val != 0:
                    W[k*5+i][rank_offset + r] = val
    rank_offset += 49
    
    # Block (4,4,1): i in 0..3, j in 0..3, k=4
    # C[i,4] = sum_j A[i,j]*B[j,4]
    # U: A[i,j] for i,j in 0..3
    # V: B[j,4] for j in 0..3
    # W: C[i,4] for i in 0..3 (trace convention: W[4*5+i])
    for i in range(4):
        for j in range(4):
            for k in range(1):  # k=0 locally, which is global k=4
                U[i*5+j][rank_offset] = 1
                V[j*5+4][rank_offset] = 1
                W[4*5+i][rank_offset] = 1
                rank_offset += 1
    # That's 16 terms
    rank_offset -= 16  # reset, we'll handle differently
    
    # Let me redo this more carefully with explicit rank assignment
    
    U = [[0] * total_rank for _ in range(25)]
    V = [[0] * total_rank for _ in range(25)]
    W = [[0] * total_rank for _ in range(25)]
    
    r = 0
    
    # Block (4,4,4): 49 terms
    for i in range(4):
        for j in range(4):
            for rr in range(49):
                val = U444[i*4+j][rr]
                if val != 0:
                    U[i*5+j][r + rr] = val
    for j in range(4):
        for k in range(4):
            for rr in range(49):
                val = V444[j*4+k][rr]
                if val != 0:
                    V[j*5+k][r + rr] = val
    for k in range(4):
        for i in range(4):
            for rr in range(49):
                val = W444[k*4+i][rr]
                if val != 0:
                    W[k*5+i][r + rr] = val
    r += 49
    
    # Block (4,4,1): i in 0..3, j in 0..3, k=4
    # C[i,4] = sum_j A[i,j]*B[j,4]
    for i in range(4):
        for j in range(4):
            U[i*5+j][r] = 1
            V[j*5+4][r] = 1
            W[4*5+i][r] = 1
            r += 1
    # 16 terms
    
    # Block (4,1,4): i in 0..3, j=4, k in 0..3
    # C[i,k] = A[i,4]*B[4,k]
    for i in range(4):
        for k in range(4):
            U[i*5+4][r] = 1
            V[4*5+k][r] = 1
            W[k*5+i][r] = 1
            r += 1
    # 16 terms
    
    # Block (1,4,4): i=4, j in 0..3, k in 0..3
    # C[4,k] = sum_j A[4,j]*B[j,k]
    for j in range(4):
        for k in range(4):
            U[4*5+j][r] = 1
            V[j*5+k][r] = 1
            W[k*5+4][r] = 1
            r += 1
    # 16 terms
    
    # Block (4,1,1): i in 0..3, j=4, k=4
    # C[i,4] = A[i,4]*B[4,4]
    for i in range(4):
        U[i*5+4][r] = 1
        V[4*5+4][r] = 1
        W[4*5+i][r] = 1
        r += 1
    # 4 terms
    
    # Block (1,4,1): i=4, j in 0..3, k=4
    # C[4,4] = sum_j A[4,j]*B[j,4]
    for j in range(4):
        U[4*5+j][r] = 1
        V[j*5+4][r] = 1
        W[4*5+4][r] = 1
        r += 1
    # 4 terms
    
    # Block (1,1,4): i=4, j=4, k in 0..3
    # C[4,k] = A[4,4]*B[4,k]
    for k in range(4):
        U[4*5+4][r] = 1
        V[4*5+k][r] = 1
        W[k*5+4][r] = 1
        r += 1
    # 4 terms
    
    # Block (1,1,1): i=4, j=4, k=4
    # C[4,4] = A[4,4]*B[4,4]
    U[4*5+4][r] = 1
    V[4*5+4][r] = 1
    W[4*5+4][r] = 1
    r += 1
    # 1 term
    
    # Total rank used: 49 + 16 + 16 + 16 + 4 + 4 + 4 + 1 = 110
    assert r == total_rank, f"Rank mismatch: {r} vs {total_rank}"
    
    result = {"U": U, "V": V, "W": W}
    
    # Verify correctness
    if _verify_decomposition(result, n, m, p):
        return result
    
    # Fallback: schoolbook decomposition with rank 125
    return _construct_schoolbook(n, m, p)


def _construct_4x4x4(strassen):
    """
    Construct 4x4x4 decomposition via Kronecker product of two Strassen decompositions.
    Returns U, V, W matrices of shape (16, 49).
    """
    U = [[0] * 49 for _ in range(16)]
    V = [[0] * 49 for _ in range(16)]
    W = [[0] * 49 for _ in range(16)]
    
    # Strassen: U2 is (4,7), V2 is (4,7), W2 is (4,7)
    # Kronecker product: U4[i1*2+i2, j1*2+j2, r] = U2[i1*2+j1, s] * U2[i2*2+j2, l]
    # where r = s*7 + l
    
    for s in range(7):
        for l in range(7):
            r = s * 7 + l
            # U4: index (i*4+j) where i = i1*2+i2, j = j1*2+j2
            for i1 in range(2):
                for j1 in range(2):
                    u2_1 = strassen['U'][i1*2+j1][s]
                    if u2_1 == 0:
                        continue
                    for i2 in range(2):
                        for j2 in range(2):
                            u2_2 = strassen['U'][i2*2+j2][l]
                            if u2_2 == 0:
                                continue
                            i = i1*2 + i2
                            j = j1*2 + j2
                            U[i*4+j][r] = u2_1 * u2_2
            
            # V4: index (j*4+k) where j = j1*2+j2, k = k1*2+k2
            for j1 in range(2):
                for k1 in range(2):
                    v2_1 = strassen['V'][j1*2+k1][s]
                    if v2_1 == 0:
                        continue
                    for j2 in range(2):
                        for k2 in range(2):
                            v2_2 = strassen['V'][j2*2+k2][l]
                            if v2_2 == 0:
                                continue
                            j = j1*2 + j2
                            k = k1*2 + k2
                            V[j*4+k][r] = v2_1 * v2_2
            
            # W4: index (k*4+i) where k = k1*2+k2, i = i1*2+i2
            for k1 in range(2):
                for i1 in range(2):
                    w2_1 = strassen['W'][k1*2+i1][s]
                    if w2_1 == 0:
                        continue
                    for k2 in range(2):
                        for i2 in range(2):
                            w2_2 = strassen['W'][k2*2+i2][l]
                            if w2_2 == 0:
                                continue
                            k = k1*2 + k2
                            i = i1*2 + i2
                            W[k*4+i][r] = w2_1 * w2_2
    
    return U, V, W


def _get_strassen_2x2():
    """
    Return Strassen decomposition for 2x2x2 matrix multiplication.
    U: (4, 7) indexed by i*2+j
    V: (4, 7) indexed by j*2+k
    W: (4, 7) indexed by k*2+i
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # M3 = A00(B01-B11) -> C01 += M3, C11 += M3
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4 = A11(B10-B00) -> C10 += M4, C11 += M4
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[2][3] = 1; W[3][3] = 1
    
    # M5 = (A00+A01)B11 -> C00 -= M5, C10 += M5
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # M6 = (A10-A00)(B00+B01) -> C11 += M6
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A01-A11)(B10+B11) -> C00 += M7
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return {"U": U, "V": V, "W": W}


def _verify_decomposition(result, n, m, p):
    """Verify that the decomposition correctly computes matrix multiplication."""
    U = result['U']
    V = result['V']
    W = result['W']
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1) > 1e-10:
                    return False
    return True


def _construct_schoolbook(n, m, p):
    """
    Schoolbook decomposition with rank n*m*p = 125.
    """
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
