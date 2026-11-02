"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
import time

def _verify(U, V, W, n=3, m=3, p=3):
    """Verify that U,V,W correctly decompose matrix multiplication.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Check: for all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    """
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _construct_strassen_2x2_block():
    """
    Construct a decomposition for 3x3 matrix multiplication by
    partitioning into 2x2 (top-left) and 1x1 (row/col 3) blocks.
    Uses Strassen's 7-term algorithm for the 2x2 block and direct
    multiplication for cross-terms.
    
    Indexing: A[i,j] -> U[i*m+j], B[j,k] -> V[j*p+k], C[i,k] -> W[k*n+i]
    For 3x3: A[i,j] -> U[3i+j], B[j,k] -> V[3j+k], C[i,k] -> W[3k+i]
    """
    n, m, p = 3, 3, 3
    
    # Build list of (u_indices, u_coeffs, v_indices, v_coeffs, w_indices, w_coeffs)
    terms = []
    
    # Strassen's 7 terms for the 2x2 top-left block
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Term 1: M1 = (A00+A11)(B00+B11)
    # U: A[0,0]=0, A[1,1]=4
    # V: B[0,0]=0, B[1,1]=4
    # W: C[0,0]=W[0] (+1), C[1,1]=W[4] (+1)
    terms.append(([0, 4], [1, 1], [0, 4], [1, 1], [0, 4], [1, 1]))
    
    # Term 2: M2 = (A10+A11)B00
    # U: A[1,0]=3, A[1,1]=4
    # V: B[0,0]=0
    # W: C[1,0]=W[3] (+1), C[1,1]=W[4] (-1)
    terms.append(([3, 4], [1, 1], [0], [1], [3, 4], [1, -1]))
    
    # Term 3: M3 = A00(B10-B11)
    # U: A[0,0]=0
    # V: B[1,0]=3, B[1,1]=4
    # W: C[0,1]=W[1] (+1), C[1,1]=W[4] (+1)
    terms.append(([0], [1], [3, 4], [1, -1], [1, 4], [1, 1]))
    
    # Term 4: M4 = A11(B01-B00)
    # U: A[1,1]=4
    # V: B[0,1]=1, B[0,0]=0
    # W: C[0,0]=W[0] (-1), C[1,0]=W[3] (+1)
    terms.append(([4], [1], [1, 0], [1, -1], [0, 3], [-1, 1]))
    
    # Term 5: M5 = (A00+A01)B11
    # U: A[0,0]=0, A[0,1]=1
    # V: B[1,1]=4
    # W: C[0,0]=W[0] (+1), C[0,1]=W[1] (+1)
    terms.append(([0, 1], [1, 1], [4], [1], [0, 1], [1, 1]))
    
    # Term 6: M6 = (A10-A00)(B00+B01)
    # U: A[1,0]=3, A[0,0]=0
    # V: B[0,0]=0, B[0,1]=1
    # W: C[1,1]=W[4] (+1)
    terms.append(([3, 0], [1, -1], [0, 1], [1, 1], [4], [1]))
    
    # Term 7: M7 = (A01-A11)(B10+B11)
    # U: A[0,1]=1, A[1,1]=4
    # V: B[1,0]=3, B[1,1]=4
    # W: C[0,0]=W[0] (+1), C[0,1]=W[1] (-1)
    terms.append(([1, 4], [1, -1], [3, 4], [1, 1], [0, 1], [1, -1]))
    
    # Cross terms - direct products:
    # C[0,2] = A00*B02 + A01*B12 + A02*B22
    terms.append(([0], [1], [2], [1], [2], [1]))   # A00*B02 -> C02
    terms.append(([1], [1], [5], [1], [2], [1]))   # A01*B12 -> C02
    terms.append(([2], [1], [8], [1], [2], [1]))   # A02*B22 -> C02
    
    # C[1,2] = A10*B02 + A11*B12 + A12*B22
    terms.append(([3], [1], [2], [1], [5], [1]))   # A10*B02 -> C12
    terms.append(([4], [1], [5], [1], [5], [1]))   # A11*B12 -> C12
    terms.append(([5], [1], [8], [1], [5], [1]))   # A12*B22 -> C12
    
    # C[2,0] = A20*B00 + A21*B10 + A22*B20
    terms.append(([6], [1], [0], [1], [6], [1]))   # A20*B00 -> C20
    terms.append(([7], [1], [3], [1], [6], [1]))   # A21*B10 -> C20
    terms.append(([8], [1], [6], [1], [6], [1]))   # A22*B20 -> C20
    
    # C[2,1] = A20*B01 + A21*B11 + A22*B21
    terms.append(([6], [1], [1], [1], [7], [1]))   # A20*B01 -> C21
    terms.append(([7], [1], [4], [1], [7], [1]))   # A21*B11 -> C21
    terms.append(([8], [1], [7], [1], [7], [1]))   # A22*B21 -> C21
    
    # C[2,2] = A20*B02 + A21*B12 + A22*B22
    terms.append(([6], [1], [2], [1], [8], [1]))   # A20*B02 -> C22
    terms.append(([7], [1], [5], [1], [8], [1]))   # A21*B12 -> C22
    terms.append(([8], [1], [8], [1], [8], [1]))   # A22*B22 -> C22
    
    rank = len(terms)  # 7 + 15 = 22
    U = [[0] * rank for _ in range(9)]
    V = [[0] * rank for _ in range(9)]
    W = [[0] * rank for _ in range(9)]
    
    for col_idx, (u_idx, u_c, v_idx, v_c, w_idx, w_c) in enumerate(terms):
        for idx, coeff in zip(u_idx, u_c):
            U[idx][col_idx] += coeff
        for idx, coeff in zip(v_idx, v_c):
            V[idx][col_idx] += coeff
        for idx, coeff in zip(w_idx, w_c):
            W[idx][col_idx] += coeff
    
    if _verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    return None

def _construct_schoolbook():
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

def construct():
    overall_deadline = time.time() + 140
    
    # Try the Strassen 2x2 block decomposition (rank 22)
    start = time.time()
    r22 = _construct_strassen_2x2_block()
    if r22 is not None:
        return r22
    
    if time.time() > overall_deadline:
        return _construct_schoolbook()
    
    # Fall back to schoolbook (rank 27)
    return _construct_schoolbook()
# EVOLVE-BLOCK-END
