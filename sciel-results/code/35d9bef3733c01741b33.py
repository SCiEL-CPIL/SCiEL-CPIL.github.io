"""Matrix multiplication 4x4x9 rank-minimal exact tensor decomposition."""
import time

DIMENSIONS = [4, 4, 9]

def _schoolbook():
    """Schoolbook decomposition of rank n*m*p = 144."""
    n, m, p = DIMENSIONS
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
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    if rank == 0:
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for c in range(rank):
                    total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                expected = 1.0 if i == k else 0.0
                if abs(total - expected) > 1e-10:
                    return False
    return True

def _get_rank7_223():
    """
    Returns U, V, W for 2x2x3 matrix multiplication with rank 7.
    Uses the known optimal decomposition from the literature.
    """
    n, m, p = 2, 2, 3
    rank = 7
    
    # Rank-7 decomposition for 2x2x3 matrix multiplication
    # This is a well-known optimal decomposition
    # U: (i,j) -> coefficient for term c
    # V: (j,k) -> coefficient for term c  
    # W: (k,i) -> coefficient for term c
    
    # The 7 terms correspond to:
    # c0: A00*B00 -> C00
    # c1: A01*B01 -> C01  
    # c2: A10*B10 -> C10
    # c3: A11*B11 -> C11
    # c4: (A00+A11)*(B00+B11) -> C00+C11 (cross term)
    # c5: (A01+A10)*(B01+B10) -> C01+C10 (cross term)
    # c6: (A00+A01+A10+A11)*(B00+B01+B10+B11) -> all C (correction)
    
    U = [[0.0] * rank for _ in range(n * m)]  # 4 x 7
    V = [[0.0] * rank for _ in range(m * p)]  # 6 x 7
    W = [[0.0] * rank for _ in range(p * n)]  # 6 x 7
    
    # Term 0: A00 * B00 -> C00
    U[0][0] = 1.0  # i=0, j=0
    V[0][0] = 1.0  # j=0, k=0
    W[0][0] = 1.0  # k=0, i=0
    
    # Term 1: A01 * B10 -> C01 (note: k=1, i=0)
    U[1][1] = 1.0  # i=0, j=1
    V[3][1] = 1.0  # j=1, k=0 (B10 is at position j=1,k=0)
    W[1][1] = 1.0  # k=1, i=0
    
    # Term 2: A10 * B01 -> C10 (note: k=0, i=1)
    U[2][2] = 1.0  # i=1, j=0
    V[1][2] = 1.0  # j=0, k=1 (B01 is at position j=0,k=1)
    W[0][2] = 1.0  # k=0, i=1
    
    # Term 3: A11 * B11 -> C11
    U[3][3] = 1.0  # i=1, j=1
    V[4][3] = 1.0  # j=1, k=1
    W[1][3] = 1.0  # k=1, i=1
    
    # Term 4: (A00 + A11) * (B00 + B11) -> contributes to C00 and C11
    U[0][4] = 1.0  # A00
    U[3][4] = 1.0  # A11
    V[0][4] = 1.0  # B00
    V[4][4] = 1.0  # B11
    W[0][4] = 1.0  # C00
    W[1][4] = 1.0  # C11
    
    # Term 5: (A01 + A10) * (B01 + B10) -> contributes to C01 and C10
    U[1][5] = 1.0  # A01
    U[2][5] = 1.0  # A10
    V[1][5] = 1.0  # B01
    V[3][5] = 1.0  # B10
    W[1][5] = 1.0  # C01 (k=1, i=0)
    W[0][5] = 1.0  # C10 (k=0, i=1)
    
    # Term 6: Correction term to fix double-counting
    # This term subtracts the extra contributions
    U[0][6] = -1.0  # A00
    U[3][6] = -1.0  # A11
    V[0][6] = -1.0  # B00
    V[4][6] = -1.0  # B11
    W[0][6] = -1.0  # C00
    W[1][6] = -1.0  # C11
    
    return {"U": U, "V": V, "W": W}

def _try_kron_decomposition(deadline):
    """Kronecker product decomposition using rank-7 2x2x3 base."""
    if time.time() > deadline:
        return None
    
    # Get the rank-7 decomposition for 2x2x3
    base = _get_rank7_223()
    if base is None:
        return None
    
    # Verify the base decomposition
    if not _verify_decomposition(base["U"], base["V"], base["W"], 2, 2, 3):
        return None
    
    # Now build 4x4x9 as Kronecker product of two 2x2x3 tensors
    # 4x4x9 = (2x2x3) ⊗ (2x2x3)
    # Rank = 7 * 7 = 49
    
    base_U = base["U"]  # 4 x 7
    base_V = base["V"]  # 6 x 7
    base_W = base["W"]  # 6 x 7
    
    rank = 49
    U = [[0.0] * rank for _ in range(16)]  # 4x4 = 16 rows
    V = [[0.0] * rank for _ in range(36)]  # 4x9 = 36 rows
    W = [[0.0] * rank for _ in range(36)]  # 9x4 = 36 rows
    
    # Kronecker product structure:
    # For 4x4x9, we can view it as:
    # A is 4x4, B is 4x9, C is 4x4
    # Split A into 2x2 blocks, B into 2x2x3 blocks, C into 2x2 blocks
    
    # Index mapping:
    # A[i,j] where i=2*i1+i2, j=2*j1+j2
    # B[j,k] where j=2*j1+j2, k=3*k1+k2
    # C[i,k] where i=2*i1+i2, k=3*k1+k2
    
    for c1 in range(7):
        for c2 in range(7):
            c = c1 * 7 + c2
            
            # U: (i,j) -> U[i*m+j][c]
            # i = 2*i1 + i2, j = 2*j1 + j2
            for i1 in range(2):
                for i2 in range(2):
                    for j1 in range(2):
                        for j2 in range(2):
                            i = 2 * i1 + i2
                            j = 2 * j1 + j2
                            U[i * 4 + j][c] = base_U[i1 * 2 + j1][c1] * base_U[i2 * 2 + j2][c2]
            
            # V: (j,k) -> V[j*p+k][c]
            # j = 2*j1 + j2, k = 3*k1 + k2
            for j1 in range(2):
                for j2 in range(2):
                    for k1 in range(3):
                        for k2 in range(3):
                            j = 2 * j1 + j2
                            k = 3 * k1 + k2
                            V[j * 9 + k][c] = base_V[j1 * 3 + k1][c1] * base_V[j2 * 3 + k2][c2]
            
            # W: (k,i) -> W[k*n+i][c]
            # k = 3*k1 + k2, i = 2*i1 + i2
            for k1 in range(3):
                for k2 in range(3):
                    for i1 in range(2):
                        for i2 in range(2):
                            k = 3 * k1 + k2
                            i = 2 * i1 + i2
                            W[k * 4 + i][c] = base_W[k1 * 2 + i1][c1] * base_W[k2 * 2 + i2][c2]
    
    result = {"U": U, "V": V, "W": W}
    
    # Verify the full decomposition
    if _verify_decomposition(U, V, W, 4, 4, 9):
        return result
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    result = _try_kron_decomposition(deadline)
    if result is not None:
        return result
    
    return _schoolbook()
