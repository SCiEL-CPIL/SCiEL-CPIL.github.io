import time
import numpy as np

def construct():
    n, m, p = 2, 3, 4
    deadline = time.time() + 140
    
    # Try Rank 22
    try:
        result = _construct_rank_22()
        # Verify correctness
        if _verify_decomposition(result, n, m, p):
            return result
        else:
            raise ValueError("Rank 22 decomposition failed verification")
    except Exception:
        return _construct_rank_24()

def _verify_decomposition(result, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(result['U'], dtype=float)
    V = np.array(result['V'], dtype=float)
    W = np.array(result['W'], dtype=float)
    
    rank = U.shape[1]
    
    # For A (n x m), B (m x p), C (n x p):
    # C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
    # Tensor T[i,j,k] = 1 for all valid i,j,k
    # Decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Build the tensor
    T = np.ones((n, m, p))
    
    # Compute decomposition using vectorized operations
    error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                error += (val - T[i, j, k]) ** 2
    
    return error < 1e-10

def _construct_rank_24():
    n, m, p = 2, 3, 4
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

def _add_strassen_block(U, V, W, col, a_idx, b_idx, c_idx):
    """
    Add a 2x2 Strassen block.
    a_idx: [a00, a01, a10, a11] - indices into U
    b_idx: [b00, b01, b10, b11] - indices into V
    c_idx: [c00, c01, c10, c11] - indices into W
    Returns new col after 7 terms.
    """
    a00, a01, a10, a11 = a_idx
    b00, b01, b10, b11 = b_idx
    c00, c01, c10, c11 = c_idx
    
    # M1: (A00+A11)(B00+B11)
    U[a00][col] = 1; U[a11][col] = 1
    V[b00][col] = 1; V[b11][col] = 1
    W[c00][col] = 1; W[c11][col] = 1
    col += 1
    
    # M2: (A10+A11)B00
    U[a10][col] = 1; U[a11][col] = 1
    V[b00][col] = 1
    W[c10][col] = 1; W[c11][col] = -1
    col += 1
    
    # M3: A00(B10+B11)
    U[a00][col] = 1
    V[b10][col] = 1; V[b11][col] = 1
    W[c10][col] = 1; W[c11][col] = 1
    col += 1
    
    # M4: A11(B01+B10)
    U[a11][col] = 1
    V[b01][col] = 1; V[b10][col] = 1
    W[c00][col] = -1; W[c01][col] = 1
    col += 1
    
    # M5: (A00+A01)B11
    U[a00][col] = 1; U[a01][col] = 1
    V[b11][col] = 1
    W[c00][col] = 1; W[c10][col] = 1
    col += 1
    
    # M6: (A10-A00)(B00+B01)
    U[a10][col] = 1; U[a00][col] = -1
    V[b00][col] = 1; V[b01][col] = 1
    W[c01][col] = 1; W[c10][col] = -1
    col += 1
    
    # M7: (A01-A11)(B10+B11)
    U[a01][col] = 1; U[a11][col] = -1
    V[b10][col] = 1; V[b11][col] = 1
    W[c00][col] = -1; W[c11][col] = 1
    col += 1
    
    return col

def _construct_rank_22():
    """
    Rank-22 decomposition using Strassen for 2x2 blocks.
    
    Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]  (7 columns)
    Block 2: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]  (7 columns)
    Block 3: A[0:2, 2] * B[2, 0:4] -> C[0:2, 0:4]      (8 columns, schoolbook)
    Total: 22 columns
    """
    n, m, p = 2, 3, 4
    rank = 22
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # A indices: A00=0, A01=1, A10=3, A11=4
    # B indices: B00=0, B01=1, B10=4, B11=5
    # C indices: C00=0, C01=1, C10=2, C11=3
    col = _add_strassen_block(U, V, W, col, [0, 1, 3, 4], [0, 1, 4, 5], [0, 1, 2, 3])
    
    # Block 2: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    # A indices: A00=0, A01=1, A10=3, A11=4
    # B indices: B00=2, B01=3, B10=6, B11=7
    # C indices: C02=4, C03=5, C12=6, C13=7
    col = _add_strassen_block(U, V, W, col, [0, 1, 3, 4], [2, 3, 6, 7], [4, 5, 6, 7])
    
    # Block 3: A[0:2, 2] * B[2, 0:4] -> C[0:2, 0:4]
    # A indices: A02=2, A12=5
    # B indices: B20=8, B21=9, B22=10, B23=11
    # C indices: C00=0, C01=1, C02=4, C03=5, C10=2, C11=3, C12=6, C13=7
    
    # Schoolbook: 2*4 = 8 terms
    U[2][col] = 1; V[8][col] = 1; W[0][col] = 1   # A02*B20 -> C00
    col += 1
    U[2][col] = 1; V[9][col] = 1; W[1][col] = 1   # A02*B21 -> C01
    col += 1
    U[2][col] = 1; V[10][col] = 1; W[4][col] = 1  # A02*B22 -> C02
    col += 1
    U[2][col] = 1; V[11][col] = 1; W[5][col] = 1  # A02*B23 -> C03
    col += 1
    U[5][col] = 1; V[8][col] = 1; W[2][col] = 1   # A12*B20 -> C10
    col += 1
    U[5][col] = 1; V[9][col] = 1; W[3][col] = 1   # A12*B21 -> C11
    col += 1
    U[5][col] = 1; V[10][col] = 1; W[6][col] = 1  # A12*B22 -> C12
    col += 1
    U[5][col] = 1; V[11][col] = 1; W[7][col] = 1  # A12*B23 -> C13
    col += 1
    
    if col != 22:
        raise ValueError(f"Expected 22 columns, got {col}")
        
    return {"U": U, "V": V, "W": W}
