import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def _build_strassen_2x2():
    """
    Build Strassen decomposition for 2x2 x 2x2 matrix multiplication.
    Returns U (4,7), V (4,7), W (4,7) where:
    U indexed by (i,k), V by (k,j), W by (j,i)
    T[i,k,j] = sum_r U[i*2+k, r] * V[k*2+j, r] * W[j*2+i, r]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A11+A22)(B11+B22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A21+A22)B11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # M3 = A11(B12-B22)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # M4 = A22(B21-B11)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # M5 = (A11+A12)B22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # M6 = (A21-A11)(B11+B12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A12-A22)(B21+B22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _build_4x4x4_strassen():
    """
    Build Strassen-based rank-49 decomposition for 4x4 x 4x4.
    Uses the tensor product structure: 4x4x4 = (2x2x2) ⊗ (2x2x2)
    """
    U2, V2, W2 = _build_strassen_2x2()
    
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    # The 4x4x4 tensor can be seen as 4 blocks of 2x2x2
    # Using Kronecker product structure:
    # A = [[A00, A01], [A10, A11]] where each Ai is 2x2
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    
    # For the block structure, we can use the same Strassen scheme at the block level
    # but each "scalar" multiplication is replaced by a 2x2x2 multiplication
    
    # Actually, the simplest correct approach: 
    # 4x4x4 tensor rank is at most 7*7 = 49 using tensor product
    # The decomposition is: for each pair (r1, r2) of 2x2 Strassen indices,
    # we get one rank-1 term for the 4x4x4 tensor
    
    # U_4x4[i*4+j, r1*7+r2] = U_2x2[i2*2+j2, r1] * U_2x2[i1*2+j1, r2]
    # where i = 2*i1+i2, j = 2*j1+j2 (or some consistent ordering)
    
    # Let's use the standard Kronecker product:
    # Index i in 0..3, j in 0..3 for A (4x4)
    # i = 2*i1 + i2, j = 2*j1 + j2
    # Similarly for B: j = 2*j1+j2, k = 2*k1+k2
    # And for C: k = 2*k1+k2, i = 2*i1+i2
    
    for r1 in range(7):
        for r2 in range(7):
            col = r1 * 7 + r2
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            I = 2*i1 + i2
                            J = 2*j1 + j2
                            U[I*4+J, col] = U2[i1*2+i2, r1] * U2[j1*2+j2, r2]
                            
            for j1 in range(2):
                for k1 in range(2):
                    for j2 in range(2):
                        for k2 in range(2):
                            J = 2*j1 + j2
                            K = 2*k1 + k2
                            V[J*4+K, col] = V2[j1*2+j2, r1] * V2[k1*2+k2, r2]
                            
            for k1 in range(2):
                for i1 in range(2):
                    for k2 in range(2):
                        for i2 in range(2):
                            K = 2*k1 + k2
                            I = 2*i1 + i2
                            W[K*4+I, col] = W2[k1*2+k2, r1] * W2[i1*2+i2, r2]
    
    return U, V, W

def _build_4x4x5_strassen():
    """
    Build a rank-64 decomposition for 4x4 x 4x5.
    Uses 4x4x4 Strassen (rank 49) for first 4 columns + schoolbook (rank 16) for 5th column.
    Total: 65. But let's try to do better.
    
    Actually, let's just use 49 + 16 = 65 for simplicity and correctness.
    """
    U444, V444, W444 = _build_4x4x4_strassen()
    
    rank = 49 + 16
    U = np.zeros((16, rank))
    V = np.zeros((20, rank))  # 4*5 = 20
    W = np.zeros((20, rank))  # 5*4 = 20
    
    # First 4 columns: use 4x4x4 Strassen
    U[:, :49] = U444
    V[:, :49] = V444
    W[:, :49] = W444
    
    # 5th column (k=4): schoolbook rank 16
    for idx in range(16):
        i = idx // 4
        j = idx % 4
        col = 49 + idx
        U[i*4+j, col] = 1
        V[j*5+4, col] = 1
        W[4*4+i, col] = 1
    
    return U, V, W

def _schoolbook_fallback():
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

def _validate_numpy(U, V, W, n, m, p):
    """Validate using NumPy for speed."""
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    rank = U_np.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_np[i*m+j, :] * V_np[j*p+k, :] * W_np[k*n+i, :])
                if abs(val - 1) > 1e-9:
                    return False
    return True

def _build_4x4x9_optimized():
    """
    Build 4x4x9 decomposition using block structure.
    
    Strategy: Split the 9 columns into 4 + 4 + 1.
    - First 4 columns: 4x4x4 Strassen (rank 49)
    - Next 4 columns: 4x4x4 Strassen (rank 49)  
    - Last column: 4x4x1 schoolbook (rank 16)
    Total: 114
    
    Alternative: Split into 5 + 4.
    - First 5 columns: 4x4x5 (rank 65)
    - Next 4 columns: 4x4x4 Strassen (rank 49)
    Total: 114
    
    Let's try a different split: 3 + 3 + 3
    Each 4x4x3: rank at most 4*4*3 = 48 (schoolbook) or less with optimization.
    Total: 3*48 = 144. Worse.
    
    Let's stick with 49 + 49 + 16 = 114 for now, but verify it works.
    """
    U444, V444, W444 = _build_4x4x4_strassen()
    
    total_rank = 49 + 49 + 16
    U = np.zeros((16, total_rank))
    V = np.zeros((36, total_rank))  # 4*9 = 36
    W = np.zeros((36, total_rank))  # 9*4 = 36
    
    # Block 1: columns 0-3
    U[:, 0:49] = U444
    V[:, 0:49] = V444
    W[:, 0:49] = W444
    
    # Block 2: columns 4-7
    offset2 = 49
    U[:, offset2:offset2+49] = U444
    # V: rows j*9+k where k in 4..7
    for j in range(4):
        for k in range(4):
            V[j*9+k+4, offset2:offset2+49] = V444[j*4+k, :]
    # W: rows k*4+i where k in 4..7
    for k in range(4):
        for i in range(4):
            W[(k+4)*4+i, offset2:offset2+49] = W444[k*4+i, :]
    
    # Block 3: column 8 (schoolbook, rank 16)
    offset3 = 98
    for idx in range(16):
        i = idx // 4
        j = idx % 4
        col = offset3 + idx
        U[i*4+j, col] = 1
        V[j*9+8, col] = 1
        W[8*4+i, col] = 1
    
    return U, V, W

def _try_winograd_4x4x9():
    """
    Attempt to use Winograd-style frequency domain decomposition for 4x4x9.
    
    Winograd's method for matrix multiplication uses polynomial evaluation/interpolation.
    For 4x4x4, the minimal rank is known to be at most 49 (Strassen), but might be lower.
    
    The Winograd algorithm for 2x2x2 uses rank 7 (same as Strassen).
    For 4x4x4, Winograd's method might give rank < 49.
    
    However, implementing a verified Winograd 4x4x4 decomposition is complex.
    Let's try a simpler approach: use the known fact that 4x4x4 can be done
    in rank at most 49, and see if we can find a better decomposition for 4x4x9.
    
    For now, return None to indicate this approach isn't implemented.
    """
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # Try Winograd approach first
    result = _try_winograd_4x4x9()
    if result is not None:
        U, V, W = result
        if _validate_numpy(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fall back to Strassen-based approach
    U, V, W = _build_4x4x9_optimized()
    
    # Validate
    if _validate_numpy(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Final fallback: schoolbook
    return _schoolbook_fallback()
