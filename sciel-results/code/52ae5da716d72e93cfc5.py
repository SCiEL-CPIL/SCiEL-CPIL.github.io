"""Program for matrix multiplication tensor decomposition: 2x4 * 4x5."""
import time
import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def _validate_numpy(U, V, W, n=2, m=4, p=5, tol=1e-9):
    """Vectorized validation using numpy for speed."""
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    rank = U_np.shape[1]
    
    # Reshape to (n, m, rank), (m, p, rank), (p, n, rank)
    U_reshaped = U_np.reshape(n, m, rank)
    V_reshaped = V_np.reshape(m, p, rank)
    W_reshaped = W_np.reshape(p, n, rank)
    
    # We need sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] == 1 for all i,j,k
    # This is a tensor contraction.
    # We can iterate over r and accumulate? No, we need the sum to be 1.
    # Actually, the tensor M[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # should be 1 if i,j,k form a valid path? No.
    # The decomposition is for the matrix multiplication tensor.
    # The tensor M is such that M[i,j,k] = 1 if the indices correspond to the identity?
    # No, the standard tensor for matrix multiplication M_{n,m,p} has entries M[i,j,k] = 1 
    # where i is row of C, j is col of A/row of B, k is col of C.
    # Wait, the tensor T such that C = AB is T[i, j, k] = 1?
    # No, T[i, j, k] is the coefficient of A[i,j] B[j,k] in C[i,k].
    # So T[i, j, k] = 1 for all valid i,j,k.
    # And the decomposition is T = sum_r U_r \otimes V_r \otimes W_r.
    # So we check if sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1.
    
    # Efficient check:
    # Create tensors of shape (n, m, p, rank)
    # U part: U[i, j, r] -> broadcast to (n, m, 1, rank)
    # V part: V[j, k, r] -> broadcast to (1, m, p, rank)
    # W part: W[k, i, r] -> broadcast to (1, 1, p, n, rank)? No.
    
    # Let's just use the loop but vectorize over r.
    # prod = U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :]
    # val = sum(prod)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                prod = U_np[i*m+j] * V_np[j*p+k] * W_np[k*n+i]
                val = np.sum(prod)
                if abs(val - 1.0) > tol:
                    return False
    return True

def _clean_entry(v):
    if abs(v - round(v)) < 1e-9:
        return int(round(v))
    doubled = v * 2
    if abs(doubled - round(doubled)) < 1e-9:
        return round(doubled) / 2.0
    return v

def _construct_rank_36():
    """
    Rank-36 decomposition for 2x4 x 4x5 matrix multiplication.
    Uses Strassen for 2x2x2 blocks and schoolbook for 2x2x1.
    """
    n, m, p = 2, 4, 5
    rank = 36
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_222(i1, i2, j1, j2, k1, k2, col_offset):
        a11 = i1 * m + j1
        a12 = i1 * m + j2
        a21 = i2 * m + j1
        a22 = i2 * m + j2
        b11 = j1 * p + k1
        b12 = j1 * p + k2
        b21 = j2 * p + k1
        b22 = j2 * p + k2
        c11 = k1 * n + i1
        c12 = k2 * n + i1
        c21 = k1 * n + i2
        c22 = k2 * n + i2
        
        col = col_offset + 0
        U[a11][col] = 1; U[a22][col] = 1
        V[b11][col] = 1; V[b22][col] = 1
        W[c11][col] = 1; W[c22][col] = 1
        
        col = col_offset + 1
        U[a21][col] = 1; U[a22][col] = 1
        V[b11][col] = 1
        W[c21][col] = 1; W[c22][col] = -1
        
        col = col_offset + 2
        U[a11][col] = 1
        V[b12][col] = 1; V[b22][col] = -1
        W[c12][col] = 1; W[c22][col] = 1
        
        col = col_offset + 3
        U[a22][col] = 1
        V[b21][col] = 1; V[b11][col] = -1
        W[c11][col] = 1; W[c21][col] = 1
        
        col = col_offset + 4
        U[a11][col] = 1; U[a12][col] = 1
        V[b22][col] = 1
        W[c11][col] = -1; W[c12][col] = 1
        
        col = col_offset + 5
        U[a21][col] = 1; U[a11][col] = -1
        V[b11][col] = 1; V[b12][col] = 1
        W[c22][col] = 1
        
        col = col_offset + 6
        U[a12][col] = 1; U[a22][col] = -1
        V[b21][col] = 1; V[b22][col] = 1
        W[c11][col] = 1

    def add_schoolbook_221(i1, i2, j1, j2, k1, col_offset):
        a11 = i1 * m + j1
        a12 = i1 * m + j2
        a21 = i2 * m + j1
        a22 = i2 * m + j2
        b11 = j1 * p + k1
        b21 = j2 * p + k1
        c11 = k1 * n + i1
        c21 = k1 * n + i2
        
        col = col_offset + 0
        U[a11][col] = 1; V[b11][col] = 1; W[c11][col] = 1
        col = col_offset + 1
        U[a12][col] = 1; V[b21][col] = 1; W[c11][col] = 1
        col = col_offset + 2
        U[a21][col] = 1; V[b11][col] = 1; W[c21][col] = 1
        col = col_offset + 3
        U[a22][col] = 1; V[b21][col] = 1; W[c21][col] = 1

    col = 0
    # Sub-problem 1: A rows 0-1, cols 0-1; B rows 0-1, cols 0-4
    add_strassen_222(0, 1, 0, 1, 0, 1, col); col += 7
    add_strassen_222(0, 1, 0, 1, 2, 3, col); col += 7
    add_schoolbook_221(0, 1, 0, 1, 4, col); col += 4
    
    # Sub-problem 2: A rows 0-1, cols 2-3; B rows 2-3, cols 0-4
    add_strassen_222(0, 1, 2, 3, 0, 1, col); col += 7
    add_strassen_222(0, 1, 2, 3, 2, 3, col); col += 7
    add_schoolbook_221(0, 1, 2, 3, 4, col); col += 4
    
    assert col == rank, f"Expected {rank} columns, got {col}"
    return U, V, W

def _construct_schoolbook():
    """Fallback schoolbook decomposition with rank n*m*p = 40."""
    n, m, p = 2, 4, 5
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
    
    return U, V, W

def construct():
    start_time = time.time()
    DEADLINE = 140
    
    n, m, p = DIMENSIONS
    
    # Try Rank-36 decomposition
    try:
        U_r36, V_r36, W_r36 = _construct_rank_36()
        if _validate_numpy(U_r36, V_r36, W_r36, n, m, p):
            result_U = [[_clean_entry(v) for v in row] for row in U_r36]
            result_V = [[_clean_entry(v) for v in row] for row in V_r36]
            result_W = [[_clean_entry(v) for v in row] for row in W_r36]
            return {"U": result_U, "V": result_V, "W": result_W}
    except Exception:
        pass
    
    # Fallback to Schoolbook
    U_school, V_school, W_school = _construct_schoolbook()
    assert _validate_numpy(U_school, V_school, W_school, n, m, p), "Schoolbook decomposition failed!"
    
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
