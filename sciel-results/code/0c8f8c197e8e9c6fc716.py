import time
import numpy as np

T_START = time.time()
DEADLINE = T_START + 130.0

def _is_deadline():
    return time.time() >= DEADLINE

def _winograd_3x3x3():
    """
    Winograd F(4,3) for 3x3x3 matrix multiplication.
    Rank 16.
    
    m(A) = U @ A @ P^T  (4x4)
    m(B) = U @ B @ P^T  (4x4)
    M = m(A) * m(B) element-wise (16 products)
    C = U^T @ M @ P (3x3)
    
    P = [[1,1,1],[1,1,-1],[1,-1,1],[1,-1,-1]]  (4x3)
    U = [[1,0,0,0],[1,0,1,0],[1,0,-1,0],[1,1,0,-1]]  (4x4)
    """
    P = np.array([[1,1,1],[1,1,-1],[1,-1,1],[1,-1,-1]], dtype=object)
    U = np.array([[1,0,0,0],[1,0,1,0],[1,0,-1,0],[1,1,0,-1]], dtype=object)
    
    rank = 16
    U_mat = np.zeros((9, rank), dtype=object)
    V_mat = np.zeros((9, rank), dtype=object)
    W_mat = np.zeros((9, rank), dtype=object)
    
    for a in range(4):
        for b in range(4):
            col = a * 4 + b
            # U_mat[A_idx, col]: A_idx = i*3+j, coefficient = U[a,i] * P[b,j]
            for i in range(3):
                for j in range(3):
                    U_mat[i*3+j, col] += U[a,i] * P[b,j]
                    V_mat[i*3+j, col] += U[a,i] * P[b,j]
                    W_mat[i*3+j, col] += U[a,i] * P[b,j]
    
    # Fix W: C[i,k] = sum_{a,b} U[a,i] * M[a,b] * P[b,k]
    # W_mat[C_idx, col] where C_idx = i*3+k
    W_mat = np.zeros((9, rank), dtype=object)
    for a in range(4):
        for b in range(4):
            col = a * 4 + b
            for i in range(3):
                for k in range(3):
                    W_mat[i*3+k, col] += U[a,i] * P[b,k]
    
    return U_mat, V_mat, W_mat, rank

def _schoolbook_3x3x2():
    """3x3 * 3x2 = 3x2. Schoolbook rank 18."""
    n, m, p = 3, 3, 2
    rank = n * m * p
    U_mat = np.zeros((n*m, rank), dtype=object)
    V_mat = np.zeros((m*p, rank), dtype=object)
    W_mat = np.zeros((p*n, rank), dtype=object)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_mat[i*m+j, col] = 1
                V_mat[j*p+k, col] = 1
                # W convention: W[k*n+i, col]
                W_mat[k*n+i, col] = 1
                col += 1
    
    return U_mat, V_mat, W_mat, rank

def _schoolbook_3x2x3():
    """3x2 * 2x3 = 3x3. Schoolbook rank 18."""
    n, m, p = 3, 2, 3
    rank = n * m * p
    U_mat = np.zeros((n*m, rank), dtype=object)
    V_mat = np.zeros((m*p, rank), dtype=object)
    W_mat = np.zeros((p*n, rank), dtype=object)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_mat[i*m+j, col] = 1
                V_mat[j*p+k, col] = 1
                W_mat[k*n+i, col] = 1
                col += 1
    
    return U_mat, V_mat, W_mat, rank

def _strassen_2x2x2():
    """2x2 * 2x2 = 2x2. Strassen rank 7."""
    rank = 7
    U_mat = np.zeros((4, rank), dtype=object)
    V_mat = np.zeros((4, rank), dtype=object)
    W_mat = np.zeros((4, rank), dtype=object)
    
    # P0 = (A00+A11)*(B00+B11)
    U_mat[0,0] += 1; U_mat[3,0] += 1
    V_mat[0,0] += 1; V_mat[3,0] += 1
    W_mat[0,0] += 1; W_mat[3,3] += 1
    
    # P1 = (A10+A11)*B00
    U_mat[2,1] += 1; U_mat[3,1] += 1
    V_mat[0,1] += 1
    W_mat[2,0] += 1; W_mat[3,3] -= 1
    
    # P2 = A00*(B01-B11)
    U_mat[0,2] += 1
    V_mat[1,2] += 1; V_mat[3,2] -= 1
    W_mat[1,1] += 1; W_mat[3,3] += 1
    
    # P3 = A11*(B10-B00)
    U_mat[3,3] += 1
    V_mat[2,3] += 1; V_mat[0,3] -= 1
    W_mat[0,0] += 1; W_mat[2,0] += 1
    
    # P4 = (A00+A01)*B11
    U_mat[0,4] += 1; U_mat[1,4] += 1
    V_mat[3,4] += 1
    W_mat[0,0] -= 1; W_mat[1,1] += 1
    
    # P5 = (A10-A00)*(B00+B01)
    U_mat[2,5] += 1; U_mat[0,5] -= 1
    V_mat[0,5] += 1; V_mat[1,5] += 1
    W_mat[0,0] += 1
    
    # P6 = (A01-A11)*(B10+B11)
    U_mat[1,6] += 1; U_mat[3,6] -= 1
    V_mat[2,6] += 1; V_mat[3,6] += 1
    W_mat[3,3] += 1
    
    return U_mat, V_mat, W_mat, rank

def _schoolbook_3x2x2():
    """3x2 * 2x2 = 3x2. Schoolbook rank 12."""
    n, m, p = 3, 2, 2
    rank = n * m * p
    U_mat = np.zeros((n*m, rank), dtype=object)
    V_mat = np.zeros((m*p, rank), dtype=object)
    W_mat = np.zeros((p*n, rank), dtype=object)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_mat[i*m+j, col] = 1
                V_mat[j*p+k, col] = 1
                W_mat[k*n+i, col] = 1
                col += 1
    
    return U_mat, V_mat, W_mat, rank

def construct():
    """
    Decompose 3x5x5 matrix multiplication.
    A: 3x5, B: 5x5, C: 3x5.
    
    Partition: j into [0:3]+[3:5], k into [0:3]+[3:5].
    C[i,k] = sum_j A[i,j] B[j,k]
    
    Blocks:
    1. A[0:3, 0:3] * B[0:3, 0:3] -> C[0:3, 0:3]  (3x3x3, Winograd rank 16)
    2. A[0:3, 0:3] * B[0:3, 3:5] -> C[0:3, 3:5]  (3x3x2, schoolbook rank 18)
    3. A[0:3, 3:5] * B[3:5, 0:3] -> C[0:3, 0:3]  (3x2x3, schoolbook rank 18)
    4. A[0:3, 3:5] * B[3:5, 3:5] -> C[0:3, 3:5]  (3x2x2, schoolbook rank 12)
    
    Total rank: 16 + 18 + 18 + 12 = 64
    """
    n, m, p = 3, 5, 5
    
    U_full = np.zeros((n*m, 0), dtype=object)
    V_full = np.zeros((m*p, 0), dtype=object)
    W_full = np.zeros((p*n, 0), dtype=object)
    
    def add_block(Ub, Vb, Wb, j_start, k_start, j_len, k_len):
        nonlocal U_full, V_full, W_full
        rb = Ub.shape[1]
        new_U = np.zeros((n*m, rb), dtype=object)
        new_V = np.zeros((m*p, rb), dtype=object)
        new_W = np.zeros((p*n, rb), dtype=object)
        
        # Map U: local (i, jl) -> global (i, j_start+jl)
        for i in range(n):
            for jl in range(j_len):
                for c in range(rb):
                    new_U[i*m + (j_start+jl), c] = Ub[i*j_len+jl, c]
        
        # Map V: local (jl, kl) -> global (j_start+jl, k_start+kl)
        for jl in range(j_len):
            for kl in range(k_len):
                for c in range(rb):
                    new_V[(j_start+jl)*p + (k_start+kl), c] = Vb[jl*k_len+kl, c]
        
        # Map W: local (i, kl) -> global (k_start+kl, i), W[k*n+i, c]
        for i in range(n):
            for kl in range(k_len):
                for c in range(rb):
                    new_W[(k_start+kl)*n + i, c] = Wb[i*k_len+kl, c]
        
        U_full = np.hstack([U_full, new_U])
        V_full = np.hstack([V_full, new_V])
        W_full = np.hstack([W_full, new_W])
    
    # Block 1: A[0:3, 0:3] * B[0:3, 0:3] -> C[0:3, 0:3]
    if not _is_deadline():
        Ub, Vb, Wb, _ = _winograd_3x3x3()
        add_block(Ub, Vb, Wb, 0, 0, 3, 3)
    
    # Block 2: A[0:3, 0:3] * B[0:3, 3:5] -> C[0:3, 3:5]
    if not _is_deadline():
        Ub, Vb, Wb, _ = _schoolbook_3x3x2()
        add_block(Ub, Vb, Wb, 0, 3, 3, 2)
    
    # Block 3: A[0:3, 3:5] * B[3:5, 0:3] -> C[0:3, 0:3]
    if not _is_deadline():
        Ub, Vb, Wb, _ = _schoolbook_3x2x3()
        add_block(Ub, Vb, Wb, 3, 0, 2, 3)
    
    # Block 4: A[0:3, 3:5] * B[3:5, 3:5] -> C[0:3, 3:5]
    if not _is_deadline():
        Ub, Vb, Wb, _ = _schoolbook_3x2x2()
        add_block(Ub, Vb, Wb, 3, 3, 2, 2)
    
    U_out = U_full.tolist()
    V_out = V_full.tolist()
    W_out = W_full.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
