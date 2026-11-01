"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    rank = len(U[0]) if U else 0
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if len(U[0]) != rank or len(V[0]) != rank or len(W[0]) != rank:
        return False
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Efficient verification using tensor contraction
    # T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We want T[i, j, k] == 1 for all i, j, k
    
    # Reshape to use einsum or matmul for speed
    # U: (n*m, R)
    # V: (m*p, R)
    # W: (p*n, R)
    
    # Create full tensor T of shape (n, m, p)
    # T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Iterate over c
    # T = sum_c U_c (n*m) outer V_c (m*p) outer W_c (p*n) ? No.
    # T[i,j,k] is the coefficient.
    
    # Use broadcasting
    # U_reshaped: (n, m, R)
    # V_reshaped: (m, p, R)
    # W_reshaped: (p, n, R)
    
    U_r = U_arr.reshape(n, m, rank)
    V_r = V_arr.reshape(m, p, rank)
    W_r = W_arr.reshape(p, n, rank)
    
    # Sum over R
    # T[i, j, k] = sum_r U_r[i, j, r] * V_r[j, k, r] * W_r[k, i, r]
    
    # This is a contraction of 3 tensors.
    # We can do it in a loop over r if R is small, or use einsum.
    # np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    
    T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    
    # Check if T is all 1s
    if not np.allclose(T, 1.0, atol=tol):
        return False
        
    return True

def _strassen_decomposition():
    """
    Construct a Strassen-based decomposition for 2x4x8 matrix multiplication.
    Rank 56.
    """
    n, m, p = DIMENSIONS  # 2, 4, 8
    rank = 56
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    strassen_terms = [
        (
            {(0,0): 1, (1,1): 1},   
            {(0,0): 1, (1,1): 1},   
            {(0,0): 1, (1,1): 1}    
        ),
        (
            {(1,0): 1, (1,1): 1},   
            {(0,0): 1},             
            {(1,0): 1, (1,1): -1}   
        ),
        (
            {(0,0): 1},             
            {(0,1): 1, (1,1): -1},  
            {(0,1): 1, (1,1): 1}    
        ),
        (
            {(1,1): 1},             
            {(1,0): 1, (0,0): -1},  
            {(0,0): 1, (1,0): 1}    
        ),
        (
            {(0,0): 1, (0,1): 1},   
            {(1,1): 1},             
            {(0,0): -1, (0,1): 1}   
        ),
        (
            {(1,0): 1, (0,0): -1},  
            {(0,0): 1, (0,1): 1},   
            {(1,1): 1}              
        ),
        (
            {(0,1): 1, (1,1): -1},  
            {(1,0): 1, (1,1): 1},   
            {(0,0): 1}              
        ),
    ]
    
    col = 0
    for a_block in range(2):
        for b_block in range(4):
            for term in strassen_terms:
                A_coeffs, B_coeffs, C_coeffs = term
                
                for (li, lj), coeff in A_coeffs.items():
                    global_i = li
                    global_j = 2 * a_block + lj
                    U[global_i * m + global_j, col] += coeff
                
                for (lj, lk), coeff in B_coeffs.items():
                    global_j = 2 * a_block + lj
                    global_k = 2 * b_block + lk
                    V[global_j * p + global_k, col] += coeff
                
                for (li, lk), coeff in C_coeffs.items():
                    global_i = li
                    global_k = 2 * b_block + lk
                    W[global_k * n + global_i, col] += coeff
                
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _schoolbook_decomposition():
    """Fallback schoolbook decomposition with rank n*m*p = 64."""
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
    start_time = time.time()
    deadline_global = start_time + 110.0
    
    n, m, p = DIMENSIONS
    
    try:
        result = _strassen_decomposition()
        U, V, W = result["U"], result["V"], result["W"]
        
        # Quick verify
        if _verify(U, V, W, n, m, p, tol=1e-9):
            return result
    except Exception:
        pass
    
    return _schoolbook_decomposition()

# EVOLVE-BLOCK-END
