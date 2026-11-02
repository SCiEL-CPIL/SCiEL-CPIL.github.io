import numpy as np
import time

# EVOLVE-BLOCK-START

def _generate_strassen_4x4():
    """
    Generates U, V, W for 4x4 Strassen decomposition (Rank 49).
    Uses the standard 2x2 Strassen recursion applied to 4x4 matrices.
    
    Convention:
      U[i*m + j, r] for A[i,j]
      V[j*p + k, r] for B[j,k]  
      W[k*n + i, r] for C[i,k] (trace convention)
    """
    n = 4
    rank = 49
    U = np.zeros((n*n, rank), dtype=float)
    V = np.zeros((n*n, rank), dtype=float)
    W = np.zeros((n*n, rank), dtype=float)
    
    # 2x2 Strassen: M1..M7
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For each Strassen step t (0..6), define which A-blocks, B-blocks, C-blocks are involved
    # A_t: list of (block_row, block_col, sign)
    # B_t: list of (block_row, block_col, sign)
    # C_t: list of (block_row, block_col, sign)
    
    A_terms = [
        [(0,0,1), (1,1,1)],    # M1: A11 + A22
        [(1,0,1), (1,1,1)],    # M2: A21 + A22
        [(0,0,1)],              # M3: A11
        [(1,1,1)],              # M4: A22
        [(0,0,1), (0,1,1)],    # M5: A11 + A12
        [(1,0,1), (0,0,-1)],   # M6: A21 - A11
        [(0,1,1), (1,1,-1)],   # M7: A12 - A22
    ]
    
    B_terms = [
        [(0,0,1), (1,1,1)],    # M1: B11 + B22
        [(0,0,1)],              # M2: B11
        [(0,1,1), (1,1,-1)],   # M3: B12 - B22
        [(1,0,1), (0,0,-1)],   # M4: B21 - B11
        [(1,1,1)],              # M5: B22
        [(0,0,1), (0,1,1)],    # M6: B11 + B12
        [(1,0,1), (1,1,1)],    # M7: B21 + B22
    ]
    
    C_terms = [
        [(0,0,1), (1,1,1)],    # M1: C11 + C22
        [(1,0,1), (1,1,-1)],   # M2: C21 - C22
        [(0,1,1), (1,1,1)],    # M3: C12 + C22
        [(0,0,1), (1,0,1)],    # M4: C11 + C21
        [(0,1,1), (0,0,-1)],   # M5: C12 - C11
        [(1,1,1)],              # M6: C22
        [(0,0,1)],              # M7: C11
    ]
    
    for t1 in range(7):
        for t2 in range(7):
            r_idx = t1 * 7 + t2
            
            # U: A[i,j] coefficient for rank-r term
            for (br1, bc1, s1) in A_terms[t1]:
                for (br2, bc2, s2) in A_terms[t2]:
                    i_global = br1 * 2 + br2
                    j_global = bc1 * 2 + bc2
                    u_idx = i_global * n + j_global
                    U[u_idx, r_idx] += s1 * s2
            
            # V: B[j,k] coefficient for rank-r term
            for (br1, bc1, s1) in B_terms[t1]:
                for (br2, bc2, s2) in B_terms[t2]:
                    j_global = br1 * 2 + br2
                    k_global = bc1 * 2 + bc2
                    v_idx = j_global * n + k_global
                    V[v_idx, r_idx] += s1 * s2
            
            # W: C[i,k] coefficient, stored at W[k*n + i, r] (trace convention)
            for (br1, bc1, s1) in C_terms[t1]:
                for (br2, bc2, s2) in C_terms[t2]:
                    i_global = br1 * 2 + br2
                    k_global = bc1 * 2 + bc2
                    w_idx = k_global * n + i_global
                    W[w_idx, r_idx] += s1 * s2
    
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p, atol=1e-9):
    """
    Verify that the tensor decomposition exactly reproduces matrix multiplication.
    """
    rank = U.shape[1]
    
    U_r = np.asarray(U, dtype=float).reshape(n, m, rank)
    V_r = np.asarray(V, dtype=float).reshape(m, p, rank)
    W_r = np.asarray(W, dtype=float).reshape(p, n, rank)
    
    T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    
    target = np.ones((n, m, p))
    
    return bool(np.allclose(T, target, atol=atol))


def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    n, m, p = 4, 4, 4
    
    # Generate the rank-49 Strassen decomposition
    U, V, W = _generate_strassen_4x4()
    
    # Verify the decomposition
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook decomposition (rank 64)
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
