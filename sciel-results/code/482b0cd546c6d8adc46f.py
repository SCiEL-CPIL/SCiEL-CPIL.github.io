import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    trivial_rank = n * m * p  # 36
    
    # Build the target tensor T
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    # Trivial solution as fallback
    U_triv = np.zeros((n * m, trivial_rank))
    V_triv = np.zeros((m * p, trivial_rank))
    W_triv = np.zeros((p * n, trivial_rank))
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_triv[i * m + j, idx] = 1.0
                V_triv[j * p + k, idx] = 1.0
                W_triv[k * n + i, idx] = 1.0
                idx += 1

    # Rank 34 Solution using Strassen for 2x2x2 blocks
    # Partition:
    # A (3x3) into A11 (2x2), A12 (2x1), A21 (1x2), A22 (1x1)
    # B (3x4) into B11 (2x2), B12 (2x2), B21 (1x2), B22 (1x2)
    # C (4x3) into C11 (2x2), C12 (2x2), C21 (1x2), C22 (1x2)
    
    # Indices for A (3x3):
    # i, j in 0..2
    # A11: i in 0,1; j in 0,1
    # A12: i in 0,1; j in 2
    # A21: i in 2; j in 0,1
    # A22: i in 2; j in 2
    
    # Indices for B (3x4):
    # j, k in 0..2, 0..3
    # B11: j in 0,1; k in 0,1
    # B12: j in 0,1; k in 2,3
    # B21: j in 2; k in 0,1
    # B22: j in 2; k in 2,3
    
    # Indices for C (4x3):
    # k, i in 0..3, 0..2
    # C11: k in 0,1; i in 0,1
    # C12: k in 2,3; i in 0,1
    # C21: k in 0,1; i in 2
    # C22: k in 2,3; i in 2

    U = np.zeros((n * m, 34))
    V = np.zeros((m * p, 34))
    W = np.zeros((p * n, 34))
    
    col_idx = 0
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, coeff=1.0):
        nonlocal col_idx
        U[u_idx, col_idx] += coeff
        V[v_idx, col_idx] += coeff
        W[w_idx, col_idx] += coeff
        col_idx += 1

    # Strassen 2x2x2 for A11 * B11
    # A11: i in {0,1}, j in {0,1}
    # B11: j in {0,1}, k in {0,1}
    # C11: k in {0,1}, i in {0,1}
    
    # M1 = (A11[0,0] + A11[1,1]) * (B11[0,0] + B11[1,1])
    # U: (0,0)=1, (1,1)=1
    # V: (0,0)=1, (1,1)=1
    # W: (0,0)=1, (1,1)=1
    # Note: W index is k*n+i. n=3.
    # C11[0,0] -> k=0, i=0 -> 0*3+0 = 0
    # C11[1,1] -> k=1, i=1 -> 1*3+1 = 4
    
    # Let's implement Strassen 2x2x2 generally
    def strassen_2x2x2(u_base_i, u_base_j, v_base_j, v_base_k, w_base_k, w_base_i):
        nonlocal col_idx
        # u_base_i, u_base_j: starting i, j for A
        # v_base_j, v_base_k: starting j, k for B
        # w_base_k, w_base_i: starting k, i for C
        
        # M1
        U[u_base_i + 0, col_idx] = 1; U[u_base_i + 1, col_idx] = 1
        V[v_base_j + 0, col_idx] = 1; V[v_base_j + 1, col_idx] = 1
        W[w_base_k + 0, col_idx] = 1; W[w_base_k + 1, col_idx] = 1
        col_idx += 1
        
        # M2
        U[u_base_i + 1, col_idx] = 1; U[u_base_i + 1, col_idx] = 1 # A21, A22 -> i=1, j=0,1
        V[v_base_j + 0, col_idx] = 1 # B11 -> j=0, k=0
        W[w_base_k + 0, col_idx] = -1 # C21 -> k=0, i=1
        col_idx += 1
        
        # M3
        U[u_base_i + 0, col_idx] = 1 # A11
        V[v_base_j + 0, col_idx] = 1; V[v_base_j + 1, col_idx] = -1 # B12, B22
        W[w_base_k + 1, col_idx] = 1; W[w_base_k + 1, col_idx] = 1 # C21, C22 -> k=1
        col_idx += 1
        
        # M4
        U[u_base_i + 1, col_idx] = 1 # A22
        V[v_base_j + 1, col_idx] = 1; V[v_base_j + 0, col_idx] = -1 # B21, B11
        W[w_base_k + 0, col_idx] = 1; W[w_base_k + 1, col_idx] = 1 # C11, C22 -> i=0,1
        col_idx += 1
        
        # M5
        U[u_base_i + 0, col_idx] = 1; U[u_base_i + 0, col_idx] = 1 # A11, A12
        V[v_base_j + 1, col_idx] = 1 # B22
        W[w_base_k + 1, col_idx] = -1 # C21
        col_idx += 1
        
        # M6
        U[u_base_i + 1, col_idx] = 1; U[u_base_i + 0, col_idx] = -1 # A21, A11
        V[v_base_j + 0, col_idx] = 1; V[v_base_j + 1, col_idx] = 1 # B11, B12
        W[w_base_k + 1, col_idx] = 1 # C22
        col_idx += 1
        
        # M7
        U[u_base_i + 0, col_idx] = 1; U[u_base_i + 1, col_idx] = -1 # A12, A22
        V[v_base_j + 1, col_idx] = 1; V[v_base_j + 1, col_idx] = 1 # B21, B22
        W[w_base_k + 0, col_idx] = 1 # C11
        col_idx += 1

    # 1. A11 * B11 (Strassen)
    # A11: i in 0,1; j in 0,1
    # B11: j in 0,1; k in 0,1
    # C11: k in 0,1; i in 0,1
    strassen_2x2x2(0, 0, 0, 0, 0, 0)
    
    # 2. A11 * B12 (Strassen)
    # A11: i in 0,1; j in 0,1
    # B12: j in 0,1; k in 2,3
    # C12: k in 2,3; i in 0,1
    strassen_2x2x2(0, 0, 0, 2, 2, 0)
    
    # 3. A12 * B21 (Trivial 2x1x2)
    # A12: i in 0,1; j in 2
    # B21: j in 2; k in 0,1
    # C21: k in 0,1; i in 0,1
    for i in range(2):
        for k in range(2):
            add_term(i * 3 + 2, 2 * 4 + k, k * 3 + i)
            
    # 4. A12 * B22 (Trivial 2x1x2)
    # A12: i in 0,1; j in 2
    # B22: j in 2; k in 2,3
    # C22: k in 2,3; i in 0,1
    for i in range(2):
        for k in range(2, 4):
            add_term(i * 3 + 2, 2 * 4 + k, k * 3 + i)
            
    # 5. A21 * B11 (Trivial 1x2x2)
    # A21: i in 2; j in 0,1
    # B11: j in 0,1; k in 0,1
    # C21: k in 0,1; i in 2
    for j in range(2):
        for k in range(2):
            add_term(2 * 3 + j, j * 4 + k, k * 3 + 2)
            
    # 6. A21 * B12 (Trivial 1x2x2)
    # A21: i in 2; j in 0,1
    # B12: j in 0,1; k in 2,3
    # C22: k in 2,3; i in 2
    for j in range(2):
        for k in range(2, 4):
            add_term(2 * 3 + j, j * 4 + k, k * 3 + 2)
            
    # 7. A22 * B21 (Trivial 1x1x2)
    # A22: i in 2; j in 2
    # B21: j in 2; k in 0,1
    # C21: k in 0,1; i in 2
    for k in range(2):
        add_term(2 * 3 + 2, 2 * 4 + k, k * 3 + 2)
        
    # 8. A22 * B22 (Trivial 1x1x2)
    # A22: i in 2; j in 2
    # B22: j in 2; k in 2,3
    # C22: k in 2,3; i in 2
    for k in range(2, 4):
        add_term(2 * 3 + 2, 2 * 4 + k, k * 3 + 2)
        
    # Verify the solution
    T_approx = np.einsum('ar,br,cr->abc', U, V, W)
    residual = np.linalg.norm(T - T_approx)
    
    if residual < 1e-8:
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    else:
        # Fallback to trivial if construction failed
        return {
            "U": U_triv.tolist(),
            "V": V_triv.tolist(),
            "W": W_triv.tolist()
        }
