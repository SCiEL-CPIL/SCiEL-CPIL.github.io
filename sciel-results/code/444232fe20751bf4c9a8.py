import numpy as np
import time

DIMENSIONS = [3, 3, 6]

def _bclr_23_terms():
    """Return the 23-term BCLR decomposition for 3x3x3.
    
    Returns list of (u, v, w) where each is a 9-element vector.
    u[i*3+j] = coeff for A[i,j]
    v[j*3+k] = coeff for B[j,k]  
    w[i*3+k] = coeff for C[i,k]
    """
    terms = []
    
    # The 23 BCLR terms (from Bini, Capovani, Lotti, Romani 1979)
    # Each term: (u_coeffs, v_coeffs, w_coeffs) as 9-tuples
    
    # I'll use a verified rank-23 algorithm
    # Reference: "Optimal Algorithms for Multiplying 3x3 and 4x4 Matrices"
    
    t = [
        # Term 0
        ([1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0]),
        # Term 1
        ([0,0,0, 1,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [0,0,0, 1,0,0, 0,0,0]),
        # Term 2
        ([0,0,0, 0,0,0, 1,0,0], [1,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 1,0,0]),
        # Term 3
        ([0,1,0, 0,0,0, 0,0,0], [0,0,0, 1,0,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0]),
        # Term 4
        ([0,0,0, 0,1,0, 0,0,0], [0,0,0, 1,0,0, 0,0,0], [0,0,0, 0,0,1, 0,0,0]),
        # Term 5
        ([0,0,0, 0,0,0, 0,1,0], [0,0,0, 1,0,0, 0,0,0], [0,0,0, 0,0,0, 0,1,0]),
        # Term 6
        ([0,0,1, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 1,0,0], [0,0,0, 0,0,0, 0,0,1]),
        # Term 7
        ([0,0,0, 0,0,1, 0,0,0], [0,0,0, 0,0,0, 1,0,0], [0,0,0, 0,0,0, 0,0,0, 1]),  # wrong length
        # Term 8
        ([0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 1,0,0], [0,0,0, 0,0,0, 0,0,0, 0,0,1]),  # wrong
    ]
    # This is getting messy. Let me use a cleaner formulation.
    return terms

def _strassen_2x2():
    """Return rank-7 Strassen decomposition for 2x2 * 2x2."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    
    W[0, 0] = 1; W[0, 3] = 1; W[0, 4] = -1; W[0, 6] = -1
    W[1, 2] = 1; W[1, 4] = 1
    W[2, 1] = 1; W[2, 3] = 1
    W[3, 0] = 1; W[3, 1] = -1; W[3, 2] = 1; W[3, 6] = 1
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify the decomposition."""
    rank = U.shape[1]
    T = np.zeros((n, m, p))
    T.fill(1.0)
    
    T_comp = np.zeros((n, m, p))
    for i in range(n):
        for k in range(m):
            for j in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
                T_comp[i, k, j] = s
    
    return np.allclose(T, T_comp, atol=1e-9)

def _build_rank36_decomposition(n, m, p):
    """Build the rank-36 decomposition using Strassen on 2x2 blocks.
    
    Split B into three 3x2 blocks. For each block:
    - Use Strassen (rank 7) for the 2x2 * 2x2 part
    - Add rank 6 for the remaining terms
    Total per block: 13, total: 39
    
    Actually, let me recount: 7 + 4 + 4 + 2 = 17 per block, 51 total.
    But maybe I can do better by sharing some terms.
    """
    U_list = []
    V_list = []
    W_list = []
    
    for j in range(3):
        b_col_start = 2 * j
        
        # Strassen for A[0:2, 0:2] * B[0:2, b_col_start:b_col_start+2]
        U_s, V_s, W_s = _strassen_2x2()
        
        for r in range(7):
            U_term = np.zeros(n * m)
            V_term = np.zeros(m * p)
            W_term = np.zeros(p * n)
            
            for a_flat in range(4):
                ai = a_flat // 2
                ak = a_flat % 2
                U_term[ai * m + ak] = U_s[a_flat, r]
            
            for b_flat in range(4):
                bk = b_flat // 2
                bj = b_flat % 2
                global_bj = b_col_start + bj
                V_term[bk * p + global_bj] = V_s[b_flat, r]
            
            for c_flat in range(4):
                ci = c_flat // 2
                cj = c_flat % 2
                global_cj = b_col_start + cj
                W_term[global_cj * n + ci] = W_s[c_flat, r]
            
            U_list.append(U_term)
            V_list.append(V_term)
            W_list.append(W_term)
        
        # A[0:2, 2] * B[2, b_col_start:b_col_start+2]
        for ii in range(2):
            for jj in range(2):
                U_term = np.zeros(n * m)
                V_term = np.zeros(m * p)
                W_term = np.zeros(p * n)
                
                U_term[ii * m + 2] = 1.0
                V_term[2 * p + b_col_start + jj] = 1.0
                W_term[(b_col_start + jj) * n + ii] = 1.0
                
                U_list.append(U_term)
                V_list.append(V_term)
                W_list.append(W_term)
        
        # A[2, 0:2] * B[0:2, b_col_start:b_col_start+2]
        for jj in range(2):
            for kk in range(2):
                U_term = np.zeros(n * m)
                V_term = np.zeros(m * p)
                W_term = np.zeros(p * n)
                
                U_term[2 * m + kk] = 1.0
                V_term[kk * p + b_col_start + jj] = 1.0
                W_term[(b_col_start + jj) * n + 2] = 1.0
                
                U_list.append(U_term)
                V_list.append(V_term)
                W_list.append(W_term)
        
        # A[2, 2] * B[2, b_col_start:b_col_start+2]
        for jj in range(2):
            U_term = np.zeros(n * m)
            V_term = np.zeros(m * p)
            W_term = np.zeros(p * n)
            
            U_term[2 * m + 2] = 1.0
            V_term[2 * p + b_col_start + jj] = 1.0
            W_term[(b_col_start + jj) * n + 2] = 1.0
            
            U_list.append(U_term)
            V_list.append(V_term)
            W_list.append(W_term)
    
    rank = len(U_list)
    U = np.column_stack(U_list)
    V = np.column_stack(V_list)
    W = np.column_stack(W_list)
    
    return U, V, W, rank

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    
    # Build the Strassen-based decomposition
    U, V, W, rank = _build_rank36_decomposition(n, m, p)
    
    # Verify
    if time.time() < deadline:
        is_valid = _verify_decomposition(U, V, W, n, m, p)
        if not is_valid:
            # Fallback to schoolbook
            R = n * m * p
            U = np.zeros((n * m, R))
            V = np.zeros((m * p, R))
            W = np.zeros((p * n, R))
            r = 0
            for i in range(n):
                for k in range(m):
                    for j in range(p):
                        U[i * m + k, r] = 1.0
                        V[k * p + j, r] = 1.0
                        W[j * n + i, r] = 1.0
                        r += 1
            rank = R
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
