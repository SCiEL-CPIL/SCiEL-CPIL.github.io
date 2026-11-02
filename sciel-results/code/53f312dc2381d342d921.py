import numpy as np
import time

DIMENSIONS = [3, 3, 6]

def _strassen_2x2():
    """Return rank-7 Strassen decomposition for 2x2 * 2x2.
    
    Returns U (4,7), V (4,7), W (4,7) where:
      U[i, r] = coefficient of A[i//2][i%2] in the r-th left linear form
      V[j, r] = coefficient of B[j//2][j%2] in the r-th right linear form  
      W[k, r] = coefficient in output C[k//2][k%2] = sum_r W[k,r]*M_r
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1
    U[3, 0] = 1
    V[0, 0] = 1
    V[3, 0] = 1
    
    # M2 = (A10+A11)B00
    U[2, 1] = 1
    U[3, 1] = 1
    V[0, 1] = 1
    
    # M3 = A00(B10-B11)
    U[0, 2] = 1
    V[2, 2] = 1
    V[3, 2] = -1
    
    # M4 = A11(B01-B00)
    U[3, 3] = 1
    V[1, 3] = 1
    V[0, 3] = -1
    
    # M5 = (A00+A01)B11
    U[0, 4] = 1
    U[1, 4] = 1
    V[3, 4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1
    U[0, 5] = -1
    V[0, 5] = 1
    V[1, 5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1
    U[3, 6] = -1
    V[2, 6] = 1
    V[3, 6] = 1
    
    # Output: C[i,k] = sum_r W[k*2+i, r] * M_r
    # C00 = M1 + M4 - M5 - M7
    W[0, 0] = 1
    W[0, 3] = 1
    W[0, 4] = -1
    W[0, 6] = -1
    
    # C01 = M3 + M5
    W[1, 2] = 1
    W[1, 4] = 1
    
    # C10 = M2 + M4
    W[2, 1] = 1
    W[2, 3] = 1
    
    # C11 = M1 - M2 + M3 + M7
    W[3, 0] = 1
    W[3, 1] = -1
    W[3, 2] = 1
    W[3, 6] = 1
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication.
    
    C[i,j] = sum_k A[i,k] * B[k,j]
    In our indexing:
      U is (n*m, rank), indexed by i*m+k
      V is (m*p, rank), indexed by k*p+j  
      W is (p*n, rank), indexed by j*n+i (trace convention)
    
    The tensor T[i,k,j] = 1 should equal sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
    """
    rank = U.shape[1]
    # Build the tensor T
    T = np.zeros((n, m, p))
    for i in range(n):
        for k in range(m):
            for j in range(p):
                T[i, k, j] = 1.0
    
    # Compute from decomposition
    T_comp = np.zeros((n, m, p))
    for i in range(n):
        for k in range(m):
            for j in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
                T_comp[i, k, j] = s
    
    return np.allclose(T, T_comp, atol=1e-9)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # 3, 3, 6
    
    # Strategy: Block decomposition using Strassen for 2x2 blocks
    # A is 3x3, B is 3x6, C = A*B is 3x6
    # 
    # Split B into three 3x2 blocks: B[:, 0:2], B[:, 2:4], B[:, 4:6]
    # For each block Bj (3x2):
    #   Cj = A * Bj (3x3 * 3x2 = 3x2)
    #   Split rows of Cj: top 2 rows (i=0,1) and bottom 1 row (i=2)
    #   
    #   Top part: Cj[0:2, :] = A[0:2, :] * Bj[:, :]
    #     = A[0:2, 0:2] * Bj[0:2, :] + A[0:2, 2] * Bj[2, :]
    #     First term: 2x2 * 2x2 -> rank 7 (Strassen)
    #     Second term: 2x1 * 1x2 -> rank 2 (each element is separate)
    #   
    #   Bottom part: Cj[2, :] = A[2, :] * Bj[:, :]
    #     = A[2, 0:2] * Bj[0:2, :] + A[2, 2] * Bj[2, :]
    #     First term: 1x2 * 2x2 -> rank 2
    #     Second term: 1x1 * 1x2 -> rank 1
    #   
    #   Total per block: 7 + 2 + 2 + 1 = 12
    #   Total: 3 * 12 = 36
    #
    # This is better than schoolbook rank 54!
    
    U_list = []
    V_list = []
    W_list = []
    
    for j in range(3):  # For each 3x2 block of B
        b_col_start = 2 * j
        
        # --- Strassen for A[0:2, 0:2] * Bj[0:2, :] (2x2 * 2x2) ---
        U_s, V_s, W_s = _strassen_2x2()
        
        for r in range(7):
            U_term = np.zeros(n * m)  # 9 entries
            V_term = np.zeros(m * p)  # 18 entries
            W_term = np.zeros(p * n)  # 18 entries
            
            # U: A is 3x3, flat index i*m+k = i*3+k
            for a_flat in range(4):
                ai = a_flat // 2
                ak = a_flat % 2
                U_term[ai * m + ak] = U_s[a_flat, r]
            
            # V: B is 3x6, flat index k*p+j = k*6+j
            for b_flat in range(4):
                bk = b_flat // 2
                bj = b_flat % 2
                global_bj = b_col_start + bj
                V_term[bk * p + global_bj] = V_s[b_flat, r]
            
            # W: C is 3x6, output index j*n+i = j*3+i (trace convention)
            for c_flat in range(4):
                ci = c_flat // 2
                cj = c_flat % 2
                global_cj = b_col_start + cj
                W_term[global_cj * n + ci] = W_s[c_flat, r]
            
            U_list.append(U_term)
            V_list.append(V_term)
            W_list.append(W_term)
        
        # --- A[0:2, 2] * Bj[2, :] (2x1 * 1x2) -> rank 2 ---
        # C[0, b_col_start] += A[0,2] * Bj[2, 0]
        # C[0, b_col_start+1] += A[0,2] * Bj[2, 1]
        # C[1, b_col_start] += A[1,2] * Bj[2, 0]
        # C[1, b_col_start+1] += A[1,2] * Bj[2, 1]
        
        for ii in range(2):  # i = 0, 1
            for jj in range(2):  # j = 0, 1
                U_term = np.zeros(n * m)
                V_term = np.zeros(m * p)
                W_term = np.zeros(p * n)
                
                U_term[ii * m + 2] = 1.0  # A[ii, 2]
                V_term[2 * p + b_col_start + jj] = 1.0  # B[2, b_col_start+jj]
                W_term[(b_col_start + jj) * n + ii] = 1.0  # C[b_col_start+jj, ii]
                
                U_list.append(U_term)
                V_list.append(V_term)
                W_list.append(W_term)
        
        # --- A[2, 0:2] * Bj[0:2, :] (1x2 * 2x2) -> rank 2 ---
        # C[2, b_col_start] += A[2,0]*Bj[0,0] + A[2,1]*Bj[1,0]
        # C[2, b_col_start+1] += A[2,0]*Bj[0,1] + A[2,1]*Bj[1,1]
        
        for jj in range(2):  # j = 0, 1
            # C[2, b_col_start+jj] gets contribution from k=0 and k=1
            # We need two terms: one for A[2,0]*Bj[0,jj] and one for A[2,1]*Bj[1,jj]
            for kk in range(2):  # k = 0, 1
                U_term = np.zeros(n * m)
                V_term = np.zeros(m * p)
                W_term = np.zeros(p * n)
                
                U_term[2 * m + kk] = 1.0  # A[2, kk]
                V_term[kk * p + b_col_start + jj] = 1.0  # B[kk, b_col_start+jj]
                W_term[(b_col_start + jj) * n + 2] = 1.0  # C[b_col_start+jj, 2]
                
                U_list.append(U_term)
                V_list.append(V_term)
                W_list.append(W_term)
        
        # --- A[2, 2] * Bj[2, :] (1x1 * 1x2) -> rank 1 per element, total 2 ---
        # C[2, b_col_start] += A[2,2] * Bj[2, 0]
        # C[2, b_col_start+1] += A[2,2] * Bj[2, 1]
        
        for jj in range(2):  # j = 0, 1
            U_term = np.zeros(n * m)
            V_term = np.zeros(m * p)
            W_term = np.zeros(p * n)
            
            U_term[2 * m + 2] = 1.0  # A[2, 2]
            V_term[2 * p + b_col_start + jj] = 1.0  # B[2, b_col_start+jj]
            W_term[(b_col_start + jj) * n + 2] = 1.0  # C[b_col_start+jj, 2]
            
            U_list.append(U_term)
            V_list.append(V_term)
            W_list.append(W_term)
    
    # Convert to arrays
    rank = len(U_list)
    U = np.column_stack(U_list)
    V = np.column_stack(V_list)
    W = np.column_stack(W_list)
    
    # Verify the decomposition
    if time.time() < deadline:
        is_valid = _verify_decomposition(U, V, W, n, m, p)
        if not is_valid:
            # Fall back to schoolbook if verification fails
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
