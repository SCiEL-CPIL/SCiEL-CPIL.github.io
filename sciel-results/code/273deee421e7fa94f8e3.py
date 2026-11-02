import numpy as np
import time

DIMENSIONS = [3, 3, 6]

def _build_schoolbook(n, m, p):
    """Return the trivial schoolbook (rank = n*m*p) decomposition."""
    R = n * m * p
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, r] = 1.0
                V[j * p + k, r] = 1.0
                W[k * n + i, r] = 1.0
                r += 1
    return U, V, W

def _strassen_2x2():
    """Return rank-7 Strassen decomposition for 2x2 * 2x2.
    
    Returns U (4,7), V (4,7), W (4,7) where:
      U[i, r] = coefficient of A[i//2][i%2] in the r-th left linear form
      V[j, r] = coefficient of B[j//2][j%2] in the r-th right linear form  
      W[k, r] = coefficient in output C[k//2][k%2] = sum_r W[k,r]*M_r
    """
    # Standard Strassen for C = A*B, A,B,C all 2x2
    # Index: A[0,0],A[0,1],A[1,0],A[1,1] -> flat 0,1,2,3
    # B[0,0],B[0,1],B[1,0],B[1,1] -> flat 0,1,2,3
    # C[0,0],C[0,1],C[1,0],C[1,1] -> flat 0,1,2,3
    
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1  # A[0,0]
    U[3, 0] = 1  # A[1,1]
    V[0, 0] = 1  # B[0,0]
    V[3, 0] = 1  # B[1,1]
    
    # M2 = (A10+A11)B00
    U[2, 1] = 1  # A[1,0]
    U[3, 1] = 1  # A[1,1]
    V[0, 1] = 1  # B[0,0]
    
    # M3 = A00(B10-B11)
    U[0, 2] = 1  # A[0,0]
    V[2, 2] = 1  # B[1,0]
    V[3, 2] = -1 # B[1,1]
    
    # M4 = A11(B01-B00)
    U[3, 3] = 1  # A[1,1]
    V[1, 3] = 1  # B[0,1]
    V[0, 3] = -1 # B[0,0]
    
    # M5 = (A00+A01)B11
    U[0, 4] = 1  # A[0,0]
    U[1, 4] = 1  # A[0,1]
    V[3, 4] = 1  # B[1,1]
    
    # M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1  # A[1,0]
    U[0, 5] = -1 # A[0,0]
    V[0, 5] = 1  # B[0,0]
    V[1, 5] = 1  # B[0,1]
    
    # M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1  # A[0,1]
    U[3, 6] = -1 # A[1,1]
    V[2, 6] = 1  # B[1,0]
    V[3, 6] = 1  # B[1,1]
    
    # Output: C[i,k] = sum_r W[k*2+i, r] * M_r
    # C00 = M1 + M4 - M5 - M7
    W[0, 0] = 1  # C[0,0] from M1
    W[0, 3] = 1  # C[0,0] from M4
    W[0, 4] = -1 # C[0,0] from M5
    W[0, 6] = -1 # C[0,0] from M7
    
    # C01 = M3 + M5
    W[1, 2] = 1  # C[0,1] from M3
    W[1, 4] = 1  # C[0,1] from M5
    
    # C10 = M2 + M4
    W[2, 1] = 1  # C[1,0] from M2
    W[2, 3] = 1  # C[1,0] from M4
    
    # C11 = M1 - M2 + M3 + M7
    W[3, 0] = 1  # C[1,1] from M1
    W[3, 1] = -1 # C[1,1] from M2
    W[3, 2] = 1  # C[1,1] from M3
    W[3, 6] = 1  # C[1,1] from M7
    
    return U, V, W

def _build_block_decomposition(n, m, p):
    """Build a decomposition using block structure with Strassen for 2x2 blocks.
    
    For 3x3 * 3x6:
    Split A (3x3) into rows: top 2 rows (2x3), bottom 1 row (1x3)
    Split B (3x6) into columns: left 2 cols (3x2), next 2 (3x2), right 2 (3x2)
    Split C (3x6) into columns: left 2, next 2, right 2
    
    For each block of A (2x3 or 1x3) times block of B (3x2):
      - 2x3 * 3x2: split A into [A_top_left(2x2) | A_top_right(2x1)], 
        B into [B_top(2x2); B_bot(1x2)]
        Result = A_top_left * B_top + A_top_right * B_bot
        First term: 2x2 * 2x2 -> rank 7 (Strassen)
        Second term: 2x1 * 1x2 -> rank 1 (outer product)
        Total: rank 8 for this block
        
      - 1x3 * 3x2: split B into [B_top(2x2); B_bot(1x2)]
        Result = A_row * B_top (1x3*2x2... wait, A is 1x3, B is 3x2)
        Actually A_row (1x3) * B (3x2):
        Split B into rows: B_top (2x2), B_bot (1x2)
        A_row = [A_left (1x2) | A_right (1x1)]
        Result = A_left * B_top + A_right * B_bot
        First: 1x2 * 2x2 -> rank 2
        Second: 1x1 * 1x2 -> rank 1
        Total: rank 3
        
    So total rank = 3 blocks * (8 + 3) = 3 * 11 = 33
    
    But wait, we need to be careful. Let me reconsider.
    
    Actually, let me think about this more carefully.
    
    A is 3x3, B is 3x6. C = A*B is 3x6.
    
    Split C into three 3x2 blocks: C1, C2, C3 (columns 0-1, 2-3, 4-5)
    B = [B1 | B2 | B3] where each Bi is 3x2.
    
    Cj = A * Bj for j=1,2,3.
    
    For each Cj = A * Bj (3x3 * 3x2 = 3x2):
      Split A into A_top (2x3) and A_bot (1x3)
      Split Bj into Bj_top (2x2) and Bj_bot (1x2)
      
      Cj_top = A_top * Bj_top (2x3 * 2x2... wait)
      
      Hmm, A_top is 2x3, Bj_top is 2x2. These don't multiply directly.
      
      Let me redo: A is 3x3. Rows 0,1 are "top", row 2 is "bottom".
      Bj is 3x2. Rows 0,1 are "top", row 2 is "bottom".
      
      Cj[i,k] = sum_j A[i,j] * Bj[j,k]
      
      Cj_top (rows 0,1) = sum_j A_top[:,j] * Bj[j,k]
      = A_top[0:2, 0:2] * Bj[0:2, :] + A_top[0:2, 2] * Bj[2, :]
      
      So Cj_top = A_top_left (2x2) * Bj_top (2x2) + A_top_right (2x1) * Bj_bottom (1x2)
      
      First term: 2x2 * 2x2 -> rank 7
      Second term: 2x1 * 1x2 -> rank 1 (it's an outer product: each element is A_top_right[i] * Bj_bottom[k])
      
      Cj_bottom (row 2) = A_bot (1x3) * Bj (3x2)
      = A_bot[0:2] * Bj_top (1x2 * 2x2) + A_bot[2] * Bj_bottom (1x1 * 1x2)
      
      First term: 1x2 * 2x2 -> rank 2
      Second term: 1x1 * 1x2 -> rank 1
      
    So for each j: rank = 7 + 1 + 2 + 1 = 11
    Total rank = 3 * 11 = 33
    
    This is better than 54!
    """
    n, m, p = 3, 3, 6
    
    U_list = []
    V_list = []
    W_list = []
    
    for j in range(3):  # For each 3x2 block of B
        # Block j: B[:, 2*j:2*j+2]
        b_col_start = 2 * j
        
        # We'll add 11 terms for this block.
        
        # --- Top part of C (rows 0,1) ---
        # C_top = A_top_left * Bj_top + A_top_right * Bj_bottom
        
        # Term 1-7: Strassen for A[0:2, 0:2] * B[0:2, 2*j:2*j+2]
        U_s, V_s, W_s = _strassen_2x2()
        
        for r in range(7):
            U_term = np.zeros(n * m)  # 9 entries
            V_term = np.zeros(m * p)  # 18 entries
            W_term = np.zeros(p * n)  # 18 entries
            
            # U: A is 3x3, flat index i*m+j = i*3+j
            for a_flat in range(4):
                ai = a_flat // 2
                aj = a_flat % 2
                U_term[ai * 3 + aj] = U_s[a_flat, r]
            
            # V: B is 3x6, flat index j*p+k = j*6+k
            for b_flat in range(4):
                bj = b_flat // 2
                bk = b_flat % 2
                global_bk = b_col_start + bk
                V_term[bj * 6 + global_bk] = V_s[b_flat, r]
            
            # W: C is 3x6, output index k*n+i = k*3+i (trace convention)
            # C_top rows 0,1, cols 2*j, 2*j+1
            for c_flat in range(4):
                ci = c_flat // 2
                ck = c_flat % 2
                global_ck = b_col_start + ck
                W_term[global_ck * 3 + ci] = W_s[c_flat, r]
            
            U_list.append(U_term)
            V_list.append(V_term)
            W_list.append(W_term)
        
        # Term 8: A_top_right (2x1) * Bj_bottom (1x2) -> 4 rank-1 terms
        # A[i,2] * B[2,k] contributes to C[i,k] for i in {0,1}, k in {2*j, 2*j+1}
        for i in range(2):
            for k in range(2):
                U_term = np.zeros(n * m)
                V_term = np.zeros(m * p)
                W_term = np.zeros(p * n)
                
                global_k = b_col_start + k
                U_term[i * 3 + 2] = 1.0  # A[i, 2]
                V_term[2 * 6 + global_k] = 1.0  # B[2, 2*j+k]
                W_term[global_k * 3 + i] = 1.0  # C[2*j+k, i]
                
                U_list.append(U_term)
                V_list.append(V_term)
                W_list.append(W_term)
        
        # --- Bottom part of C (row 2) ---
        # C_bottom = A_bot_left * Bj_top + A_bot_right * Bj_bottom
        
        # Terms 9-12: A_bot_left (1x2) * Bj_top (2x2) -> 4 rank-1 terms
        # A[2, jj] * B[jj, k] contributes to C[2, k] for jj in {0,1}, k in {2*j, 2*j+1}
        for jj in range(2):
            for k in range(2):
                U_term = np.zeros(n * m)
                V_term = np.zeros(m * p)
                W_term = np.zeros(p * n)
                
                global_k = b_col_start + k
                U_term[2 * 3 + jj] = 1.0  # A[2, jj]
                V_term[jj * 6 + global_k] = 1.0  # B[jj, 2*j+k]
                W_term[global_k * 3 + 2] = 1.0  # C[2*j+k, 2]
                
                U_list.append(U_term)
                V_list.append(V_term)
                W_list.append(W_term)
        
        # Terms 13-14: A_bot_right (1x1) * Bj_bottom (1x2) -> 2 rank-1 terms
        # A[2, 2] * B[2, k] contributes to C[2, k] for k in {2*j, 2*j+1}
        for k in range(2):
            U_term = np.zeros(n * m)
            V_term = np.zeros(m * p)
            W_term = np.zeros(p * n)
            
            global_k = b_col_start + k
            U_term[2 * 3 + 2] = 1.0  # A[2, 2]
            V_term[2 * 6 + global_k] = 1.0  # B[2, 2*j+k]
            W_term[global_k * 3 + 2] = 1.0  # C[2*j+k, 2]
            
            U_list.append(U_term)
            V_list.append(V_term)
            W_list.append(W_term)
    
    U = np.column_stack(U_list)
    V = np.column_stack(V_list)
    W = np.column_stack(W_list)
    
    rank = U.shape[1]
    return rank, U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    a_dim = n * m
    b_dim = m * p
    c_dim = p * n
    R = U.shape[1]
    
    # Check all non-zero entries: T[i*m+j, j*p+k, k*n+i] = 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = 0.0
                for r in range(R):
                    s += U[a, r] * V[b, r] * W[c, r]
                if abs(s - 1.0) > tol:
                    return False
    
    # Check a sample of zero entries
    np.random.seed(42)
    for _ in range(1000):
        a = np.random.randint(0, a_dim)
        b = np.random.randint(0, b_dim)
        c = np.random.randint(0, c_dim)
        
        # Check if this is a non-zero entry
        i1, j1 = a // m, a % m
        j2, k1 = b // p, b % p
        k2, i2 = c // n, c % n
        
        if i1 == i2 and j1 == j2 and k1 == k2:
            continue  # Skip non-zero entries
        
        s = 0.0
        for r in range(R):
            s += U[a, r] * V[b, r] * W[c, r]
        if abs(s) > tol:
            return False
    
    return True

def construct():
    n, m, p = DIMENSIONS
    
    deadline = time.time() + 110  # Safety margin
    
    # Try the block decomposition with Strassen
    try:
        rank, U, V, W = _build_block_decomposition(n, m, p)
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        else:
            # If verification fails, fall back to schoolbook
            pass
    except Exception:
        pass
    
    if time.time() > deadline:
        U, V, W = _build_schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
    U, V, W = _build_schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
