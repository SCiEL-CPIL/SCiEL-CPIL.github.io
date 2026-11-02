import numpy as np
import time

DIMENSIONS = [3, 3, 6]

def _strassen_2x2():
    """Return rank-7 Strassen decomposition for 2x2 * 2x2.
    
    U: (4, 7) where U[i*2+j, r] for A[i,j]
    V: (4, 7) where V[i*2+j, r] for B[i,j]  
    W: (4, 7) where W[i*2+j, r] for C[i,j]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    
    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    
    # M3 = A00(B10-B11)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    
    # M4 = A11(B01-B00)
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    
    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    
    # Output: C[i,k] = sum_r W[k*2+i, r] * M_r
    # C00 = M1 + M4 - M5 - M7
    W[0, 0] = 1; W[0, 3] = 1; W[0, 4] = -1; W[0, 6] = -1
    # C01 = M3 + M5
    W[1, 2] = 1; W[1, 4] = 1
    # C10 = M2 + M4
    W[2, 1] = 1; W[2, 3] = 1
    # C11 = M1 - M2 + M3 + M7
    W[3, 0] = 1; W[3, 1] = -1; W[3, 2] = 1; W[3, 6] = 1
    
    return U, V, W

def _build_block_decomposition(n, m, p):
    """Build a decomposition using block structure with Strassen for 2x2 blocks.
    
    For 3x3 * 3x6:
    Split B into three 3x2 blocks (columns 0-1, 2-3, 4-5).
    For each 3x2 block of B:
      - Top 2 rows of C: A_top(2x3) * Bj_top(2x2) [Strassen rank 7] 
                         + A_top_right(2x1) * Bj_bottom(1x2) [4 rank-1]
      - Bottom 1 row of C: A_bot(1x3) * Bj_top(2x2) [4 rank-1]
                         + A_bot_right(1x1) * Bj_bottom(1x2) [2 rank-1]
    Total per block: 7 + 4 + 4 + 2 = 17
    Total rank = 3 * 17 = 51
    """
    n, m, p = 3, 3, 6
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    for block in range(3):  # For each 3x2 block of B
        b_col_start = 2 * block
        
        # --- Strassen for A[0:2, 0:2] * B[0:2, b_col_start:b_col_start+2] ---
        U_s, V_s, W_s = _strassen_2x2()
        
        for r in range(7):
            U_term = np.zeros(n * m)  # 9 entries
            V_term = np.zeros(m * p)  # 18 entries
            W_term = np.zeros(p * n)  # 18 entries
            
            # Map Strassen U (4 entries for 2x2) to full U (9 entries for 3x3)
            # Strassen index: a_flat = ai*2 + aj, where ai, aj in {0,1}
            for a_flat in range(4):
                ai = a_flat // 2
                aj = a_flat % 2
                U_term[ai * m + aj] = U_s[a_flat, r]
            
            # Map Strassen V (4 entries for 2x2) to full V (18 entries for 3x6)
            # Strassen index: b_flat = bj*2 + bk, where bj, bk in {0,1}
            for b_flat in range(4):
                bj = b_flat // 2
                bk = b_flat % 2
                global_bk = b_col_start + bk
                V_term[bj * p + global_bk] = V_s[b_flat, r]
            
            # Map Strassen W (4 entries for 2x2) to full W (18 entries for 6x3)
            # Strassen index: c_flat = ck*2 + ci, where ck, ci in {0,1}
            for c_flat in range(4):
                ci = c_flat // 2
                ck = c_flat % 2
                global_ck = b_col_start + ck
                W_term[global_ck * n + ci] = W_s[c_flat, r]
            
            U_cols.append(U_term)
            V_cols.append(V_term)
            W_cols.append(W_term)
        
        # --- A_top_right (2x1) * Bj_bottom (1x2): 4 rank-1 terms ---
        # A[i, 2] * B[2, b_col_start+k] for i in {0,1}, k in {0,1}
        for i in range(2):
            for k in range(2):
                U_term = np.zeros(n * m)
                V_term = np.zeros(m * p)
                W_term = np.zeros(p * n)
                
                global_k = b_col_start + k
                U_term[i * m + 2] = 1.0
                V_term[2 * p + global_k] = 1.0
                W_term[global_k * n + i] = 1.0
                
                U_cols.append(U_term)
                V_cols.append(V_term)
                W_cols.append(W_term)
        
        # --- A_bot_left (1x2) * Bj_top (2x2): 4 rank-1 terms ---
        # A[2, jj] * B[jj, b_col_start+k] for jj in {0,1}, k in {0,1}
        for jj in range(2):
            for k in range(2):
                U_term = np.zeros(n * m)
                V_term = np.zeros(m * p)
                W_term = np.zeros(p * n)
                
                global_k = b_col_start + k
                U_term[2 * m + jj] = 1.0
                V_term[jj * p + global_k] = 1.0
                W_term[global_k * n + 2] = 1.0
                
                U_cols.append(U_term)
                V_cols.append(V_term)
                W_cols.append(W_term)
        
        # --- A_bot_right (1x1) * Bj_bottom (1x2): 2 rank-1 terms ---
        # A[2, 2] * B[2, b_col_start+k] for k in {0,1}
        for k in range(2):
            U_term = np.zeros(n * m)
            V_term = np.zeros(m * p)
            W_term = np.zeros(p * n)
            
            global_k = b_col_start + k
            U_term[2 * m + 2] = 1.0
            V_term[2 * p + global_k] = 1.0
            W_term[global_k * n + 2] = 1.0
            
            U_cols.append(U_term)
            V_cols.append(V_term)
            W_cols.append(W_term)
    
    U = np.column_stack(U_cols)
    V = np.column_stack(V_cols)
    W = np.column_stack(W_cols)
    
    rank = U.shape[1]
    return rank, U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
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
    for _ in range(3000):
        a = np.random.randint(0, a_dim)
        b = np.random.randint(0, b_dim)
        c = np.random.randint(0, c_dim)
        
        i1, j1 = a // m, a % m
        j2, k1 = b // p, b % p
        k2, i2 = c // n, c % n
        
        if i1 == i2 and j1 == j2 and k1 == k2:
            continue
        
        s = 0.0
        for r in range(R):
            s += U[a, r] * V[b, r] * W[c, r]
        if abs(s) > tol:
            return False
    
    return True

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

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120 - 15  # 105s safety margin
    
    # Try the block decomposition with Strassen (rank 51)
    rank, U, V, W = _build_block_decomposition(n, m, p)
    
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook rank 54
    U, V, W = _build_schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
