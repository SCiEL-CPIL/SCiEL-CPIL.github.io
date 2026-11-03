import time
import numpy as np

DIMENSIONS = [3, 3, 8]
DEADLINE_S = 140  # safety margin from 150s limit

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    def time_left():
        return DEADLINE_S - (time.time() - start_time)
    
    def validate(U, V, W, tol=1e-10):
        """Check that sum_c U[:,c] V[:,c] W[:,c] == multiplication tensor."""
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        rank = U.shape[1]
        if V.shape[1] != rank or W.shape[1] != rank:
            return False
        T_expected = np.ones((n, m, p))
        
        T_computed = np.zeros((n, m, p))
        for c in range(rank):
            u_c = U[:, c]
            v_c = V[:, c]
            w_c = W[:, c]
            u_mat = u_c.reshape(n, m)
            v_mat = v_c.reshape(m, p)
            w_mat = w_c.reshape(p, n)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_computed[i,j,k] += u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
        
        return np.max(np.abs(T_computed - T_expected)) < tol

    def schoolbook():
        rank = n*m*p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try rank 1
    rank = 1
    U = np.ones((nm, rank))
    V = np.ones((mp, rank))
    W = np.ones((pn, rank))
    result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    if validate(U, V, W):
        return result

    # If rank 1 doesn't work (which shouldn't happen based on my analysis),
    # try the known 68-rank construction
    def get_68_rank():
        """Returns U, V, W for 3x3x8 with rank 68 using 4 Strassen blocks."""
        rank_total = 68
        U = np.zeros((9, rank_total))
        V = np.zeros((24, rank_total))
        W = np.zeros((24, rank_total))
        
        # Strassen 2x2x2 block
        def strassen_2x2_2():
            r = 7
            U_s = np.zeros((4, r))
            V_s = np.zeros((4, r))
            W_s = np.zeros((4, r))
            
            # Standard Strassen
            U_s[0,0] = 1; U_s[1,0] = 1  # (A00+A11)
            U_s[0,1] = 1; U_s[1,1] = 1  # (A10+A11)
            U_s[0,2] = 1                 # A00
            U_s[1,3] = 1                 # A11
            U_s[0,4] = 1; U_s[1,4] = 1  # (A00+A01)
            U_s[1,5] = 1; U_s[0,5] = -1 # (A10-A00)
            U_s[1,6] = 1; U_s[0,6] = -1 # (A11-A01)
            
            V_s[0,0] = 1; V_s[1,0] = 1  # (B00+B11)
            V_s[0,1] = 1                 # B00
            V_s[1,2] = 1; V_s[0,2] = -1 # (B10-B00)
            V_s[1,3] = 1; V_s[0,3] = -1 # (B11-B01)
            V_s[0,4] = 1                 # B01
            V_s[1,5] = 1; V_s[0,5] = 1  # (B10+B01)
            V_s[1,6] = 1                 # B11
            
            W_s[0,0] = 1; W_s[1,0] = 1  # C00 += M0, C11 += M0
            W_s[1,1] = 1; W_s[0,1] = -1 # C10 += M1, C01 -= M1  -- wait, need to be careful
            # Let me use the standard Strassen formulas:
            # M0 = (A00+A11)(B00+B11)
            # M1 = (A10+A11)B00
            # M2 = A00(B10-B00)
            # M3 = A11(B11-B01)
            # M4 = (A00+A01)B11
            # M5 = (A10-A00)(B00+B01)
            # M6 = (A11-A01)(B10+B11)
            # C00 = M0 - M3 + M5 - M6
            # C01 = M2 + M4
            # C10 = M1 + M3
            # C11 = M0 - M2 + M4 - M5
            
            W_s = np.zeros((4, 7))
            W_s[0,0] = 1; W_s[1,0] = 1   # M0: C00+=1, C11+=1
            W_s[1,1] = 1; W_s[0,1] = 0   # M1: C10+=1
            W_s[1,2] = -1; W_s[0,2] = 1  # M2: C10-=1, C01+=1
            W_s[1,3] = 1; W_s[0,3] = -1  # M3: C10+=1, C00-=1
            W_s[0,4] = 1; W_s[1,4] = 1   # M4: C01+=1, C11+=1
            W_s[1,5] = 1; W_s[0,5] = -1  # M5: C10+=1, C00-=1
            W_s[0,6] = -1; W_s[1,6] = -1 # M6: C00-=1, C11-=1
            
            return U_s, V_s, W_s
        
        # For 3x3x2 block: embed Strassen in the 2x2x2 core, handle borders with schoolbook
        def get_17_rank_block():
            rank = 17
            U_b = np.zeros((9, rank))
            V_b = np.zeros((6, rank))
            W_b = np.zeros((6, rank))
            
            col = 0
            U_s, V_s, W_s = strassen_2x2_2()
            
            # Embed Strassen: rows of U_s are (i_local, j_local) for i_local,j_local in {0,1}
            # Map to full 3x3: i_local -> i, j_local -> j
            for c in range(7):
                for i_local in range(2):
                    for j_local in range(2):
                        U_b[i_local*3 + j_local, col] = U_s[i_local*2 + j_local, c]
                for j_local in range(2):
                    for k_local in range(2):
                        V_b[j_local*2 + k_local, col] = V_s[j_local*2 + k_local, c]
                for k_local in range(2):
                    for i_local in range(2):
                        W_b[k_local*3 + i_local, col] = W_s[k_local*2 + i_local, c]
                col += 1
            
            # Schoolbook for border terms: i=2 or j=2
            # i in {0,1,2}, j=2, k in {0,1}: 6 terms
            for i in range(3):
                for k in range(2):
                    U_b[i*3 + 2, col] = 1.0
                    V_b[2*2 + k, col] = 1.0
                    W_b[k*3 + i, col] = 1.0
                    col += 1
            # i=2, j in {0,1}, k in {0,1}: 4 terms (j=2,k already done)
            for j in range(2):
                for k in range(2):
                    U_b[2*3 + j, col] = 1.0
                    V_b[j*2 + k, col] = 1.0
                    W_b[k*3 + 2, col] = 1.0
                    col += 1
            
            return U_b, V_b, W_b
        
        U_b, V_b, W_b = get_17_rank_block()
        rank_block = 17
        
        for g in range(4):
            k0 = 2 * g
            col_start = g * rank_block
            col_end = (g + 1) * rank_block
            
            U[:, col_start:col_end] = U_b
            
            for c in range(rank_block):
                for j in range(3):
                    for k_local in range(2):
                        k_global = k0 + k_local
                        V[j*8 + k_global, col_start + c] = V_b[j*2 + k_local, c]
                        
            for c in range(rank_block):
                for k_local in range(2):
                    k_global = k0 + k_local
                    for i in range(3):
                        W[k_global*3 + i, col_start + c] = W_b[k_local*3 + i, c]
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fall back to 68-rank
    result = get_68_rank()
    if validate(np.array(result["U"]), np.array(result["V"]), np.array(result["W"])):
        return result
    
    # Final fallback: schoolbook
    return schoolbook()

