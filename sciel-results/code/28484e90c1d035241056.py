"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline, 10s safety margin
    
    n, m, p = DIMENSIONS  # 2, 4, 8
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def verify(U, V, W):
        """Verify that the decomposition is correct."""
        rank = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for seed in [42, 123, 456, 789, 101]:
            np.random.seed(seed)
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            C_true = A @ B
            
            # Compute using decomposition
            C_comp = np.zeros((n, p))
            for r in range(rank):
                # X_r[i] = sum_j U[i*m+j, r] * A[i,j]
                X_r = np.array([sum(U_arr[i*m+j, r] * A[i, j] for j in range(m)) for i in range(n)])
                # Y_r[k] = sum_j V[j*p+k, r] * B[j,k]
                Y_r = np.array([sum(V_arr[j*p+k, r] * B[j, k] for j in range(m)) for k in range(p)])
                # C[i,k] += W[k*n+i, r] * X_r[i] * Y_r[k]
                for i in range(n):
                    for k in range(p):
                        C_comp[i, k] += W_arr[k*n+i, r] * X_r[i] * Y_r[k]
            
            if not np.allclose(C_comp, C_true, atol=1e-10):
                return False
        
        return True
    
    def strassen_2x2x2():
        """Return Strassen decomposition for 2x2x2 matrix multiplication.
        C = A * B where A, B are 2x2, C is 2x2.
        Returns U, V, W as lists of shape (4, 7) each.
        U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
        """
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # A = [[A00, A01], [A10, A11]], indices: A00->0, A01->1, A10->2, A11->3
        # B = [[B00, B01], [B10, B11]], indices: B00->0, B01->1, B10->2, B11->3
        # C = [[C00, C01], [C10, C11]], indices: C00->0, C01->1, C10->2, C11->3
        
        # M1 = (A00+A11)(B00+B11)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # M2 = (A10+A11)B00
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1; W[3][1] = -1
        
        # M3 = A00(B01-B11)
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[1][2] = 1; W[3][2] = 1
        
        # M4 = A11(B10-B00)
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = -1; W[2][3] = 1
        
        # M5 = (A00+A01)B11
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = 1; W[1][4] = 1
        
        # M6 = (A10-A00)(B00+B01)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        # M7 = (A01-A11)(B10+B11)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = -1
        
        return U, V, W
    
    # Build 2x4x8 decomposition using 8 Strassen 2x2x2 blocks
    # Structure:
    # A = [A1 | A2] where A1 = A[:,0:2], A2 = A[:,2:4] (both 2x2)
    # B = [B_top; B_bot] where B_top = B[0:2,:], B_bot = B[2:4,:] (both 2x8)
    # C = A1*B_top + A2*B_bot
    # 
    # Split columns of B into 4 pairs: cols (0,1), (2,3), (4,5), (6,7)
    # For each pair cp and each half h (0=top, 1=bot):
    #   sub-problem: A_{h+1} (2x2) * B_{h,pair} (2x2) -> C_{:, pair} (2x2)
    #   This is a 2x2x2 multiplication with Strassen rank 7
    # Total: 8 sub-problems * 7 = 56
    
    U_s, V_s, W_s = strassen_2x2x2()
    
    # Full decomposition
    rank = 56
    U_full = [[0]*rank for _ in range(8)]   # n*m = 2*4 = 8 rows
    V_full = [[0]*rank for _ in range(32)]  # m*p = 4*8 = 32 rows
    W_full = [[0]*rank for _ in range(16)]  # p*n = 8*2 = 16 rows
    
    # Mapping for each Strassen block (h, cp) where h in {0,1}, cp in {0,1,2,3}:
    # A_half = A[:, 2*h : 2*h+2]  (2x2)
    #   A element (i, j_local) -> global A index (i, 2*h + j_local) -> U row i*4 + (2*h + j_local)
    # B_half = B[2*h : 2*h+2, 2*cp : 2*cp+2]  (2x2)
    #   B element (j_local, k_local) -> global B index (2*h + j_local, 2*cp + k_local) -> V row (2*h+j_local)*8 + (2*cp+k_local)
    # C output: C[:, 2*cp : 2*cp+2] (2x2)
    #   C element (i, k_local) -> global C index (i, 2*cp + k_local) -> W row (2*cp+k_local)*2 + i
    
    for h in range(2):       # 0 = top (A1, B_top), 1 = bottom (A2, B_bot)
        for cp in range(4):  # column pair
            base_col = (h * 4 + cp) * 7  # starting column in the rank dimension
            
            for r in range(7):
                col = base_col + r
                # U: A_half elements
                for i in range(2):
                    for j_local in range(2):
                        val = U_s[i*2 + j_local][r]
                        if val != 0:
                            u_row = i * 4 + (2*h + j_local)
                            U_full[u_row][col] += val
                
                # V: B_half elements
                for j_local in range(2):
                    for k_local in range(2):
                        val = V_s[j_local*2 + k_local][r]
                        if val != 0:
                            v_row = (2*h + j_local) * 8 + (2*cp + k_local)
                            V_full[v_row][col] += val
                
                # W: C output elements
                for k_local in range(2):
                    for i in range(2):
                        val = W_s[k_local*2 + i][r]
                        if val != 0:
                            w_row = (2*cp + k_local) * 2 + i
                            W_full[w_row][col] += val
    
    result = {"U": U_full, "V": V_full, "W": W_full}
    
    # Verify
    if verify(U_full, V_full, W_full):
        return result
    
    # Fallback to schoolbook
    fallback = schoolbook()
    return fallback
# EVOLVE-BLOCK-END
