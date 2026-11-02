import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    n, m, p = 2, 4, 6
    
    # Rank 42 decomposition: 3 column blocks × 2 row sub-blocks × 7 Strassen
    rank = 42
    U = [[0] * rank for _ in range(n * m)]  # (8, 42)
    V = [[0] * rank for _ in range(m * p)]  # (24, 42)
    W = [[0] * rank for _ in range(p * n)]  # (12, 42)
    
    def add_strassen_block(A_idx, B_idx, C_idx, start_col):
        """
        Add a 2x2x2 Strassen block to the decomposition.
        A_idx: [a, b, c, d] - indices in U for A = [[a,b],[c,d]]
        B_idx: [e, f, g, h] - indices in V for B = [[e,f],[g,h]]
        C_idx: [c11, c12, c21, c22] - indices in W for C = [[c11,c12],[c21,c22]]
        """
        a, b, c, d = A_idx
        e, f, g, h = B_idx
        c11, c12, c21, c22 = C_idx
        
        # M1 = (a+d)(e+h)
        U[a][start_col] += 1
        U[d][start_col] += 1
        V[e][start_col] += 1
        V[h][start_col] += 1
        W[c11][start_col] += 1
        W[c22][start_col] += 1
        start_col += 1
        
        # M2 = (c+d)e
        U[c][start_col] += 1
        U[d][start_col] += 1
        V[e][start_col] += 1
        W[c21][start_col] += 1
        W[c22][start_col] -= 1
        start_col += 1
        
        # M3 = a(f-h)
        U[a][start_col] += 1
        V[f][start_col] += 1
        V[h][start_col] -= 1
        W[c12][start_col] += 1
        W[c22][start_col] += 1
        start_col += 1
        
        # M4 = d(g-e)
        U[d][start_col] += 1
        V[g][start_col] += 1
        V[e][start_col] -= 1
        W[c11][start_col] += 1
        W[c21][start_col] += 1
        start_col += 1
        
        # M5 = (a+b)h
        U[a][start_col] += 1
        U[b][start_col] += 1
        V[h][start_col] += 1
        W[c11][start_col] -= 1
        W[c12][start_col] += 1
        start_col += 1
        
        # M6 = (c-a)(e+f)
        U[c][start_col] += 1
        U[a][start_col] -= 1
        V[e][start_col] += 1
        V[f][start_col] += 1
        W[c22][start_col] += 1
        start_col += 1
        
        # M7 = (b-d)(g+h)
        U[b][start_col] += 1
        U[d][start_col] -= 1
        V[g][start_col] += 1
        V[h][start_col] += 1
        W[c11][start_col] += 1
        start_col += 1
        
        return start_col
    
    # Loop over 3 column pairs of B (and C)
    col = 0
    for k in range(3):
        # A_L: cols 0,1 of A (2x2 block)
        # A_R: cols 2,3 of A (2x2 block)
        # B_Tk: rows 0,1; cols 2k, 2k+1 of B (2x2 block)
        # B_Bk: rows 2,3; cols 2k, 2k+1 of B (2x2 block)
        
        # A_L indices in U:
        # a = A[0,0] -> 0*4+0 = 0
        # b = A[0,1] -> 0*4+1 = 1
        # c = A[1,0] -> 1*4+0 = 4
        # d = A[1,1] -> 1*4+1 = 5
        A_L_idx = [0, 1, 4, 5]
        
        # A_R indices in U:
        # a = A[0,2] -> 0*4+2 = 2
        # b = A[0,3] -> 0*4+3 = 3
        # c = A[1,2] -> 1*4+2 = 6
        # d = A[1,3] -> 1*4+3 = 7
        A_R_idx = [2, 3, 6, 7]
        
        # B_Tk indices in V:
        # e = B[0, 2k] -> 0*6+2k
        # f = B[0, 2k+1] -> 0*6+2k+1
        # g = B[1, 2k] -> 1*6+2k
        # h = B[1, 2k+1] -> 1*6+2k+1
        B_Tk_idx = [0*6 + 2*k, 0*6 + 2*k + 1, 1*6 + 2*k, 1*6 + 2*k + 1]
        
        # B_Bk indices in V:
        # e = B[2, 2k] -> 2*6+2k
        # f = B[2, 2k+1] -> 2*6+2k+1
        # g = B[3, 2k] -> 3*6+2k
        # h = B[3, 2k+1] -> 3*6+2k+1
        B_Bk_idx = [2*6 + 2*k, 2*6 + 2*k + 1, 3*6 + 2*k, 3*6 + 2*k + 1]
        
        # C indices in W (trace convention: W[k*n+i, r] where k is column, i is row)
        # c11 = C[0, 2k] -> W[2k*2+0] = W[4k]
        # c12 = C[0, 2k+1] -> W[(2k+1)*2+0] = W[4k+2]
        # c21 = C[1, 2k] -> W[2k*2+1] = W[4k+1]
        # c22 = C[1, 2k+1] -> W[(2k+1)*2+1] = W[4k+3]
        C_idx = [2*k*2 + 0, (2*k+1)*2 + 0, 2*k*2 + 1, (2*k+1)*2 + 1]
        
        col = add_strassen_block(A_L_idx, B_Tk_idx, C_idx, col)
        col = add_strassen_block(A_R_idx, B_Bk_idx, C_idx, col)
    
    # Verify the decomposition with a random test
    if time.time() - start_time < deadline - 5.0:
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Generate random test matrices
        np.random.seed(42)
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_expected = A_test @ B_test
        
        # Compute C from decomposition
        # C[i,j] = sum_r U[i*m+j, r] * V[j*p+i... wait, need to be careful
        # U has shape (n*m, rank), indexed by A[i,a] -> row i*m+a
        # V has shape (m*p, rank), indexed by B[a,b] -> row a*p+b
        # W has shape (p*n, rank), indexed by C[i,b] -> row b*n+i (trace convention)
        
        C_computed = np.zeros((n, p))
        for i in range(n):
            for b in range(p):
                val = 0.0
                for r in range(rank):
                    # Sum over a: U[i*m+a, r] * V[a*p+b, r] * W[b*n+i, r]
                    for a in range(m):
                        val += U_np[i*m+a, r] * V_np[a*p+b, r] * W_np[b*n+i, r]
                C_computed[i, b] = val
        
        max_err = np.max(np.abs(C_computed - C_expected))
        if max_err > 1e-10:
            # If verification fails, fall back to naive rank-48
            rank = 48
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            
            col = 0
            for i in range(n):
                for a in range(m):
                    for b in range(p):
                        U[i*m+a][col] = 1
                        V[a*p+b][col] = 1
                        W[b*n+i][col] = 1
                        col += 1
    
    return {"U": U, "V": V, "W": W}
