"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 7
    start_time = time.time()
    deadline = start_time + 120  # Safety margin
    
    U_rows = []
    V_rows = []
    W_rows = []
    
    def add_term(u_vec, v_vec, w_vec):
        """Add a rank-1 term."""
        U_rows.append(list(u_vec))
        V_rows.append(list(v_vec))
        W_rows.append(list(w_vec))
    
    def zeros(length):
        return [0] * length
    
    def time_check():
        return time.time() > deadline
    
    # Strassen 2x2x2 decomposition
    # For C = A * B where A, B, C are 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A10)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # For the tensor T[i*m+j, j*p+k, k*n+i], we need:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if (i,j,k) valid, else 0
    
    # Strategy: Use Strassen on 2x2x2 sub-blocks to save rank.
    # Split the problem into sub-problems based on i, j, k ranges.
    
    # For 2x2x2 sub-block (i in {i0,i0+1}, j in {j0,j0+1}, k in {k0,k0+1}):
    # We can use 7 terms instead of 8.
    
    # Let's apply Strassen to all possible 2x2x2 sub-blocks.
    
    # First, handle the 3x4x7 problem by splitting:
    # i: 0,1 (2 rows) and 2 (1 row)
    # j: 0,1 (2 cols) and 2,3 (2 cols)  
    # k: 0,1 (2 cols), 2,3 (2 cols), 4,5 (2 cols), 6 (1 col)
    
    # For each combination, use Strassen if 2x2x2, else use direct terms.
    
    # Helper: add direct terms for a sub-block
    def add_direct_terms(i_range, j_range, k_range):
        """Add one term per (i,j,k) in the ranges."""
        for i in i_range:
            for j in j_range:
                for k in k_range:
                    u = zeros(n * m)
                    v = zeros(m * p)
                    w = zeros(p * n)
                    u[i * m + j] = 1
                    v[j * p + k] = 1
                    w[k * n + i] = 1
                    add_term(u, v, w)
    
    # Helper: add Strassen terms for 2x2x2 sub-block
    # C = A*B where A is 2x2 (rows i0,i1; cols j0,j1), B is 2x2 (rows j0,j1; cols k0,k1)
    # C[i0,k0] = M1 + M4 - M5 + M7
    # C[i0,k1] = M3 + M5
    # C[i1,k0] = M2 + M4
    # C[i1,k1] = M1 - M2 + M3 + M6
    def add_strassen_terms(i0, j0, k0):
        """Add 7 Strassen terms for 2x2x2 sub-block starting at (i0,j0,k0)."""
        i1 = i0 + 1
        j1 = j0 + 1
        k1 = k0 + 1
        
        # M1 = (A[i0,j0] + A[i1,j1]) * (B[j0,k0] + B[j1,k1])
        # Contributes: C[i0,k0]+1, C[i1,k1]+1
        u1 = zeros(n * m)
        v1 = zeros(m * p)
        w1 = zeros(p * n)
        u1[i0 * m + j0] = 1
        u1[i1 * m + j1] = 1
        v1[j0 * p + k0] = 1
        v1[j1 * p + k1] = 1
        w1[k0 * n + i0] = 1
        w1[k1 * n + i1] = 1
        add_term(u1, v1, w1)
        
        # M2 = (A[i1,j0] + A[i1,j1]) * B[j0,k0]
        # Contributes: C[i1,k0]+1, C[i1,k1]-1
        u2 = zeros(n * m)
        v2 = zeros(m * p)
        w2 = zeros(p * n)
        u2[i1 * m + j0] = 1
        u2[i1 * m + j1] = 1
        v2[j0 * p + k0] = 1
        w2[k0 * n + i1] = 1
        w2[k1 * n + i1] = -1
        add_term(u2, v2, w2)
        
        # M3 = A[i0,j0] * (B[j1,k0] - B[j1,k1])
        # Contributes: C[i0,k1]+1, C[i1,k1]+1
        u3 = zeros(n * m)
        v3 = zeros(m * p)
        w3 = zeros(p * n)
        u3[i0 * m + j0] = 1
        v3[j1 * p + k0] = 1
        v3[j1 * p + k1] = -1
        w3[k1 * n + i0] = 1
        w3[k1 * n + i1] = 1
        add_term(u3, v3, w3)
        
        # M4 = A[i1,j1] * (B[j0,k1] - B[j0,k0])
        # Contributes: C[i0,k0]+1, C[i1,k0]+1
        u4 = zeros(n * m)
        v4 = zeros(m * p)
        w4 = zeros(p * n)
        u4[i1 * m + j1] = 1
        v4[j0 * p + k1] = 1
        v4[j0 * p + k0] = -1
        w4[k0 * n + i0] = 1
        w4[k0 * n + i1] = 1
        add_term(u4, v4, w4)
        
        # M5 = (A[i0,j0] + A[i0,j1]) * B[j1,k1]
        # Contributes: C[i0,k0]-1, C[i0,k1]+1
        u5 = zeros(n * m)
        v5 = zeros(m * p)
        w5 = zeros(p * n)
        u5[i0 * m + j0] = 1
        u5[i0 * m + j1] = 1
        v5[j1 * p + k1] = 1
        w5[k0 * n + i0] = -1
        w5[k1 * n + i0] = 1
        add_term(u5, v5, w5)
        
        # M6 = (A[i1,j0] - A[i0,j0]) * (B[j0,k0] + B[j1,k0])
        # Contributes: C[i1,k1]+1
        u6 = zeros(n * m)
        v6 = zeros(m * p)
        w6 = zeros(p * n)
        u6[i1 * m + j0] = 1
        u6[i0 * m + j0] = -1
        v6[j0 * p + k0] = 1
        v6[j1 * p + k0] = 1
        w6[k1 * n + i1] = 1
        add_term(u6, v6, w6)
        
        # M7 = (A[i0,j1] - A[i1,j1]) * (B[j0,k1] + B[j1,k1])
        # Contributes: C[i0,k0]+1, C[i0,k1]+1
        u7 = zeros(n * m)
        v7 = zeros(m * p)
        w7 = zeros(p * n)
        u7[i0 * m + j1] = 1
        u7[i1 * m + j1] = -1
        v7[j0 * p + k1] = 1
        v7[j1 * p + k1] = 1
        w7[k0 * n + i0] = 1
        w7[k1 * n + i0] = 1
        add_term(u7, v7, w7)
    
    # Now build the decomposition:
    # Split i into {0,1} and {2}
    # Split j into {0,1} and {2,3}
    # Split k into {0,1}, {2,3}, {4,5}, {6}
    
    # For i in {0,1}, j in {0,1}:
    for k0 in [0, 2, 4]:
        if k0 + 1 < p:
            add_strassen_terms(0, 0, k0)
        else:
            add_direct_terms([0, 1], [0, 1], [k0])
    
    # For i in {0,1}, j in {2,3}:
    for k0 in [0, 2, 4]:
        if k0 + 1 < p:
            add_strassen_terms(0, 2, k0)
        else:
            add_direct_terms([0, 1], [2, 3], [k0])
    
    # For i = 2: use batched terms - one per k value
    # For each k, U has all j entries set to 1 for i=2, V has all j entries for that k, W has (k,n+i)
    for k in range(p):
        u = zeros(n * m)
        v = zeros(m * p)
        w = zeros(p * n)
        for j in range(m):
            u[2 * m + j] = 1
            v[j * p + k] = 1
        w[k * n + 2] = 1
        add_term(u, v, w)
    
    rank = len(U_rows)
    
    # Convert to proper matrices: U_rows is (rank, n*m), we want (n*m, rank)
    U = [list(col) for col in zip(*U_rows)]
    V = [list(col) for col in zip(*V_rows)]
    W = [list(col) for col in zip(*W_rows)]
    
    # Verify correctness
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Reconstruct the tensor
    T_reconstructed = np.zeros((n * m, m * p, p * n))
    for r in range(rank):
        T_reconstructed += np.einsum('a,b,c->abc', U_np[:, r], V_np[:, r], W_np[:, r])
    
    # Check against the true multiplication tensor
    T_true = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_true[i * m + j, j * p + k, k * n + i] = 1
    
    if not np.allclose(T_reconstructed, T_true, atol=1e-9):
        # Fallback to rank-21 approach
        U_rows_fb = []
        V_rows_fb = []
        W_rows_fb = []
        
        for i in range(n):
            for k in range(p):
                u = zeros(n * m)
                v = zeros(m * p)
                w = zeros(p * n)
                for j in range(m):
                    u[i * m + j] = 1
                    v[j * p + k] = 1
                w[k * n + i] = 1
                U_rows_fb.append(u)
                V_rows_fb.append(v)
                W_rows_fb.append(w)
        
        U = [list(col) for col in zip(*U_rows_fb)]
        V = [list(col) for col in zip(*V_rows_fb)]
        W = [list(col) for col in zip(*W_rows_fb)]
        
        # Final verification
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        T_reconstructed = np.zeros((n * m, m * p, p * n))
        for r in range(len(U_rows_fb)):
            T_reconstructed += np.einsum('a,b,c->abc', U_np[:, r], V_np[:, r], W_np[:, r])
        
        if not np.allclose(T_reconstructed, T_true, atol=1e-10):
            # Schoolbook fallback
            rank_schoolbook = n * m * p
            U = [[0] * rank_schoolbook for _ in range(n * m)]
            V = [[0] * rank_schoolbook for _ in range(m * p)]
            W = [[0] * rank_schoolbook for _ in range(p * n)]
            
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j][col] = 1
                        V[j * p + k][col] = 1
                        W[k * n + i][col] = 1
                        col += 1
        
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
