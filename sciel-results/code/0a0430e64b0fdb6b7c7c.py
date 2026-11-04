import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    # Global dimensions
    U = np.zeros((n * m, 112), dtype=float)
    V = np.zeros((m * p, 112), dtype=float)
    W = np.zeros((p * n, 112), dtype=float)
    
    # Helper to add a local decomposition to global
    # local_u: shape (a*b, rank_local)
    # local_v: shape (b*c, rank_local)
    # local_w: shape (c*a, rank_local)
    # row_offsets: (i_start, j_start) for U, (j_start, k_start) for V, (k_start, i_start) for W
    # col_offset: starting column index in global U, V, W
    def add_decomposition(local_u, local_v, local_w, i_start, j_start, k_start, col_offset):
        a, b, c = local_u.shape[0] // b, local_v.shape[0] // c, local_w.shape[0] // a
        rank_local = local_u.shape[1]
        
        # U rows: i in [i_start, i_start+a), j in [j_start, j_start+b)
        # Global row index: i*m + j
        for i in range(a):
            for j in range(b):
                local_row_u = i * b + j
                global_row_u = (i_start + i) * m + (j_start + j)
                U[global_row_u, col_offset:col_offset+rank_local] += local_u[local_row_u]
                
        # V rows: j in [j_start, j_start+b), k in [k_start, k_start+c)
        # Global row index: j*p + k
        for j in range(b):
            for k in range(c):
                local_row_v = j * c + k
                global_row_v = (j_start + j) * p + (k_start + k)
                V[global_row_v, col_offset:col_offset+rank_local] += local_v[local_row_v]
                
        # W rows: k in [k_start, k_start+c), i in [i_start, i_start+a)
        # Global row index: k*n + i
        for k in range(c):
            for i in range(a):
                local_row_w = k * a + i
                global_row_w = (k_start + k) * n + (i_start + i)
                W[global_row_w, col_offset:col_offset+rank_local] += local_w[local_row_w]

    # Strassen 2x2x2 decomposition (Rank 7)
    U_222 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 1, 0, 1],
        [1, 1, 0, 1, 0, 0, -1]
    ])
    V_222 = np.array([
        [1, 1, 0, 0, 0, 1, 0],
        [1, 0, 0, 1, 0, 0, 1],
        [1, 0, 1, -1, 0, 1, 0],
        [1, 0, -1, 0, 1, 0, 1]
    ])
    W_222 = np.array([
        [1, 0, 0, 1, -1, 0, -1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 0, 1]
    ])

    # Schoolbook 2x2x1 decomposition (Rank 4)
    # A 2x2, B 2x1 -> C 2x1
    # C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0]
    # C[1,0] = A[1,0]B[0,0] + A[1,1]B[1,0]
    U_221 = np.array([
        [1, 0, 0, 0], # A[0,0]
        [0, 1, 0, 0], # A[0,1]
        [0, 0, 1, 0], # A[1,0]
        [0, 0, 0, 1]  # A[1,1]
    ])
    V_221 = np.array([
        [1, 0, 1, 0], # B[0,0]
        [0, 1, 0, 1]  # B[1,0]
    ])
    W_221 = np.array([
        [1, 0, 0, 0], # C[0,0]
        [0, 0, 1, 0]  # C[1,0]
    ])
    
    # Note: W_221 rows are k*n+i. Here c=1, a=2.
    # k=0, i=0 -> row 0. k=0, i=1 -> row 1.
    # C[0,0] corresponds to i=0, k=0.
    # C[1,0] corresponds to i=1, k=0.
    # So W_221[0] is for C[0,0], W_221[1] is for C[1,0].
    # My W_222 above had rows k*n+i.
    # For 2x2x2:
    # Row 0: k=0, i=0 -> C[0,0]
    # Row 1: k=0, i=1 -> C[1,0]
    # Row 2: k=1, i=0 -> C[0,1]
    # Row 3: k=1, i=1 -> C[1,1]
    # Let's re-verify W_222.
    # C00 = M1+M4-M5-M7. W[0] = [1, 0, 0, 1, -1, 0, -1]. Correct.
    # C10 = M2+M4. W[1] = [0, 1, 0, 1, 0, 0, 0]. Correct.
    # C01 = M3+M5. W[2] = [0, 0, 1, 0, 1, 0, 0]. Correct.
    # C11 = M1-M2+M3+M7. W[3] = [1, -1, 1, 0, 0, 0, 1]. Correct.
    
    # For 2x2x1:
    # C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0].
    # Terms: (0,0,0), (0,1,0).
    # U: A[0,0]=1, A[0,1]=1.
    # V: B[0,0]=1, B[1,0]=1.
    # W: C[0,0] gets 1 from both.
    # My U_221:
    # Col 0: A[0,0]=1.
    # Col 1: A[0,1]=1.
    # Col 2: A[1,0]=1.
    # Col 3: A[1,1]=1.
    # My V_221:
    # Col 0: B[0,0]=1.
    # Col 1: B[1,0]=1.
    # Col 2: B[0,0]=1.
    # Col 3: B[1,0]=1.
    # My W_221:
    # Col 0: C[0,0]=1.
    # Col 1: C[1,0]=1.
    # Col 2: C[0,0]=1.
    # Col 3: C[1,0]=1.
    # Wait, W_221 rows:
    # Row 0 (k=0, i=0): C[0,0]. Should be 1, 0, 1, 0.
    # Row 1 (k=0, i=1): C[1,0]. Should be 0, 1, 0, 1.
    # My W_221 is:
    # [[1, 0, 0, 0],
    #  [0, 0, 1, 0]] -> This is wrong.
    # Let's fix W_221.
    W_221 = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ])

    # Helper to generate 2x2x3 decomposition (Rank 11)
    def get_223():
        # A 2x2, B 2x3 -> C 2x3
        # B = [B_L (2x2), B_R (2x1)]
        # C = A B_L + A B_R
        # A B_L: Rank 7
        # A B_R: Rank 4
        u_223 = np.zeros((4, 11))
        v_223 = np.zeros((6, 11))
        w_223 = np.zeros((6, 11)) # c*a = 3*2 = 6
        
        # Part 1: A B_L (cols 0,1 of B)
        u_223[:, 0:7] = U_222
        v_223[0:4, 0:7] = V_222 # B_L is rows 0,1 of B
        w_223[0:4, 0:7] = W_222 # C_L is cols 0,1 of C. k in 0..1, i in 0..1.
        # W_222 rows are k*2+i.
        # C_L has k=0,1.
        # C_R has k=2.
        
        # Part 2: A B_R (col 2 of B)
        # B_R is 2x1.
        u_223[:, 7:11] = U_221
        # V_221 rows: j*p+k. j in 0..1, k=0.
        # B_R is rows 0,1 of B, col 2.
        # So V_223 rows 2,3 (j=0,k=2 and j=1,k=2) should get V_221.
        v_223[2, 7:11] = V_221[0]
        v_223[3, 7:11] = V_221[1]
        
        # W_221 rows: k*n+i. k=0, i in 0..1.
        # C_R is col 2 of C. k=2.
        # So W_223 rows 4,5 (k=2, i=0 and k=2, i=1) should get W_221.
        w_223[4, 7:11] = W_221[0]
        w_223[5, 7:11] = W_221[1]
        
        return u_223, v_223, w_223

    # Helper to generate 2x3x3 decomposition (Rank 17)
    def get_233():
        # A 2x3, B 3x3 -> C 2x3
        # A = [A_L (2x2), A_R (2x1)]
        # B = [B_T (2x3); B_B (1x3)]
        # C = A_L B_T + A_R B_B
        # A_L B_T: Rank 11 (2x2x3)
        # A_R B_B: Rank 6 (2x1x3)
        
        u_233 = np.zeros((6, 17))
        v_233 = np.zeros((9, 17))
        w_233 = np.zeros((6, 17)) # c*a = 3*2 = 6
        
        # Part 1: A_L B_T
        u_lt, v_lt, w_lt = get_223()
        # A_L is cols 0,1 of A.
        u_233[0:4, 0:11] = u_lt
        # B_T is rows 0,1 of B.
        v_233[0:6, 0:11] = v_lt
        # C is cols 0,1,2.
        w_233[:, 0:11] = w_lt
        
        # Part 2: A_R B_B
        # A_R is 2x1. B_B is 1x3.
        # C is 2x3.
        # Terms: (i, 0, k) for i in 0..1, k in 0..2.
        # A_R[i, 0] * B_B[0, k] -> C[i, k].
        # Rank 6.
        # U: A_R. Rows i*1+0 = i.
        u_rb = np.zeros((2, 6))
        for i in range(2):
            for k in range(3):
                u_rb[i, i*3+k] = 1
        # V: B_B. Rows 0*3+k = k.
        v_rb = np.zeros((1, 6))
        for i in range(2):
            for k in range(3):
                v_rb[0, i*3+k] = 1
        # W: C. Rows k*2+i.
        w_rb = np.zeros((6, 6))
        for i in range(2):
            for k in range(3):
                w_rb[k*2+i, i*3+k] = 1
        
        u_233[4:6, 11:17] = u_rb
        # B_B is row 2 of B.
        v_233[6, 11:17] = v_rb
        w_233[:, 11:17] = w_rb
        
        return u_233, v_233, w_233

    # Assemble 4x5x6
    # A blocks:
    # A00 (2x2) at (0,0)
    # A01 (2x3) at (0,2)
    # A10 (2x2) at (2,0)
    # A11 (2x3) at (2,2)
    
    # B blocks:
    # B00 (2x3) at (0,0)
    # B01 (2x3) at (0,3)
    # B10 (3x3) at (2,0)
    # B11 (3x3) at (2,3)
    
    col_offset = 0
    
    # C00 = A00 B00 + A01 B10
    # A00 B00: 2x2x3. Rank 11.
    u, v, w = get_223()
    add_decomposition(u, v, w, 0, 0, 0, col_offset)
    col_offset += 11
    
    # A01 B10: 2x3x3. Rank 17.
    u, v, w = get_233()
    add_decomposition(u, v, w, 0, 2, 2, col_offset)
    col_offset += 17
    
    # C01 = A00 B01 + A01 B11
    # A00 B01: 2x2x3. Rank 11.
    u, v, w = get_223()
    add_decomposition(u, v, w, 0, 0, 3, col_offset)
    col_offset += 11
    
    # A01 B11: 2x3x3. Rank 17.
    u, v, w = get_233()
    add_decomposition(u, v, w, 0, 2, 2, col_offset)
    col_offset += 17
    
    # C10 = A10 B00 + A11 B10
    # A10 B00: 2x2x3. Rank 11.
    u, v, w = get_223()
    add_decomposition(u, v, w, 2, 0, 0, col_offset)
    col_offset += 11
    
    # A11 B10: 2x3x3. Rank 17.
    u, v, w = get_233()
    add_decomposition(u, v, w, 2, 2, 2, col_offset)
    col_offset += 17
    
    # C11 = A10 B01 + A11 B11
    # A10 B01: 2x2x3. Rank 11.
    u, v, w = get_223()
    add_decomposition(u, v, w, 2, 0, 3, col_offset)
    col_offset += 11
    
    # A11 B11: 2x3x3. Rank 17.
    u, v, w = get_233()
    add_decomposition(u, v, w, 2, 2, 2, col_offset)
    col_offset += 17
    
    # Verify
    if time.time() > deadline - 10:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    T = np.einsum('ijc,jkc,kic->ijk', U.reshape(n,m,-1), V.reshape(m,p,-1), W.reshape(p,n,-1))
    # Check against ones
    # The tensor is sum U[i,j,c] V[j,k,c] W[k,i,c]
    # This should equal 1 if i,j,k correspond to a valid term?
    # No, the decomposition is for the multiplication tensor.
    # T[i,j,k] should be 1 if the term A[i,j]B[j,k] exists?
    # Actually, the tensor T_{ijk} = 1 for all i,j,k?
    # No, the multiplication tensor has T_{ijk} = 1 for all i,j,k.
    # Wait, the tensor is defined such that C[i,k] = sum_j A[i,j] B[j,k].
    # The decomposition is sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1 for all i,j,k.
    # So T should be all ones.
    
    if not np.allclose(T, np.ones((n, m, p)), atol=1e-9):
        # Fallback to schoolbook if verification fails
        U_s = np.zeros((n * m, n * m * p))
        V_s = np.zeros((m * p, n * m * p))
        W_s = np.zeros((p * n, n * m * p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i * m + j, col] = 1
                    V_s[j * p + k, col] = 1
                    W_s[k * n + i, col] = 1
                    col += 1
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
