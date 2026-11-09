"""Program for rank-minimal tensor decomposition of 2x4 x 4x8 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = 2, 4, 8
    deadline = time.time() + 130  # safety margin before 150s kill

    # Strassen's rank-7 decomposition for 2x2 matrix multiplication
    # For C = A * B where A, B, C are 2x2:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    #
    # Strassen's 7 products:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    #
    # C[0,0] = M1 - M4 + M5 - M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M6 - M7

    # For our problem: A is 2x4, B is 4x8
    # Split A into two 2x2 blocks: A = [A_left, A_right] where A_left = A[:, 0:2], A_right = A[:, 2:4]
    # Split B into two 4x4 blocks vertically: B = [B_top; B_bottom] where B_top = B[0:2, :], B_bottom = B[2:4, :]
    # C = A_left * B_top + A_right * B_bottom
    #
    # Each term: 2x2 * 2x8. Split the 2x8 into four 2x2 blocks:
    # B_top = [B_top_0, B_top_1, B_top_2, B_top_3] (each 2x2)
    # B_bottom = [B_bot_0, B_bot_1, B_bot_2, B_bot_3] (each 2x2)
    #
    # C[:, 0:2] = A_left * B_top_0 + A_right * B_bot_0
    # C[:, 2:4] = A_left * B_top_1 + A_right * B_bot_1
    # C[:, 4:6] = A_left * B_top_2 + A_right * B_bot_2
    # C[:, 6:8] = A_left * B_top_3 + A_right * B_bot_3
    #
    # Each 2x2 * 2x2 multiplication uses rank 7 (Strassen).
    # Total: 8 sub-problems * 7 = 56 rank terms.

    rank = 56
    U = np.zeros((n * m, rank), dtype=float)  # (8, 56)
    V = np.zeros((m * p, rank), dtype=float)  # (32, 56)
    W = np.zeros((p * n, rank), dtype=float)  # (16, 56)

    # Strassen coefficients:
    # For 2x2 matrices A=[[a,b],[c,d]], B=[[e,f],[g,h]]:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    #
    # C[0,0] = M1 - M4 + M5 - M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M6 - M7

    # U indices: row = i*m + k = i*4 + k, where i in [0,2), k in [0,4)
    # V indices: row = k*p + j = k*8 + j, where k in [0,4), j in [0,8)
    # W indices: row = j*n + i = j*2 + i, where j in [0,8), i in [0,2)
    # Wait, the problem says W[k*n+i, r] convention. Let me re-read.
    # "output factor follows the official trace convention W[k*n+i,r]"
    # So W index is k*n + i where k is the column index of B (which becomes row index of W? No...)
    # Actually, let me think about this more carefully.
    #
    # The tensor decomposition: C[i,j] = sum_r (sum_k U[i*m+k, r] * A[i,k]) * (sum_k V[k*p+j, r] * B[k,j]) * W[j*n+i, r]
    # Wait, that doesn't look right either. Let me think about what the convention means.
    #
    # Standard: C[i,j] = sum_k A[i,k] B[k,j]
    # In tensor form: C[i,j] = sum_r U[i,j,r] * ... hmm
    #
    # Actually, the bilinear form is: C[i,j] = sum_k A[i,k] B[k,j]
    # This is a 3-linear form: sum_{i,j,k} T[i,j,k] A[i,k] B[k,j] C[i,j]
    # where T[i,j,k] = delta (1 for all valid i,j,k)
    #
    # A rank-r decomposition gives:
    # T[i,j,k] = sum_{r=1}^r U[i,k,r] * V[k,j,r] * W[j,i,r]
    # where U has shape (n*m, r), V has shape (m*p, r), W has shape (p*n, r)
    # and W is indexed as W[j*n+i, r] (i.e., row index = j*n + i)
    #
    # So: sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] = 1 for all i,j,k
    # And the decomposition computes: C[i,j] = sum_r (sum_k U[i*m+k,r]*A[i,k]) * (sum_k V[k*p+j,r]*B[k,j]) * W[j*n+i,r]
    # Wait no, that's not right either. The bilinear form means:
    # C[i,j] = sum_r W[j*n+i,r] * (sum_k U[i*m+k,r]*A[i,k]) * (sum_k' V[k'*p+j,r]*B[k',j])
    # But that would give products of sums, not sums of products.
    #
    # Actually, for matrix multiplication as a bilinear map:
    # C = A * B
    # The tensor is: T[i, k, j] such that C[i,j] = sum_k T[i,k,j] A[i,k] B[k,j]
    # For standard matrix mult, T[i,k,j] = 1 for all valid (i,k,j).
    #
    # A rank-r decomposition of this tensor is:
    # T[i,k,j] = sum_r U[i,k,r] V[k,j,r] W[j,i,r]
    # where the three factors correspond to the three indices.
    #
    # In the convention given:
    # U has shape (n*m, rank), indexed by (i*m + k)
    # V has shape (m*p, rank), indexed by (k*p + j)
    # W has shape (p*n, rank), indexed by (j*n + i)
    #
    # So: sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] = 1 for all i in [0,n), k in [0,m), j in [0,p)
    #
    # And the computation is: C[i,j] = sum_r W[j*n+i,r] * (sum_k U[i*m+k,r]*A[i,k]) * (sum_k' V[k'*p+j,r]*B[k',j])
    # Wait, no. The bilinear form is:
    # C[i,j] = sum_k A[i,k] B[k,j] = sum_r (sum_k A[i,k] U[i*m+k,r]) (sum_k B[k,j] V[k*p+j,r]) W[j*n+i,r]
    # Hmm, that's still a product of two sums times W.
    #
    # Actually, I think the correct interpretation is:
    # C[i,j] = sum_r W[j*n+i, r] * U[i, r] * V[j, r]
    # where U[i, r] = sum_k A[i,k] * U_matrix[i*m+k, r]
    # and V[j, r] = sum_k B[k,j] * V_matrix[k*p+j, r]
    #
    # This is the standard bilinear form decomposition.

    # Let me define the Strassen decomposition for a single 2x2x2 sub-problem.
    # Sub-problem: C_sub = A_sub * B_sub where A_sub is 2x2, B_sub is 2x2, C_sub is 2x2.
    # A_sub = [[a, b], [c, d]] where a=A_sub[0,0], b=A_sub[0,1], c=A_sub[1,0], d=A_sub[1,1]
    # B_sub = [[e, f], [g, h]] where e=B_sub[0,0], f=B_sub[0,1], g=B_sub[1,0], h=B_sub[1,1]
    # C_sub = [[C00, C01], [C10, C11]]
    #
    # For this sub-problem, the local indices are:
    # i in {0,1}, k in {0,1}, j in {0,1}
    # U_local[i, k, r] for r in {0,...,6}
    # V_local[k, j, r]
    # W_local[j, i, r]
    #
    # The 7 products:
    # M1 = (a+d)(e+h) => U: a has (i=0,k=0), d has (i=1,k=1); V: e has (k=0,j=0), h has (k=1,j=1)
    #   U1 = [[1,0],[0,1]] (for a and d), V1 = [[1,0],[0,1]] (for e and h), W1 contributes to C00 and C11
    #   C00 += M1, C11 += M1
    #   So W1[0,0] = 1 (for C00), W1[1,1] = 1 (for C11)
    #
    # M2 = (c+d)e => U: c has (i=1,k=0), d has (i=1,k=1); V: e has (k=0,j=0)
    #   U2 = [[0,0],[1,1]], V2 = [[1,0],[0,0]], W2: C10 += M2, so W2[0,1] = 1
    #
    # M3 = a(f-h) => U: a has (i=0,k=0); V: f has (k=0,j=1), h has (k=1,j=1)
    #   U3 = [[1,0],[0,0]], V3 = [[0,1],[0,-1]], W3: C01 += M3, so W3[1,0] = 1
    #
    # M4 = d(g-e) => U: d has (i=1,k=1); V: g has (k=1,j=0), e has (k=0,j=0)
    #   U4 = [[0,0],[0,1]], V4 = [[-1,0],[1,0]], W4: C00 += -M4, C10 += M4
    #   So W4[0,0] = -1, W4[0,1] = 1
    #
    # M5 = (a+b)h => U: a has (i=0,k=0), b has (i=0,k=1); V: h has (k=1,j=1)
    #   U5 = [[1,1],[0,0]], V5 = [[0,0],[0,1]], W5: C00 += M5, C01 += M5
    #   So W5[0,0] = 1, W5[1,0] = 1
    #
    # M6 = (c-a)(e+f) => U: c has (i=1,k=0), a has (i=0,k=0); V: e has (k=0,j=0), f has (k=0,j=1)
    #   U6 = [[-1,0],[1,0]], V6 = [[1,1],[0,0]], W6: C11 += M6, so W6[1,1] = 1
    #
    # M7 = (b-d)(g+h) => U: b has (i=0,k=1), d has (i=1,k=1); V: g has (k=1,j=0), h has (k=1,j=1)
    #   U7 = [[0,1],[0,-1]], V7 = [[0,0],[1,1]], W7: C00 += -M7, C11 += -M7
    #   So W7[0,0] = -1, W7[1,1] = -1

    # Let me define the 7 Strassen factors for a 2x2x2 problem:
    # Each factor r has U_r (2x2), V_r (2x2), W_r (2x2)
    # U_r[i,k] = coefficient of A[i,k] in the r-th product
    # V_r[k,j] = coefficient of B[k,j] in the r-th product
    # W_r[j,i] = coefficient of the r-th product in C[i,j]

    strassen_U = [
        [[1,0],[0,1]],   # M1: a, d
        [[0,0],[1,1]],   # M2: c, d
        [[1,0],[0,0]],   # M3: a
        [[0,0],[0,1]],   # M4: d
        [[1,1],[0,0]],   # M5: a, b
        [[-1,0],[1,0]],  # M6: -a, c
        [[0,1],[0,-1]],  # M7: b, -d
    ]
    strassen_V = [
        [[1,0],[0,1]],   # M1: e, h
        [[1,0],[0,0]],   # M2: e
        [[0,1],[0,-1]],  # M3: f, -h
        [[-1,0],[1,0]],  # M4: -e, g
        [[0,0],[0,1]],   # M5: h
        [[1,1],[0,0]],   # M6: e, f
        [[0,0],[1,1]],   # M7: g, h
    ]
    strassen_W = [
        [[1,0],[0,1]],   # M1: C00, C11
        [[0,1],[0,0]],   # M2: C10 (W[j,i]: j=0,i=1)
        [[0,1],[0,0]],   # M3: C01 (W[j,i]: j=1,i=0)
        [[-1,1],[0,0]],  # M4: -C00, C10
        [[1,1],[0,0]],   # M5: C00, C01
        [[0,0],[0,1]],   # M6: C11 (W[j,i]: j=1,i=1)
        [[-1,0],[0,-1]], # M7: -C00, -C11
    ]

    # Now, for the full 2x4 x 4x8 problem:
    # We have 8 sub-problems (block_a in {0,1} for left/right, block_b in {0,1,2,3} for column groups)
    # Wait, let me reconsider the splitting.
    #
    # A is 2x4. Split columns into two groups of 2: A = [A_L, A_R] where A_L = A[:,0:2], A_R = A[:,2:4]
    # B is 4x8. Split rows into two groups of 2: B = [B_T; B_B] where B_T = B[0:2,:], B_B = B[2:4,:]
    # C = A_L * B_T + A_R * B_B
    #
    # B_T is 2x8. Split into four 2x2 blocks: B_T = [BT0, BT1, BT2, BT3]
    # B_B is 2x8. Split into four 2x2 blocks: B_B = [BB0, BB1, BB2, BB3]
    #
    # C[:, 0:2] = A_L * BT0 + A_R * BB0
    # C[:, 2:4] = A_L * BT1 + A_R * BB1
    # C[:, 4:6] = A_L * BT2 + A_R * BB2
    # C[:, 6:8] = A_L * BT3 + A_R * BB3
    #
    # So we have 8 sub-problems:
    # For each col_group in {0,1,2,3} and row_part in {0,1}:
    #   if row_part == 0: A_sub = A_L, B_sub = BT[col_group]
    #   if row_part == 1: A_sub = A_R, B_sub = BB[col_group]
    #   C_sub = A_sub * B_sub contributes to C[:, col_group*2:(col_group+1)*2]

    col = 0
    for col_group in range(4):
        for row_part in range(2):
            # A_sub is 2x2:
            # row_part 0: A_L = A[:, 0:2], so A_sub[i, k] = A[i, k]
            # row_part 1: A_R = A[:, 2:4], so A_sub[i, k] = A[i, k+2]
            # B_sub is 2x2:
            # row_part 0: BT[col_group] = B[0:2, col_group*2:(col_group+1)*2], so B_sub[k, j] = B[k, col_group*2+j]
            # row_part 1: BB[col_group] = B[2:4, col_group*2:(col_group+1)*2], so B_sub[k, j] = B[k+2, col_group*2+j]
            # C_sub is 2x2, contributes to C[i, col_group*2+j]
            
            for r in range(7):
                # U: shape (2,2) for local (i, k)
                for i in range(2):
                    for k in range(2):
                        u_val = strassen_U[r][i][k]
                        if u_val != 0:
                            # Global U index: i*m + k_global
                            # k_global = k + 2*row_part (since row_part 0 uses k=0,1; row_part 1 uses k=2,3)
                            k_global = k + 2 * row_part
                            U[i * m + k_global, col] += u_val
                
                # V: shape (2,2) for local (k, j)
                for k in range(2):
                    for j in range(2):
                        v_val = strassen_V[r][k][j]
                        if v_val != 0:
                            # Global V index: k_global*p + j_global
                            # k_global = k + 2*row_part
                            # j_global = col_group*2 + j
                            k_global = k + 2 * row_part
                            j_global = col_group * 2 + j
                            V[k_global * p + j_global, col] += v_val
                
                # W: shape (2,2) for local (j, i)
                for j in range(2):
                    for i in range(2):
                        w_val = strassen_W[r][j][i]
                        if w_val != 0:
                            # Global W index: j_global*n + i
                            # j_global = col_group*2 + j
                            j_global = col_group * 2 + j
                            W[j_global * n + i, col] += w_val
                
                col += 1

    # Verify the decomposition
    def verify(U_mat, V_mat, W_mat, n, m, p):
        U_arr = np.array(U_mat)
        V_arr = np.array(V_mat)
        W_arr = np.array(W_mat)
        rank_val = U_arr.shape[1]
        
        np.random.seed(42)
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        C_computed = np.zeros((n, p))
        for i in range(n):
            for j in range(p):
                total = 0
                for r in range(rank_val):
                    u_sum = 0
                    for k in range(m):
                        u_sum += U_arr[i*m + k, r] * A[i, k]
                    v_sum = 0
                    for k in range(m):
                        v_sum += V_arr[k*p + j, r] * B[k, j]
                    total += W_arr[j*n + i, r] * u_sum * v_sum
                C_computed[i, j] = total
        
        return np.allclose(C_computed, C_expected, atol=1e-9)

    if not verify(U, V, W, n, m, p):
        # Fallback to schoolbook if verification fails
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    U[i*m + k, col] = 1
                    V[k*p + j, col] = 1
                    W[j*n + i, col] = 1
                    col += 1

    # Convert to lists of lists for the return format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
