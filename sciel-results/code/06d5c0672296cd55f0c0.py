"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Schoolbook fallback
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    def strassen_2x2():
        """Return (U, V, W) for 2x2*2x2 Strassen, each of shape (4, 7)"""
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # A: [a, b, c, d] (indices 0,1,2,3)
        # B: [e, f, g, h] (indices 0,1,2,3)
        # C: [C11, C12, C21, C22] (indices 0,1,2,3)
        
        # M1 = (a+d)(e+h)
        U[0][0] += 1; U[3][0] += 1
        V[0][0] += 1; V[3][0] += 1
        W[0][0] += 1; W[3][0] += 1
        
        # M2 = (c+d)e
        U[2][1] += 1; U[3][1] += 1
        V[0][1] += 1
        W[2][1] += 1; W[3][1] += -1
        
        # M3 = a(f-h)
        U[0][2] += 1
        V[1][2] += 1; V[3][2] += -1
        W[1][2] += 1; W[3][2] += 1
        
        # M4 = d(g-e)
        U[3][3] += 1
        V[2][3] += 1; V[0][3] += -1
        W[0][3] += 1; W[2][3] += 1
        
        # M5 = (a+b)h
        U[0][4] += 1; U[1][4] += 1
        V[3][4] += 1
        W[0][4] += -1; W[1][4] += 1
        
        # M6 = (c-a)(e+f)
        U[2][5] += 1; U[0][5] += -1
        V[0][5] += 1; V[1][5] += 1
        W[3][5] += 1
        
        # M7 = (b-d)(g+h)
        U[1][6] += 1; U[3][6] += -1
        V[2][6] += 1; V[3][6] += 1
        W[0][6] += 1
        
        return U, V, W
    
    # Build 4x4*4x4 Strassen decomposition
    def build_4x4_strassen():
        U_total = [[0]*49 for _ in range(16)]
        V_total = [[0]*49 for _ in range(16)]
        W_total = [[0]*49 for _ in range(16)]
        
        A11 = [0, 1, 4, 5]
        A12 = [2, 3, 6, 7]
        A21 = [8, 9, 12, 13]
        A22 = [10, 11, 14, 15]
        
        B11 = [0, 1, 4, 5]
        B12 = [2, 3, 6, 7]
        B21 = [8, 9, 12, 13]
        B22 = [10, 11, 14, 15]
        
        C11 = [0, 1, 4, 5]
        C12 = [2, 3, 6, 7]
        C21 = [8, 9, 12, 13]
        C22 = [10, 11, 14, 15]
        
        products = [
            ([ (A11, 1), (A22, 1) ], [ (B11, 1), (B22, 1) ], [ (C11, 1), (C22, 1) ]),
            ([ (A21, 1), (A22, 1) ], [ (B11, 1) ], [ (C21, 1), (C22, -1) ]),
            ([ (A11, 1) ], [ (B12, 1), (B22, -1) ], [ (C12, 1), (C22, 1) ]),
            ([ (A22, 1) ], [ (B21, 1), (B11, -1) ], [ (C11, 1), (C21, 1) ]),
            ([ (A11, 1), (A12, 1) ], [ (B22, 1) ], [ (C11, -1), (C12, 1) ]),
            ([ (A21, 1), (A11, -1) ], [ (B11, 1), (B12, 1) ], [ (C22, 1) ]),
            ([ (A12, 1), (A22, -1) ], [ (B21, 1), (B22, 1) ], [ (C11, 1) ]),
        ]
        
        for prod_idx, (a_defs, b_defs, c_defs) in enumerate(products):
            a_contrib = [dict() for _ in range(4)]
            for block, sign in a_defs:
                for local_i, global_i in enumerate(block):
                    a_contrib[local_i][global_i] = a_contrib[local_i].get(global_i, 0) + sign
            
            b_contrib = [dict() for _ in range(4)]
            for block, sign in b_defs:
                for local_i, global_i in enumerate(block):
                    b_contrib[local_i][global_i] = b_contrib[local_i].get(global_i, 0) + sign
            
            c_contrib = [dict() for _ in range(4)]
            for block, sign in c_defs:
                for local_i, global_i in enumerate(block):
                    c_contrib[local_i][global_i] = c_contrib[local_i].get(global_i, 0) + sign
            
            U2, V2, W2 = strassen_2x2()
            base_col = prod_idx * 7
            for local_i in range(4):
                for global_i, sign_a in a_contrib[local_i].items():
                    for col in range(7):
                        U_total[global_i][base_col + col] += sign_a * U2[local_i][col]
                for global_i, sign_b in b_contrib[local_i].items():
                    for col in range(7):
                        V_total[global_i][base_col + col] += sign_b * V2[local_i][col]
                for global_i, sign_c in c_contrib[local_i].items():
                    for col in range(7):
                        W_total[global_i][base_col + col] += sign_c * W2[local_i][col]
        
        return U_total, V_total, W_total
    
    # Build 4x4*4x2 decomposition
    def build_4x4x2():
        U_total = [[0]*28 for _ in range(16)]
        V_total = [[0]*28 for _ in range(8)]
        W_total = [[0]*28 for _ in range(8)]
        
        A11 = [0, 1, 4, 5]
        A12 = [2, 3, 6, 7]
        A21 = [8, 9, 12, 13]
        A22 = [10, 11, 14, 15]
        
        Bt1 = [0, 1, 2, 3]  # top 2x2 of 4x2
        Bt2 = [4, 5, 6, 7]  # bottom 2x2 of 4x2
        
        Cr_top = [0, 1, 4, 5]      # top 2x2 of 4x2 output
        Cr_bottom = [8, 9, 12, 13]  # bottom 2x2 of 4x2 output
        
        products = [
            (A11, Bt1, Cr_top),
            (A12, Bt2, Cr_top),
            (A21, Bt1, Cr_bottom),
            (A22, Bt2, Cr_bottom),
        ]
        
        for prod_idx, (a_block, b_block, c_block) in enumerate(products):
            U2, V2, W2 = strassen_2x2()
            base_col = prod_idx * 7
            for local_i in range(4):
                for col in range(7):
                    U_total[a_block[local_i]][base_col + col] += U2[local_i][col]
                    V_total[b_block[local_i]][base_col + col] += V2[local_i][col]
                    W_total[c_block[local_i]][base_col + col] += W2[local_i][col]
        
        return U_total, V_total, W_total
    
    # Try to build the optimized decomposition
    try:
        if time.time() > deadline:
            return {"U": U_sb, "V": V_sb, "W": W_sb}
        
        U1, V1, W1 = build_4x4_strassen()
        U2, V2, W2 = build_4x4x2()
        
        # Verify with a test
        import numpy as np
        U1_np = np.array(U1, dtype=float)
        V1_np = np.array(V1, dtype=float)
        W1_np = np.array(W1, dtype=float)
        
        # Test 4x4*4x4
        A_test = np.random.randn(4, 4)
        B_test = np.random.randn(4, 4)
        C_expected = A_test @ B_test
        
        # Compute using decomposition: C[i,k] = sum_c U[i,c] * V[j,c] * W[k,i,c]... 
        # Actually: C[i,k] = sum_j A[i,j]*B[j,k]
        # Decomposition: C[i,k] = sum_c U[i*m+j... ] 
        # The tensor: T[i,j,k] = delta (identity)
        # U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should sum to T[i,j,k]
        
        # Let me verify: for each (i,j,k), sum_c U[i*4+j, c] * V[j*4+k, c] * W[k*4+i, c] == 1 if this is the identity, else 0
        # Wait, the matrix multiplication tensor T[i,j,k] = 1 for all i,j,k (it's the "multiplication" tensor)
        # Actually no. The tensor for matrix mult is T[i,j,k] where the product is sum_j A[i,j]*B[j,k]
        # The decomposition satisfies: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta for the identity
        # But for matrix multiplication, it should be: the bilinear form gives C[i,k] = sum_j A[i,j]*B[j,k]
        # So the tensor element T[i,j,k] = 1 (all ones) in the sense that C[i,k] = sum_j T[i,j,k]*A[i,j]*B[j,k] with T=1
        
        # Let me just verify numerically
        # C[i,k] = sum_c (sum_j U[i*4+j, c] * A[i,j]) * (sum_j V[j*4+k, c] * B[j,k]) * W[k*4+i, c]
        # No, that's not right either.
        
        # The correct verification: 
        # C[i,k] should equal sum_j A[i,j]*B[j,k]
        # And C[i,k] = sum_c U_part[i,c] * V_part[k,c] * W_part[i,k,c]... 
        # I'm getting confused. Let me just check:
        # The decomposition says: C[i,k] = sum_c (U[i*4+j] for all j) ... 
        # Actually the standard form is:
        # C[i,k] = sum_c sum_j U[i*m+j, c] * A[i,j] * V[j*p+k, c] * B[j,k] * W[k*n+i, c]
        # But that has A and B inside the sum over c, which doesn't make sense for bilinear.
        
        # The correct form: C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k])
        # No, that's not bilinear in A and B.
        
        # I think the correct interpretation is:
        # C[i,k] = sum_c (U_col_c · A_row_i) * (V_col_c · B_col_k) * W[k*n+i, c]
        # where U_col_c is the c-th column of U, A_row_i is the i-th row of A, etc.
        # But U has shape (n*m, rank), so U_col_c has n*m elements.
        
        # Let me just do: for each c, compute:
        # u_c = U[:, c] (length n*m)
        # v_c = V[:, c] (length m*p)  
        # w_c = W[:, c] (length p*n)
        # Then C[i,k] += w_c[k*n+i] * (sum_j u_c[i*m+j] * A[i,j]) * (sum_j v_c[j*p+k] * B[j,k])
        
        # Let me test this
        C_computed = np.zeros((4, 4))
        for c in range(49):
            u_c = U1_np[:, c]
            v_c = V1_np[:, c]
            w_c = W1_np[:, c]
            for i in range(4):
                for k in range(4):
                    a_part = sum(u_c[i*4+j] * A_test[i, j] for j in range(4))
                    b_part = sum(v_c[j*4+k] * B_test[j, k] for j in range(4))
                    C_computed[i, k] += w_c[k*4+i] * a_part * b_part
        
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            print(f"Verification failed for 4x4*4x4: max diff = {np.max(np.abs(C_computed - C_expected))}")
            return {"U": U_sb, "V": V_sb, "W": W_sb}
        
        # Verify 4x4*4x2
        U2_np = np.array(U2, dtype=float)
        V2_np = np.array(V2, dtype=float)
        W2_np = np.array(W2, dtype=float)
        
        B_test2 = np.random.randn(4, 2)
        C_expected2 = A_test @ B_test2
        C_computed2 = np.zeros((4, 2))
        for c in range(28):
            u_c = U2_np[:, c]
            v_c = V2_np[:, c]
            w_c = W2_np[:, c]
            for i in range(4):
                for k in range(2):
                    a_part = sum(u_c[i*4+j] * A_test[i, j] for j in range(4))
                    b_part = sum(v_c[j*2+k] * B_test2[j, k] for j in range(4))
                    C_computed2[i, k] += w_c[k*4+i] * a_part * b_part
        
        if not np.allclose(C_computed2, C_expected2, atol=1e-10):
            print(f"Verification failed for 4x4*4x2: max diff = {np.max(np.abs(C_computed2 - C_expected2))}")
            return {"U": U_sb, "V": V_sb, "W": W_sb}
        
        # Combine into final 4x4*4x6
        rank_total = 49 + 28
        U_final = [[0]*rank_total for _ in range(16)]
        V_final = [[0]*rank_total for _ in range(24)]
        W_final = [[0]*rank_total for _ in range(24)]
        
        for i in range(16):
            for col in range(49):
                U_final[i][col] = U1[i][col]
                V_final[i][col] = V1[i][col]
                W_final[i][col] = W1[i][col]
        
        for i in range(16):
            for col in range(28):
                U_final[i][49 + col] = U2[i][col]
                V_final[i][49 + col] = V2[i][col]
                W_final[i][49 + col] = W2[i][col]
        for i in range(16, 24):
            for col in range(28):
                V_final[i][49 + col] = V2[i - 16][col]
                W_final[i][49 + col] = W2[i - 16][col]
        
        # Final verification
        U_final_np = np.array(U_final, dtype=float)
        V_final_np = np.array(V_final, dtype=float)
        W_final_np = np.array(W_final, dtype=float)
        
        B_full = np.random.randn(4, 6)
        C_full_expected = A_test @ B_full
        C_full_computed = np.zeros((4, 6))
        for c in range(rank_total):
            u_c = U_final_np[:, c]
            v_c = V_final_np[:, c]
            w_c = W_final_np[:, c]
            for i in range(4):
                for k in range(6):
                    a_part = sum(u_c[i*4+j] * A_test[i, j] for j in range(4))
                    b_part = sum(v_c[j*6+k] * B_full[j, k] for j in range(4))
                    C_full_computed[i, k] += w_c[k*4+i] * a_part * b_part
        
        if not np.allclose(C_full_computed, C_full_expected, atol=1e-10):
            print(f"Final verification failed: max diff = {np.max(np.abs(C_full_computed - C_full_expected))}")
            return {"U": U_sb, "V": V_sb, "W": W_sb}
        
        return {"U": U_final, "V": V_final, "W": W_final}
    
    except Exception as e:
        print(f"Error: {e}")
        return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END
