"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Build the rank-18 decomposition for 4x4x4 matrix multiplication
    # Then extend to 4x4x6 by adding 2 more columns with sharing
    
    # The rank-18 decomposition for 4x4x4 uses the following structure:
    # We'll construct it explicitly using known formulas.
    
    # Actually, let me use a verified approach: 
    # For 4x4 * 4x4, use two Strassen 2x2x2 decompositions with sharing.
    # The standard rank-18 algorithm for 4x4x4:
    
    # Let me use a different strategy: build the 4x4x6 tensor directly
    # using a combination of Strassen blocks with careful sharing.
    
    # Key insight: C[i,k] = sum_j A[i,j] * B[j,k]
    # For i in {0,1} (top half of A rows) and k in {0,1,2} (column pairs of B):
    # C_top[:,k_pair] = A_top_left * B_left[:,k_pair] + A_top_right * B_right[:,k_pair]
    # For i in {2,3} (bottom half of A rows):
    # C_bot[:,k_pair] = A_bot_left * B_left[:,k_pair] + A_bot_right * B_right[:,k_pair]
    
    # Each "A_block * B_block" is a 2x2 * 2x2 multiplication (7 Strassen mults)
    # But we can share: the B blocks are the same for top and bottom A rows!
    
    # Better: use the rank-18 4x4x4 decomposition as building blocks.
    # The 4x4x4 rank-18 decomposition can be seen as:
    # 4 blocks of 2x2x2, each with 7 mults = 28, but with 10 shared/eliminated = 18
    
    # For 4x4x6, let's use a block approach:
    # Split into 3 column-pairs. For each pair, we need a 4x4 * 4x2 product.
    # A 4x4 * 4x2 can be done with rank at most 4*4*2 = 32, but we can do better.
    
    # Let me try: use Strassen on 2x2 blocks with sharing across the 3 column groups.
    # A is 4x4: split into A00,A01,A10,A11 (each 2x2)
    # B is 4x6: split into B00,B01,B02 (each 2x2) for top 2 rows, 
    #            B10,B11,B12 (each 2x2) for bottom 2 rows
    
    # C is 4x6: C00,C01,C02 (top 2 rows, each 2x2), C10,C11,C12 (bottom 2 rows)
    
    # C0k = A00*B0k + A01*B1k for k=0,1,2
    # C1k = A10*B0k + A11*B1k for k=0,1,2
    
    # For each k, C0k and C1k each need a 2x2*2x2 + 2x2*2x2.
    # Using Strassen for each 2x2*2x2: 7 mults each, 14 per k, 42 total for 3 k's.
    # But we can share: the B blocks B0k, B1k are used in both C0k and C1k.
    
    # Actually, the sharing is in the A blocks. Let me think again...
    
    # For fixed k: C0k = A00*B0k + A01*B1k, C1k = A10*B0k + A11*B1k
    # This is like a 4x4 * 4x2 product. Rank-18 for 4x4x4 suggests 
    # 4x4x2 might need ~9-10 mults? No, that's not right.
    
    # Let me just use the straightforward Strassen approach but optimize:
    # For each k (3 column pairs), compute C[:,k] = A * B[:,k]
    # This is a 4x4 * 4x2 matrix product.
    # Using Strassen: split A into 2x2 blocks, B[:,k] into 2x2 blocks.
    # C_top[:,k] = A00*Btop[:,k] + A01*Bbot[:,k]  (2x2 = 2x2*2x2 + 2x2*2x2)
    # C_bot[:,k] = A10*Btop[:,k] + A11*Bbot[:,k]
    
    # Each 2x2*2x2 uses 7 Strassen mults. 4 such products per k, 3 k's = 84.
    # Same as before. To improve, I need cross-k sharing.
    
    # Alternative: treat B as 4x6 and use a 4x4 * 4x6 specific decomposition.
    # Upper bound: 4*4*6/2 = 48 (using some sharing), or use known results.
    
    # Let me try rank 42: for each of the 3 column pairs, use a 4x4*4x2 
    # decomposition of rank 14 (7 Strassen for the sum of two 2x2 products, 
    # but with sharing between the two terms).
    
    # Actually, C[:,k] = A * B[:,k] where B[:,k] is 4x2.
    # Write B[:,k] = [Btop_k; Bbot_k] where Btop_k, Bbot_k are 2x2.
    # C[:,k] = [A00 A01; A10 A11] * [Btop_k; Bbot_k]
    #        = [A00*Btop_k + A01*Bbot_k; A10*Btop_k + A11*Bbot_k]
    
    # The 4x2 result has 8 elements. Minimum rank for 4x4*4x2 is at least 
    # max(4*2, 4*4, 4*2) ... no, tensor rank lower bounds are complex.
    
    # Let me just verify the current rank-84 solution works and try to reduce.
    
    # For now, let me use a cleaner rank-84 construction and verify it.
    
    rank = 84
    U = [[0] * rank for _ in range(n * m)]  # 16 x 84
    V = [[0] * rank for _ in range(m * p)]  # 24 x 84
    W = [[0] * rank for _ in range(p * n)]  # 24 x 84
    
    col = 0
    for i in range(2):  # A row block index (0 or 1)
        for k in range(3):  # C col block index (0, 1, or 2)
            for j in range(2):  # Sum over j: A_ij * B_jk
                # Block A_ij: rows [2*i, 2*i+1], cols [2*j, 2*j+1]
                # Block B_jk: rows [2*j, 2*j+1], cols [2*k, 2*k+1]
                
                # A_ij elements:
                a_idx = 2*i * m + 2*j
                b_idx = 2*i * m + 2*j + 1
                c_idx = (2*i+1) * m + 2*j
                d_idx = (2*i+1) * m + 2*j + 1
                
                # B_jk elements:
                e_idx = 2*j * p + 2*k
                f_idx = 2*j * p + 2*k + 1
                g_idx = (2*j+1) * p + 2*k
                h_idx = (2*j+1) * p + 2*k + 1
                
                # C_ik block: rows [2*i, 2*i+1], cols [2*k, 2*k+1]
                # W index: W[col*n + row, r]
                c11_idx = (2*k) * n + 2*i
                c12_idx = (2*k+1) * n + 2*i
                c21_idx = (2*k) * n + 2*i + 1
                c22_idx = (2*k+1) * n + 2*i + 1
                
                # Strassen 7 multiplications
                # M1 = (a+d)(e+h)
                U[a_idx][col] = 1
                U[d_idx][col] = 1
                V[e_idx][col] = 1
                V[h_idx][col] = 1
                W[c11_idx][col] = 1
                W[c22_idx][col] = 1
                col += 1
                
                # M2 = (c+d)e
                U[c_idx][col] = 1
                U[d_idx][col] = 1
                V[e_idx][col] = 1
                W[c21_idx][col] = 1
                W[c22_idx][col] = -1
                col += 1
                
                # M3 = a(f-h)
                U[a_idx][col] = 1
                V[f_idx][col] = 1
                V[h_idx][col] = -1
                W[c12_idx][col] = 1
                W[c22_idx][col] = 1
                col += 1
                
                # M4 = d(g-e)
                U[d_idx][col] = 1
                V[g_idx][col] = 1
                V[e_idx][col] = -1
                W[c11_idx][col] = 1
                W[c21_idx][col] = 1
                col += 1
                
                # M5 = (a+b)h
                U[a_idx][col] = 1
                U[b_idx][col] = 1
                V[h_idx][col] = 1
                W[c11_idx][col] = -1
                W[c12_idx][col] = 1
                col += 1
                
                # M6 = (c-a)(e+f)
                U[c_idx][col] = 1
                U[a_idx][col] = -1
                V[e_idx][col] = 1
                V[f_idx][col] = 1
                W[c22_idx][col] = 1
                col += 1
                
                # M7 = (b-d)(g+h)
                U[b_idx][col] = 1
                U[d_idx][col] = -1
                V[g_idx][col] = 1
                V[h_idx][col] = 1
                W[c11_idx][col] = 1
                col += 1
    
    # Verify the decomposition
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check: sum_r U[:,r] * V[:,r] * W[:,r] should equal the matmul tensor
    # Tensor T[i,j,k] = 1 if we're computing C[k,i] = sum_j A[i,j]*B[j,k]
    # In W convention: W[k*n+i, r]
    
    # Build expected tensor
    T_expected = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_expected[i*m+j, j*p+k, k*n+i] = 1
    
    # Compute from decomposition
    T_decomp = np.zeros_like(T_expected)
    for r in range(rank):
        T_decomp += np.einsum('a,b,c->abc', U_arr[:,r], V_arr[:,r], W_arr[:,r])
    
    # Check correctness
    if not np.allclose(T_decomp, T_expected, atol=1e-10):
        # If incorrect, fall back to naive
        rank_naive = n * m * p
        U_naive = [[0] * rank_naive for _ in range(n * m)]
        V_naive = [[0] * rank_naive for _ in range(m * p)]
        W_naive = [[0] * rank_naive for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_naive[i*m+j][col] = 1
                    V_naive[j*p+k][col] = 1
                    W_naive[k*n+i][col] = 1
                    col += 1
        return {"U": U_naive, "V": V_naive, "W": W_naive}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
