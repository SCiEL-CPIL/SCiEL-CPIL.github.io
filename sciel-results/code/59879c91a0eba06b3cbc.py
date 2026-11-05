import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 51
    rank = 51
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # --- Part 1: 4x4x4 using Strassen (Rank 49) ---
    # Strassen 2x2x2 coefficients
    # P_r(A), Q_r(B), W_r(C)
    # Indices for 2x2: i in {0,1}, j in {0,1}, k in {0,1}
    # U index: i*2 + j
    # V index: j*2 + k
    # W index: k*2 + i
    
    # Define the 7 terms for 2x2x2
    # Each term is (U_coeffs, V_coeffs, W_coeffs)
    # U_coeffs is a dict {index: val}
    # V_coeffs is a dict {index: val}
    # W_coeffs is a dict {index: val}
    
    def strassen_2x2():
        terms = []
        # M1
        U1 = {(0,0): 1, (1,1): 1}
        V1 = {(0,0): 1, (1,1): 1}
        W1 = {(0,0): 1, (1,1): 1} # C00 += M1, C11 += M1
        terms.append((U1, V1, W1))
        
        # M2
        U2 = {(1,0): 1, (1,1): 1}
        V2 = {(0,0): 1}
        W2 = {(0,1): 1, (1,1): -1} # C10 += M2, C11 -= M2
        terms.append((U2, V2, W2))
        
        # M3
        U3 = {(0,0): 1}
        V3 = {(1,0): 1, (1,1): -1}
        W3 = {(0,0): 1, (1,0): 1} # C00 += M3, C10 += M3
        terms.append((U3, V3, W3))
        
        # M4
        U4 = {(1,1): 1}
        V4 = {(0,1): 1, (0,0): -1}
        W4 = {(0,0): 1, (1,0): 1} # C00 += M4, C10 += M4
        terms.append((U4, V4, W4))
        
        # M5
        U5 = {(0,0): 1, (0,1): 1}
        V5 = {(1,1): 1}
        W5 = {(0,0): -1, (1,1): 1} # C00 -= M5, C11 += M5
        terms.append((U5, V5, W5))
        
        # M6
        U6 = {(1,0): 1, (0,0): -1}
        V6 = {(0,0): 1, (1,0): 1}
        W6 = {(1,1): 1} # C11 += M6
        terms.append((U6, V6, W6))
        
        # M7
        U7 = {(0,1): 1, (1,1): -1}
        V7 = {(1,1): 1, (0,1): 1}
        W7 = {(0,0): 1, (1,1): 1} # C00 += M7, C11 += M7
        terms.append((U7, V7, W7))
        
        return terms

    s2_terms = strassen_2x2()
    
    # Outer Strassen for 4x4x4
    # A blocks: A00 (0:2, 0:2), A01 (0:2, 2:4), A10 (2:4, 0:2), A11 (2:4, 2:4)
    # B blocks: B00, B01, B10, B11
    
    # We need to define the outer linear forms on the blocks.
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B11+B01)
    
    # For each outer term r, we have a linear combination of blocks for A and B.
    # Let's represent the block coefficients.
    # A_blocks: 0:A00, 1:A01, 2:A10, 3:A11
    # B_blocks: 0:B00, 1:B01, 2:B10, 3:B11
    
    outer_A = [
        {0: 1, 3: 1},      # M1
        {2: 1, 3: 1},      # M2
        {0: 1},            # M3
        {3: 1},            # M4
        {0: 1, 1: 1},      # M5
        {2: 1, 0: -1},     # M6
        {1: 1, 3: -1}      # M7
    ]
    
    outer_B = [
        {0: 1, 3: 1},      # M1
        {0: 1},            # M2
        {2: 1, 3: -1},     # M3
        {1: 1, 0: -1},     # M4
        {3: 1},            # M5
        {0: 1, 2: 1},      # M6
        {3: 1, 1: 1}       # M7
    ]
    
    # W coefficients for outer terms (mapping to 4x4 output blocks)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    # Block indices: 0:C00, 1:C01, 2:C10, 3:C11
    
    outer_W = [
        {0: 1, 3: 1},      # M1
        {2: 1, 3: -1},     # M2
        {1: 1, 3: 1},      # M3
        {0: 1, 2: 1},      # M4
        {0: -1, 1: 1},     # M5
        {3: 1},            # M6
        {0: 1, 3: 1}       # M7
    ]
    
    # Now iterate
    col_idx = 0
    for r in range(7):
        A_coeffs = outer_A[r]
        B_coeffs = outer_B[r]
        W_coeffs_outer = outer_W[r]
        
        # For this outer term, we perform a 2x2 Strassen multiplication
        # The "matrix" A' is formed by combining the blocks.
        # But Strassen operates on the 4 entries of the 2x2 block matrix.
        # We need to map the inner Strassen terms to the actual 4x4 indices.
        
        for s in range(7):
            U_inner, V_inner, W_inner = s2_terms[s]
            
            # U_inner has keys (i, j) for the 2x2 "block matrix"
            # These correspond to the 4 blocks of A.
            # (0,0) -> Block 0 (A00)
            # (0,1) -> Block 1 (A01)
            # (1,0) -> Block 2 (A10)
            # (1,1) -> Block 3 (A11)
            
            # We need to construct the final U vector for this term.
            # U_final[i, j] for i,j in 0..3
            
            # For each block index b in U_inner:
            for (bi, bj), coeff in U_inner.items():
                block_idx = bi * 2 + bj
                # The coefficient for this block in the outer combination is A_coeffs.get(block_idx, 0)
                outer_coeff = A_coeffs.get(block_idx, 0)
                total_coeff = coeff * outer_coeff
                if total_coeff == 0:
                    continue
                
                # This block corresponds to rows 2*bi : 2*bi+2 and cols 2*bj : 2*bj+2
                for di in range(2):
                    for dj in range(2):
                        i = 2 * bi + di
                        j = 2 * bj + dj
                        U[i * m + j][col_idx] += total_coeff
            
            # Similarly for V
            for (bj, bk), coeff in V_inner.items():
                block_idx = bj * 2 + bk
                outer_coeff = B_coeffs.get(block_idx, 0)
                total_coeff = coeff * outer_coeff
                if total_coeff == 0:
                    continue
                
                for dj in range(2):
                    for dk in range(2):
                        j = 2 * bj + dj
                        k = 2 * bk + dk
                        V[j * p + k][col_idx] += total_coeff
            
            # Similarly for W
            # W_inner keys (k, i) for the 2x2 block output
            for (bk, bi), coeff in W_inner.items():
                block_idx = bk * 2 + bi
                outer_coeff = W_coeffs_outer.get(block_idx, 0)
                total_coeff = coeff * outer_coeff
                if total_coeff == 0:
                    continue
                
                for dk in range(2):
                    for di in range(2):
                        k = 2 * bk + dk
                        i = 2 * bi + di
                        W[k * n + i][col_idx] += total_coeff
            
            col_idx += 1
    
    # --- Part 2: 4x4x1 (Rank 1) ---
    # i in 0..3, j in 0..3, k=4
    # Term: U[i, j]=1, V[j, 4]=1, W[4, i]=1
    # This sums to 1 for all i, j.
    # Contribution to C[i, 4] is A[i, j] B[j, 4].
    # Sum over j gives correct result.
    # Wait, if U[i, j]=1 for all j, then P(A) = sum_j A[i, j].
    # Q(B) = sum_j B[j, 4].
    # Product is (sum A) (sum B).
    # This is NOT sum A[i, j] B[j, 4].
    # So Rank 1 is NOT sufficient for 4x4x1 if we use scalar bilinear forms.
    # We need to preserve the index j.
    # So we need separate terms for each j?
    # Or can we use a different structure?
    # The tensor T[i, j, k] = 1 for k=4.
    # This is a matrix of 1s in the i-j plane for each k? No, k is fixed.
    # It's a slice.
    # The rank of the tensor with T[i, j, 4] = 1 is 4?
    # T = sum_j e_i^T \otimes e_j \otimes e_4?
    # No, T[i, j, 4] = 1.
    # This can be written as sum_j (e_i \otimes e_j \otimes e_4)?
    # If we sum over j, we get 1 for all i.
    # So yes, we need 4 terms (one for each j).
    # Rank 4.
    
    # Let's check 4x1x5.
    # T[i, 4, k] = 1.
    # Sum over i?
    # We need 5 terms (one for each k)?
    # Or 4 terms (one for each i)?
    # If we use 4 terms, indexed by i:
    # Term i: U[i, 4]=1, V[4, k]=1 (for all k?), W[k, i]=1.
    # If V[4, k]=1 for all k, then Q(B) = sum_k B[4, k].
    # Then C[i, k] gets sum_k B[4, k] for all k.
    # This is wrong. We need B[4, k].
    # So we need to isolate k.
    # So we need 5 terms, indexed by k?
    # Term k: U[i, 4]=1 (for all i), V[4, k]=1, W[k, i]=1 (for all i).
    # If U[i, 4]=1 for all i, P(A) = sum_i A[i, 4].
    # Then C[i, k] gets sum_i A[i, 4] for all i.
    # Wrong.
    
    # So we need to isolate both i and k?
    # Then we need 4*5 = 20 terms?
    # That's schoolbook.
    
    # Is there a better way?
    # The tensor T[i, j, k] with T[i, 4, k] = 1.
    # This is a rank-1 matrix in the i-k plane?
    # The slice is a matrix of 1s.
    # The rank of the tensor is the rank of the matrix?
    # No, tensor rank.
    # For a tensor with only one non-zero slice, the rank is the rank of that slice matrix.
    # The slice is a 4x5 matrix of 1s.
    # The rank of this matrix is 1.
    # So the tensor rank is 1?
    # Let's verify.
    # T = u \otimes v \otimes w?
    # T[i, 4, k] = u_i v_4 w_k.
    # We want this to be 1 for all i, k.
    # So u_i = 1, v_4 = 1, w_k = 1.
    # Then T[i, 4, k] = 1.
    # And T[i, j, k] = 0 for j != 4.
    # So yes, Rank 1 is sufficient!
    
    # Why did I think it wasn't?
    # Because I was thinking of the bilinear form P(A)Q(B).
    # If P(A) = sum_i A[i, 4] and Q(B) = sum_k B[4, k],
    # The product is a scalar.
    # But W[k, i] can distribute it?
    # W[k, i] = 1 for all k, i.
    # Then C[i, k] += 1 * (sum A) * (sum B).
    # This is not A[i, 4] B[4, k].
    
    # The issue is that the bilinear form must match the specific entry.
    # The tensor decomposition T = sum U \otimes V \otimes W implies:
    # C[i, k] = sum_r W[k, i, r] P_r(A) Q_r(B).
    # If T[i, 4, k] = 1, it means the coefficient of A[i, 4] B[4, k] in C[i, k] is 1.
    # So we need a term where P_r(A) contains A[i, 4] and Q_r(B) contains B[4, k].
    # If P_r(A) = A[i, 4], then r must depend on i.
    # If Q_r(B) = B[4, k], then r must depend on k.
    # So r must depend on (i, k).
    # So we need 20 terms.
    
    # UNLESS... we can use a different decomposition.
    # But the task requires integer/half-integer entries.
    # And the structure is fixed.
    
    # So 4x4x1 is Rank 16?
    # T[i, j, 4] = 1.
    # Coefficient of A[i, j] B[j, 4] in C[i, 4] is 1.
    # P_r(A) must contain A[i, j].
    # Q_r(B) must contain B[j, 4].
    # So r depends on (i, j).
    # 16 terms.
    
    # So total rank is 49 + 16 + 20 = 85.
    
    # This is still better than 100.
    
    # Let's implement the 16 and 20 terms.
    
    # 4x4x1: i in 0..3, j in 0..3, k=4
    for i in range(4):
        for j in range(4):
            U[i * m + j][col_idx] = 1
            V[j * p + 4][col_idx] = 1
            W[4 * n + i][col_idx] = 1
            col_idx += 1
            
    # 4x1x5: i in 0..3, j=4, k in 0..4
    for i in range(4):
        for k in range(5):
            U[i * m + 4][col_idx] = 1
            V[4 * p + k][col_idx] = 1
            W[k * n + i][col_idx] = 1
            col_idx += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
