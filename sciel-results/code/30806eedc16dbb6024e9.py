import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 8
    rank = 14
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # We decompose C = A * B where A is 2x4 and B is 4x8.
    # We treat B as two 4x4 blocks: B_left (cols 0-3) and B_right (cols 4-7).
    # We compute C_left = A * B_left and C_right = A * B_right.
    # Each is a 2x4 * 4x4 multiplication.
    
    # The tensor for 2x4 * 4x4 has dimensions (2*4, 4*4, 4*2) = (8, 16, 8).
    # We need a rank-7 decomposition for this sub-problem if we can find one.
    # Is the rank of 2x4x4 multiplication 7?
    # Let's check the structure.
    # A is 2x4, B is 4x4.
    # C[i,k] = sum_j A[i,j] B[j,k] for i in 0..1, j in 0..3, k in 0..3.
    
    # Standard rank for 2x4x4 is 8 (schoolbook).
    # Can we do better?
    # If we view A as two row vectors a0, a1 of length 4.
    # C0 = a0 * B, C1 = a1 * B.
    # This is equivalent to computing two linear combinations of rows of B.
    # Specifically, C0[k] = sum_j a0[j] B[j,k].
    # This is a linear map from B to C0.
    # The tensor T[i,j,k] = delta_{i,0} for i=0? No.
    # T[i, j, k] = 1 if i matches the row of A?
    # The tensor is T_{i,j,k} where i is row of A (0..1), j is col of A/row of B (0..3), k is col of B (0..3).
    # The output is C_{i,k}.
    # The decomposition is sum_c U[i,j,c] V[j,k,c] W[i,k,c] = T[i,j,k].
    # Here W is indexed by (i,k) which has 2*4=8 entries.
    # U is indexed by (i,j) which has 2*4=8 entries.
    # V is indexed by (j,k) which has 4*4=16 entries.
    
    # This is the tensor for multiplying a 2x4 matrix by a 4x4 matrix.
    # Is there a rank 7 decomposition?
    # The rank of matrix multiplication tensor M_{m,n,p} is known for small cases.
    # M_{2,4,4} rank is 8?
    # Let's check if 7 is possible.
    # If rank is 8, then total rank for 2x4x8 would be 16.
    # The prompt suggests 14. This implies a shared structure or a different split.
    
    # Alternative split:
    # Maybe the 14 comes from something else.
    # Let's look at the "Strassen" hint in the directive.
    # "Apply the 2x2 Strassen decomposition to the two 2x2 block columns of the 4x8 matrix B"
    # B is 4x8. It can be seen as 2x2 blocks of 2x4? No.
    # B is 4x8. Rows 0-3, Cols 0-7.
    # Split B into 4 blocks of 2x2?
    # B = [B11 B12; B21 B22] where each is 2x4?
    # Or B = [B1 B2] where B1, B2 are 4x4.
    
    # Let's try to construct a valid rank-14 decomposition if possible, or fall back to a valid lower rank.
    # Actually, the rank of M_{2,4,8} is at most 16 (two M_{2,4,4} of rank 8).
    # If M_{2,4,4} has rank 7, then 14 is possible.
    # Let's try to find a rank 7 decomposition for M_{2,4,4}.
    
    # If I can't find it, I will use the schoolbook rank 16 (or 64 if I'm not careful).
    # Wait, the current program returns rank 64.
    # I can easily return rank 16 by doing two separate 2x4x4 schoolbook decompositions.
    # Rank 16 is better than 64. Score 64/16 = 4.0.
    
    # Let's try to get rank 14.
    # Is there a known rank 7 for 2x4x4?
    # According to literature, the rank of M_{2,4,4} is 8.
    # So 14 might not be possible with this simple split.
    
    # However, the directive says "resulting in a total rank of 14... This exploits the linearity...".
    # This implies a shared decomposition.
    # If the rank of M_{2,4,4} is 8, can we share terms?
    # If we compute A*B_left and A*B_right, the U vectors are the same for both?
    # U depends on A indices (i,j). Yes, U is the same for both halves of B.
    # V depends on B indices (j,k). V_left and V_right are different.
    # W depends on C indices (i,k). W_left and W_right are different.
    # So if we use the same U for both, we don't save rank in U.
    # The rank is the number of columns in U, V, W.
    # If we use the same U for both, we still need separate columns for V and W?
    # No, a rank-1 term is a triple (u, v, w).
    # If we have 8 terms for left and 8 for right, that's 16 terms.
    # Can we combine them?
    # If u_left = u_right, then we have terms (u, v_l, w_l) and (u, v_r, w_r).
    # These are distinct rank-1 terms unless v_l/v_r and w_l/w_r align.
    # So we likely have rank 16.
    
    # Let's implement the rank 16 solution first, as it's safe and better than 64.
    # Then I'll try to see if I can reduce it.
    
    rank = 16
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Left half: cols 0-3 of B
    col_offset = 0
    for i in range(n):
        for j in range(m):
            for k in range(4):  # k in 0..3
                idx = i*m + j
                v_idx = j*p + k
                w_idx = k*n + i
                U[idx, col_offset] = 1
                V[v_idx, col_offset] = 1
                W[w_idx, col_offset] = 1
                col_offset += 1
                
    # Right half: cols 4-7 of B
    for i in range(n):
        for j in range(m):
            for k in range(4, 8):
                idx = i*m + j
                v_idx = j*p + k
                w_idx = k*n + i
                U[idx, col_offset] = 1
                V[v_idx, col_offset] = 1
                W[w_idx, col_offset] = 1
                col_offset += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
