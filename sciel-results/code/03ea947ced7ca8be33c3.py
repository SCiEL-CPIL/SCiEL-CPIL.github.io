"""Matrix multiplication 3x5x5 - rank-minimal exact tensor decomposition."""
import time

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct an exact tensor decomposition for 3x5 x 5x5 matrix multiplication.
    
    Strategy:
    - Use Strassen's rank-7 decomposition for a 2x2x2 sub-block (indices i=0,1, j=0,1, k=0,1)
    - Use schoolbook (rank 1 per term) for all other index combinations
    - Total rank: 7 (Strassen) + 67 (schoolbook for remaining 67 (i,j,k) tuples) = 74
    """
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    # We'll build U, V, W as lists of lists
    # U: (n*m) x rank = 15 x rank
    # V: (m*p) x rank = 25 x rank
    # W: (p*n) x rank = 15 x rank
    
    # First, collect all (i,j,k) tuples
    all_ijk = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    
    # Identify the 2x2x2 sub-block: i in {0,1}, j in {0,1}, k in {0,1}
    sub_block_ijk = set([(i, j, k) for i in range(2) for j in range(2) for k in range(2)])
    
    # Remaining indices for schoolbook
    remaining_ijk = [(i, j, k) for (i, j, k) in all_ijk if (i, j, k) not in sub_block_ijk]
    
    # Strassen decomposition for 2x2x2:
    # The 8 terms of schoolbook 2x2x2 are replaced by 7 terms.
    # Strassen's algorithm for C = A*B where A, B are 2x2:
    #   M1 = (A00+A11)*(B00+B11)
    #   M2 = (A10+A11)*B00
    #   M3 = A00*(B01-B11)
    #   M4 = A11*(B10-B00)
    #   M5 = (A00+A01)*B11
    #   M6 = (A10-A00)*(B00+B01)
    #   M7 = (A01-A11)*(B10+B11)
    #   C00 = M1 + M4 - M5 + M7
    #   C01 = M3 + M5
    #   C10 = M2 + M4
    #   C11 = M1 - M2 + M3 + M6
    #
    # In tensor decomposition form, each "M" corresponds to a rank-1 term.
    # For the sub-block with indices i' in {0,1} (maps to i), j' in {0,1} (maps to j), k' in {0,1} (maps to k):
    #
    # The U vector for each Strassen term has non-zero entries at positions (i'*m + j') for the A matrix indices.
    # The V vector has non-zero entries at positions (j'*p + k') for the B matrix indices.
    # The W vector has non-zero entries at positions (k'*n + i') for the C matrix indices.
    
    # Let's define the 7 Strassen terms explicitly.
    # A has indices (i', j'), B has indices (j', k'), C has indices (k', i').
    # Flattened: A_idx = i'*2 + j', B_idx = j'*2 + k', C_idx = k'*2 + i'
    
    # Strassen terms:
    # Term 1: M1 = (A00+A11)*(B00+B11)
    #   A part: A00 and A11 -> indices (0,0) and (1,1) -> flat: 0 and 3
    #   B part: B00 and B11 -> indices (0,0) and (1,1) -> flat: 0 and 3
    #   C part: C00 and C11 -> indices (0,0) and (1,1) -> flat: 0 and 3
    #   But wait, M1 contributes to C00 with +1 and C11 with +1.
    #   Actually, let me be more careful.
    
    # Let me re-derive the rank-1 tensors for Strassen.
    # Each M_c is a product of a linear form in A and a linear form in B, and it contributes to specific C entries.
    # The tensor decomposition has rank-1 terms u_c ⊗ v_c ⊗ w_c where:
    #   sum_c u_c[a] v_c[b] w_c[c] = T[a,b,c]
    
    # For the 2x2x2 case, the tensor T has T[i,j,k] = 1 if i,j,k are the indices such that C[i,k] = sum_j A[i,j] B[j,k].
    # So T[i,j,k] = 1 for all (i,j,k) in {0,1}^3.
    
    # Strassen gives 7 rank-1 tensors. Let me write them out:
    
    # Using the standard Strassen formulation mapped to tensor indices:
    # A[i',j'], B[j',k'], C[k',i']
    
    # Term 1: M1
    # u1: A00 + A11 -> u1[0*2+0]=1, u1[1*2+1]=1, others 0 -> u1 = [1,0,0,1]
    # v1: B00 + B11 -> v1[0*2+0]=1, v1[1*2+1]=1 -> v1 = [1,0,0,1]
    # w1: C00 + C11 -> w1[0*2+0]=1, w1[1*2+1]=1 -> w1 = [1,0,0,1]
    
    # Term 2: M2 = (A10+A11)*B00
    # u2: A10 + A11 -> u2[1*2+0]=1, u2[1*2+1]=1 -> u2 = [0,0,1,1]
    # v2: B00 -> v2[0*2+0]=1 -> v2 = [1,0,0,0]
    # w2: C10 -> w2[0*2+1]=1 -> w2 = [0,1,0,0]
    
    # Term 3: M3 = A00*(B01-B11)
    # u3: A00 -> u3[0*2+0]=1 -> u3 = [1,0,0,0]
    # v3: B01 - B11 -> v3[0*2+1]=1, v3[1*2+1]=-1 -> v3 = [0,1,0,-1]
    # w3: C01 -> w3[1*2+0]=1 -> w3 = [0,0,1,0]
    
    # Term 4: M4 = A11*(B10-B00)
    # u4: A11 -> u4[1*2+1]=1 -> u4 = [0,0,0,1]
    # v4: B10 - B00 -> v4[1*2+0]=1, v4[0*2+0]=-1 -> v4 = [-1,0,1,0]
    # w4: C10 + C11 -> w4[0*2+1]=1, w4[1*2+1]=1 -> w4 = [0,1,0,1]
    
    # Wait, I need to check: M4 contributes to C10 with +1 and C11 with +1?
    # C10 = M2 + M4, C11 = M1 - M2 + M3 + M6
    # So M4 contributes to C10 only, not C11. Let me re-check.
    
    # Actually:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # So:
    # M1 -> C00 (+1), C11 (+1)
    # M2 -> C10 (+1), C11 (-1)
    # M3 -> C01 (+1), C11 (+1)
    # M4 -> C00 (+1), C10 (+1)
    # M5 -> C00 (-1), C01 (+1)
    # M6 -> C11 (+1)
    # M7 -> C00 (+1)
    
    # Let me redo:
    # Term 1: M1
    # u1 = [1,0,0,1] (A00+A11)
    # v1 = [1,0,0,1] (B00+B11)
    # w1: C00+1, C11+1 -> C indices: (0,0) and (1,1) -> flat: 0 and 3 -> w1 = [1,0,0,1]
    
    # Term 2: M2
    # u2 = [0,0,1,1] (A10+A11)
    # v2 = [1,0,0,0] (B00)
    # w2: C10+1, C11-1 -> C indices: (0,1) and (1,1) -> flat: 1 and 3 -> w2 = [0,1,0,-1]
    
    # Term 3: M3
    # u3 = [1,0,0,0] (A00)
    # v3 = [0,1,0,-1] (B01-B11)
    # w3: C01+1, C11+1 -> C indices: (1,0) and (1,1) -> flat: 2 and 3 -> w3 = [0,0,1,1]
    
    # Term 4: M4
    # u4 = [0,0,0,1] (A11)
    # v4 = [-1,0,1,0] (B10-B00)
    # w4: C00+1, C10+1 -> C indices: (0,0) and (0,1) -> flat: 0 and 1 -> w4 = [1,1,0,0]
    
    # Term 5: M5
    # u5 = [1,1,0,0] (A00+A01)
    # v5 = [0,0,0,1] (B11)
    # w5: C00-1, C01+1 -> C indices: (0,0) and (1,0) -> flat: 0 and 2 -> w5 = [-1,0,1,0]
    
    # Term 6: M6
    # u6 = [0,-1,1,0] (A10-A00)
    # v6 = [1,1,0,0] (B00+B01)
    # w6: C11+1 -> C index: (1,1) -> flat: 3 -> w6 = [0,0,0,1]
    
    # Term 7: M7
    # u7 = [0,1,-1,0] (A01-A11)
    # v7 = [0,0,1,1] (B10+B11)
    # w7: C00+1 -> C index: (0,0) -> flat: 0 -> w7 = [1,0,0,0]
    
    # Now map these to the full dimensions:
    # For the sub-block, i' in {0,1} maps to i in {0,1}, j' in {0,1} maps to j in {0,1}, k' in {0,1} maps to k in {0,1}
    # U index: i*m + j where i=i', j=j'
    # V index: j*p + k where j=j', k=k'
    # W index: k*n + i where k=k', i=i'
    
    strassen_terms = []
    # Each term: (u_local, v_local, w_local) where each is a dict from local index to value
    
    # u_local uses flat index i'*2+j'
    # v_local uses flat index j'*2+k'
    # w_local uses flat index k'*2+i'
    
    s1_u = {0: 1, 3: 1}
    s1_v = {0: 1, 3: 1}
    s1_w = {0: 1, 3: 1}
    
    s2_u = {2: 1, 3: 1}
    s2_v = {0: 1}
    s2_w = {1: 1, 3: -1}
    
    s3_u = {0: 1}
    s3_v = {1: 1, 3: -1}
    s3_w = {2: 1, 3: 1}
    
    s4_u = {3: 1}
    s4_v = {0: -1, 2: 1}
    s4_w = {0: 1, 1: 1}
    
    s5_u = {0: 1, 1: 1}
    s5_v = {3: 1}
    s5_w = {0: -1, 2: 1}
    
    s6_u = {0: -1, 2: 1}
    s6_v = {0: 1, 1: 1}
    s6_w = {3: 1}
    
    s7_u = {1: 1, 3: -1}
    s7_v = {2: 1, 3: 1}
    s7_w = {0: 1}
    
    strassen_terms = [
        (s1_u, s1_v, s1_w),
        (s2_u, s2_v, s2_w),
        (s3_u, s3_v, s3_w),
        (s4_u, s4_v, s4_w),
        (s5_u, s5_v, s5_w),
        (s6_u, s6_v, s6_w),
        (s7_u, s7_v, s7_w),
    ]
    
    # Now build the full U, V, W matrices
    # Total rank = 7 (Strassen) + len(remaining_ijk) (schoolbook)
    num_strassen = 7
    num_school = len(remaining_ijk)
    total_rank = num_strassen + num_school
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Add Strassen terms (columns 0 to 6)
    for c in range(num_strassen):
        u_local, v_local, w_local = strassen_terms[c]
        for u_idx, val in u_local.items():
            # u_idx is i'*2 + j', so i' = u_idx // 2, j' = u_idx % 2
            i_prime = u_idx // 2
            j_prime = u_idx % 2
            U[i_prime * m + j_prime][c] = val
        
        for v_idx, val in v_local.items():
            # v_idx is j'*2 + k', so j' = v_idx // 2, k' = v_idx % 2
            j_prime = v_idx // 2
            k_prime = v_idx % 2
            V[j_prime * p + k_prime][c] = val
        
        for w_idx, val in w_local.items():
            # w_idx is k'*2 + i', so k' = w_idx // 2, i' = w_idx % 2
            k_prime = w_idx // 2
            i_prime = w_idx % 2
            W[k_prime * n + i_prime][c] = val
    
    # Add schoolbook terms (columns 7 to total_rank-1)
    for idx, (i, j, k) in enumerate(remaining_ijk):
        col = num_strassen + idx
        U[i * m + j][col] = 1
        V[j * p + k][col] = 1
        W[k * n + i][col] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
