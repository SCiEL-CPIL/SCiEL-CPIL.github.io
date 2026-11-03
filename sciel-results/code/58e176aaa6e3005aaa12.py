"""Program for 3x4x6 matrix multiplication tensor decomposition."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 4, 6
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Each column c represents a rank-1 tensor term.
    # U is (n*m) x rank, V is (m*p) x rank, W is (p*n) x rank.
    # The decomposition is: T_{i,j,k} = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Note: W index is k*n + i per the task description "W[k*n+i,r]".
    
    # We use the decomposition M_{3,4,6} = 4 * M_{3,2,3} (splitting the middle dimension 4 into two 2s)
    # Wait, M_{3,4,6} involves summing over j=0..3.
    # If we split j into two groups of size 2: j=0,1 and j=2,3.
    # This gives two M_{3,2,6} blocks? No, the indices for U depend on j, V depends on j, W depends on i,k.
    # T_{i,j,k} = A_{i,j} B_{j,k}.
    # Sum over j.
    # Let's just build the terms explicitly using the derived rank-5 decomposition for M_{3,2,3}.
    # Actually, let's stick to the plan:
    # Total rank = 20.
    # We have 4 blocks of M_{3,2,3}? No.
    # M_{3,4,6}: i in 0..2, j in 0..3, k in 0..5.
    # We can view this as 2 blocks of M_{3,2,6} (j=0,1 and j=2,3).
    # Rank(M_{3,2,6})?
    # M_{3,2,6} = M_{3,2,3} + M_{3,2,3} (split k).
    # Rank(M_{3,2,3}) = 5.
    # So Rank(M_{3,2,6}) = 10.
    # Total Rank = 2 * 10 = 20.
    
    # Let's generate the 5 terms for a generic M_{3,2,3} sub-problem.
    # Sub-problem: A is 3x2 (indices i, j_local), B is 2x3 (indices j_local, k_local).
    # Result C is 3x3 (indices i, k_local).
    # We need 5 rank-1 terms.
    
    # Recall the derivation:
    # M_{3,2,3} = M_{3,2,2} + M_{3,2,1}.
    # M_{3,2,1} has rank 3.
    # M_{3,2,2} has rank 2.
    # Total 5.
    
    # Let's explicitly construct the 5 terms for M_{3,2,3}.
    # Let A be 3x2: A_{i,0}, A_{i,1}. B be 2x3: B_{0,k}, B_{1,k}.
    # C_{i,k} = A_{i,0} B_{0,k} + A_{i,1} B_{1,k}.
    
    # Term 1, 2, 3 for M_{3,2,1} part (assuming we treat the second dimension of B as 1? No, M_{3,2,1} means B is 2x1).
    # This decomposition path is tricky to generalize without explicit formulas.
    # Let's use a known explicit rank 5 algorithm for 3x2x3 or just use the schoolbook for the subblocks if 5 is hard to code blindly.
    # Schoolbook for 3x2x3 is 18. 2 * 10 (if M_{3,2,6} is 10) -> 20.
    # Schoolbook for M_{3,2,6} is 36. Total 72. No gain.
    
    # Let's try to implement the specific rank 5 terms for M_{3,2,3}.
    # Based on "Rank(M_{3,2,3}) <= 5":
    # This is equivalent to M_{3,2,2} (rank 2) + M_{3,2,1} (rank 3).
    
    # M_{3,2,1}: C_{i,0} = A_{i,0}B_{0,0} + A_{i,1}B_{1,0}.
    # Rank 3?
    # Usually M_{n,2,1} has rank n.
    # Terms:
    # t1: U=[1,0,0, 0,1,0, 0,0,1]^T (wait U is 3x2=6 entries).
    # Let's define the vector shapes.
    # U_vec: length 6 (i=0..2, j=0..1).
    # V_vec: length 6 (j=0..1, k=0..2).
    # W_vec: length 9 (k=0..2, i=0..2).
    
    # For M_{3,2,1} (k_local=0 only):
    # We need to compute C_{i,0}.
    # C_{i,0} = A_{i,0} B_{0,0} + A_{i,1} B_{1,0}.
    # This is a sum of 2 products. Rank <= 2?
    # No, the tensor T_{i,j,k} for fixed k=0 is T_{i,j,0} = A_{i,j} B_{j,0}.
    # This is a bilinear form in A and B.
    # T = sum_j A_{i,j} B_{j,0}.
    # This is just the vector product.
    # The tensor rank of M_{n, m, 1} is m? Or n?
    # M_{n, m, 1}: C_i = sum_j A_{ij} B_{j1}.
    # This is linear in A and B.
    # T_{i,j,1} = delta_{i,i} delta_{j,j}? No.
    # T_{i,j,k} = 1 if k=1 (or 0) else 0.
    # T_{i,j,0} = 1 for all i,j?
    # No, C_i = sum_j A_{ij} B_{j0}.
    # The tensor is T_{i,j,0} = 1.
    # We need sum_c U_{i,j,c} V_{j,0,c} W_{0,i,c} = 1.
    # If we set W_{0,i,c} = delta_{i, c_mod_n}?
    # Let's check rank of M_{n,m,1}.
    # It is known that R(M_{n,m,1}) = m.
    # Proof:
    # C = A B. B is a column vector.
    # C_i = sum_j A_{ij} B_j.
    # This is equivalent to computing m linear combinations of A?
    # Actually, consider the matrix A. We are computing C = A b.
    # This is a map from (A,b) to C.
    # Rank is m.
    # Construction:
    # For c in 0..m-1:
    # U_{i,j,c} = delta_{j,c}
    # V_{j,k,c} = delta_{j,c} (k is fixed to 0, so V_{j,0,c} = delta_{j,c})
    # W_{k,i,c} = 1 (k=0)
    # Sum_c delta_{j,c} delta_{j,c} * 1 = sum_c delta_{j,c}^2 = 1 if j is in 0..m-1?
    # Wait, sum_c U_{i,j,c} V_{j,k,c} W_{k,i,c}
    # = sum_{c=0}^{m-1} delta_{j,c} * delta_{j,c} * 1
    # = 1 if j is in 0..m-1, else 0.
    # But we want 1 for ALL j?
    # No, the tensor T_{i,j,k} is 1 for all i,j when k=0.
    # If m < number of j's, this doesn't work.
    # But in M_{3,2,1}, m=2. j ranges 0..1.
    # So sum_{c=0}^1 delta_{j,c}^2 = 1 for j=0,1.
    # So yes, rank is m.
    # So M_{3,2,1} has rank 2?
    # My previous thought said 3. Let's re-verify.
    # T_{i,j,0} = 1.
    # We need sum_c U_{ijc} V_{j0c} W_{0ic} = 1.
    # If we use c=0..1:
    # c=0: U_{ij0}=delta_{j0}, V_{j00}=1, W_{0i0}=1. Term: delta_{j0}.
    # c=1: U_{ij1}=delta_{j1}, V_{j01}=1, W_{0i1}=1. Term: delta_{j1}.
    # Sum = 1 for j=0,1.
    # So Rank(M_{3,2,1}) = 2.
    
    # Now M_{3,2,2}.
    # C_{i,k} = sum_{j=0}^1 A_{ij} B_{jk} for k=0,1.
    # Rank(M_{3,2,2})?
    # This is M_{3,2,2} <= M_{2,2,2} + M_{1,2,2}?
    # R(M_{2,2,2}) = 7 (Strassen).
    # R(M_{1,2,2}) = 2?
    # M_{1,2,2}: C_{0,k} = A_{00}B_{0k} + A_{01}B_{1k}.
    # This is a rank 1 matrix times vectors?
    # C = A b^T? No, A is 1x2, B is 2x2. C is 1x2.
    # C_k = sum_j A_j B_{jk}.
    # This is vector product. Rank 2?
    # T_{0,j,k} = 1.
    # Same as above, rank m=2.
    # So R(M_{3,2,2}) <= 7 + 2 = 9?
    # Or better:
    # M_{3,2,2} = M_{3,1,2} + M_{2,1,2}?
    # R(M_{3,1,2}) = 1 (rank 1 matrix multiplication).
    # R(M_{2,1,2}) = 1.
    # Total 2?
    # Let's check.
    # C_{i,k} = A_{i,0}B_{0,k} + A_{i,1}B_{1,k}.
    # Term 1: A_{i,0} B_{0,k}.
    # U1_{i,j,k} = delta_{j,0}. V1 = 1. W1 = 1.
    # Term 2: A_{i,1} B_{1,k}.
    # U2_{i,j,k} = delta_{j,1}. V2 = 1. W2 = 1.
    # Sum = A_{i,0}B_{0,k} + A_{i,1}B_{1,k}.
    # So Rank(M_{3,2,2}) = 2.
    
    # So Rank(M_{3,2,3}) = Rank(M_{3,2,2}) + Rank(M_{3,2,1}) = 2 + 2 = 4.
    # Wait, is M_{3,2,3} = M_{3,2,2} + M_{3,2,1}?
    # Yes, split k.
    # So Rank(M_{3,2,3}) <= 4.
    
    # Then Rank(M_{3,2,6}) = 2 * Rank(M_{3,2,3}) <= 8.
    # Rank(M_{3,4,6}) = 2 * Rank(M_{3,2,6}) <= 16.
    
    # Let's verify Rank(M_{3,1,2}) = 1.
    # C_{i,k} = A_{i,0} B_{0,k}.
    # This is a rank-1 matrix product.
    # T_{i,0,k} = 1.
    # Sum_c U_{i,0,c} V_{0,k,c} W_{k,i,c} = 1.
    # c=0: U=1, V=1, W=1.
    # Yes, rank 1.
    
    # So the total rank is 16.
    # Let's construct the 16 terms.
    
    # Structure:
    # j_global in 0..3.
    # k_global in 0..5.
    
    # We split j_global into two blocks: j1 in 0..1, j2 in 0..1 (mapped to 2,3).
    # We split k_global into three blocks: k1 in 0..1, k2 in 0..1, k3 in 0..1 (mapped to 0,1, 2,3, 4,5).
    # Wait, M_{3,2,6} split k into 2 blocks of size 3?
    # My previous logic: M_{3,2,3} has rank 4.
    # M_{3,2,6} = M_{3,2,3} (k=0..2) + M_{3,2,3} (k=3..5).
    # So 2 blocks of M_{3,2,3}.
    # Each M_{3,2,3} has rank 4.
    # Total 8 terms for one j-block (size 2).
    # We have 2 j-blocks (j=0,1 and j=2,3).
    # Total 16 terms.
    
    # Let's generate the 4 terms for a generic M_{3,2,3} instance.
    # Instance defined by:
    # i in 0..2
    # j_local in 0..1
    # k_local in 0..2
    
    # Terms for M_{3,2,3}:
    # Decomposition: M_{3,2,2} (k=0,1) + M_{3,2,1} (k=2).
    
    # Part 1: M_{3,2,2} (k_local=0,1).
    # Split j_local.
    # Term 1.1: A_{i,0} B_{0, k_local} (for k_local=0,1).
    # This term covers j_local=0.
    # U_{i,j,k,c} = delta_{j,0}.
    # V_{j,k,c} = 1. (V depends on j_local, k_local).
    # Wait, V index is j_local * 3 + k_local? Or just the sub-matrix?
    # In the global V matrix, V_{j_global, k_global}.
    # For this term, V is 1 if j_global corresponds to j_local=0.
    # So V_{j,g,c} = 1 if j_local=0.
    # W_{k,i,c} = 1. (W depends on k_local, i).
    # But this term is only valid for k_local=0,1?
    # The term A_{i,0} B_{0,k} computes C_{i,k} for all k.
    # We are summing these terms.
    # If we just use U=delta_{j,0}, V=1, W=1, it computes A_{i,0} B_{0,k} for ALL k.
    # But we only want it for k=0,1 in the M_{3,2,2} part?
    # No, M_{3,2,2} IS the part where k is 0,1.
    # So we need to ensure this term is 0 for k=2.
    # We can multiply by W which is 0 for k=2?
    # W_{k,i,c}. If we set W_{k,i,c} = delta_{k,0} + delta_{k,1}?
    # Then Sum = delta_{j,0} * 1 * (delta_{k,0}+delta_{k,1}) * A_{i,0} B_{0,k}??
    # No, the term value is U*V*W.
    # We want the sum of terms to equal A_{i,0} B_{0,k} for k=0,1 and 0 for k=2.
    # So for this specific term, W should be 1 if k_local=0 or 1, else 0.
    # And U should be delta_{j_local, 0}.
    # And V should be 1.
    # So Term 1.1:
    # U_{i,j_local, c} = 1 if j_local==0 else 0.
    # V_{j_local, k_local, c} = 1.
    # W_{k_local, i, c} = 1 if k_local < 2 else 0.
    
    # Term 1.2: A_{i,1} B_{1, k_local} (for k_local=0,1).
    # U_{i,j_local, c} = 1 if j_local==1 else 0.
    # V_{j_local, k_local, c} = 1.
    # W_{k_local, i, c} = 1 if k_local < 2 else 0.
    
    # Part 2: M_{3,2,1} (k_local=2).
    # C_{i,2} = A_{i,0} B_{0,2} + A_{i,1} B_{1,2}.
    # Rank 2 terms.
    # Term 2.1: A_{i,0} B_{0,2}.
    # U = delta_{j_local, 0}.
    # V = delta_{k_local, 2}. (So V_{j,k,c} = 1 if k_local==2 else 0).
    # W = 1. (But must be 0 for k!=2? No, V is already 0).
    # Wait, if V is 0 for k!=2, then the term is 0.
    # So W can be 1.
    
    # Term 2.2: A_{i,1} B_{1,2}.
    # U = delta_{j_local, 1}.
    # V = delta_{k_local, 2}.
    # W = 1.
    
    # Let's check if this works.
    # Sum for fixed i, j_global, k_global.
    # Let j_local, k_local be the local indices.
    # Sum = (U1 V1 W1) + (U2 V2 W2) + (U3 V3 W3) + (U4 V4 W4).
    
    # If k_local < 2:
    # Term 1.1: delta_{j,0} * 1 * 1 = delta_{j,0}.
    # Term 1.2: delta_{j,1} * 1 * 1 = delta_{j,1}.
    # Term 2.1: delta_{j,0} * 0 * 1 = 0.
    # Term 2.2: delta_{j,1} * 0 * 1 = 0.
    # Sum = delta_{j,0} + delta_{j,1} = 1.
    # But we want A_{i,j} B_{j,k}.
    # The tensor value is 1 if we are at (i,j,k).
    # The product U*V*W is 1.
    # This corresponds to the coefficient of A_{i,j} B_{j,k}.
    # So yes, if the sum is 1, it means the term contributes 1 * A_{i,j} B_{j,k}.
    # So for k<2, we get A_{i,0}B_{0,k} + A_{i,1}B_{1,k}. Correct.
    
    # If k_local == 2:
    # Term 1.1: delta_{j,0} * 1 * 0 = 0.
    # Term 1.2: delta_{j,1} * 1 * 0 = 0.
    # Term 2.1: delta_{j,0} * 1 * 1 = delta_{j,0}.
    # Term 2.2: delta_{j,1} * 1 * 1 = delta_{j,1}.
    # Sum = delta_{j,0} + delta_{j,1} = 1.
    # So we get A_{i,0}B_{0,2} + A_{i,1}B_{1,2}. Correct.
    
    # So the 4 terms for M_{3,2,3} are:
    # 1. U=delta_{j,0}, V=1, W=(k<2)
    # 2. U=delta_{j,1}, V=1, W=(k<2)
    # 3. U=delta_{j,0}, V=(k==2), W=1
    # 4. U=delta_{j,1}, V=(k==2), W=1
    
    # Now map to global indices.
    # Global j = j_block * 2 + j_local.
    # Global k = k_block * 3 + k_local.
    
    # We have 2 j_blocks (0, 1) and 2 k_blocks (0, 1).
    # Total 4 M_{3,2,3} instances.
    # Each has 4 terms.
    # Total 16 terms.
    
    U = [[0] * 16 for _ in range(n * m)]
    V = [[0] * 16 for _ in range(m * p)]
    W = [[0] * 16 for _ in range(p * n)]
    
    col = 0
    for jb in range(2): # j_block 0, 1
        for kb in range(2): # k_block 0, 1
            # Terms 1-4 for this (jb, kb) instance
            for t in range(4):
                # t=0: U=d_j0, V=1, W=k<2
                # t=1: U=d_j1, V=1, W=k<2
                # t=2: U=d_j0, V=k==2, W=1
                # t=3: U=d_j1, V=k==2, W=1
                
                for i in range(n):
                    for jl in range(2):
                        jg = jb * 2 + jl
                        # U index: i*m + jg
                        if t == 0 or t == 2:
                            if jl == 0:
                                U[i * m + jg][col] = 1
                        else:
                            if jl == 1:
                                U[i * m + jg][col] = 1
                        
                        for kl in range(3):
                            kg = kb * 3 + kl
                            # V index: jg*p + kg
                            # V depends on jl, kl
                            if t == 0 or t == 1:
                                # V = 1
                                V[jg * p + kg][col] = 1
                            else:
                                # V = delta_{kl, 2}
                                if kl == 2:
                                    V[jg * p + kg][col] = 1
                            
                            # W index: kg*n + i
                            # W depends on kl
                            if t == 0 or t == 1:
                                # W = 1 if kl < 2
                                if kl < 2:
                                    W[kg * n + i][col] = 1
                            else:
                                # W = 1
                                W[kg * n + i][col] = 1
                
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
