"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 5, 6
    
    # We'll build a decomposition by splitting the middle dimension
    # m = 5 = 2 + 3
    # For A (2x5) @ B (5x6) -> C (2x6)
    # C = A[:, :2] @ B[:2, :] + A[:, 2:] @ B[2:, :]
    # 
    # Block 1: (2x2) @ (2x6) -> 2x6
    # Block 2: (2x3) @ (3x6) -> 2x6
    #
    # For 2x2 @ 2x6: split 6 into 3 blocks of 2x2, each rank 4, total 12
    # For 2x3 @ 3x6: we can use a decomposition
    #   Let X = A[:,2:] (2x3), Y = B[2:,:] (3x6)
    #   Z = X @ Y (2x6)
    #   We can write this with rank 6 (schoolbook) or try to do better
    #   Actually 2x3x6: lower bound is max(6, 18, 12) = 18? No.
    #   The tensor rank for 2x3x6 is at least max(2*3, 3*6, 2*6) = max(6,18,12)=18
    #   But that's not right either. The rank is at least the max of the three dimensions' products in a certain sense.
    #   
    # Let me just use schoolbook for the sub-blocks but combine smartly.
    # 
    # Alternative: just use a direct rank-30 decomposition
    # The key insight: we can share structure.
    
    # Let me try a different approach: use the formula
    # C[i,k] = sum_j A[i,j] B[j,k]
    # We want to write this as sum_r U[i,r] V[r,k] where U depends on A and V on B
    # Actually the decomposition is: C = sum_r (U_r . A) * (V_r . B) where . denotes appropriate contraction
    # 
    # More precisely: C[i,k] = sum_r U[i, r, :] contracted with A gives a scalar, times V[:, r, k] contracted with B
    # 
    # The standard form: C[i,k] = sum_r (sum_j U[i,j,r] A[i,j]) * (sum_l V[l,r,k] B[l,k])
    # Wait, that's not right either.
    #
    # The correct form for matrix multiplication tensor:
    # C[i,k] = sum_j A[i,j] B[j,k] = sum_r U[i,r] * (A contracted) * V[r,k] * (B contracted)
    # 
    # Actually the bilinear form is:
    # C[i,k] = sum_r (sum_j U[i,j,r] A[i,j]) * (sum_l V[l,r,k] B[l,k])
    # This is wrong because A[i,j] appears in multiple terms.
    #
    # The correct decomposition: we have rank-r tensors such that
    # T[i,j,k,l] = sum_r U[i,j,r] * V[k,l,r]
    # where T[i,j,k,l] = delta_{j,k} (since C[i,l] = sum_j A[i,j] B[j,l])
    #
    # So we need: delta_{j,k} = sum_r U[i,j,r] V[k,l,r] ... no that's not right.
    #
    # Let me be very careful. The matrix multiplication tensor M has:
    # M[i, j, k, l] = 1 if j == k, else 0
    # where i indexes rows of A (0..n-1), j indexes cols of A/rows of B (0..m-1),
    # k indexes cols of B (0..p-1), l... wait.
    #
    # Standard: A is n x m, B is m x p, C = A @ B is n x p.
    # C[i, l] = sum_j A[i, j] B[j, l]
    # The tensor is T[i, j, l] with the multiplication being bilinear in A and B.
    # T[i, j, l] = delta_{j, ?} ... 
    #
    # The bilinear map: (A, B) -> C where C[i,l] = sum_j A[i,j] B[j,l]
    # This can be written as: C[i,l] = sum_r U[i, r] * (A_part) * V[r, l] * (B_part)
    # More precisely: C[i,l] = sum_r (sum_j U[i,j,r] A[i,j]) * (sum_s V[s,r,l] B[s,l])
    # But this double-counts. The correct form is:
    # C[i,l] = sum_r U[i, r, :] dot A_row_i  times  V[:, r, l] dot B_col_l
    # where U[i, r, :] is a vector in R^m and V[:, r, l] is a vector in R^m.
    # So: C[i,l] = sum_r (sum_j U[i,j,r] A[i,j]) * (sum_s V[s,r,l] B[s,l])
    # And we need: sum_r U[i,j,r] V[s,r,l] = delta_{j,s} for all i, l
    # (the i and l just index which "copy" but the constraint is the same for all i,l)
    #
    # So the problem reduces to: find rank-r decomposition of the identity-like tensor
    # delta_{j,s} = sum_r U[i,j,r] V[s,r,l] for all i, l
    # Since the RHS doesn't depend on i or l in a meaningful way (U and V can depend on i and l),
    # we need: for each i, l: sum_r U[i,j,r] V[s,r,l] = delta_{j,s}
    #
    # This means: the matrices M_{i,l} = U_i @ V_l^T (where U_i is m x r, V_l is m x r)
    # must equal I_m for all i, l.
    #
    # So we need: for all i in [0,n), l in [0,p): U_i @ V_l^T = I_m
    # where U_i is m x r and V_l is m x r.
    #
    # This is a much cleaner formulation! We need n*p pairs of matrices (U_i, V_l) such that
    # U_i @ V_l^T = I_m for all i, l.
    #
    # For m=5, we need 2*6=12 pairs of 5x5 products equaling I_5.
    # The rank r must be at least 5 (since U_i V_l^T = I_5 requires rank at least 5).
    # But we need ALL 12 pairs to work simultaneously.
    #
    # If we set U_i = U for all i and V_l = V for all l, then we need U V^T = I_5,
    # which requires r >= 5 and U, V to be 5x5 with U V^T = I_5. Then rank = 5.
    # But wait, that would give rank 5 total, which is impossible since we need to encode
    # different i and l.
    
    # I think I'm overcomplicating this. Let me go back to basics.
    #
    # The standard tensor decomposition for matrix multiplication:
    # The multiplication tensor for n x m * m x p -> n x p has rank r if there exist
    # r terms such that for all i, j, k:
    # delta_{j,k} = sum_{r=1}^R u[i,j,r] * v[k,r]  ... no.
    #
    # Let me look at this from the evaluator's perspective. The format is:
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # And the decomposition is: T = sum_r U[:,r] outer V[:,r] outer W[:,r]
    # where T is the 3-way tensor of shape (n*m, m*p, p*n)
    #
    # The tensor T[a, b, c] where a = i*m+j, b = j'*p+k, c = k'*n+i'
    # T[a, b, c] = 1 if j == j' and i == i' and k == k', else 0
    # Wait, that's not right either.
    #
    # For matrix multiplication: C[i,k] = sum_j A[i,j] B[j,k]
    # The bilinear form: C[i,k] = sum_{j} A[i,j] B[j,k]
    # As a tensor: T[i, j, k] where the first index is (i,j) for A, second is (j,k) for B,
    # and the output is (i,k).
    #
    # The tensor T has shape (n*m) x (m*p) x (n*p) and:
    # T[i*m+j, j'*p+k, k'*n+i'] = delta_{j,j'} * delta_{i,i'} * delta_{k,k'}
    #
    # So T is 1 exactly when j=j', i=i', k=k', which means a=i*m+j, b=j*p+k, c=k*n+i
    # with the same i, j, k.
    #
    # We need: T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
    #
    # For the schoolbook: rank = n*m*p, with one term for each (i,j,k):
    # U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1
    #
    # Now, can we do better? Yes, if we can find a smaller rank decomposition.
    #
    # For 2x5x6, let me try to find a decomposition with rank < 60.
    #
    # Strategy: use block structure. Split m=5 into 2+3.
    # Then A @ B = A1 @ B1 + A2 @ B2 where A1 is 2x2, B1 is 2x6, A2 is 2x3, B2 is 3x6.
    #
    # The 2x2 @ 2x6 part: this is like multiplying a 2x2 by a 2x6.
    # We can write 2x6 as three 2x2 blocks. Each 2x2 @ 2x2 has rank 4 (schoolbook) or 7 (Strassen, worse).
    # So 2x2 @ 2x6 has rank 12 with schoolbook.
    #
    # The 2x3 @ 3x6 part: schoolbook rank is 2*3*6 = 36. Can we do better?
    # 
    # Total schoolbook for this split: 12 + 36 = 48. Better than 60!
    #
    # Can we do even better? Let me try m=5 = 1+4:
    # 2x1 @ 1x6: rank 2*1*6 = 12
    # 2x4 @ 4x6: rank 2*4*6 = 48
    # Total: 60. No improvement.
    #
    # m=5 = 3+2:
    # 2x3 @ 3x6: rank 36
    # 2x2 @ 2x6: rank 12
    # Total: 48. Same.
    #
    # So splitting m gives rank 48. Can we do better with a different approach?
    #
    # What if we also split n or p? n=2 is already small. p=6 = 3+3:
    # For 2x5 @ 5x3: rank 2*5*3 = 30
    # For 2x5 @ 5x3: rank 30
    # Total: 60. No help.
    #
    # What about splitting both m and p?
    # m=5=2+3, p=6=3+3:
    # A1(2x2) @ B1(2x3): rank 12
    # A1(2x2) @ B2(2x3): rank 12
    # A2(2x3) @ B1(3x3): rank 18
    # A2(2x3) @ B2(3x3): rank 18
    # Total: 60. Worse.
    #
    # Let me try m=5=2+3, p=6=2+4:
    # A1(2x2) @ B1(2x2): rank 4
    # A1(2x2) @ B2(2x4): rank 16
    # A2(2x3) @ B1(3x2): rank 12
    # A2(2x3) @ B2(3x4): rank 24
    # Total: 56. Better than 48? No, 56 > 48.
    #
    # So the best simple block decomposition gives rank 48.
    #
    # Can we share terms between blocks? The key insight is that the U, V, W matrices
    # can have overlapping structure.
    #
    # Let me just implement the rank-48 decomposition and see if it's valid.
    
    n, m, p = 2, 5, 6
    # Split m into m1=2, m2=3
    m1, m2 = 2, 3
    
    # Block 1: A[:, :m1] @ B[:m1, :]  (2x2 @ 2x6)
    # Block 2: A[:, m1:] @ B[m1:, :]  (2x3 @ 3x6)
    
    rank1 = n * m1 * p  # 2*2*6 = 24... wait that's schoolbook for the sub-block
    # Actually for 2x2 @ 2x6: the schoolbook rank is 2*2*6 = 24? No.
    # For n x m @ m x p, schoolbook rank is n*m*p.
    # 2x2 @ 2x6: n=2, m=2, p=6, rank = 2*2*6 = 24. That's not right.
    # 
    # Wait, 2x2 @ 2x6 means A is 2x2, B is 2x6, C is 2x6.
    # C[i,k] = sum_j A[i,j] B[j,k] for j in 0..1
    # Schoolbook: 2*2*6 = 24 terms. But that seems high.
    # Actually yes, for each (i,j,k) triple, one term. i in 0..1, j in 0..1, k in 0..5.
    # That's 2*2*6 = 24 terms.
    #
    # Hmm, but earlier I said 12. Let me recheck.
    # 2x2 @ 2x2: i in 0..1, j in 0..1, k in 0..1: 8 terms. Rank 8 schoolbook.
    # But Strassen does 2x2 @ 2x2 in rank 7.
    # 2x2 @ 2x6: 2*2*6 = 24 schoolbook terms.
    #
    # So my earlier calculation was wrong. Let me redo.
    #
    # m=5=2+3:
    # Block 1: 2x2 @ 2x6: rank 2*2*6 = 24
    # Block 2: 2x3 @ 3x6: rank 2*3*6 = 36
    # Total: 60. Same as original!
    #
    # So simple splitting doesn't help because the schoolbook rank is multiplicative.
    #
    # The only way to beat 60 is to use a non-schoolbook decomposition for the sub-blocks
    # or to share terms across blocks.
    #
    # For 2x2 @ 2x2, Strassen gives rank 7 instead of 8.
    # So 2x2 @ 2x6 using Strassen on each 2x2 block: 3 * 7 = 21 instead of 24.
    # 
    # For 2x3 @ 3x6: can we do better than 36?
    # 2x3 @ 3x6: this is less standard. Let me think.
    # We can write 3x6 as 3x2 + 3x2 + 3x2, so 2x3 @ 3x2 three times.
    # 2x3 @ 3x2: schoolbook rank 2*3*2 = 12.
    # Can we do 2x3 @ 3x2 in less than 12? The lower bound is max(2*3, 3*2, 2*2) = max(6,6,4) = 6.
    # But achieving rank 6 for 2x3x2 is unlikely.
    #
    # This is getting complicated. Let me just implement the rank-60 schoolbook for now
    # and see if I can find any improvement.
    
    # Actually, let me try a completely different approach.
    # The rank of the 2x5x6 matrix multiplication tensor.
    # Known results: rank of n x m x p tensor.
    # For 2 x m x p, the rank is known to be 2m + 2p - 2 for certain cases? No.
    #
    # Let me just use the schoolbook decomposition. It's guaranteed correct.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
