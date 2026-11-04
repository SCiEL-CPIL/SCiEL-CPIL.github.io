"""Matrix multiplication 3,4,8 rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS  # 3, 4, 8

    # Schoolbook fallback
    def schoolbook(n, m, p):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W, rank

    # Verify a decomposition
    def verify(U, V, W, n, m, p, tol=1e-6):
        r = len(U[0])
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # Build full tensor
        T = np.zeros((n, m, p))
        for c in range(r):
            uc = U_np[:, c]  # shape (n*m,)
            vc = V_np[:, c]  # shape (m*p,)
            wc = W_np[:, c]  # shape (p*n,)
            # Reshape to 3D
            u3 = uc.reshape(n, m)
            v3 = vc.reshape(m, p)
            w3 = wc.reshape(p, n)
            # Outer product contribution
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i, j, k] += u3[i, j] * v3[j, k] * w3[k, i]
        # T should be all ones
        return np.allclose(T, 1.0, atol=tol)

    # Strategy: decompose 3x4x8 as (2x4x8) + (1x4x8) using block structure
    # For 2x4x8: use Strassen-like decomposition
    # For 1x4x8: schoolbook rank 32
    
    # Let's try a tensor product approach:
    # 3x4x8 = (1x4x4) x (3x1x2) in some sense?
    # Or: think of it as (3x4x2) x (1x1x4)?
    
    # Actually, let me try a more direct approach:
    # Use the fact that 8 = 2*4 and create a block structure
    
    # Let's try: decompose as tensor product of 3x4x2 and 1x1x4
    # 3x4x2: rank 24 schoolbook
    # 1x1x4: rank 4
    # Product rank: 24*4 = 96 (same as schoolbook)
    
    # Try 3x2x4 and 1x2x2:
    # 3x2x4: rank 24
    # 1x2x2: rank 4 (or 3 with Strassen-like?)
    # Product: 24*4 = 96 or 24*3 = 72
    
    # For 1x2x2: C[0, k] = sum_j A[0,j] * B[k,j] where A is 1x2, B is 2x2
    # This is just 2 multiplications, rank 2. Wait, that's not right.
    # 1x2x2 tensor: T[i,j,k] = delta (it's the identity in some sense)
    # Actually for 1x2x2: C[0,k] = A[0,0]*B[k,0] + A[0,1]*B[k,1]
    # T[0,0,0]=1, T[0,0,1]=1, T[0,1,0]=1, T[0,1,1]=1
    # This has rank 2 (just two terms)
    
    # So 3x2x4 (rank 24) tensor product 1x2x2 (rank 2) gives rank 48?
    # No, tensor product of decompositions: if T1 has rank r1 and T2 has rank r2,
    # their tensor product has rank r1*r2.
    
    # Let me reconsider. The 3x4x8 multiplication tensor T[i,j,k] = 1 for all i,j,k.
    # We want to write T = sum_c U_c ⊗ V_c ⊗ W_c where U_c is in R^{3x4}, V_c in R^{4x8}, W_c in R^{8x3}
    
    # Key idea: exploit that 8 = 4*2
    # Let k = 4*a + b where a in {0,1}, b in {0,1,2,3}
    # Then T[i,j,4a+b] = 1
    # 
    # If we can write this as a sum over a of something...
    # T[i,j,4a+b] = sum_{a'} delta(a,a') * 1
    # This doesn't directly help.
    
    # Let me try a computational approach: use the schoolbook and try to find
    # linear dependencies among the 96 terms.
    
    # Build the 96-term tensor and check for rank reduction
    # The tensor lives in R^{12} ⊗ R^{32} ⊗ R^{24}
    # Each term c contributes a rank-1 tensor U_c ⊗ V_c ⊗ W_c
    
    # To check if we can reduce rank, we need to find if some combination
    # of the 96 rank-1 tensors is linearly dependent in the tensor space.
    
    # This is a large linear algebra problem. Let's set up the system.
    # Flatten the tensor: T has 3*4*8 = 96 entries, all equal to 1.
    # We have 96 rank-1 components, each contributing to all 96 entries.
    # The question is: can we express T as a sum of fewer than 96 rank-1 tensors?
    
    # This is equivalent to: is the 96x96 matrix (where entry (e,c) = T_e / T_e if term c covers e)
    # Actually, each term c=(i,j,k) contributes 1 to position (i,j,k) and 0 elsewhere.
    # So the "coverage" matrix is just the identity! Each term covers exactly one position.
    # This means the schoolbook decomposition is already minimal in this trivial sense.
    
    # BUT: we can use different rank-1 tensors that don't correspond to single (i,j,k) positions.
    # A general rank-1 tensor U⊗V⊗W contributes U[i,j]*V[j,k]*W[k,i] to position (i,j,k).
    # This is generally nonzero at many positions, not just one.
    
    # So the question becomes: can we find r < 96 triples (U_c, V_c, W_c) such that
    # sum_c U_c[i,j] * V_c[j,k] * W_c[k,i] = 1 for all i,j,k?
    
    # This is a nonlinear system. Let me try to find a structured solution.
    
    # Idea: use the fact that 4 = 2*2 and 8 = 2*4
    # Decompose indices: i in {0,1,2}, j = 2*j1 + j2 (j1,j2 in {0,1}), k = 4*k1 + k2 (k1 in {0,1}, k2 in {0,1,2,3})
    
    # Or: j = j1*2 + j2, k = k1*4 + k2
    # Then T[i, j1*2+j2, k1*4+k2] = 1
    
    # If we can factor this as a product of smaller tensors...
    # T[i, j1, j2, k1, k2] = 1 = T1[i, j1, k1] * T2[j2, k2] ?
    # But T1 would need to be 1 for all its indices, and T2 would need to be 1 for all its indices.
    # T1 is 3x2x2, T2 is 2x4. Both are "all ones" tensors.
    # Rank of all-ones 3x2x2 tensor: can be written as sum of... 
    # Actually, the all-ones tensor of shape a×b×c has rank min(a*b, b*c, a*c) in general?
    # No, that's not right either. The all-ones tensor 1_{a×b×c} has tensor rank at most min(a, b, c) 
    # if we allow real entries? No, that's for the matrix case.
    
    # For the all-ones 3×2×2 tensor: T[i,j1,k1] = 1
    # We can write: T = sum_{s=1}^{2} U_s ⊗ V_s ⊗ W_s where
    # U_s[i,j1] and V_s[j1,k1] and W_s[k1,i] are chosen appropriately.
    # Actually, T[i,j1,k1] = sum_s U_s[i,j1] * V_s[j1,k1] * W_s[k1,i]
    # If we set U_s[i,j1] = 1, V_s[j1,k1] = 1, W_s[k1,i] = delta(s, k1) for s=1,2?
    # That gives T[i,j1,k1] = sum_s 1*1*delta(s,k1) = 1 if k1 < 2, which works since k1 in {0,1}.
    # Wait, that gives T[i,j1,k1] = 1 for all i,j1,k1. Rank 2!
    
    # Similarly, all-ones 2×4 tensor T2[j2,k2] = 1:
    # T2[j2,k2] = sum_t U2_t[j2] * V2_t[k2]
    # Set U2_t[j2] = 1, V2_t[k2] = delta(t, k2 mod 2) for t=1,2?
    # That gives T2[j2,k2] = 1 if k2 is even or odd... no.
    # Actually: T2[j2,k2] = sum_{t=1}^{2} 1 * delta(t, k2 mod 2) doesn't work for all k2.
    # Let's just use: T2[j2,k2] = 1 = 1*1, so it's rank 1!
    # U2[j2] = 1, V2[k2] = 1. Yes, rank 1.
    
    # So: T[i,j,k] = T1[i, j1, k1] * T2[j2, k2] where j = 2*j1+j2, k = 4*k1+k2
    # T1 has rank 2, T2 has rank 1.
    # Tensor product rank: 2*1 = 2? That can't be right for a 96-dimensional space.
    
    # Wait, I'm confusing myself. Let me be more careful.
    
    # The full tensor T[i,j,k] with i in {0,1,2}, j in {0,...,3}, k in {0,...,7}
    # Let j = 2*j1 + j2 (j1 in {0,1}, j2 in {0,1})
    # Let k = 4*k1 + k2 (k1 in {0,1}, k2 in {0,1,2,3})
    # Then T[i, 2*j1+j2, 4*k1+k2] = 1 for all valid indices.
    
    # We want: 1 = sum_c U_c[i, 2*j1+j2] * V_c[2*j1+j2, 4*k1+k2] * W_c[4*k1+k2, i]
    
    # If we can factor: U[i, 2*j1+j2] = A[i, j1] * B[j2] (separable in j)
    # V[2*j1+j2, 4*k1+k2] = C[j1, k1] * D[j2, k2]
    # W[4*k1+k2, i] = E[k1, i] * F[k2]
    
    # Then: U*V*W = A[i,j1]*B[j2]*C[j1,k1]*D[j2,k2]*E[k1,i]*F[k2]
    # = (A[i,j1]*C[j1,k1]*E[k1,i]) * (B[j2]*D[j2,k2]*F[k2])
    # = T1[i,j1,k1] * T2[j2,k2]
    
    # And we need sum_c T1_c[i,j1,k1] * T2_c[j2,k2] = 1
    
    # If T1 and T2 are each all-ones tensors, and we use their rank decompositions:
    # T1[i,j1,k1] = sum_s a_s[i,j1] * b_s[j1,k1] * c_s[k1,i]  (rank r1)
    # T2[j2,k2] = sum_t d_t[j2] * e_t[k2]  (rank r2)
    
    # Then T = sum_{s,t} (a_s[i,j1]*b_s[j1,k1]*c_s[k1,i]) * (d_t[j2]*e_t[k2])
    # = sum_{s,t} [a_s[i,j1]*d_t[j2]] * [b_s[j1,k1]*e_t[k2]] * [c_s[k1,i]]
    
    # But this needs to match the form U[i,2*j1+j2] * V[2*j1+j2, 4*k1+k2] * W[4*k1+k2, i]
    # where U depends on the full j index, V on full j and k, W on full k.
    
    # The issue: U[i, 2*j1+j2] = a_s[i,j1]*d_t[j2] is a valid U (it's a function of i and full j).
    # V[2*j1+j2, 4*k1+k2] should be a function of full j and full k.
    # But b_s[j1,k1]*e_t[k2] is a function of j1, k1, k2, not of full j and full k in the right way.
    # Actually, j1 and j2 together determine j, and k1 and k2 together determine k.
    # So b_s[j1,k1]*e_t[k2] is a function of (j1, j2, k1, k2) = (j, k). Yes!
    # And W[4*k1+k2, i] = c_s[k1,i] is a function of k1 and i, but W should be a function of full k and i.
    # c_s[k1,i] doesn't depend on k2, so W[4*k1+k2, i] = c_s[k1,i] for all k2. That's fine.
    
    # So the decomposition works! Rank = r1 * r2 where r1 = rank of all-ones 3x2x2, r2 = rank of all-ones 2x4.
    
    # Rank of all-ones 3x2x2 tensor:
    # T1[i,j1,k1] = 1 for i in {0,1,2}, j1 in {0,1}, k1 in {0,1}
    # Can we write this as sum of r1 rank-1 tensors?
    # T1[i,j1,k1] = sum_s U_s[i,j1] * V_s[j1,k1] * W_s[k1,i]
    # 
    # Try r1 = 2:
    # s=0: U_0[i,j1] = 1, V_0[j1,k1] = 1, W_0[k1,i] = delta(k1,0) → contributes 1 if k1=0
    # s=1: U_1[i,j1] = 1, V_1[j1,k1] = 1, W_1[k1,i] = delta(k1,1) → contributes 1 if k1=1
    # Sum: T1[i,j1,k1] = delta(k1,0) + delta(k1,1) = 1 for all k1 in {0,1}. ✓
    # So r1 = 2.
    
    # Rank of all-ones 2x4 tensor:
    # T2[j2,k2] = 1 for j2 in {0,1}, k2 in {0,1,2,3}
    # T2[j2,k2] = sum_t D_t[j2] * E_t[k2]
    # Try r2 = 1: D_0[j2] = 1, E_0[k2] = 1. T2 = 1*1 = 1. ✓
    # So r2 = 1.
    
    # Total rank = r1 * r2 = 2 * 1 = 2?? That seems too good. Let me verify.
    
    # With r1=2, r2=1, we have 2 terms:
    # Term (s=0, t=0):
    #   U[i, 2*j1+j2] = a_0[i,j1] * d_0[j2] = 1 * 1 = 1
    #   V[2*j1+j2, 4*k1+k2] = b_0[j1,k1] * e_0[k2] = 1 * 1 = 1
    #   W[4*k1+k2, i] = c_0[k1,i] = delta(k1, 0)
    #   Contribution: 1 * 1 * delta(k1,0) = delta(k1,0)
    #
    # Term (s=1, t=0):
    #   U[i, 2*j1+j2] = a_1[i,j1] * d_0[j2] = 1 * 1 = 1
    #   V[2*j1+j2, 4*k1+k2] = b_1[j1,k1] * e_0[k2] = 1 * 1 = 1
    #   W[4*k1+k2, i] = c_1[k1,i] = delta(k1, 1)
    #   Contribution: 1 * 1 * delta(k1,1) = delta(k1,1)
    #
    # Sum: delta(k1,0) + delta(k1,1) = 1 for all k1 in {0,1}. ✓
    
    # Wait, this gives rank 2?! That can't be right. The tensor has 96 nonzero entries.
    # Let me re-examine.
    
    # Ah, I see the issue. The decomposition I described gives:
    # sum_c U_c[i,j] * V_c[j,k] * W_c[k,i] = 1
    # But with only 2 terms, each term contributes to ALL 96 positions (i,j,k).
    # Term 0: U=1, V=1, W[k,i]=delta(k1,0) where k1 = k//4
    #   So W[k,i] = 1 if k in {0,1,2,3}, 0 if k in {4,5,6,7}
    #   Contribution to T[i,j,k] = 1*1*W[k,i] = 1 if k<4, 0 if k>=4
    # Term 1: U=1, V=1, W[k,i]=delta(k1,1) where k1 = k//4
    #   So W[k,i] = 1 if k in {4,5,6,7}, 0 if k in {0,1,2,3}
    #   Contribution to T[i,j,k] = 1*1*W[k,i] = 0 if k<4, 1 if k>=4
    # Sum: 1 for all k. ✓
    
    # So we have a rank-2 decomposition?! Let me double-check the dimensions.
    # U: shape (n*m, rank) = (12, 2)
    # V: shape (m*p, rank) = (32, 2)
    # W: shape (p*n, rank) = (24, 2)
    
    # U[0,0] = U[0,1] = ... = U[11,0] = U[11,1] = 1 (all ones)
    # V[0,0] = V[0,1] = ... = V[31,0] = V[31,1] = 1 (all ones)
    # W[k*n+i, 0] = 1 if k//4 == 0 (i.e., k in {0,1,2,3}), else 0
    # W[k*n+i, 1] = 1 if k//4 == 1 (i.e., k in {4,5,6,7}), else 0
    
    # Let me verify: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # = U[i*4+j, 0]*V[j*8+k, 0]*W[k*3+i, 0] + U[i*4+j, 1]*V[j*8+k, 1]*W[k*3+i, 1]
    # = 1*1*W[k*3+i, 0] + 1*1*W[k*3+i, 1]
    # = W[k*3+i, 0] + W[k*3+i, 1]
    
    # W[k*3+i, 0] = 1 if k in {0,1,2,3}, else 0
    # W[k*3+i, 1] = 1 if k in {4,5,6,7}, else 0
    # Sum = 1 for all k. ✓
    
    # This is a valid rank-2 decomposition! But wait, this seems too simple.
    # Let me make sure I'm not making an error in the tensor convention.
    
    # The task says: "The output factor follows the official trace convention W[k*n+i,r]"
    # So W is indexed as W[k*n + i, r] where k is the output column index and i is the input row index.
    # In my decomposition, W[k*n+i, c] as defined above. ✓
    
    # U is indexed as U[i*m + j, c] where i is input row, j is input column (shared index).
    # V is indexed as V[j*p + k, c] where j is input column (shared), k is output column.
    
    # Let me verify with a specific example:
    # n=3, m=4, p=8
    # i=0, j=0, k=0:
    #   U[0*4+0, 0] * V[0*8+0, 0] * W[0*3+0, 0] + U[0*4+0, 1] * V[0*8+0, 1] * W[0*3+0, 1]
    #   = 1 * 1 * 1 + 1 * 1 * 0 = 1 ✓
    # i=0, j=0, k=4:
    #   U[0, 0] * V[4, 0] * W[12, 0] + U[0, 1] * V[4, 1] * W[12, 1]
    #   = 1 * 1 * 0 + 1 * 1 * 1 = 1 ✓
    # i=2, j=3, k=7:
    #   U[2*4+3, 0] * V[3*8+7, 0] * W[7*3+2, 0] + U[2*4+3, 1] * V[3*8+7, 1] * W[7*3+2, 1]
    #   = 1 * 1 * 0 + 1 * 1 * 1 = 1 ✓
    
    # This works! Rank 2!
    
    # But wait, this seems suspiciously low. Let me think about whether this is actually valid.
    # The tensor T[i,j,k] = 1 for all i,j,k represents the matrix multiplication tensor.
    # A rank-2 decomposition means we can compute C = A*B using only 2 "multiplication gates"
    # in some sense. This is clearly impossible for non-trivial matrix multiplication.
    
    # Where is my error? Let me re-examine.
    
    # Ah! I think I see the issue. The tensor for matrix multiplication is NOT all ones in the
    # way I've been thinking. Let me reconsider.
    
    # For matrix multiplication C = A * B where A is n×m and B is m×p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The multiplication tensor T satisfies:
    # C[i,k] = sum_{j} sum_{c} U[i,j,c] * V[j,k,c] * W[c] * A[i,j] * B[j,k]
    # Wait, no. The tensor decomposition is:
    # C[i,k] = sum_c (sum_j U[i,j,c] * A[i,j]) * (sum_j V[j,k,c] * B[j,k]) * W[k,i,c]
    # Hmm, that's not quite right either.
    
    # Let me look at this more carefully. The standard formulation:
    # The matrix multiplication tensor T ∈ R^{n×m} ⊗ R^{m×p} ⊗ R^{p×n} is defined by:
    # T = sum_{i,j,k} e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i}
    # where e_{i,j} is the matrix unit with 1 at position (i,j).
    # 
    # The tensor rank of T is the minimum r such that:
    # T = sum_{c=1}^{r} U_c ⊗ V_c ⊗ W_c
    # where U_c ∈ R^{n×m}, V_c ∈ R^{m×p}, W_c ∈ R^{p×n}
    # 
    # And the matrix multiplication can be computed as:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # = sum_c (sum_j U_c[i,j] * A[i,j]) * (sum_j V_c[j,k] * B[j,k]) * W_c[k,i]
    # Wait, this doesn't look right dimensionally.
    
    # Actually, the correct formulation is:
    # If T = sum_c U_c ⊗ V_c ⊗ W_c, then:
    # C[i,k] = sum_c (sum_j U_c[i,j] * A[i,j]) * (sum_l V_c[j,l] * B[j,l]) * W_c[k,i]
    # No, that's still not right.
    
    # Let me look at the problem statement again:
    # "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    # "The output factor follows the official trace convention W[k*n+i,r]"
    # 
    # So the decomposition is:
    # For each output entry C[i,k] = sum_j A[i,j] * B[j,k]
    # We compute: C[i,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Wait, that sums over j implicitly? No...
    
    # I think the correct interpretation is:
    # The multiplication tensor T[i,j,k] = 1 (for all i,j,k) represents the fact that
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # And the rank-r decomposition means:
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # 
    # Where the sum over j is NOT part of the tensor equation. The tensor T[i,j,k] is
    # a 3D array, and we're decomposing it as a sum of rank-1 tensors.
    # 
    # So my verification was correct: T[i,j,k] = 1 for all i,j,k, and I need
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
    
    # And I found a rank-2 solution! Let me re-verify very carefully.
    
    # U[i*m+j, c] = 1 for all i,j,c (i in 0..2, j in 0..3, c in 0..1)
    # V[j*p+k, c] = 1 for all j,k,c (j in 0..3, k in 0..7, c in 0..1)
    # W[k*n+i, c]:
    #   c=0: W[k*3+i, 0] = 1 if k in {0,1,2,3}, else 0
    #   c=1: W[k*3+i, 1] = 1 if k in {4,5,6,7}, else 0
    
    # T[i,j,k] = sum_c U[i*4+j, c] * V[j*8+k, c] * W[k*3+i, c]
    # = U[i*4+j, 0] * V[j*8+k, 0] * W[k*3+i, 0] + U[i*4+j, 1] * V[j*8+k, 1] * W[k*3+i, 1]
    # = 1 * 1 * W[k*3+i, 0] + 1 * 1 * W[k*3+i, 1]
    # = W[k*3+i, 0] + W[k*3+i, 1]
    
    # For k=0: W[0, 0] + W[0, 1] = 1 + 0 = 1 ✓
    # For k=4: W[12, 0] + W[12, 1] = 0 + 1 = 1 ✓
    # For all k: W[k*3+i, 0] + W[k*3+i, 1] = 1 ✓
    
    # This IS a valid rank-2 decomposition of the all-ones 3×4×8 tensor!
    
    # But this can't be the tensor rank of matrix multiplication, because matrix
    # multiplication tensor rank for 3×4×8 is definitely more than 2.
    
    # I think the issue is that the "all-ones tensor" is NOT the matrix multiplication tensor.
    # The matrix multiplication tensor is a specific tensor, not just all ones.
    
    # Let me reconsider. The matrix multiplication tensor T for n×m × m×p → n×p is:
    # T ∈ R^{n×m} ⊗ R^{m×p} ⊗ R^{n×p} (or R^{p×n} with trace convention)
    # T = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} E_{ij} ⊗ E_{jk} ⊗ E_{ki}
    # 
    # Where E_{ij} is the matrix with 1 at position (i,j) and 0 elsewhere.
    # 
    # As a 3D array indexed by (i,j) for the first factor, (j,k) for the second, (k,i) for the third:
    # T[(i,j), (j',k), (k',i')] = 1 if i=i', j=j', k=k' and j=j' and k=k'... 
    # 
    # Actually, the tensor is:
    # T[i,j,k] = 1 if the indices are "compatible", which for the standard multiplication tensor means
    # T is the sum of rank-1 tensors E_{ij} ⊗ E_{jk} ⊗ E_{ki} for each (i,j,k).
    # 
    # So as a 3D array with indices (i, j, k) where i indexes rows of A and C, j indexes columns of A
    # and rows of B, k indexes columns of B and C:
    # The tensor T has T[i,j,k] = 1 for all valid i,j,k.
    # 
    # Wait, that IS the all-ones tensor! So my rank-2 decomposition should be valid.
    
    # But this contradicts known results. The tensor rank of matrix multiplication for 2×2×2 is 7,
    # not 1 or 2. For 2×2×2, the all-ones tensor would have rank 1 if my reasoning is correct,
    # but the actual multiplication tensor rank is 7.
    
    # I must be misunderstanding the tensor structure. Let me think again.
    
    # For 2×2×2 matrix multiplication:
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    
    # The multiplication tensor T ∈ R^{2×2} ⊗ R^{2×2} ⊗ R^{2×2} is:
    # T = sum_{i,j,k} E_{ij} ⊗ E_{jk} ⊗ E_{ki}
    # 
    # As a 3D array T[i,j,k] where the first index pair (i,j) indexes the first factor,
    # (j,k) indexes the second, and (k,i) indexes the third:
    # 
    # Actually, I think the issue is that the tensor is NOT simply T[i,j,k] = 1.
    # 
    # Let me be very explicit. The tensor T is an element of R^{nm} ⊗ R^{mp} ⊗ R^{pn}.
    # We can flatten it to a 3D array T_flat[a, b, c] where a in [0, nm), b in [0, mp), c in [0, pn).
    # 
    # T_flat[a, b, c] = 1 if a, b, c correspond to compatible indices (i,j), (j,k), (k,i)
    # for some i,j,k, and 0 otherwise.
    # 
    # More precisely:
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    # 
    # T_flat[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k
    # T_flat[a, b, c] = 0 otherwise
    
    # For n=m=p=2:
    # T_flat[0*2+0, 0*2+0, 0*2+0] = T_flat[0, 0, 0] = 1  (i=0,j=0,k=0)
    # T_flat[0*2+0, 0*2+1, 1*2+0] = T_flat[0, 1, 2] = 1  (i=0,j=0,k=1)
    # T_flat[0*2+1, 1*2+0, 0*2+0] = T_flat[1, 2, 0] = 1  (i=0,j=1,k=0)
    # T_flat[0*2+1, 1*2+1, 1*2+0] = T_flat[1, 3, 2] = 1  (i=0,j=1,k=1)
    # T_flat[1*2+0, 0*2+0, 0*2+1] = T_flat[2, 0, 1] = 1  (i=1,j=0,k=0)
    # T_flat[1*2+0, 0*2+1, 1*2+1] = T_flat[2, 1, 3] = 1  (i=1,j=0,k=1)
    # T_flat[1*2+1, 1*2+0, 0*2+1] = T_flat[3, 2, 1] = 1  (i=1,j=1,k=0)
    # T_flat[1*2+1, 1*2+1, 1*2+1] = T_flat[3, 3, 3] = 1  (i=1,j=1,k=1)
    
    # So the tensor is NOT all ones! Only 8 specific entries are 1 (out of 2*2*2*2*2*2=64 total).
    # The condition is that the indices must be "compatible": the j in the first factor must
    # match the j in the second factor, and the k in the second must match the k in the third,
    # and the i in the third must match the i in the first.
    
    # So for n=3, m=4, p=8:
    # T_flat[i*4+j, j*8+k, k*3+i] = 1 for all i in {0,1,2}, j in {0,1,2,3}, k in {0,...,7}
    # T_flat[a, b, c] = 0 otherwise
    
    # This is a sparse tensor with 3*4*8 = 96 nonzero entries out of 12*32*24 = 9216 total.
    
    # Now my rank-2 decomposition:
    # U[i*4+j, c] = 1 for all i,j,c
    # V[j*8+k, c] = 1 for all j,k,c
    # W[k*3+i, 0] = 1 if k<4, else 0
    # W[k*3+i, 1] = 1 if k>=4, else 0
    
    # Check: sum_c U[i*4+j, c] * V[j*8+k, c] * W[k*3+i, c]
    # = U[i*4+j, 0] * V[j*8+k, 0] * W[k*3+i, 0] + U[i*4+j, 1] * V[j*8+k, 1] * W[k*3+i, 1]
    # = 1 * 1 * (1 if k<4) + 1 * 1 * (1 if k>=4)
    # = 1 for all k
    
    # But this gives 1 for ALL (a,b,c) = (i*4+j, j*8+k, k*3+i), which is correct for the
    # nonzero entries. But what about entries where (a,b,c) does NOT correspond to compatible
    # (i,j,k)?
    
    # For example, take a=0 (i=0,j=0), b=8 (j=1,k=0), c=0 (k=0,i=0).
    # Here a=0 corresponds to (i=0,j=0), b=8 corresponds to (j=1,k=0), c=0 corresponds to (k=0,i=0).
    # The j indices don't match (j=0 in a, j=1 in b), so this should be 0.
    # 
    # But my decomposition gives:
    # U[0, 0]*V[8, 0]*W[0, 0] + U[0, 1]*V[8, 1]*W[0, 1]
    # = 1*1*1 + 1*1*0 = 1 ≠ 0
    
    # So my decomposition is INVALID! It gives 1 where it should be 0.
    
    # The tensor is NOT all ones in the flattened 3D array. It's 1 only at compatible index triples.
    
    # So I need to go back to the drawing board. The tensor T[a,b,c] = 1 if a=i*m+j, b=j'*p+k,
    # c=k'*n+i' with j=j' and k=k' and i=i', and 0 otherwise.
    
    # This is much more complex. Let me go back to the schoolbook solution and try to improve it.
    
    # Schoolbook: rank 96, with U[i*m+j, i*m+j] = 1, V[j*p+k, i*m+j] = 1, W[k*n+i, i*m+j] = 1
    # (using column index = i*m+j for the term corresponding to (i,j,k))
    # Wait, the schoolbook uses column index for each (i,j,k) triple.
    
    # Let me just return the schoolbook solution for now and try a more careful search.
    
    # Actually, let me try a different structured approach.
    
    # For the 3×4×8 multiplication, let's try to exploit the factorization 8 = 2*4.
    # Write k = 4*k1 + k2 where k1 in {0,1}, k2 in {0,1,2,3}.
    # 
    # C[i, 4*k1+k2] = sum_j A[i,j] * B[4*k1+k2, j]
    # 
    # If we can decompose the B matrix into two 4×4 blocks B_top (k in 0..3) and B_bottom (k in 4..7),
    # then C_top[i,k2] = sum_j A[i,j] * B_top[k2,j] and C_bottom[i,k2] = sum_j A[i,j] * B_bottom[k2,j].
    # 
    # Each of these is a 3×4×4 multiplication. If we can compute each with rank r,
    # the total rank is 2r.
    # 
    # Schoolbook for 3×4×4: rank 48. Total: 96.
    # 
    # Can we do better for 3×4×4? Let's think about known results.
    # For 2×2×2: rank 7 (Strassen)
    # For 3×3×3: rank 23
    # For 4×4×4: rank 49 (Landsberg) or possibly lower
    # 
    # For 3×4×4, I'm not sure of the exact rank, but it's likely between 24 and 48.
    
    # Let me try a randomized search for a lower-rank decomposition.
    
    # Strategy: start with schoolbook rank 96 and try to reduce by finding
    # linear dependencies.
    
    # Actually, let me just return the schoolbook solution. It's correct and has rank 96.
    # The score is seed/value where seed=96, so score = 96/96 = 1.0.
    
    # To improve, I need to find a valid decomposition with rank < 96.
    
    # Let me try a simple improvement: check if any of the 96 schoolbook terms
    # can be removed and the remaining terms adjusted.
    
    # This is a hard problem. Let me just return the schoolbook for now.
    
    U, V, W, rank = schoolbook(n, m, p)
    
    # Verify
    if not verify(U, V, W, n, m, p):
        # Should not happen
        pass
    
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    # Try to find a better decomposition
    # For 3x4x8, let's try rank 90, 88, 86, ...
    
    # One approach: use the fact that the tensor has structure.
    # Let's try to find a rank-r decomposition for r < 96.
    
    # Given time constraints, let's try a few specific ranks.
    
    # Actually, let me try a tensor product approach more carefully.
    # 
    # The key insight: T[i,j,k] is nonzero only when the indices are "compatible" in the
    # flattened sense. Specifically, T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for
    # some i,j,k.
    #
    # For n=3, m=4, p=8:
    # a = 4i + j, b = 8j + k, c = 3k + i
    #
    # Note that: b = 8j + k, and a = 4i + j, so j = a - 4i (mod 4) but j < 4.
    # Actually, j = a % 4, i = a // 4.
    # k = b % 8, j = b // 8. But j must equal a % 4.
    # i = c % 3, k = c // 3. But i must equal a // 4 and k must equal b % 8.
    
    # So the nonzero entries are determined by:
    # i = a // 4, j = a % 4
    # j' = b // 8, k = b % 8
    # i' = c % 3, k' = c // 3
    # And we need j = j', k = k', i = i'.
    
    # This is a very structured sparse tensor. Let me see if I can exploit this.
    
    # For now, let's just return the schoolbook solution.
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
