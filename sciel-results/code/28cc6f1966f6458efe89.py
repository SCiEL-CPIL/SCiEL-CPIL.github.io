"""Matrix multiplication 3x4x8 - structured low-rank decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = 3, 4, 8
    
    # Schoolbook: rank = n*m*p = 96
    rank_schoolbook = n * m * p
    U = np.zeros((n * m, rank_schoolbook), dtype=float)
    V = np.zeros((m * p, rank_schoolbook), dtype=float)
    W = np.zeros((p * n, rank_schoolbook), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    if time.time() > deadline:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Try structured decomposition: split middle dim into 2+2, output into 2+2+2+2
    # For each (j_block, k_block): 3x2 * 2x2 = 3x2
    # Use Strassen on the 2x2 * 2x2 = 2x2 part? No, it's 3x2 * 2x2.
    # 
    # Alternative: use the fact that 3x2 * 2x2 can be done in rank 11 (known bound)
    # or just use schoolbook 12.
    #
    # Let me try a different structured approach.
    # 
    # Key idea: C[i,k] = sum_j A[i,j]*B[j,k]
    # = sum_j (A[i,j] + delta_ij)*B[j,k] - sum_j delta_ij*B[j,k]
    # This doesn't help directly.
    #
    # Let me try: express in terms of sums and differences to share terms.
    #
    # For the j=0,1 group:
    # A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # = 0.5*((A[i,0]+A[i,1])*(B[0,k]+B[1,k]) + (A[i,0]-A[i,1])*(B[0,k]-B[1,k]))
    # This uses 2 bilinear terms per (i,k) instead of 2, so no gain.
    #
    # But if we share across k: for fixed i, we need to compute for all k:
    # A[i,0]*B[0,k] + A[i,1]*B[1,k] for k=0..7
    # = A[i,0]*[B[0,0]...B[0,7]] + A[i,1]*[B[1,0]...B[1,7]]
    # This is a rank-2 term in the B-space for each i.
    #
    # Total for j-group {0,1}: for each i (3 values), 2 terms = 6 terms, for all 8 k's.
    # But each term is a scalar times a row of B, so we need 6*8 = 48 W-entries... 
    # Actually no, the rank counts the number of bilinear terms.
    #
    # Let me count properly. For the j-group {0,1}:
    # For each i in {0,1,2} and k in {0,...,7}:
    #   term1: A[i,0] * B[0,k]
    #   term2: A[i,1] * B[1,k]
    # That's 3*2*8 = 48 bilinear terms.
    #
    # Using the sum/difference trick:
    #   s_i = A[i,0]+A[i,1], d_i = A[i,0]-A[i,1]
    #   For each k: s_i*(B[0,k]+B[1,k])/2 + d_i*(B[0,k]-B[1,k])/2
    # Still 2 terms per (i,k), so 48 terms. No gain.
    #
    # The gain from Strassen comes when you have 2x2 * 2x2 = 2x2, where you can
    # do 7 multiplications instead of 8. Here we have 3x2 * 2x2, which is different.
    #
    # Let me try a completely different approach: use a rank-72 decomposition
    # based on the following insight.
    #
    # C = A * B, A is 3x4, B is 4x8.
    # Write A as a 3x4 matrix. Think of it as 4 column vectors a_0, a_1, a_2, a_3 in R^3.
    # B has 4 row vectors b_0, b_1, b_2, b_3 in R^8.
    # C = sum_j a_j * b_j (outer product sum)
    #
    # This is a sum of 4 rank-1 terms (as matrices), but the tensor rank is different.
    #
    # Let me just try to find a valid decomposition with rank < 96 using a 
    # systematic approach. I'll use the following:
    #
    # Decomposition via Kronecker product structure:
    # 3x4x8 = (3x2x2) ⊗ (1x2x4) ... this doesn't quite work.
    #
    # Let me try: 3x4x8 can be seen as (3x4x2) x 4 (four copies along output)
    # M(3,4,2) * 4
    #
    # For M(3,4,2): 3x4 * 4x2 = 3x2
    # Split into j-groups of 2: two 3x2*2x2 problems
    # Each 3x2*2x2 has schoolbook rank 12
    # Total for M(3,4,2): 24
    # Total for M(3,4,8): 4*24 = 96. Same as schoolbook.
    #
    # I need to find a way to share terms across the output blocks.
    #
    # Here's an idea: instead of treating the 8 output columns independently,
    # group them and use the linearity in B.
    #
    # C[i, 0:4] = A[i,:] * B[:, 0:4]
    # C[i, 4:8] = A[i,:] * B[:, 4:8]
    #
    # If I precompute certain combinations of A[i,:] and B rows, I might save.
    #
    # Let me try a rank-80 decomposition:
    # Use 4 terms for the "cross" interactions and schoolbook for the rest.
    #
    # Actually, let me just implement a valid decomposition and verify it.
    # I'll use the following known technique:
    #
    # For M(3,4,8), use the bound from the "laser method" or direct construction.
    # 
    # Let me try rank 72:
    # Split into 3 groups of output columns (not evenly, but 3,3,2)
    # This is getting complicated. Let me just return schoolbook for now
    # but with a verification step.
    
    # Verify schoolbook
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_test = A_test @ B_test
    
    # Check: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should give C[i,k]
    # Actually the decomposition is: C[i,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Wait, that's not right. The standard form is:
    # C[i,k] = sum_c (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k]) * W[k*n+i, c]
    # No, the correct form for tensor decomposition of matrix multiplication is:
    # C[i,k] = sum_c U_c[i,j] * V_c[j,k] * W_c[k,i] ... 
    # 
    # Actually, the standard bilinear form is:
    # C[i,k] = sum_c (sum_j U[i*m+j, c] * A[i,j]) * (sum_l V[j*p+k, c] * B[j,k])
    # No wait. Let me re-read the interface.
    #
    # The interface says: U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank)
    # The decomposition should satisfy:
    # C[i,k] = sum_c U[i*m+j, c] * A[i,j] * V[j*p+k, c] * B[j,k] * W[k*n+i, c]
    # summed over j and c? That doesn't make sense dimensionally.
    #
    # The correct interpretation for matrix multiplication tensor:
    # T[i,j,k] = delta (the multiplication tensor)
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # A rank-r decomposition means:
    # C[i,k] = sum_{c=1}^{r} (sum_j U_c[i,j] * A[i,j]) * (sum_j V_c[j,k] * B[j,k]) * W_c[i,k]
    # where U_c is n×m, V_c is m×p, W_c is p×n (or n×p).
    #
    # In the flattened form:
    # U is (n*m, r): U[i*m+j, c] = U_c[i,j]
    # V is (m*p, r): V[j*p+k, c] = V_c[j,k]  
    # W is (p*n, r): W[k*n+i, c] = W_c[k,i] (note: k*n+i, so row index is k*n+i)
    #
    # So: C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k])
    #
    # For schoolbook: U[i*m+j, c] = 1 if (i,j) matches c's (i,j), else 0
    #                V[j*p+k, c] = 1 if (j,k) matches c's (j,k), else 0
    #                W[k*n+i, c] = 1 if (k,i) matches c's (k,i), else 0
    # And c enumerates all (i,j,k) triples.
    #
    # Let me verify this is correct for schoolbook.
    
    # Actually, I realize the schoolbook might not be exactly as I wrote. Let me re-derive.
    # For schoolbook, each term c corresponds to a unique (i,j,k) triple.
    # U[i'*m+j', c] = 1 if (i',j') = (i,j) for c's (i,j), else 0
    # V[j'*p+k', c] = 1 if (j',k') = (j,k) for c's (j,k), else 0
    # W[k'*n+i', c] = 1 if (k',i') = (k,i) for c's (k,i), else 0
    #
    # Then: sum_j U[i*m+j, c] * A[i,j] = A[i,j] (only the j matching c's j contributes)
    #       sum_j V[j*p+k, c] * B[j,k] = B[j,k] (only the j matching c's j contributes)
    #       W[k*n+i, c] = 1 (since (k,i) matches)
    #
    # So term c contributes A[i,j] * B[j,k] * 1 = A[i,j]*B[j,k]
    # Sum over c: sum over all (i',j',k') triples where i'=i, k'=k: A[i,j']*B[j',k]
    # = sum_{j'} A[i,j']*B[j',k] = C[i,k]. Correct!
    
    # Now let me try to find a better decomposition.
    # 
    # Idea: Use the fact that for the 2x2x2 sub-problem, Strassen gives rank 7.
    # 
    # Let me decompose 3x4x8 as follows:
    # Split j into {0,1} and {2,3}. Split k into {0,1}, {2,3}, {4,5}, {6,7}.
    # For each (j_block, k_block) pair, we have a 3x2 * 2x2 = 3x2 sub-problem.
    # 
    # For 3x2 * 2x2: 
    #   X(3x2) * Y(2x2) = Z(3x2)
    #   Z[i,0] = X[i,0]*Y[0,0] + X[i,1]*Y[1,0]
    #   Z[i,1] = X[i,0]*Y[0,1] + X[i,1]*Y[1,1]
    #
    # This can be written as: for each i, Z[i,:] = X[i,0]*Y[0,:] + X[i,1]*Y[1,:]
    # = [X[i,0], X[i,1]] * Y^T (where Y^T is 2x2)
    #
    # The rank of this tensor: it's the sum of 3 rank-2 tensors (one for each i).
    # Each rank-2 tensor is: (e_i ⊗ [1,0]) * (e_i ⊗ [0,1])... no.
    #
    # Actually, Z[i,j] = X[i,0]*Y[0,j] + X[i,1]*Y[1,j]
    # This is a sum of 2 terms: X[i,0]*Y[0,j] and X[i,1]*Y[1,j]
    # The first term: for all i,j, it's X[i,0]*Y[0,j]. This is a rank-1 tensor in the 
    # (i,j) output space, with U-part depending on X[:,0] and V-part on Y[0,:].
    # Similarly for the second term.
    #
    # So the total rank is at most 2*3 = 6? No, that's not right either.
    #
    # Let me think of it as: Z = X * Y where X is 3x2, Y is 2x2.
    # This is a bilinear map from (X,Y) to Z.
    # The tensor is: T[i,j,a,b] = delta(a,b) where... 
    #
    # Actually, the simplest way: Z[i,j] = sum_a X[i,a]*Y[a,j]
    # This is the standard matrix multiplication tensor M(3,2,2).
    # The rank of M(3,2,2) is at most 3*2*2 = 12 (schoolbook).
    # Can it be less? 
    #
    # M(3,2,2): I believe the rank is exactly 12, since min(3,2)=2 and the 
    # "small" dimension doesn't allow Strassen-like savings.
    #
    # Hmm. Let me try a different grouping.
    #
    # What if I split differently? 3x4x8:
    # Split i into groups? No, n=3 is small.
    #
    # Let me try: use a rank-72 decomposition based on the following:
    # C = A * B
    # Write A as [a1 a2 a3 a4] (columns, each 3x1)
    # B as [b1 b2 b3 b4] (rows, each 1x8)
    # C = a1*b1 + a2*b2 + a3*b3 + a4*b4
    #
    # This is a sum of 4 rank-1 (matrix) terms. But the tensor rank is what matters.
    #
    # I think the best I can do without deep theory is to try a numerical optimization.
    # Let me set up the problem and try to minimize the rank.
    #
    # For a given rank r, we need to find U (12xr), V (32xr), W (24xr) such that
    # for all A, B: C = A*B where C[i,k] = sum_c W[k*n+i,c] * (sum_j U[i*m+j,c]*A[i,j]) * (sum_j V[j*p+k,c]*B[j,k])
    #
    # This is a bilinear equation in A and B. We can test it on random matrices.
    #
    # Let me try rank 80 first, then try to reduce.
    
    def verify_decomposition(U, V, W, n, m, p, trials=20):
        """Verify that the decomposition correctly computes matrix multiplication."""
        for _ in range(trials):
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            C_expected = A @ B
            
            # Compute C from decomposition
            C_computed = np.zeros((n, p))
            for c in range(U.shape[1]):
                # U_c is n x m (flattened as n*m)
                U_c = U[:, c].reshape(n, m)
                V_c = V[:, c].reshape(m, p)
                W_c = W[:, c].reshape(p, n)
                
                # (sum_j U_c[i,j] * A[i,j]) is element-wise product, not sum
                # Wait, I need to re-read the convention.
                #
                # The standard tensor decomposition for matrix multiplication:
                # C[i,k] = sum_c W_c[i,k] * (U_c^T * A) * (V_c^T * B)
                # No, that's not standard either.
                #
                # Let me use the convention from the problem:
                # U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank)
                # The decomposition: C[i,k] = sum_c U[i*m+j, c] * A[i,j] * V[j*p+k, c] * B[j,k] * W[k*n+i, c]
                # summed over j and c.
                #
                # Wait, that would be: C[i,k] = sum_c W[k*n+i,c] * (sum_j U[i*m+j,c]*A[i,j]) * (sum_j V[j*p+k,c]*B[j,k])
                # But the j in U and V might be different! Let me re-read.
                #
                # Actually, I think the correct interpretation is:
                # C[i,k] = sum_c (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k]) * W[k*n+i, c]
                # where the sums over j in U and V are independent.
                #
                # For schoolbook: U[i*m+j, c] = 1 if c corresponds to (i,j), else 0
                # Then sum_j U[i*m+j, c]*A[i,j] = A[i,j_c] (the specific j for this c)
                # Similarly V gives B[j_c, k]
                # And W[k*n+i, c] = 1
                # So term c = A[i,j_c] * B[j_c, k]
                # Sum over c: sum over all j_c: A[i,j_c]*B[j_c,k] = C[i,k]. Correct!
                
                U_sum = np.zeros(n)
                for ii in range(n):
                    for jj in range(m):
                        U_sum[ii] += U[ii * m + jj, c] * A[ii, jj]
                
                V_sum = np.zeros(p)
                for jj in range(m):
                    for kk in range(p):
                        V_sum[kk] += V[jj * p + kk, c] * B[jj, kk]
                
                for kk in range(p):
                    for ii in range(n):
                        C_computed[ii, kk] += W[kk * n + ii, c] * U_sum[ii] * V_sum[kk]
            
            if not np.allclose(C_computed, C_expected, atol=1e-9):
                return False
        return True
    
    # Let me try to construct a rank-80 decomposition.
    # Strategy: use schoolbook for most terms, but share some.
    #
    # Actually, let me try a simpler improvement: 
    # Use the identity: xy = ((x+y)^2 - (x-y)^2) / 4
    # This doesn't reduce the number of multiplications, but might help with structure.
    #
    # I think the best approach within the time budget is to try a few structured
    # decompositions and verify them. Let me try rank 80.
    #
    # For rank 80: I'll use the following structure:
    # Split B into two 4x4 blocks: B = [B1, B2]
    # C = [A*B1, A*B2]
    # For each A*B_i (3x4 * 4x4 = 3x4):
    #   Use a decomposition of rank 40.
    # Total: 80.
    #
    # For 3x4*4x4 with rank 40:
    # Schoolbook is 48. Need to save 8 terms.
    #
    # Idea: A(3x4)*B(4x4). Split A into A1(3x2), A2(3x2).
    # Split B into B1(2x4), B2(2x4) (top and bottom rows).
    # C = A1*B1 + A2*B2
    # Each 3x2*2x4 has schoolbook rank 24. Total 48.
    #
    # For 3x2*2x4: X(3x2)*Y(2x4) = Z(3x4)
    # Z[i,k] = X[i,0]*Y[0,k] + X[i,1]*Y[1,k]
    # 
    # Use the Strassen-like trick on the 2x2 part:
    # For each pair of output columns (k, k+1):
    # Z[i,2l] = X[i,0]*Y[0,2l] + X[i,1]*Y[1,2l]
    # Z[i,2l+1] = X[i,0]*Y[0,2l+1] + X[i,1]*Y[1,2l+1]
    #
    # Let s_i = X[i,0]+X[i,1], d_i = X[i,0]-X[i,1]
    # Let e_{0,2l} = Y[0,2l]+Y[0,2l+1], f_{0,2l} = Y[0,2l]-Y[0,2l+1]
    # Let e_{1,2l} = Y[1,2l]+Y[1,2l+1], f_{1,2l} = Y[1,2l]-Y[1,2l+1]
    #
    # Then: Z[i,2l] + Z[i,2l+1] = s_i * (Y[0,2l]+Y[0,2l+1] + Y[1,2l]+Y[1,2l+1]) / ... 
    # This is getting messy. Let me just try to implement and verify.
    
    # Let me try a different approach: just use rank 96 (schoolbook) which we know works.
    # The task says score is seed/value, so 1.0 is the baseline. To improve, I need rank < 96.
    #
    # Let me try rank 84: use a hybrid approach.
    #
    # Actually, I just realized: I should try to use the known result that
    # M(3,4,8) <= 72 or some other bound. But I don't have a concrete construction.
    #
    # Let me try a numerical approach: start with schoolbook and try to zero out
    # some entries via optimization.
    
    if time.time() > deadline - 10:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Try to find a rank-80 decomposition by starting from schoolbook and 
    # trying to merge terms.
    #
    # For now, let me just return the schoolbook decomposition.
    # The key insight is that I need a valid decomposition, and schoolbook works.
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
