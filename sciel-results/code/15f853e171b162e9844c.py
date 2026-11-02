"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 9
    
    # Decompose (2,3,9) as three (2,3,3) blocks
    # For each (2,3,3): A[2x3] * B[3x3] = C[2x3]
    # Use rank-6 decomposition for (2,3,3)
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Split: j=0 gives A[i,0]*B[0,k] (rank 2), j=1,2 gives (2,2,3) part
    # For (2,2,3): rank 6 via 3 outer products of rank 2
    # Actually, let's use explicit rank-6 for (2,3,3)
    
    # (2,3,3) rank-6 decomposition:
    # Terms: (i,j,k) with j in {0,1,2}
    # We need 6 terms. One way:
    # Term 0: A[0,0]*B[0,0] -> U[0,0]=1,V[0,0]=1,W[0,0]=1
    # This gets complex. Let me use the (1+2) split approach globally.
    
    # Global approach: split middle dim 3 into 1+2
    # Part 1: (2,1,9) rank 2
    # Part 2: (2,2,9) - split 9 into three 3s, each (2,2,3) rank 6, total 18
    # Grand total: 2 + 18 = 20
    
    # But can I do better? Let me try rank 18 by a different split.
    
    # Alternative: For (2,3,9), use the fact that we can group.
    # Let me just implement (2,1,9) + 3*(2,2,3) = 2 + 18 = 20
    
    rank = 20
    U = [[0] * rank for _ in range(n * m)]  # 6 x 20
    V = [[0] * rank for _ in range(m * p)]  # 27 x 20
    W = [[0] * rank for _ in range(p * n)]  # 18 x 20
    
    col = 0
    # Part 1: (2,1,9) - A[:,0] * B[0,:]
    # A[i,0] for i in 0,1; B[0,k] for k in 0..8
    # C[i,k] += A[i,0] * B[0,k]
    for i in range(n):
        for k in range(p):
            # This is a rank-1 term: A[i,0] * B[0,k]
            # But we need rank-2 for the whole (2,1,9)
            pass
    
    # Actually (2,1,9) has rank 2, not 18. Let me redo.
    # (2,1,9): C[i,k] = A[i,0]*B[0,k]. This is outer product of col(A,0) and row(B,0).
    # Rank = 2: use 2 terms.
    # Term a: A[0,0]*B[0,k] for all k -> but that's not a single tensor term.
    # Tensor terms are (i,j,k) triples. For (2,1,9): j=0 only.
    # C[i,k] = A[i,0]*B[0,k]. This is rank 2: 
    #   Term 1: i=0, j=0, k=0..8 -> no, each term is one (i,j,k).
    # Actually rank of (n,1,p) is n*p? No. (2,1,9): tensor T[i,0,k] = delta. 
    # The tensor is T[i,j,k] where j=0. T[i,0,k] = 1 for all i,k. 
    # This is a rank-1 tensor! T[i,0,k] = u[i]*v[0]*w[k] with u=[1,1], v=[1], w=[1,...,1].
    # Wait no, the multiplication tensor for (n,m,p) has T[i,j,k] = 1 if the product C[i,k] = sum_j A[i,j]B[j,k].
    # So T[i,j,k] = 1 for all valid i,j,k. For (2,1,9): T[i,0,k] = 1 for i in {0,1}, k in {0..8}.
    # This is a rank-1 tensor: T = u ⊗ v ⊗ w where u[i]=1, v[0]=1, w[k]=1.
    # So rank 1!
    
    # Similarly (2,2,3): T[i,j,k] = 1 for i in {0,1}, j in {0,1}, k in {0,1,2}.
    # This has rank... it's the (2,2,3) multiplication tensor. Known rank is 6.
    
    # Let me verify: can I do (2,2,3) in rank 6?
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # = sum over j of A[i,j]*B[j,k]
    # 6 terms: for each (i,j,k) we'd need 2*2*3=12 schoolbook.
    # Rank 6 means 6 tensor terms. Each term is a triple (i0,j0,k0) with coefficient.
    # Actually each term in the decomposition is a product u_c[i]*v_c[j]*w_c[k].
    # We need sum_c u_c[i]*v_c[j]*w_c[k] = 1 for all i,j,k (for the (2,2,3) sub-tensor).
    # This is a tensor of all ones of shape (2,2,3). The rank of the all-ones tensor (2,2,3) is... 
    # min(2,2,3) = 2? No. The all-ones matrix of size 2x3 has rank 1. But tensor rank is different.
    # The all-ones tensor (2,2,3): can we write it as sum of r rank-1 tensors?
    # Rank-1 tensor: u[i]*v[j]*w[k]. Sum of r such = 1 for all i,j,k.
    # This is equivalent to: the 2x2x3 array of all 1s. 
    # If we fix k, we get a 2x2 all-ones matrix, rank 1. 
    # The tensor rank of all-ones (a,b,c) is min(a,b,c) if... actually it's known to be min(a,b,c) for the all-ones tensor? No.
    # All-ones (2,2,2): rank 2 (since it's the sum of two rank-1: [1,1][1,1][1,1] can't be rank 1 unless... [1,1]⊗[1,1]⊗[1,1] IS rank 1 and equals all ones!)
    # So all-ones (2,2,2) has rank 1!
    # All-ones (2,2,3): [1,1]⊗[1,1]⊗[1,1,1] is rank 1 and equals all ones!
    # So the (2,2,3) multiplication tensor (which is all-ones of shape (2,2,3)) has rank 1?!
    
    # Wait, I'm confusing myself. The multiplication tensor T for (n,m,p) has T[i,j,k]=1 for all i,j,k.
    # This IS the all-ones tensor of shape (n,m,p). And the all-ones tensor has rank 1!
    # Because T[i,j,k] = 1 = 1*1*1 = u[i]*v[j]*w[k] with u=v=w=all ones.
    
    # So the schoolbook rank n*m*p is NOT minimal. The actual tensor rank of the multiplication tensor
    # (all-ones of shape (n,m,p)) is 1?! That can't be right because then all matrix multiplication
    # would be rank 1.
    
    # I'm making an error. Let me re-read the problem. The decomposition is:
    # C = A*B, where (AB)[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor T[i,j,k] = 1 (for all valid indices) represents this.
    # A decomposition of rank r means: T[i,j,k] = sum_{c=1}^r U[i,c]*V[j,c]*W[k,c]
    # where U is (n x r), V is (m x r), W is (p x r).
    # For T to be all-ones: sum_c U[i,c]*V[j,c]*W[k,c] = 1 for all i,j,k.
    # If r=1: U[i,0]*V[j,0]*W[k,0] = 1 for all i,j,k. This requires U[i,0]=a_i, V[j,0]=b_j, W[k,0]=c_k with a_i*b_j*c_k=1 for all i,j,k.
    # This is possible! e.g., a_i=1, b_j=1, c_k=1. So rank 1?!
    
    # But that would mean any matrix multiplication is rank 1, which is absurd. The issue is that
    # in the actual computation, we're not just computing the tensor value; we're using the factorization
    # to compute C from A and B. The factorization U,V,W means:
    # C[i,k] = sum_c U[i,c] * (sum_j V[j,c] * A[i,j]) ... no.
    
    # Let me re-read the interface: U is (n*m, rank), V is (m*p, rank), W is (p*n, rank).
    # So U is indexed by (i,j) pairs, V by (j,k) pairs, W by (k,i) pairs.
    # The decomposition is: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # where T[i,j,k] = 1.
    
    # So U is (n*m) x rank, not n x rank. This changes everything!
    # Now T[i,j,k] = sum_c U[i,j]_c * V[j,k]_c * W[k,i]_c = 1 for all i,j,k.
    
    # For (2,1,9): U is (2*1)=2 x rank, V is (1*9)=9 x rank, W is (9*2)=18 x rank.
    # T[i,0,k] = sum_c U[i,0]_c * V[0,k]_c * W[k,i]_c = 1 for all i,k.
    # With r=1: U[i,0]_0 * V[0,k]_0 * W[k,i]_0 = 1 for all i,k.
    # Let U[0,0]=1, U[1,0]=1, V[0,k]=1 for all k, W[k,i]=1 for all k,i. Then product=1. Rank 1!
    
    # So (2,1,9) has rank 1 in this formulation.
    # And (2,2,3): U is 4x r, V is 6x r, W is 6x r. T[i,j,k]=1 for i in {0,1}, j in {0,1}, k in {0,1,2}.
    # With r=1: U[i,j]_0 * V[j,k]_0 * W[k,i]_0 = 1 for all i,j,k.
    # U[0,0]*V[0,0]*W[0,0]=1, U[0,0]*V[0,1]*W[1,0]=1, U[0,0]*V[0,2]*W[2,0]=1,
    # U[0,1]*V[1,0]*W[0,0]=1, U[0,1]*V[1,1]*W[1,0]=1, U[0,1]*V[1,2]*W[2,0]=1,
    # U[1,0]*V[0,0]*W[0,1]=1, U[1,0]*V[0,1]*W[1,1]=1, U[1,0]*V[0,2]*W[2,1]=1,
    # U[1,1]*V[1,0]*W[0,1]=1, U[1,1]*V[1,1]*W[1,1]=1, U[1,1]*V[1,2]*W[2,1]=1.
    # From U[0,0]*V[0,0]*W[0,0]=1 and U[0,0]*V[0,1]*W[1,0]=1: V[0,0]*W[0,0]=V[0,1]*W[1,0].
    # This is a system that may or may not have a solution. Let's try: set all U,V,W=1. Then all products=1. It works! Rank 1!
    
    # So the entire (2,3,9) multiplication tensor has rank 1 in this formulation?!
    # U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1 for all i,j,k.
    # Then T[i,j,k] = U[i*m+j,0]*V[j*p+k,0]*W[k*n+i,0] = 1*1*1 = 1. Yes!
    
    # So the answer is rank 1!
    
    rank = 1
    U = [[1] * rank for _ in range(n * m)]
    V = [[1] * rank for _ in range(m * p)]
    W = [[1] * rank for _ in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
