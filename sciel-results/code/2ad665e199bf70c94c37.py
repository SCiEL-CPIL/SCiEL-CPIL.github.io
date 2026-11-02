import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard: 150s wall-clock limit, target < 130s for safety
    start_time = time.time()
    deadline = start_time + 130.0
    
    n, m, p = DIMENSIONS  # 2, 3, 7
    schoolbook_rank = n * m * p  # 42
    
    def check_time():
        return time.time() > deadline
    
    # The matrix multiplication tensor T has entries T[i,j,k] = delta_{i,j,k}
    # where for C = A*B, C[i,k] = sum_j A[i,j]*B[j,k]
    # T[i,j,k] = 1 if (i,j,k) corresponds to a valid term, else 0
    # More precisely, T is a 3-tensor of shape (n, m, p) where
    # T[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B contribute to (i,k) of C
    # Actually for matrix multiplication: T_{i,j,k} = 1 for all valid (i,j,k)
    # The tensor is T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # Wait, that's not right. Let me reconsider.
    
    # For C = A * B where A is n x m and B is m x p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The multiplication tensor has entries T[i,j,k] = 1 for all (i,j,k)
    # because each (i,j,k) triple corresponds to one multiplication term.
    
    # So T is a tensor of shape (n, m, p) with all entries = 1? No!
    # T[i,j,k] = 1 for all i,j,k. That's the full tensor.
    # Rank of this tensor: we need to find minimum r such that
    # T[i,j,k] = sum_{a=1}^{r} U[i,a] * V[j,a] * W[k,a]
    
    # Actually, the standard matrix multiplication tensor M_{n,m,p} has
    # entries T[i,j,k] = 1 if the indices are valid, which they all are.
    # So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # This is the all-ones tensor!
    
    # Wait, that can't be right either. Let me think more carefully.
    
    # The matrix multiplication tensor M_{n,m,p} is defined as:
    # M_{n,m,p} = sum_{i=1}^{n} sum_{j=1}^{m} sum_{k=1}^{p} e_i ⊗ e_j ⊗ e_k * delta
    # No, that's not standard.
    
    # Standard definition: The matrix multiplication tensor is
    # T = sum_{i,j,k} x_i y_j z_k where the coefficient is 1 if (i,j,k) is a valid index triple.
    # For n x m times m x p: valid triples are all (i,j,k) with i in [1,n], j in [1,m], k in [1,p].
    # So T = sum_{i=1}^{n} sum_{j=1}^{m} sum_{k=1}^{p} e_i ⊗ e_j ⊗ e_k
    # This means T[i,j,k] = 1 for all valid (i,j,k).
    
    # So we need to decompose the all-ones tensor of shape (n,m,p) into a sum of rank-1 tensors.
    # T[i,j,k] = sum_a U[i,a] V[j,a] W[k,a]
    
    # The all-ones tensor has rank 1! Because T[i,j,k] = 1 = 1*1*1.
    # U[i,a] = 1 for all i, V[j,a] = 1 for all j, W[k,a] = 1 for all k, with rank 1.
    
    # Wait, that would mean rank 1 is sufficient. But that's clearly wrong for matrix multiplication.
    # I'm confusing the tensor structure.
    
    # Let me reconsider. The matrix multiplication tensor is NOT the all-ones tensor.
    # 
    # The correct definition: We want to compute C = A*B where A is n x m, B is m x p.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # 
    # The tensor T is such that T(A,B) = C, i.e.,
    # C[i,k] = sum_{j} T[i,j,k] A[i,j] B[j,k]
    # 
    # So T[i,j,k] = 1 for all i,j,k. That IS the all-ones tensor.
    # 
    # But wait, that means the bilinear map is just the sum over j of A[i,j]*B[j,k],
    # which is exactly matrix multiplication. And the tensor is all-ones.
    # 
    # The rank of the all-ones tensor of shape (n,m,p) is indeed 1, since
    # T[i,j,k] = 1*1*1.
    # 
    # But that contradicts known results that matrix multiplication has rank at least n*m*p/... 
    # 
    # OH WAIT. I think I'm confusing two different formulations.
    
    # In the standard tensor decomposition for matrix multiplication, we have:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # This can be written as a trilinear form:
    # sum_{i,j,k} C[i,k] * (something) = sum_{i,j,k} A[i,j] B[j,k] * (something)
    
    # Actually, the standard formulation uses the tensor:
    # T = sum_{i=1}^{n} sum_{j=1}^{m} sum_{k=1}^{p} e_i ⊗ e_j ⊗ e_k
    # where the "multiplication" is: given vectors a (length nm), b (length mp),
    # we form c (length np) such that c[i,k] = sum_j a[i,j] b[j,k].
    
    # The tensor T in this formulation has T[i,j,k] = 1 for all (i,j,k) in [n]x[m]x[p].
    # And we want T = sum_{a=1}^{r} u_a ⊗ v_a ⊗ w_a where
    # u_a is in R^{n}, v_a is in R^{m}, w_a is in R^{p}.
    # 
    # And T[i,j,k] = sum_a u_a[i] v_a[j] w_a[k] = 1 for all i,j,k.
    
    # This is indeed the all-ones tensor, and it has rank 1!
    # u_a = [1,1,...,1], v_a = [1,1,...,1], w_a = [1,1,...,1]
    
    # But this can't be right because we know matrix multiplication requires more than rank 1.
    
    # I think the issue is that the standard matrix multiplication tensor is NOT the all-ones tensor.
    # Let me look at this differently.
    
    # The correct tensor for matrix multiplication M_{n,m,p} is:
    # T[i,j,k] = 1 if the indices correspond to a valid multiplication term
    # 
    # For C = A*B: C[i,k] = sum_j A[i,j] B[j,k]
    # The "terms" are indexed by (i,j,k) and each term A[i,j]*B[j,k] contributes to C[i,k].
    # So the tensor should be:
    # T[i,j,k] = 1 for all (i,j,k)
    # 
    # And we decompose as: T[i,j,k] = sum_a U[i,a] V[j,a] W[k,a]
    # 
    # With U being n x r, V being m x r, W being p x r.
    
    # This IS the all-ones tensor and has rank 1.
    
    # I must be misunderstanding the problem. Let me re-read the task.
    
    # "Find a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix by a 3 by 7 matrix."
    # "The output factor follows the official trace convention W[k*n+i,r]."
    
    # The trace convention suggests this is about the standard matrix multiplication tensor,
    # and the rank is the number of scalar multiplications needed.
    
    # I think the key insight I'm missing: the tensor entries are NOT all 1.
    # 
    # Actually, I believe the correct formulation is:
    # The matrix multiplication tensor M_{n,m,p} has entries:
    # M[i,j,k] = 1 for all (i,j,k) in [n]x[m]x[p]
    # 
    # And we seek the minimum r such that there exist U (n x r), V (m x r), W (p x r)
    # with M[i,j,k] = sum_{a=1}^{r} U[i,a] V[j,a] W[k,a]
    # 
    # For the all-ones tensor, this is satisfied with r=1: U=V=W=ones.
    # 
    # So why is the schoolbook rank 42 = n*m*p?
    
    # AH! I think I finally understand. The schoolbook decomposition uses r = n*m*p terms,
    # but that's NOT because the tensor rank is n*m*p. The tensor rank of the all-ones
    # tensor is 1. The schoolbook method just happens to use n*m*p multiplications,
    # but that's not the tensor rank.
    
    # Wait, but the problem says "score is the rank" and the schoolbook gives rank 42.
    # If the true rank is 1, then the schoolbook would be very far from optimal.
    
    # Let me reconsider the problem statement. Maybe the tensor is NOT all-ones.
    
    # Re-reading: "multiplying a 2 by 3 matrix by a 3 by 7 matrix"
    # A is 2x3, B is 3x7, C = A*B is 2x7.
    # C[i,k] = sum_{j=0}^{2} A[i,j] B[j,k]
    # 
    # The tensor T has shape (2, 3, 7) and T[i,j,k] = 1 for all (i,j,k).
    # We want T = sum_a U_a ⊗ V_a ⊗ W_a.
    # 
    # This is the all-ones tensor, rank 1.
    
    # I'm clearly misunderstanding something fundamental. Let me think about what
    # "tensor decomposition for matrix multiplication" really means in this context.
    
    # Perhaps the issue is that the decomposition must be "exact" in the sense that
    # it computes the matrix product, and the rank counts the number of multiplications
    # in a bilinear algorithm. In that case, the rank IS the tensor rank, and for the
    # all-ones tensor it should be 1.
    
    # But the schoolbook gives 42, which suggests the tensor is NOT all-ones, or
    # there's a constraint I'm missing.
    
    # Let me try a different interpretation. Maybe the tensor is:
    # T[i,j,k] = delta_{j, something} or some other structure.
    
    # Actually, I think I've been overthinking this. Let me just implement a
    # randomized search for low-rank decompositions and verify them numerically.
    
    # The verification: for random matrices A (2x3) and B (3x7), compute C = A*B
    # using the decomposition, and check if it matches.
    
    def verify_decomposition(U, V, W, n, m, p, trials=100):
        """Verify that the decomposition correctly computes matrix multiplication."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        
        for _ in range(trials):
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            C_true = A @ B
            
            # Compute C using the decomposition
            # C[i,k] = sum_j sum_a U[i,a] V[j,a] W[k,a] A[i,j] B[j,k]
            # Wait, that's not right. Let me think about how the decomposition works.
            # 
            # If T[i,j,k] = sum_a U[i,a] V[j,a] W[k,a], then
            # C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k]
            #         = sum_j sum_a U[i,a] V[j,a] W[k,a] A[i,j] B[j,k]
            #         = sum_a U[i,a] W[k,a] (sum_j V[j,a] A[i,j] B[j,k])
            # 
            # Hmm, that's not quite right either. Let me reconsider.
            # 
            # Actually, the standard formulation is:
            # C[i,k] = sum_j A[i,j] B[j,k]
            # The tensor T satisfies T[i,j,k] = 1 for all (i,j,k).
            # A rank-r decomposition gives:
            # T[i,j,k] = sum_a U[i,a] V[j,a] W[k,a]
            # Then: C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k]
            #              = sum_j sum_a U[i,a] V[j,a] W[k,a] A[i,j] B[j,k]
            #              = sum_a U[i,a] W[k,a] sum_j V[j,a] A[i,j] B[j,k]
            # 
            # This doesn't simplify nicely. I think the correct interpretation is different.
            
            # Let me try: the decomposition gives a bilinear algorithm with r multiplications.
            # The r multiplications are:
            # M_a = (sum_i U[i,a] A[i,:]) * (sum_k W[k,a] B[:,k])  -- no, this doesn't work.
            
            # I think the correct formulation is:
            # C = sum_a U_a * V_a * W_a where U_a, V_a, W_a are vectors
            # and the "multiplication" is:
            # C[i,k] += U[i,a] * V[j,a] * W[k,a] * A[i,j] * B[j,k]  -- summed over j?
            
            # I'm going in circles. Let me just use the standard verification:
            # The tensor T[i,j,k] should equal sum_a U[i,a] V[j,a] W[k,a]
            # And T[i,j,k] = 1 for all (i,j,k) for matrix multiplication.
            
            pass
        
        # Direct tensor verification
        T_true = np.ones((n, m, p))
        T_decomp = np.einsum('ia,ja,ka->ijk', U, V, W)
        if not np.allclose(T_true, T_decomp, atol=1e-10):
            return False
        return True
    
    # Start with schoolbook as fallback
    rank_fb = n * m * p
    U_fb = np.zeros((n * m, rank_fb))
    V_fb = np.zeros((m * p, rank_fb))
    W_fb = np.zeros((p * n, rank_fb))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, col] = 1
                V_fb[j * p + k, col] = 1
                W_fb[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_fb
    best_U = U_fb.copy()
    best_V = V_fb.copy()
    best_W = W_fb.copy()
    
    # Try to find lower rank decompositions
    # The tensor T[i,j,k] = 1 for all (i,j,k) has rank 1, so we should be able
    # to find rank 1 easily. But let's verify with the actual convention.
    
    # With the trace convention W[k*n+i, r], the W matrix has shape (p*n, rank).
    # Let me try rank 1 first.
    
    # For rank 1: T[i,j,k] = U[i*m+j, 0] * V[j*p+k, 0] * W[k*n+i, 0]
    # We need this to equal 1 for all (i,j,k).
    # U[i*m+j, 0] * V[j*p+k, 0] * W[k*n+i, 0] = 1
    # This must hold for all i in [0,n), j in [0,m), k in [0,p).
    
    # Let's try U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1.
    # Then the product is 1*1*1 = 1. 
    
    # But wait, the indices in U, V, W are coupled to (i,j) and (j,k) and (k,i) respectively.
    # U has row index i*m+j, V has row index j*p+k, W has row index k*n+i.
    # So for a given (i,j,k):
    # U[i*m+j, 0] * V[j*p+k, 0] * W[k*n+i, 0]
    
    # If we set all entries to 1, this works. Let me verify.
    
    U1 = np.ones((n * m, 1))
    V1 = np.ones((m * p, 1))
    W1 = np.ones((p * n, 1))
    
    # Verify: T[i,j,k] = U[i*m+j,0] * V[j*p+k,0] * W[k*n+i,0]
    # = 1 * 1 * 1 = 1 for all (i,j,k). 
    
    # So rank 1 works! But this seems too good to be true.
    # Let me double-check by actually computing a matrix product using this decomposition.
    
    def compute_product_decomposed(A, B, U, V, W, n, m, p):
        """Compute C = A*B using the tensor decomposition."""
        A = np.array(A, dtype=float)
        B = np.array(B, dtype=float)
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        
        C = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                val = 0.0
                for j in range(m):
                    for a in range(rank):
                        val += U[i*m+j, a] * V[j*p+k, a] * W[k*n+i, a] * A[i,j] * B[j,k]
                C[i,k] = val
        return C
    
    # Test rank 1
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_true = A_test @ B_test
    C_decomp = compute_product_decomposed(A_test, B_test, U1, V1, W1, n, m, p)
    
    # This won't work because the decomposition formula is wrong.
    # The correct formula should be:
    # C[i,k] = sum_j sum_a U[i,a] V[j,a] W[k,a] A[i,j] B[j,k]  -- NO
    # 
    # I think the correct bilinear algorithm is:
    # For each term a:
    #   M_a = (sum_i U[i,a] A[i,:]) dot (sum_k W[k,a] B[:,k])  -- still wrong
    #
    # Let me look at this from the tensor perspective.
    # T[i,j,k] = sum_a U[i,a] V[j,a] W[k,a]
    # C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k]
    #         = sum_j sum_a U[i,a] V[j,a] W[k,a] A[i,j] B[j,k]
    #         = sum_a U[i,a] W[k,a] sum_j V[j,a] A[i,j] B[j,k]
    
    # This is a valid bilinear algorithm with r multiplications if we precompute
    # S_a = sum_j V[j,a] A[i,j] B[j,k] ... but that depends on i and k, so it's not a single scalar.
    
    # I think the standard formulation is different. Let me try:
    # The r scalar multiplications are:
    # M_a = (sum_{i,j} U[i*m+j, a] A[i,j]) * (sum_{j,k} V[j*p+k, a] B[j,k])
    # Then: C[i,k] = sum_a W[k*n+i, a] * M_a
    
    # Let me verify this:
    # C[i,k] = sum_a W[k*n+i, a] * (sum_{i',j'} U[i'*m+j', a] A[i',j']) * (sum_{j'',k''} V[j''*p+k'', a] B[j'',k''])
    # This is NOT equal to sum_j A[i,j] B[j,k] in general.
    
    # I think the correct formulation involves the tensor being applied differently.
    
    # Let me try yet another approach. The standard Strassen-like formulation:
    # We have linear combinations of entries of A and B, multiply them, and combine.
    # For term a:
    #   X_a = sum_{i,j} U[i*m+j, a] A[i,j]
    #   Y_a = sum_{j,k} V[j*p+k, a] B[j,k]  
    #   Z_a = X_a * Y_a
    #   C[i,k] += W[k*n+i, a] * Z_a
    #
    # For this to compute C = A*B:
    # C[i,k] = sum_a W[k*n+i, a] * (sum_{i',j'} U[i'*m+j', a] A[i',j']) * (sum_{j'',k''} V[j''*p+k'', a] B[j'',k''])
    #
    # This should equal sum_j A[i,j] B[j,k].
    #
    # This is a very different constraint than T[i,j,k] = sum_a U[i,a] V[j,a] W[k,a].
    #
    # I think the problem is using the "multiplication tensor" formulation where:
    # The tensor T has shape (nm, mp, np) and T[(i,j),(j,k),(i,k)] = 1 if the indices are consistent,
    # i.e., the same j appears in all three, and 0 otherwise.
    #
    # Wait, that makes more sense! The tensor is:
    # T[(i,j),(j',k),(i',k')] = delta_{j,j'} * delta_{i,i'} * delta_{k,k'}
    # No, that's the identity, not matrix multiplication.
    
    # I think the correct tensor for matrix multiplication is:
    # T[(i,j),(j',k),(i',k')] = 1 if i=i' and k=k' and j=j', else 0
    # That's just the identity tensor, which has rank nm*mp*np... no.
    
    # I'm clearly struggling with the exact formulation. Let me just go with the
    # schoolbook rank of 42 and try to improve it using a randomized approach
    # that directly verifies the matrix product computation.
    
    def try_random_rank(rank, max_iters=5000):
        """Try to find a rank-r decomposition via random initialization and refinement."""
        if check_time():
            return None
        
        for it in range(max_iters):
            if check_time():
                return None
            
            U = np.random.randn(n * m, rank)
            V = np.random.randn(m * p, rank)
            W = np.random.randn(p * n, rank)
            
            # Verify by computing the product
            A_test = np.random.randn(n, m)
            B_test = np.random.randn(m, p)
            C_true = A_test @ B_test
            
            # Compute using the bilinear algorithm:
            # X_a = sum_{i,j} U[i*m+j, a] A[i,j]
            # Y_a = sum_{j,k} V[j*p+k, a] B[j,k]
            # C[i,k] = sum_a W[k*n+i, a] * X_a * Y_a
            
            X = U.T @ A_test.flatten()  # shape (rank,)
            Y = V.T @ B_test.flatten()  # shape (rank,)
            C_decomp = np.zeros((n, p))
            for i in range(n):
                for k in range(p):
                    C_decomp[i, k] = np.sum(W[k*n+i, :] * X * Y)
            
            if np.allclose(C_true, C_decomp, atol=1e-10):
                return U, V, W
        
        return None
    
    # Try ranks from 1 up to schoolbook
    found = False
    for r in range(1, best_rank + 1):
        if check_time():
            break
        
        result = try_random_rank(r, max_iters=2000)
        if result is not None:
            U, V, W = result
            best_rank = r
            best_U = U
            best_V = V
            best_W = W
            found = True
            break
    
    # Convert to lists
    best_U_list = best_U.tolist()
    best_V_list = best_V.tolist()
    best_W_list = best_W.tolist()
    
    return {"U": best_U_list, "V": best_V_list, "W": best_W_list}
# EVOLVE-BLOCK-END
