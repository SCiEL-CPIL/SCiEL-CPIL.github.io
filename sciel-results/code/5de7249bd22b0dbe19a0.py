import numpy as np
from scipy.optimize import lsq_linear

def construct():
    n, m, p = 3, 4, 8
    # Start with schoolbook and try to reduce rank
    # For (3,4,8), we can likely achieve rank around 60-70
    
    # Try a structured approach: for each j, use a transform on k
    # We'll build U, V, W with rank r and solve for the entries
    
    # Let's try rank = 64 first (which is 2*4*8, using 2x compression on k)
    # Actually, let's be more aggressive and try lower ranks
    
    # The tensor equation is: for all i,j,k:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # and for all other combinations, it should be 0
    
    # This is a system of equations in the entries of U, V, W
    
    # Let's use a different strategy: exploit the Kronecker product structure
    # For fixed j, the sub-tensor is: sum_k A[i,j] * B[j,k] * C[k,i]
    # = A[i,j] * (B[j,:] · C[:,i])
    
    # So for each j, we need to decompose the outer product structure
    
    # Let me try: for each j, use r_j terms to represent the 3x8 multiplication
    # The 3x8 matrix multiplication (A[j,:] as row, B[j,:] as row) has a tensor
    
    # Actually, let me just try to find a valid decomposition with reduced rank
    # by using linear dependencies
    
    # Strategy: Use the fact that we can combine the 8 k-values using 4 basis vectors
    # if we accept some loss, but we need exactness
    
    # Let me try a direct construction with rank = n*m*p/2 = 48
    # This would require each "effective" term to cover 2 k-values
    
    # For exactness with integer/half-integer entries, we can use:
    # For k = 2t and k = 2t+1, use combinations
    
    # Let me try rank = 48 with the following scheme:
    # For each i,j and each t in 0..3, create one term that handles k=2t and k=2t+1
    
    # But this requires the contributions to be separable, which they're not directly
    
    # Alternative: use the fact that sum over k can be split
    # sum_{k=0}^{7} A[i,j]*B[j,k]*C[k,i] = 
    # A[i,j] * (sum_{t=0}^{3} B[j,2t]*C[2t,i] + B[j,2t+1]*C[2t+1,i])
    
    # This doesn't directly give a low-rank decomposition
    
    # Let me try a more practical approach: use numpy to find a low-rank approximation
    # and then round to half-integers
    
    # First, get the full tensor
    tensor = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                tensor[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Try to find a low-rank decomposition
    # We'll use the CP decomposition via optimization
    
    # Start with a reasonable rank estimate
    # For (3,4,8), literature suggests rank might be around 60-70
    
    # Let's try rank = 64
    rank = 64
    np.random.seed(42)
    
    # Initialize
    U = np.random.randn(n*m, rank)
    V = np.random.randn(m*p, rank)
    W = np.random.randn(p*n, rank)
    
    # Optimize: minimize ||tensor - sum_c U_c ⊗ V_c ⊗ W_c||_F
    # This is non-convex, but we can try alternating least squares
    
    best_error = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for iteration in range(50):
        # Update U: for each column c of U, solve least squares
        for c in range(rank):
            # tensor = U[:,c] ⊗ V[:,c] ⊗ W[:,c] + other terms
            # For fixed c, we want to find U[:,c] to minimize
            # ||tensor - sum_{c'} U[:,c'] ⊗ V[:,c'] ⊗ W[:,c']||
            # This is equivalent to: for each (a,b), 
            # tensor[a,b,:] = U[a,c]*V[b,c]*W[:,c] + sum_{c'≠c} ...
            
            # Residual for this c:
            residual = tensor.copy()
            for c2 in range(rank):
                if c2 != c:
                    residual -= np.outer(np.outer(U[:,c2], V[:,c2]), W[:,c2])
            
            # Now: residual[a,b,:] ≈ U[a,c]*V[b,c]*W[:,c]
            # For each a,b: residual[a,b,:] = U[a,c]*V[b,c]*W[:,c]
            # Let x = U[a,c]*V[b,c], then residual[a,b,:] = x * W[:,c]
            # So: residual[a,b,:] / W[:,c] should be constant = x
            
            # We can solve for U[:,c] and V[:,c] jointly
            # This is still tricky. Let me use a different approach.
            
            break  # This is getting complex; let me use a simpler method
        
        # Let me use a gradient-based approach instead
        break
    
    # The above is getting too complex. Let me try a known result or a simpler construction.
    
    # For (3,4,8), let me try the following:
    # Use the fact that we can represent the 8-dimensional k-space with 4 basis vectors
    # using a Hadamard-like transform, but that won't give exact integer results
    
    # Let me just return the schoolbook solution since finding a provably optimal
    # low-rank decomposition is research-level. But the task asks for rank-minimal.
    
    # Actually, let me try a simple improvement: note that the tensor has structure.
    # For each j, the sub-tensor involves 3*8 = 24 entries in the first and third modes.
    # The rank of each such sub-tensor (for fixed j) is at most min(3,8,3) = 3 if we
    # consider it as a 3x8x3 tensor, but it's actually 3*8 = 24 in the naive case.
    
    # Wait, for fixed j, the tensor slice is:
    # T_j[i, k, l] where i in 0..2, k in 0..7, l in 0..2
    # T_j[i,k,l] = 1 if l = i (since W[k*n+i] means the index is k*n+i, and we need
    # the third index to match i from the first mode)
    # Actually, W[k*n+i] means row k*n+i, so for the third mode, index = k*n+i.
    # So for fixed j, we have T[i, k, k*n+i] = 1 for all i,k.
    # This is a 3x8x24 tensor (but only 24 non-zero entries).
    
    # The rank of this sub-tensor: it's sum_{i,k} e_i ⊗ e_k ⊗ e_{k*n+i}
    # This has rank 24 (since all e_{k*n+i} are distinct).
    
    # So the total rank is sum_j rank(T_j) = 4*24 = 96, confirming schoolbook.
    
    # To get lower rank, we need to share terms across different j values.
    
    # Let me try: can we find terms that work for multiple j values?
    # A term (u, v, w) contributes to T[i,j,k] where i = u's index, j = v's index, k = w's index.
    # Specifically, u is in R^{12} (indices 0..11 for i*m+j), v in R^{32} (j*p+k), w in R^{24} (k*n+i).
    
    # For a term to contribute to T[i,j,k] = 1, we need:
    # u[i*m+j] * v[j*p+k] * w[k*n+i] = 1
    # And for all other (i',j',k'), u[i'*m+j'] * v[j'*p+k'] * w[k'*n+i'] = 0
    
    # This is very restrictive. The schoolbook works because each term is supported on a single (i,j,k).
    
    # To get lower rank, we need terms with larger support.
    
    # Let me try: a term that is non-zero for two different k values with the same (i,j).
    # Say, for fixed i,j, and k=0 and k=1:
    # u[i*m+j] = a, v[j*p+0] = b, v[j*p+1] = c, w[0*n+i] = d, w[1*n+i] = e
    # We need: a*b*d = 1 and a*c*e = 1, and all other products = 0.
    # So a ≠ 0, b ≠ 0, d ≠ 0, c ≠ 0, e ≠ 0.
    # And for any other (i',j',k'), the product must be 0.
    # This means u[i'*m+j'] = 0 for (i',j') ≠ (i,j), OR v[j'*p+k'] = 0, OR w[k'*n+i'] = 0.
    
    # If we set u to be non-zero only at i*m+j, v non-zero only at j*p+0 and j*p+1, and w non-zero only at 0*n+i and 1*n+i, then:
    # The term contributes to (i,j,0) and (i,j,1) only.
    # So a*b*d = 1 and a*c*e = 1.
    # We can set a=1, b=1, d=1, c=1, e=1.
    # This gives one term that covers two (i,j,k) pairs!
    
    # So we can pair up k values. For each (i,j), we have 8 k values. We can pair them into 4 pairs.
    # Each pair gives one term. So for each (i,j), we need 4 terms instead of 8.
    # Total: 3*4*4 = 48 terms!
    
    # But wait, we need to check that terms for different (i,j) don't interfere.
    # Term for (i,j) with k pair (2t, 2t+1): 
    #   u non-zero at i*m+j
    #   v non-zero at j*p+2t and j*p+2t+1
    #   w non-zero at 2t*n+i and (2t+1)*n+i
    # 
    # For a different (i',j'), if i'≠i or j'≠j, then u[i'*m+j'] = 0, so no contribution.
    # Good.
    # 
    # For the same (i,j) but different t, the v and w supports are disjoint (different k ranges), so no interference.
    # 
    # So rank 48 should work!
    
    rank = 48
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for t in range(p // 2):  # 4 pairs
                k1 = 2 * t
                k2 = 2 * t + 1
                U[i * m + j][col] = 1
                V[j * p + k1][col] = 1
                V[j * p + k2][col] = 1
                W[k1 * n + i][col] = 1
                W[k2 * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
