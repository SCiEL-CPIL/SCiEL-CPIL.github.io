import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    # Start with schoolbook rank 30, then try to reduce
    # The tensor T[i,j,k] = 1 for all i,j,k
    
    # Try rank 27, 26, ... down to some minimum
    # Use a simple iterative approach: start with random init and optimize
    
    best_rank = n * m * p  # 30
    best_U = None
    best_V = None
    best_W = None
    
    # Schoolbook solution
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    best_U, best_V, best_W = U, V, W
    best_rank = rank
    
    # Try to find lower rank solutions
    # For a given rank r, we need U (n*m x r), V (m*p x r), W (p*n x r)
    # such that for all i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    
    # This is a non-convex problem. Let's try a simple approach:
    # Parameterize and use gradient descent or a structured ansatz.
    
    # Structured approach: exploit that n=2 is small.
    # For each (i,k) pair, we have a constraint on j.
    
    # Let's try rank 28 with a structured ansatz.
    # The knowledge says rank 28 works. Let me try to construct it.
    
    # Approach: use 28 terms. Think of it as 30 - 2.
    # Can we merge two schoolbook terms?
    
    # Schoolbook term for (i,j,k): U[i*m+j,c]=1, V[j*p+k,c]=1, W[k*n+i,c]=1
    # Two terms (i1,j1,k1) and (i2,j2,k2) can be merged if there exists
    # U,V,W vectors such that the product covers both.
    
    # For merging to work with the all-ones tensor, we need:
    # For term c covering (i1,j1,k1) and (i2,j2,k2):
    # U[i1*m+j1,c]*V[j1*p+k1,c]*W[k1*n+i1,c] = 1
    # U[i2*m+j2,c]*V[j2*p+k2,c]*W[k2*n+i2,c] = 1
    # And for all other (i,j,k), the contribution should be 0 or compensated.
    
    # This is complex. Let me try a numerical optimization approach instead.
    
    def try_rank(r, max_iter=5000, lr=0.01):
        """Try to find a rank-r decomposition."""
        np.random.seed(42)
        U = np.random.randn(n*m, r) * 0.1
        V = np.random.randn(m*p, r) * 0.1
        W = np.random.randn(p*n, r) * 0.1
        
        for it in range(max_iter):
            # Compute gradient of loss = sum_{i,j,k} (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1)^2
            # This is expensive, so let's use a simplified approach.
            
            # Actually, let's use a different strategy: fix two matrices and solve for the third.
            pass
        
        return None
    
    # Let me try a more direct approach: use the known rank-28 structure.
    # For 2x3x5, one way to get rank 28:
    # Use 2 terms for the "cross" structure and 26 for the rest.
    
    # Actually, let me just return the schoolbook rank-30 solution for now,
    # but try a quick numerical search for lower rank.
    
    # Simpler approach: try specific structured decompositions.
    
    # Rank 28 attempt: 
    # Think of the 2x3 input as having 6 elements, 3x5 output as 15 elements.
    # The tensor is 2x3x5 all ones.
    
    # One known technique: for an n x m x p all-ones tensor, 
    # rank <= n*m + m*p + p*n - m - p - n + 1 (inclusion-exclusion type bound)
    # For 2x3x5: 6 + 15 + 10 - 3 - 5 - 2 + 1 = 22? That seems too optimistic.
    
    # Actually, the rank of the all-ones tensor of shape n x m x p is known.
    # For 2 x 3 x 5, let me just try to find a solution numerically.
    
    # Let me try a very simple structured ansatz for rank 28:
    # Use 28 terms where 26 are schoolbook and 2 are combined terms.
    
    # Or better: let me just do a random restart optimization for small ranks.
    
    def optimize_rank(r, n_iter=200, n_restarts=3):
        for restart in range(n_restarts):
            np.random.seed(100 + restart)
            U = np.random.randn(n*m, r) * 0.5
            V = np.random.randn(m*p, r) * 0.5
            W = np.random.randn(p*n, r) * 0.5
            
            for it in range(n_iter):
                # Compute the tensor product and loss
                # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                # We want T[i,j,k] = 1 for all i,j,k
                
                # Build the full tensor (small: 2x3x5 = 30 entries)
                T = np.zeros((n, m, p))
                for c in range(r):
                    u = U[:, c].reshape(n, m)
                    v = V[:, c].reshape(m, p)
                    w = W[:, c].reshape(p, n)
                    # T[i,j,k] += U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                T[i,j,k] += u[i,j] * v[j,k] * w[k,i]
                
                loss = np.sum((T - 1)**2)
                
                if loss < 1e-10:
                    return U, V, W, loss
                
                # Gradient descent (simplified)
                # dLoss/dU[i*m+j,c] = 2*(T[i,j,k]-1)*V[j*p+k,c]*W[k*n+i,c] summed over k
                # This is getting complex. Let me use a different approach.
                pass
        
        return None
    
    # Given the complexity, let me just return the known rank-28 solution if I can construct it,
    # otherwise return schoolbook rank 30.
    
    # Let me try to construct rank 28 explicitly.
    # Idea: the 2x3x5 all-ones tensor. 
    # Note that for fixed i, the 3x5 slice is all ones, rank 3.
    # For fixed j, the 2x5 slice is all ones, rank 2.
    # For fixed k, the 2x3 slice is all ones, rank 2.
    
    # A rank-r decomposition: T[i,j,k] = sum_c a_c[i] * b_c[j] * c_c[k]
    # where a_c is 2-vector, b_c is 3-vector, c_c is 5-vector.
    # This is the CP decomposition of the all-ones tensor.
    
    # The CP rank of the all-ones tensor of shape 2x3x5 is at most min(2*3, 3*5, 2*5) = 6?
    # No, that's not right. The CP rank of an all-ones tensor is 1!
    # T[i,j,k] = 1 = 1*1*1, so rank 1 in CP form!
    
    # Wait, but the interface is NOT CP decomposition! The interface has:
    # U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # This is NOT a simple CP decomposition. The indices are coupled differently.
    
    # In CP: T[i,j,k] = sum_c a_c[i] * b_c[j] * c_c[k]
    # Here: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    
    # The difference: in CP, each factor depends on one index. Here:
    # - U depends on (i,j)
    # - V depends on (j,k)  
    # - W depends on (k,i)
    
    # This is a "cyclic" or "tensor network" style decomposition, not CP.
    
    # So the rank can be much higher than the CP rank.
    
    # For this cyclic decomposition, the schoolbook rank is n*m*p = 30.
    # The question is whether we can do better.
    
    # Let me think about it as: for each c, we have three matrices:
    # U_c (n x m), V_c (m x p), W_c (p x n)
    # and T[i,j,k] = sum_c U_c[i,j] * V_c[j,k] * W_c[k,i]
    
    # This is equivalent to: T = sum_c (U_c ⊙ V_c ⊙ W_c) where the product is
    # (U_c ⊙ V_c ⊙ W_c)[i,j,k] = U_c[i,j] * V_c[j,k] * W_c[k,i]
    
    # For the all-ones tensor, we need: sum_c U_c[i,j] * V_c[j,k] * W_c[k,i] = 1
    
    # This is related to matrix multiplication complexity!
    # Specifically, this is the bilinear complexity of multiplying a 2x3 by 3x5 matrix,
    # but with the output indexed as (k,i) instead of (i,k).
    
    # Wait, let me re-examine. The output C[i,k] = sum_j A[i,j] * B[j,k].
    # In the decomposition: C[i,k] = sum_c sum_j U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    
    # Hmm, that doesn't look like standard bilinear complexity. Let me re-read.
    
    # Actually, I think the decomposition is:
    # C[i,k] = sum_c (sum_j U[i*m+j,c] * V[j*p+k,c]) * W[k*n+i,c]
    
    # No wait, that's not right either. Let me think about what "exactly decompose the matrix multiplication tensor" means.
    
    # The matrix multiplication tensor for (n,m,p) is T[i,j,k] = 1 where:
    # - i indexes rows of A (and C): 0..n-1
    # - j indexes columns of A / rows of B: 0..m-1  
    # - k indexes columns of B (and C): 0..p-1
    
    # The bilinear form: C[i,k] = sum_j A[i,j] * B[j,k]
    
    # A rank-r decomposition gives r bilinear forms:
    # C[i,k] = sum_c M_c[i,k] where M_c[i,k] = (sum_j U_c[i,j] * A[i,j]) * (sum_k V_c[j,k] * B[j,k])
    
    # Hmm, I'm confusing myself. Let me just focus on the interface:
    # U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank)
    # Condition: for all i,j,k: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    # This is a trilinear form that equals 1 for all (i,j,k).
    
    # Let me just try to find a solution numerically for small ranks.
    
    # Given time constraints, let me return the schoolbook solution (rank 30)
    # but with a quick attempt at rank 28 using a structured approach.
    
    # Structured rank-28 attempt:
    # Use the fact that for the all-ones tensor, we can have some terms be "sparse."
    
    # Let me try: 28 terms where most are schoolbook but 2 pairs are merged.
    
    # Actually, let me just return schoolbook rank 30 as a safe baseline.
    # The knowledge says rank 28 is achievable, but I need to construct it correctly.
    
    # Let me try one more thing: a simple optimization for rank 28.
    
    def find_decomposition(r, time_limit=10):
        start = time.time()
        for seed in range(10):
            if time.time() - start > time_limit:
                break
            np.random.seed(seed)
            U = np.random.randn(n*m, r) * 0.3
            V = np.random.randn(m*p, r) * 0.3
            W = np.random.randn(p*n, r) * 0.3
            
            for it in range(200):
                if time.time() - start > time_limit:
                    break
                # Compute T and gradient
                T = np.zeros((n, m, p))
                for c in range(r):
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                T[i,j,k] += U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                
                loss = np.sum((T - 1)**2)
                if loss < 1e-12:
                    return U, V, W
                
                # Gradient
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            err = 2 * (T[i,j,k] - 1)
                            for c in range(r):
                                dU[i*m+j,c] += err * V[j*p+k,c] * W[k*n+i,c]
                                dV[j*p+k,c] += err * U[i*m+j,c] * W[k*n+i,c]
                                dW[k*n+i,c] += err * U[i*m+j,c] * V[j*p+k,c]
                
                U -= 0.001 * dU
                V -= 0.001 * dV
                W -= 0.001 * dW
        
        return None
    
    # Try ranks from 27 down to 25
    for r in [27, 26, 25, 28]:
        result = find_decomposition(r, time_limit=5)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook rank 30
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
