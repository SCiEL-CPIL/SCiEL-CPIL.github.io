"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    U_sb, V_sb, W_sb = schoolbook()
    best_rank = U_sb.shape[1]
    best_U, best_V, best_W = U_sb, V_sb, W_sb

    # Verify a candidate decomposition
    def verify(U, V, W, tol=1e-10):
        r = U.shape[1]
        # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # We need T[i,j,k] = 1 if this is the identity element in the tensor
        # Actually, for matrix multiplication (n x m) * (m x p) -> (n x p):
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # The tensor T has T[i,j,k] = delta (it's the structure tensor)
        # The decomposition is: T = sum_c U_c outer V_c outer W_c
        # where U_c[i*m+j] * V_c[j*p+k] * W_c[k*n+i] should sum to 1 for the 
        # appropriate (i,j,k) and 0 otherwise.
        # Actually the tensor for matmul is: T[i,j,k] = 1 for all valid (i,j,k)
        # No wait. The tensor T satisfies: C[i,k] = sum_{j} T[i,j,k] * A[i,j] * B[j,k]
        # So T[i,j,k] = 1 for all i,j,k. That's just the all-ones tensor? No.
        
        # Let me reconsider. The matrix multiplication tensor T is defined by:
        # T[i, j, k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
        # So T[i,j,k] = 1 for all valid i, j, k.
        
        # Wait, that's not right either. Let me think more carefully.
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # The structure tensor T has entries T[i,j,k] such that:
        # C[i,k] = sum_{i',j',k'} T[i',j',k'] * A[i',j'] * B[j',k'] * delta_{i,i'} * delta_{k,k'}
        # No, that's overcomplicating it.
        
        # The standard formulation: The tensor T has T[i,j,k] = 1 for all i,j,k
        # and the decomposition is T = sum_c U_c ⊗ V_c ⊗ W_c where
        # U_c is a vector of length n*m, V_c of length m*p, W_c of length p*n
        # such that for each (i,j,k): sum_c U_c[i*m+j] * V_c[j*p+k] * W_c[k*n+i] = 1
        
        # Hmm, but that would mean the tensor is all-ones, which has rank 1.
        # That can't be right.
        
        # Let me look at the schoolbook decomposition:
        # For each (i,j,k), we have a term where U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1
        # and all other entries in that column are 0.
        # So sum_c U_c[i*m+j] * V_c[j*p+k] * W_c[k*n+i] = 1 for the specific (i,j,k)
        # and 0 for all other (i',j',k') != (i,j,k).
        
        # So the tensor T[i,j,k] = 1 for all (i,j,k) in the domain? No, that gives rank 1.
        
        # I think the tensor is actually the "multiplication tensor" which has:
        # T[i,j,k] = 1 for all valid indices, meaning it's the all-ones tensor.
        # But that has rank 1, which contradicts the schoolbook having rank n*m*p.
        
        # Let me re-read the problem. "Find a rank-minimal exact tensor decomposition 
        # for multiplying a 2 by 4 matrix by a 4 by 6 matrix."
        
        # The tensor for matrix multiplication A (n×m) * B (m×p) → C (n×p) is:
        # T ∈ R^{(n×m) × (m×p) × (n×p)}
        # T[a, b, c] where a indexes (i,j), b indexes (j',k), c indexes (i',k')
        # T[(i,j), (j',k), (i',k')] = δ_{i,i'} * δ_{j,j'} * δ_{k,k'}
        
        # So T[(i,j), (j',k), (i',k')] = 1 if i=i', j=j', k=k', else 0.
        # This is the identity tensor in some sense.
        
        # The decomposition: T = sum_c U_c ⊗ V_c ⊗ W_c
        # U_c has length n*m, V_c has length m*p, W_c has length n*p (or p*n per the problem)
        
        # Wait, the problem says W has shape (p*n, rank). Let me re-read.
        # "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
        
        # So W has p*n entries. The index for W is (k, i) where k is from B and i is from C.
        # W[k*n + i, c]
        
        # So the decomposition is:
        # T[(i,j), (j',k), (i',k')] = sum_c U[i*m+j, c] * V[j'*p+k, c] * W[k'*n+i', c]
        # And this should equal δ_{i,i'} * δ_{j,j'} * δ_{k,k'}
        
        # Let me verify with schoolbook:
        # For column (i,j,k): U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1
        # So for T[(i,j), (j',k'), (i'',k'')]:
        # sum_c U[i*m+j,c] * V[j'*p+k',c] * W[k''*n+i'',c]
        # = 1 * 1 * 1 if (j'=j, k'=k, k''=k, i''=i) i.e., j'=j, k'=k, i''=i
        # Wait, W index is k''*n+i'', and in schoolbook W[k*n+i]=1
        # So W[k''*n+i'',c] = 1 if k''=k and i''=i
        # V[j'*p+k',c] = 1 if j'=j and k'=k
        # U[i*m+j,c] = 1
        # So the product is 1 if j'=j, k'=k, i''=i, and k''=k
        # Which means j'=j, k'=k, i''=i, k''=k
        # So T[(i,j),(j',k'),(i'',k'')] = 1 if j'=j, k'=k, i''=i, k''=k
        # = 1 if j'=j and k'=k and i''=i and k''=k
        # = δ_{j,j'} * δ_{k,k'} * δ_{i,i''} * δ_{k,k''}
        
        # But we want δ_{i,i''} * δ_{j,j'} * δ_{k,k'}
        # The extra δ_{k,k''} is a problem.
        
        # Hmm, let me reconsider the indexing. Maybe W is indexed differently.
        # The problem says "W[k*n+i,r]" - so W is indexed by (k,i) where k is the column of B / row of C?
        
        # Actually, I think the convention might be different. Let me just verify numerically.
        
        # For the purpose of verification, let's check:
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal 1 for all (i,j,k)
        # and 0 otherwise? No, that doesn't make sense for a tensor.
        
        # I think the correct check is:
        # For all (i, j, j', k, i', k'):
        # sum_c U[i*m+j, c] * V[j'*p+k, c] * W[k'*n+i', c] = δ_{i,i'} * δ_{j,j'} * δ_{k,k'}
        
        # Let me implement this check.
        r = U.shape[1]
        # Build the tensor from the decomposition
        # T[i,j,j',k,i',k'] = sum_c U[i*m+j,c] * V[j'*p+k,c] * W[k'*n+i',c]
        # Check against δ_{i,i'} * δ_{j,j'} * δ_{k,k'}
        
        # This is expensive for large dimensions, but for 2,4,6 it's manageable
        # n=2, m=4, p=6
        # i: 0..1, j: 0..3, j': 0..3, k: 0..5, i': 0..1, k': 0..5
        # Total: 2*4*4*6*2*6 = 2304 checks, each summing over r terms
        
        # Actually, let's use a more efficient approach with einsum or broadcasting
        # T = np.einsum('ic,jc,kc->ijkl', ...) - no, the indices are mixed
        
        # Let me just do it directly for small dimensions
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        for ip in range(n):
                            for kp in range(p):
                                val = 0.0
                                for c in range(r):
                                    val += U[i*m+j, c] * V[jp*p+k, c] * W[kp*n+ip, c]
                                expected = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                                if abs(val - expected) > tol:
                                    return abs(val - expected)
        return 0.0

    # Try to find a lower rank decomposition using ALS
    def try_rank(r, n_trials=5, n_iter=100, seed=42):
        np.random.seed(seed)
        for trial in range(n_trials):
            if time.time() > deadline:
                return None
            
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)
            
            for it in range(n_iter):
                if time.time() > deadline:
                    return None
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        A_ij = np.zeros(r)
                        for k in range(p):
                            A_ij += V[j*p+k, :] * W[k*n+i, :]
                        ns = np.dot(A_ij, A_ij)
                        if ns > 1e-12:
                            U[i*m+j, :] = A_ij / ns
                        else:
                            U[i*m+j, :] = 0
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        P_jk = np.zeros(r)
                        for i in range(n):
                            P_jk += U[i*m+j, :] * W[k*n+i, :]
                        ns = np.dot(P_jk, P_jk)
                        if ns > 1e-12:
                            V[j*p+k, :] = P_jk / ns
                        else:
                            V[j*p+k, :] = 0
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        Q_ki = np.zeros(r)
                        for j in range(m):
                            Q_ki += U[i*m+j, :] * V[j*p+k, :]
                        ns = np.dot(Q_ki, Q_ki)
                        if ns > 1e-12:
                            W[k*n+i, :] = Q_ki / ns
                        else:
                            W[k*n+i, :] = 0
                
                if it % 20 == 19:
                    if verify(U, V, W) < 1e-12:
                        return U.copy(), V.copy(), W.copy()
            
            if verify(U, V, W) < 1e-12:
                return U.copy(), V.copy(), W.copy()
        
        return None

    # Try ranks below schoolbook
    for r in range(best_rank - 1, 39, -1):
        if time.time() > deadline:
            break
        result = try_rank(r, n_trials=8, n_iter=150, seed=1000 + r)
        if result is not None:
            U, V, W = result
            if verify(U, V, W) < 1e-10:
                best_U, best_V, best_W = U, V, W
                best_rank = r
                break

    if time.time() > deadline:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
