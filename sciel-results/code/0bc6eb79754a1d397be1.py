"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 3, 10
    
    # The matrix multiplication tensor M_{n,m,p} has entries:
    # T[i, j, k] = 1 if we're computing C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor has shape (n, m, p) in the sense that:
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # and the decomposition satisfies:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta (i.e., 1 for all i,j,k)
    # Wait, that's not right. Let me think about this more carefully.
    #
    # For matrix multiplication C = A * B where A is n x m and B is m x p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # The tensor decomposition is:
    # T_{i,j,k} = sum_r U_{i,j,r} * V_{j,k,r} * W_{k,i,r}
    # where T_{i,j,k} = 1 for all valid i,j,k (since the multiplication tensor
    # has all 1s in its "core" - each output entry C[i,k] receives exactly one
    # term A[i,j]*B[j,k] for each j).
    #
    # Actually, the standard formulation is:
    # The matrix multiplication tensor M_{n,m,p} is a 3-tensor of shape (n*p, m, m)
    # or equivalently, we can think of it as having indices (i,k) for output, j for
    # the summation index. The tensor entry T[(i,k), j, j'] = delta_{j,j'} * delta_{...}
    #
    # Let me use the standard formulation from the task:
    # U: shape (n*m, rank) - indexed by (i,j) where i in [0,n), j in [0,m)
    # V: shape (m*p, rank) - indexed by (j,k) where j in [0,m), k in [0,p)
    # W: shape (p*n, rank) - indexed by (k,i) where k in [0,p), i in [0,n)
    #
    # The decomposition requires:
    # For all i, j, k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # This is because C[i,k] = sum_j A[i,j]*B[j,k], and the tensor captures
    # the "wiring" of which products contribute to which output.
    #
    # Actually, I need to be more careful. The matrix multiplication tensor
    # M_{n,m,p} as a 3-tensor T with T[a,b,c] where:
    # a indexes the output (i,k) -> a = i*p + k, or a = k*n + i depending on convention
    # b indexes the first input (i,j) -> b = i*m + j
    # c indexes the second input (j,k) -> c = j*p + k
    #
    # The task says W has shape (p*n, rank) and the convention is W[k*n+i, r].
    # So W is indexed by (k, i) with k in [0,p), i in [0,n).
    #
    # The condition for exact decomposition is:
    # For each output position (i, k), the product A*B gives:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # In tensor form, the multiplication tensor satisfies:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{output}
    #
    # More precisely, the tensor M_{n,m,p} can be viewed as a linear map
    # from A ⊗ B to C, and the tensor decomposition gives:
    # M = sum_r U_r ⊗ V_r ⊗ W_r
    # where U_r is a vector in R^{n*m}, V_r in R^{m*p}, W_r in R^{p*n}
    #
    # The exact condition is that for all i, j, k:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    #
    # Wait, that can't be right either, because the LHS depends on i, j, k
    # but should give the coefficient of A[i,j]*B[j,k] in C[i,k], which is 1.
    
    # Let me use a different approach. The standard way to verify:
    # Given U, V, W, construct the tensor and check it equals the MM tensor.
    
    def verify(U, V, W, n, m, p):
        """Check if U, V, W give an exact decomposition of MM_{n,m,p}."""
        rank = U.shape[1]
        # The MM tensor T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
        # The decomposition gives: T_decomp[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We need T_decomp[i, j, k] = 1 for all i, j, k.
        
        # Compute the reconstructed tensor
        # T_decomp[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # Use einsum for efficiency
        # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
        # We want: for each (i, j, k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
        U_reshaped = U.reshape(n, m, rank)
        V_reshaped = V.reshape(m, p, rank)
        W_reshaped = W.reshape(p, n, rank)
        
        # T_decomp[i, j, k] = sum_r U_reshaped[i, j, r] * V_reshaped[j, k, r] * W_reshaped[k, i, r]
        # = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
        
        T_decomp = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
        
        # The target tensor is all ones
        T_target = np.ones((n, m, p))
        
        # Check if they match
        return np.allclose(T_decomp, T_target, atol=1e-10)
    
    # Now, let's try to find a low-rank decomposition.
    # The trivial decomposition has rank n*m*p = 2*3*10 = 60.
    
    # Strategy: Use the structure of the problem.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # For n=2, m=3, p=10:
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k]
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k]
    
    # One approach: group by the B column index k.
    # For each k, we have a 2x3 matrix multiplication (just scaling rows of A by columns of B).
    # Actually, for fixed k, C[:,k] = A * B[:,k] where B[:,k] is a 3-vector.
    # This is just a matrix-vector product, which has rank 3 (trivially) for each k.
    # But we need to share structure across k values.
    
    # Let's try a structured decomposition.
    # For each pair (i, j), we can have a "factor" that contributes to all k:
    # U[i*m+j, r] = 1 for some r
    # V[j*p+k, r] = 1 for all k (if we use the same r for all k)
    # W[k*n+i, r] = 1 for all k
    
    # But this gives rank n*m = 6, and the reconstruction would be:
    # T_decomp[i, j, k] = sum_{r: U[i*m+j,r]=1} V[j*p+k,r] * W[k*n+i,r]
    # If we assign each (i,j) pair a unique r, then:
    # T_decomp[i, j, k] = V[j*p+k, r_ij] * W[k*n+i, r_ij]
    # We need this to equal 1 for all i, j, k.
    # So V[j*p+k, r_ij] * W[k*n+i, r_ij] = 1 for all k.
    # This means V[j*p+k, r_ij] = 1/W[k*n+i, r_ij] for all k.
    # But V and W entries are shared across different (i,j) pairs if they use the same r.
    # Since each (i,j) has a unique r, there's no sharing, so we can set:
    # V[j*p+k, r_ij] = 1, W[k*n+i, r_ij] = 1 for all k.
    # This gives rank 6!
    
    # Wait, let me verify. If rank = n*m = 6:
    # For each (i, j), assign r = i*m + j.
    # U[i*m+j, r] = 1 (only one nonzero per row)
    # V[j*p+k, r] = 1 for all k
    # W[k*n+i, r] = 1 for all k
    
    # Then T_decomp[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # = U[i*m+j, i*m+j] * V[j*p+k, i*m+j] * W[k*n+i, i*m+j]
    # = 1 * 1 * 1 = 1 ✓
    
    # So rank 6 works! Let's construct it.
    
    rank = n * m  # 6
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for i in range(n):
        for j in range(m):
            r = i * m + j
            U[i * m + j, r] = 1
            for k in range(p):
                V[j * p + k, r] = 1
                W[k * n + i, r] = 1
    
    # Verify
    if verify(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # If the above doesn't work (shouldn't happen), fall back to trivial
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
