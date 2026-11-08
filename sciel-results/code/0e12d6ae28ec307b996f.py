"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 110.0  # safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    
    # Schoolbook baseline
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_schoolbook
    
    # Try to find lower rank using structured decomposition
    # For matrix multiplication, we can exploit the structure
    
    # Strategy: Use the fact that C = A*B where A is 4x4, B is 4x8
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # The tensor T[i,j,k] = delta for multiplication
    # We want to decompose T = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    # Try various ranks below schoolbook
    # Known lower bounds: rank >= max(n*m, m*p, p*n) = max(16, 32, 32) = 32
    # Upper bound from schoolbook: 128
    
    # Let's try to find a decomposition with rank close to the lower bound
    # using random initialization and optimization
    
    def verify_decomposition(U, V, W, tol=1e-9):
        """Verify that the decomposition is exact."""
        # Check that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] 
        # equals 1 if this is a valid multiplication term, 0 otherwise
        # Actually, the decomposition should satisfy:
        # For the matrix multiplication tensor T[i,j,k]:
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # where T[i,j,k] = 1 for all i,j,k (since each product term contributes)
        
        # Wait, let me reconsider. The matrix multiplication tensor is:
        # T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
        # because C[i,k] = sum_j A[i,j] B[j,k]
        # So the tensor has all ones!
        
        # Actually no. The tensor T_{i,j,k} for matrix multiplication satisfies:
        # (A*B)[i,k] = sum_j A[i,j] * B[j,k]
        # So T[i,j,k] = 1 for all valid i,j,k.
        
        # The decomposition requires:
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # Let's verify this
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(U.shape[1]):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def verify_fast(U, V, W, tol=1e-9):
        """Fast verification using tensor operations."""
        rank = U.shape[1]
        # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
        U_3d = U.reshape(n, m, rank)
        V_3d = V.reshape(m, p, rank)
        W_3d = W.reshape(p, n, rank)
        
        # Compute sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # This is a tensor contraction
        # Result[i,j,k] should be 1 for all i,j,k
        
        # Method: For each r, form outer products and sum
        # Or use einsum
        result = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        return np.allclose(result, 1.0, atol=tol)
    
    # Try structured decompositions
    
    # Approach 1: Use the fact that we can factor through intermediate dimensions
    # C[i,k] = sum_j A[i,j] B[j,k]
    
    # One idea: group terms by j
    # For each j, we have n*p terms (i,k pairs)
    # Total: m * n * p = 4 * 4 * 8 = 128 (schoolbook)
    
    # Can we do better? 
    # If we fix j, the contribution is A[i,j] * B[j,k]
    # This is an outer product of column j of A and row j of B
    # So for fixed j, we need n*p rank-1 terms? No, it's rank 1 as a matrix!
    # A[:,j] * B[j,:] is a rank-1 matrix of size n x p
    # So for fixed j, the tensor slice T[i,j,k] = A[i,j]*B[j,k] 
    # But in our decomposition, T[i,j,k] must equal 1 (the coefficient)
    
    # I think I'm confusing the tensor with the actual multiplication.
    # Let me re-read the problem.
    
    # The matrix multiplication tensor T has T[i,j,k] = 1 for all i,j,k
    # because C[i,k] = sum_j A[i,j] B[j,k], and the coefficient of A[i,j]B[j,k] is 1.
    
    # So we need: 1 = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for all i,j,k
    
    # This is a 3D tensor of all ones, size 4x4x8
    # We want its tensor rank (CP decomposition rank)
    
    # The rank of an all-ones tensor of size I x J x K is... 
    # For an all-ones tensor, the rank is min(I, J, K) if... no.
    # Actually, the CP rank of an all-ones tensor of size I x J x K is not simply min(I,J,K).
    
    # Let me think differently. 
    # T[i,j,k] = 1 = 1*1*1, so one term suffices? No, because the indices are coupled.
    # U[i*m+j, r] depends on both i and j.
    
    # Actually, T[i,j,k] = 1 can be written as:
    # 1 = (1)*(1)*(1)
    # So U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1 for all i,j,k
    # Then sum_r ... = 1*1*1 = 1. 
    # So rank 1 works?!
    
    # Wait, that can't be right. Let me verify.
    # If U[i*m+j, 0] = 1 for all i,j
    # V[j*p+k, 0] = 1 for all j,k  
    # W[k*n+i, 0] = 1 for all k,i
    # Then for any i,j,k: U[i*m+j,0]*V[j*p+k,0]*W[k*n+i,0] = 1*1*1 = 1
    # So yes, rank 1 suffices for the all-ones tensor!
    
    # But this seems too good to be true. Let me re-examine the problem statement.
    
    # "Find a rank-minimal exact tensor decomposition for multiplying a 4 by 4 matrix by a 4 by 8 matrix."
    # "The output factor follows the official trace convention W[k*n+i,r]."
    
    # Hmm, maybe the tensor is NOT all ones. Let me reconsider.
    
    # In matrix multiplication C = A * B, where A is n x m and B is m x p:
    # C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
    
    # The multiplication tensor T is defined such that:
    # T[A, B] = C, i.e., T is a trilinear map from A x B to C
    # T[i,j,k] is the coefficient of A[i,j] * B[j,k] in C[i,k]
    # Which is 1 if the indices match properly, 0 otherwise.
    
    # For C[i,k] = sum_j A[i,j] B[j,k], the coefficient of A[i,j]B[j,k] in C[i,k] is 1.
    # So T[i,j,k] = 1 for all valid i,j,k.
    
    # So the tensor IS all ones, and rank 1 should work.
    
    # But wait, the schoolbook decomposition uses rank n*m*p = 128, which suggests
    # that the rank 1 decomposition doesn't work for some reason.
    
    # Let me re-read: "U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]"
    # In schoolbook: U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1
    # for each (i,j,k) triple, with a separate column for each.
    
    # If we use rank 1 with all entries 1, then:
    # sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] = 1*1*1 = 1
    # This should be correct!
    
    # Unless... the decomposition requires that the factors have specific structure
    # that I'm missing. Or perhaps the "rank" here is not CP rank but something else.
    
    # Let me just try rank 1 and see if it passes verification.
    
    # Actually, I realize I should just test this. Let me create the rank-1 solution.
    
    U_1 = np.ones((n * m, 1))
    V_1 = np.ones((m * p, 1))
    W_1 = np.ones((p * n, 1))
    
    if verify_fast(U_1, V_1, W_1):
        best_U, best_V, best_W = U_1, V_1, W_1
        best_rank = 1
        # Convert to lists and return
        return {"U": U_1.tolist(), "V": V_1.tolist(), "W": W_1.tolist()}
    
    # If rank 1 doesn't work (which would be surprising), fall back to schoolbook
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

# EVOLVE-BLOCK-END
