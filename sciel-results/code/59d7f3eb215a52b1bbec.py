"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 7
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # Dimensions of factor matrices
    dim_u = n * m  # 9
    dim_v = m * p  # 21
    dim_w = p * n  # 21
    
    # We want to find rank-r decomposition:
    # U: (dim_u, r), V: (dim_v, r), W: (dim_w, r)
    # such that sum_t U[:,t] * V[:,t] * W[:,t] = T (the multiplication tensor)
    
    # The multiplication tensor T has T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k
    # More precisely, the tensor is: T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some valid i,j,k
    # But actually, the tensor is defined by: for each (i,j,k), there's a term e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i}
    
    # Let's think about this differently. We want to minimize rank.
    # Schoolbook gives rank 63.
    # 
    # Key insight: We can use Strassen's algorithm for 2x2x2 (rank 7 instead of 8)
    # and extend to 3x3x3 blocks.
    #
    # Strategy: Decompose 3x3x7 into overlapping 2x2x2 subproblems using Strassen.
    #
    # Actually, let's use a more systematic approach:
    # 1. Start with schoolbook (rank 63)
    # 2. Try to find lower rank decompositions using numerical optimization
    # 3. Use the known rank-23 result for 3x3x3 if we can find it
    #
    # For now, let's implement a search for low-rank decompositions.
    #
    # The tensor T for matrix multiplication M_{n,m,p} is:
    # T = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} e_i ⊗ e_j ⊗ e_k
    # where e_i is the basis vector for A[i,j], e_j for B[j,k], e_k for C[k,i]
    #
    # Actually, let me be more precise. The multiplication tensor is:
    # M = sum_{i,j,k} A_{ij} ⊗ B_{jk} ⊗ C_{ki}
    # where A_{ij} is the (i,j) entry of A, etc.
    #
    # In the flattened form:
    # U has shape (n*m, rank), indexed by (i,j) -> i*m+j
    # V has shape (m*p, rank), indexed by (j,k) -> j*p+k  
    # W has shape (p*n, rank), indexed by (k,i) -> k*n+i
    #
    # The condition is: for all A, B:
    # C = A*B
    # C[k,i] = sum_j A[i,j] * B[j,k]
    #
    # With the decomposition:
    # C[k,i] = sum_t W[k*n+i, t] * (sum_j U[i*m+j, t] * A[i,j]) * (sum_j V[j*p+k, t] * B[j,k])
    #
    # Wait, that's not quite right. Let me re-derive.
    #
    # The standard tensor decomposition for matrix multiplication:
    # We have three matrices U, V, W such that:
    # For each term t:
    #   A_t = U[:, t] reshaped to (n, m)
    #   B_t = V[:, t] reshaped to (m, p)
    #   C_t = W[:, t] reshaped to (p, n)
    #
    # Then: C = sum_t (A_t^T * A) * (B_t^T * B) * C_t ... no, that's not right either.
    #
    # Let me use the correct formulation. The multiplication tensor M_{n,m,p} is an element of
    # R^{n*m} ⊗ R^{m*p} ⊗ R^{p*n}. A rank-r decomposition is:
    # M = sum_{t=1}^r u_t ⊗ v_t ⊗ w_t
    # where u_t ∈ R^{n*m}, v_t ∈ R^{m*p}, w_t ∈ R^{p*n}
    #
    # The multiplication tensor satisfies: for any A ∈ R^{n×m}, B ∈ R^{m×p}:
    # <M, A ⊗ B ⊗ C> = tr(C^T * A * B) = sum_{i,k} C[k,i] * (A*B)[i,k]
    #
    # So: sum_{i,j,k} M[i*m+j, j*p+k, k*n+i] * A[i,j] * B[j,k] * C[k,i] = tr(C^T * A * B)
    #
    # This means M[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k, and 0 otherwise.
    #
    # A rank-r decomposition M = sum_t u_t ⊗ v_t ⊗ w_t means:
    # sum_t u_t[i*m+j] * v_t[j*p+k] * w_t[k*n+i] = M[i*m+j, j*p+k, k*n+i]
    #
    # So we need: sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1 if (i,j,k) is valid, 0 otherwise
    # But actually, since the tensor only has non-zero entries at specific positions,
    # we need the sum to equal 1 at those positions and 0 elsewhere.
    #
    # However, for the matrix multiplication to work correctly, we need:
    # For all A, B: C = A*B
    # where C[k,i] = sum_t W[k*n+i, t] * (sum_j U[i*m+j, t] * A[i,j]) * (sum_j V[j*p+k, t] * B[j,k])
    #
    # Wait, I think the correct interpretation is:
    # Let a_t = U[:,t] (length n*m), b_t = V[:,t] (length m*p), c_t = W[:,t] (length p*n)
    # Then: C = sum_t c_t * (a_t^T ⊗ A) * (b_t^T ⊗ B) ... this is getting confusing.
    #
    # Let me just use the schoolbook as baseline and try to improve it.
    
    # Schoolbook decomposition (rank 63)
    def schoolbook_decomposition():
        rank = n * m * p
        U = np.zeros((dim_u, rank))
        V = np.zeros((dim_v, rank))
        W = np.zeros((dim_w, rank))
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        
        return U, V, W, rank
    
    # Verify a decomposition
    def verify_decomposition(U, V, W, tol=1e-6):
        """Verify that the decomposition correctly computes matrix multiplication."""
        rank = U.shape[1]
        
        # Test with random matrices
        random.seed(42)
        for trial in range(5):
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            C_expected = A @ B
            
            # Compute C using the decomposition
            C_computed = np.zeros((p, n))
            for t in range(rank):
                a_t = U[:, t].reshape(n, m)
                b_t = V[:, t].reshape(m, p)
                c_t = W[:, t].reshape(p, n)
                
                # The contribution of term t is:
                # (a_t^T * A) * (b_t^T * B) * c_t
                # where a_t^T * A is a scalar, b_t^T * B is a scalar
                scalar_a = np.sum(a_t * A)
                scalar_b = np.sum(b_t * B)
                C_computed += scalar_a * scalar_b * c_t
            
            if not np.allclose(C_computed, C_expected, atol=tol):
                return False
        
        return True
    
    # Try to find a lower rank decomposition using optimization
    def optimize_decomposition(initial_rank, max_iterations=1000, lr=0.01):
        """Try to optimize a decomposition to reduce rank."""
        # This is a complex optimization problem. For now, let's just use schoolbook.
        # In a more advanced implementation, we could use gradient-based methods
        # to find lower rank decompositions.
        pass
    
    # For now, return the schoolbook decomposition
    U, V, W, rank = schoolbook_decomposition()
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
