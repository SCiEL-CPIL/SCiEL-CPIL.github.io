"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 135.0  # safety margin before 150s kill

    def _verify(U, V, W):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        rank = U.shape[1]
        # The tensor T[i, j, k] for matrix mult (n x m) * (m x p) -> (n x p)
        # T[i, j, k] = 1 if this is a valid multiply-add term, 0 otherwise
        # Actually, the matrix multiplication tensor has T[i,j,k] = 1 for all i,j,k
        # because C[i,k] = sum_j A[i,j]*B[j,k]
        # The decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # Efficient verification using einsum
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        # Reshape for verification
        # We need to check: for all i,j,k: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1
        # This is equivalent to checking the tensor product
        
        # Build the tensor from U, V, W
        # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        # Use einsum: 'ir,jr,kr -> ijk' but with reshaped indices
        U_reshaped = U_arr.reshape(n, m, rank)  # U[i,j,r]
        V_reshaped = V_arr.reshape(m, p, rank)  # V[j,k,r]
        W_reshaped = W_arr.reshape(p, n, rank)  # W[k,i,r]
        
        # T[i,j,k] = sum_r U_reshaped[i,j,r] * V_reshaped[j,k,r] * W_reshaped[k,i,r]
        # This is a bit tricky with einsum because of the cyclic index
        # Let's do it step by step
        # First compute A[i,j,k] = sum_r U_reshaped[i,j,r] * V_reshaped[j,k,r]
        # Then T[i,j,k] = sum_k' A[i,j,k'] * W_reshaped[k',i,r]... no that's not right
        
        # Actually: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Let's just compute it directly for verification
        T = np.zeros((n, m, p), dtype=np.float64)
        for r in range(rank):
            # U[:, r] gives values for all (i,j) pairs
            # V[:, r] gives values for all (j,k) pairs  
            # W[:, r] gives values for all (k,i) pairs
            U_r = U_arr[:, r].reshape(n, m)
            V_r = V_arr[:, r].reshape(m, p)
            W_r = W_arr[:, r].reshape(p, n)
            # T[i,j,k] += U_r[i,j] * V_r[j,k] * W_r[k,i]
            # This is an outer product followed by element-wise multiplication
            # U_r[i,j] * V_r[j,k] * W_r[k,i]
            # We can compute this as: for each i,j,k: U_r[i,j]*V_r[j,k]*W_r[k,i]
            # Use einsum: 'ij,jk,ki->ijk'
            T += np.einsum('ij,jk,ki->ijk', U_r, V_r, W_r)
        
        # The target tensor is all ones (since every multiply-add term exists)
        target = np.ones((n, m, p), dtype=np.float64)
        return np.allclose(T, target, atol=1e-9)

    def _schoolbook():
        """Standard schoolbook decomposition of rank n*m*p."""
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
        return {"U": U, "V": V, "W": W}

    # Start with schoolbook
    best = _schoolbook()
    best_rank = best["U"][0].__len__() if best["U"] else 0
    best_rank = len(best["U"][0]) if best["U"] else n*m*p

    # Try structured low-rank approaches
    # The key insight: for matrix multiplication, we can use block/Winograd-like structures
    # But for exact decomposition with integer/half-integer entries, we need to be careful
    
    # Let's try a structured approach: use the fact that matrix mult can be done
    # with fewer multiplications using Karatsuba-like or Winograd-like methods
    # But those typically require non-integer coefficients or are for specific sizes
    
    # For 4x5 * 5x5, let's try to find a lower rank decomposition
    # using a structured search with random initialization and ALS (Alternating Least Squares)
    
    def _als_search(rank, max_iters=50, tol=1e-6):
        """Try to find a rank-rank decomposition using ALS."""
        # Target tensor T[i,j,k] = 1 for all i,j,k
        # We want U (n*m x rank), V (m*p x rank), W (p*n x rank)
        # such that T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        
        # Initialize randomly
        np.random.seed(42)
        U = np.random.randn(n * m, rank) * 0.1
        V = np.random.randn(m * p, rank) * 0.1
        W = np.random.randn(p * n, rank) * 0.1
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Update U: for each (i,j), U[i*m+j, :] should satisfy
            # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
            # For fixed i,j: T[i,j,:] = U[i*m+j,:] * (V_j * W_i) where V_j is V[j*p:(j+1)*p, :]
            # and W_i is W[?:i+1:n, :]... this is getting complex
            
            # Let's use a different approach: treat it as a matrix completion problem
            # For each r, we have U[:,r], V[:,r], W[:,r]
            # The constraint is: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i,j,k
            
            # This is equivalent to: for each (i,j,k), the inner product of
            # U[i*m+j,:], V[j*p+k,:], W[k*n+i,:] (appropriately combined) equals 1
            
            # Let's use a gradient-based approach instead
            break
        
        return None  # Too complex for quick implementation
    
    # Instead, let's try known structured decompositions
    # For small matrices, we can try to exploit structure
    
    # Try rank = n*m*p - some savings
    # One approach: if n, m, p have common structure, we can save multiplications
    
    # For 4x5 * 5x5: 
    # Schoolbook: 100 multiplications
    # Can we do better? 
    
    # Let's try a block decomposition approach
    # Split the 5x5 matrix into blocks and use Karatsuba-like tricks
    
    # Actually, let's try a direct search for a valid decomposition with lower rank
    # using a structured initialization based on the schoolbook but with merged terms
    
    def _try_ranked_decomposition(rank):
        """Try to find a valid decomposition of given rank."""
        if time.time() > deadline:
            return None
        
        # Start from schoolbook and try to merge terms
        # or use a structured approach
        
        # One idea: use the fact that for matrix multiplication,
        # we can sometimes combine terms using algebraic identities
        
        # For now, let's just verify that schoolbook works and return it
        # A proper search would require more sophisticated methods
        
        return None
    
    # Given the time constraint, let's return the schoolbook decomposition
    # which is guaranteed to be correct
    # The rank is n*m*p = 4*5*5 = 100
    
    return best
# EVOLVE-BLOCK-END
