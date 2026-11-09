"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]
import time
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 140s deadline, 10s safety margin
    
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
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    
    def verify(U, V, W, tol=1e-9):
        """Check if decomposition is exact."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        # Reconstruct tensor
        # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        # Compare with standard multiplication tensor
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    expected = 1.0 if (i < n and j < m and k < p) else 0.0
                    # The tensor for matrix multiplication has T[i,j,k] = 1 for all valid (i,j,k)
                    # Actually no: the multiplication tensor is T[i,j,k] = A[i,j]*B[j,k]*C[i,k]
                    # So for the basis tensors, T[i,j,k] = 1 if we're computing the (i,k) entry
                    # Wait, the tensor is: for each (i,j,k), the coefficient is 1
                    # Because (A*B)[i,k] = sum_j A[i,j] * B[j,k]
                    # So the trilinear form is sum_{i,j,k} A[i,j] B[j,k] C[i,k]
                    # And the tensor T[i,j,k] = 1 for all i,j,k
                    # No wait: T[i,j,k] should be 1 only when... let me think again.
                    # The matrix multiplication tensor T has T[i,j,k] = 1 for all valid indices
                    # because (AB)[i,k] = sum_j A[i,j] B[j,k], so the coefficient of A[i,j]*B[j,k]*C[i,k] is 1
                    # But C[i,k] is the output, so in the decomposition:
                    # sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] should equal 1 for all (i,j,k)
                    # Wait no, that's not right either. The tensor is:
                    # T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k where the multiplication is (A*B)[i,k] = sum_j A[i,j]B[j,k]
                    # So T[i,j,k] = 1 for all i,j,k? No, that would mean the tensor is all ones.
                    # Actually yes! The matrix multiplication tensor for n×m by m×p has T[i,j,k] = 1
                    # for all i in [0,n), j in [0,m), k in [0,p), because each term A[i,j]*B[j,k]
                    # contributes to C[i,k] with coefficient 1.
                    # So we need sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all (i,j,k)
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def try_rank(rank):
        """Try to find a decomposition of given rank using random search."""
        if time.time() > deadline - 5:
            return None
        np.random.seed(42)
        best = None
        max_attempts = 50
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            U = np.random.randn(n * m, rank)
            V = np.random.randn(m * p, rank)
            W = np.random.randn(p * n, rank)
            # Normalize to help convergence
            U = U / (np.linalg.norm(U, axis=0, keepdims=True) + 1e-12)
            V = V / (np.linalg.norm(V, axis=0, keepdims=True) + 1e-12)
            W = W / (np.linalg.norm(W, axis=0, keepdims=True) + 1e-12)
            
            # Check if this random guess works (very unlikely, but check structure)
            # Instead, let's use a smarter approach: start from schoolbook and try to compress
            if attempt == 0:
                # Use structured initialization
                break
        
        return None
    
    def structured_search(max_rank_to_try=50):
        """Search for decompositions using structured patterns."""
        if time.time() > deadline - 5:
            return None
        
        # Try ranks from a lower bound up to schoolbook
        lower_bound = max(n * m, m * p, p * n)  # = 21
        # Try some specific ranks that might work
        candidate_ranks = list(range(lower_bound, min(max_rank_to_try, n*m*p + 1)))
        
        for rank in candidate_ranks:
            if time.time() > deadline - 10:
                break
            if rank >= n * m * p:
                break
            # For now, just return schoolbook if we can't find better
            pass
        
        return None
    
    # Start with schoolbook as fallback
    result = schoolbook()
    
    # Try to find a better decomposition
    # For 3x3x3, rank 23 is known. For 3x3x7, let's try some approaches.
    
    # Approach: Use the fact that we can share computations across k values
    # For each (i,j), we have p=7 terms. Maybe we can group them.
    
    # Let's try a block decomposition approach
    # Group the k dimension into blocks
    
    # Actually, let me try a more systematic random search with verification
    np.random.seed(123)
    
    best_rank = n * m * p
    best_result = result
    
    # Try random sparse decompositions
    for rank in range(30, 63):
        if time.time() > deadline - 10:
            break
        found = False
        for attempt in range(20):
            if time.time() > deadline - 10:
                break
            
            # Create sparse random matrices
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            
            for r in range(rank):
                # Each rank component picks a few nonzero entries
                num_nonzero_U = np.random.randint(1, 4)
                num_nonzero_V = np.random.randint(1, 4)
                num_nonzero_W = np.random.randint(1, 4)
                
                U_rows = np.random.choice(n * m, size=num_nonzero_U, replace=False)
                V_rows = np.random.choice(m * p, size=num_nonzero_V, replace=False)
                W_rows = np.random.choice(p * n, size=num_nonzero_W, replace=False)
                
                U[U_rows, r] = np.random.choice([-1, 1], size=num_nonzero_U)
                V[V_rows, r] = np.random.choice([-1, 1], size=num_nonzero_V)
                W[W_rows, r] = np.random.choice([-1, 1], size=num_nonzero_W)
            
            # Verify
            if verify(U.tolist(), V.tolist(), W.tolist()):
                found = True
                best_rank = rank
                best_result = {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
                break
        
        if found:
            break
    
    return best_result
# EVOLVE-BLOCK-END
