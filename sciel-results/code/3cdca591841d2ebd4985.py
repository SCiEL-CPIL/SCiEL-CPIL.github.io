"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Build the matrix multiplication tensor T[i,j,k] = 1 if C[i,k] involves A[i,j]*B[j,k]
    # T is a 3D array of shape (n*m, m*p, p*n) in the flattened convention
    # Actually, the tensor is defined such that C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T has entries T[i,j,k] where i in [0,n), j in [0,m), k in [0,p)
    # And the decomposition is: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Build the target tensor
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Start with schoolbook rank 128
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    
    best_rank = rank
    best_U, best_V, best_W = U, V, W
    
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Try to find lower rank decomposition using iterative rank reduction
    # Strategy: try to merge columns where possible
    
    # First, let's try a random search for lower rank
    # We'll use a greedy approach: try to find a decomposition of rank r < 128
    
    # Try ranks from 96 down to see if we can find a valid decomposition
    # Use random initialization and gradient-like descent
    
    # For efficiency, let's try a few structured approaches first
    
    # Approach 1: Use the fact that M_{4,4,8} might be decomposable using two M_{4,4,4} blocks
    # If we can find rank r for M_{4,4,4}, then rank 2r for M_{4,4,8}
    
    # Let's try to find a good decomposition for M_{4,4,4} first
    n2, m2, p2 = 4, 4, 4
    T2 = np.zeros((n2*m2, m2*p2, p2*n2), dtype=float)
    for i in range(n2):
        for j in range(m2):
            for k in range(p2):
                T2[i*m2+j, j*p2+k, k*n2+i] = 1.0
    
    # Try to find low-rank decomposition of T2
    # Start with rank 49 (known result) and try to reduce
    
    def try_decompose(tensor, dim1, dim2, dim3, target_rank, max_iter=5000, seed=42):
        """Try to find a decomposition of the tensor with given target rank."""
        np.random.seed(seed)
        
        # Scale initialization based on target rank
        scale = 1.0 / np.sqrt(target_rank)
        U = np.random.randn(n*m, target_rank) * scale
        V = np.random.randn(m*p, target_rank) * scale
        W = np.random.randn(p*n, target_rank) * scale
        best_loss = float('inf')
        best_U, best_V, best_W = U_t.copy(), V_t.copy(), W_t.copy()
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            # Compute loss: ||T - sum_c U[:,c] outer V[:,c] outer W[:,c]||_F^2
            # This is expensive, so let's use a simpler check
            
            # Actually, let's use a different approach: check if the tensor can be exactly
            # represented with this rank
            
            # For exact decomposition, we need:
            # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            
            # This is a bilinear system. Let's use a simpler heuristic.
            
            # Compute the reconstructed tensor
            recon = np.zeros_like(tensor)
            for r in range(target_rank):
                recon += np.einsum('a,b,c->abc', U_t[:,r], V_t[:,r], W_t[:,r])
            
            loss = np.sum((tensor - recon)**2)
            
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U_t.copy(), V_t.copy(), W_t.copy()
                
                if loss < 1e-10:
                    break
            
            # Simple gradient step (this is very approximate)
            lr = 0.01
            for r in range(target_rank):
                # Gradient with respect to U[:,r]
                grad_U = np.zeros(dim1)
                grad_V = np.zeros(dim2)
                grad_W = np.zeros(dim3)
                
                for a in range(dim1):
                    for b in range(dim2):
                        for c in range(dim3):
                            diff = tensor[a,b,c] - recon[a,b,c]
                            grad_U[a] += 2 * diff * V_t[b,r] * W_t[c,r]
                            grad_V[b] += 2 * diff * U_t[a,r] * W_t[c,r]
                            grad_W[c] += 2 * diff * U_t[a,r] * V_t[b,r]
                
                U_t[:,r] -= lr * grad_U
                V_t[:,r] -= lr * grad_V
                W_t[:,r] -= lr * grad_W
            
            # Recompute recon
            recon = np.zeros_like(tensor)
            for r in range(target_rank):
                recon += np.einsum('a,b,c->abc', U_t[:,r], V_t[:,r], W_t[:,r])
        
        return best_U, best_V, best_W, best_loss
    
    # This approach is too slow. Let me use a different strategy.
    
    # For M_{4,4,8}, let's use the known result that it can be done with rank <= 98
    # by using two M_{4,4,4} decompositions of rank 49 each.
    
    # But first, let's verify the schoolbook solution works and return it as fallback.
    
    # Convert to the required format (list of lists)
    def to_list(arr):
        return [[float(x) for x in row] for row in arr]
    
    # Return the best result found
    if best_rank <= 128:
        return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
    
    # Fallback: schoolbook
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
# EVOLVE-BLOCK-END
