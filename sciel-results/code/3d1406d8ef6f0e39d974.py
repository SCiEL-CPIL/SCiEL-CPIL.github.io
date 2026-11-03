"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import itertools

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s limit
    
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
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Build the matrix multiplication tensor
    # T[i, j, k] = 1 if the (i,j) entry of A times (j,k) entry of B contributes to (i,k) entry of C
    # Actually, the tensor for matrix multiplication: T[i, j, k] = 1 for the bilinear form
    # We want: C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T has T[i,j,k] = 1 for each (i,j,k) triple
    # Rank-1 decomposition: U[i,j,r] * V[j,k,r] * W[k,i,r] should sum to T[i,j,k]
    
    # The constraint is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # Let's try to find a lower rank decomposition
    # Strategy: try ranks from a lower bound up to schoolbook rank
    # Lower bound: max(n*m, m*p, p*n) = max(15, 30, 18) = 30
    
    # Try to use a randomized approach with rounding
    def try_rank(r, max_iters=2000):
        if time.time() > deadline:
            return None
        
        # Initialize randomly
        U = np.random.randn(n * m, r)
        V = np.random.randn(m * p, r)
        W = np.random.randn(p * n, r)
        
        # We want sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # This is a tensor completion problem
        
        # Use alternating least squares approach
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Update U: for each (i,j), we need sum_r U[i*m+j, r] * (V[j*p+*, r] * W[*, i, r]) = 1
            # Let's build the linear system
            
            # For fixed (i,j), the constraint is:
            # sum_r U[i*m+j, r] * S_{ij}[r] = 1
            # where S_{ij}[r] = sum_k V[j*p+k, r] * W[k*n+i, r]
            
            # But this is underdetermined. We need to use all constraints simultaneously.
            
            # Alternative: use gradient descent on the loss
            # Loss = sum_{i,j,k} (sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] - 1)^2
            
            # Compute gradient
            # dLoss/dU[i*m+j, r] = -2 * sum_k (pred[i,j,k] - 1) * V[j*p+k, r] * W[k*n+i, r]
            # where pred[i,j,k] = sum_r' U[i*m+j, r'] * V[j*p+k, r'] * W[k*n+i, r']
            
            # This is expensive. Let's use a simpler approach.
            
            # Try a different strategy: fix two matrices, solve for the third
            pass
        
        # After optimization, round to half-integers
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        
        # Verify
        if verify(U_rounded, V_rounded, W_rounded, n, m, p):
            return {'U': U_rounded.tolist(), 'V': V_rounded.tolist(), 'W': W_rounded.tolist()}
        
        return None
    
    def verify(U, V, W, n, m, p):
        """Verify that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k"""
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    # Try a structured approach: use the fact that we can combine terms
    # For now, let's try a few random initializations with a simple optimizer
    
    # Actually, let's try a different strategy: use known results
    # For small matrices, the rank is often close to the trivial bound
    
    # Let's try to find a decomposition using a greedy approach
    # Start with all-zero matrices and add rank-1 terms
    
    def greedy_decomposition(max_rank=90):
        if time.time() > deadline:
            return None
        
        # Target: T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
        # We want to represent this as sum of rank-1 tensors
        
        # A rank-1 tensor is determined by vectors u (length n*m), v (length m*p), w (length p*n)
        # such that u[i*m+j] * v[j*p+k] * w[k*n+i] = T[i,j,k] for the support
        
        # This is hard. Let's use a different approach.
        
        # Use the schoolbook and try to merge
        U = np.zeros((n * m, n * m * p))
        V = np.zeros((m * p, n * m * p))
        W = np.zeros((p * n, n * m * p))
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        
        # Try to find linear dependencies and reduce
        # This is complex. For now, return schoolbook
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    
    # Try randomized search for lower ranks
    for r in range(max(n*m, m*p, p*n), best_rank + 1):
        if time.time() > deadline:
            break
        if r >= best_rank:
            break
            
        # Try a few random initializations
        for attempt in range(5):
            if time.time() > deadline:
                break
                
            U = np.random.randint(-2, 3, (n * m, r)) / 2.0
            V = np.random.randint(-2, 3, (m * p, r)) / 2.0
            W = np.random.randint(-2, 3, (p * n, r)) / 2.0
            
            if verify(U, V, W, n, m, p):
                best_result = {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
                best_rank = r
                break
        
        if best_rank == r:
            break
    
    if best_rank < n * m * p:
        return best_result
    else:
        return best_result
# EVOLVE-BLOCK-END
