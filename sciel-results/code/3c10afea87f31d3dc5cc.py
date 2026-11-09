"""Find rank-minimal exact tensor decomposition for 2x3 * 3x7 matrix multiplication."""
DIMENSIONS = [2, 3, 7]

import time
import sys
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # The tensor T has shape (n, p, m) with T[i,k,j] = 1 for all valid (i,k,j)
    # A rank-r decomposition: T[i,k,j] = sum_{c=0}^{r-1} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Wait, let me re-read the convention.
    # U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)
    # The product: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal 1 for all (i,j,k)
    
    # Actually the bilinear identity: C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T[i,k,j] = delta (it's 1 for the contribution)
    # Rank-1 term: U[i,j] * V[j,k] * W[i,k] contributes U[i,j]*V[j,k]*W[i,k] to T[i,k,j]
    # So we need: sum_c U_c[i,j] * V_c[j,k] * W_c[i,k] = 1 for all (i,k,j)
    
    # U indexed by (i,j) -> row i*m+j, V indexed by (j,k) -> row j*p+k, W indexed by (i,k) -> row k*n+i
    
    def build_schoolbook():
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
        return U, V, W, rank
    
    def verify(U, V, W, tol=1e-10):
        """Check if the decomposition exactly decomposes the multiplication tensor."""
        rank = U.shape[1]
        # Reconstruct T[i,k,j] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    val = 0.0
                    for c in range(rank):
                        val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def verify_fast(U, V, W, tol=1e-10):
        """Faster verification using numpy."""
        rank = U.shape[1]
        # Build the tensor via einsum
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        # T[i,k,j] = sum_c U_r[i,j,c] * V_r[j,k,c] * W_r[k,i,c]
        # Use einsum
        T = np.einsum('ijc,jkc,kic->ikj', U_r, V_r, W_r)
        # T should be all 1s with shape (n, p, m)
        if T.shape != (n, p, m):
            return False
        return np.allclose(T, np.ones((n, p, m)), atol=tol)
    
    # Start with schoolbook
    U_sb, V_sb, W_sb, rank_sb = build_schoolbook()
    best_U, best_V, best_W, best_rank = U_sb, V_sb, W_sb, rank_sb
    
    if time.time() > deadline - 10:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Try to find lower rank by solving linear system
    # The constraint: for each (i,k,j), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    # This is nonlinear (bilinear in the factors).
    
    # Alternative: use the fact that if we fix two factors, the third is linear.
    # Or use optimization.
    
    # For small problems, try random restarts with optimization
    import random
    
    def try_random_decomposition(rank, max_iters=500, seed=42):
        """Try to find a rank-r decomposition using iterative optimization."""
        if rank < 1:
            return None
        
        rng = np.random.RandomState(seed)
        
        # Initialize randomly
        U = rng.randn(n * m, rank)
        V = rng.randn(m * p, rank)
        W = rng.randn(p * n, rank)
        
        # Normalize
        for c in range(rank):
            norm = max(np.max(np.abs(U[:, c])), np.max(np.abs(V[:, c])), np.max(np.abs(W[:, c])), 1e-10)
            if norm > 1e-10:
                U[:, c] /= norm
                V[:, c] /= norm
                W[:, c] *= norm
        
        best_err = float('inf')
        
        for it in range(max_iters):
            if time.time() > deadline - 20:
                break
            
            # Compute current error
            U_r = U.reshape(n, m, rank)
            V_r = V.reshape(m, p, rank)
            W_r = W.reshape(p, n, rank)
            T = np.einsum('ijc,jkc,kic->ikj', U_r, V_r, W_r)
            err = np.sum((T - 1.0)**2)
            
            if err < best_err:
                best_err = err
                if err < 1e-10:
                    # Snap to nearest integer/half-integer
                    U = np.round(U * 2) / 2
                    V = np.round(V * 2) / 2
                    W = np.round(W * 2) / 2
                    if verify_fast(U, V, W):
                        return U, V, W
                    else:
                        # Try rounding to integers
                        U = np.round(U)
                        V = np.round(V)
                        W = np.round(W)
                        if verify_fast(U, V, W):
                            return U, V, W
                        else:
                            # Revert
                            U = np.round(U * 2) / 2
                            V = np.round(V * 2) / 2
                            W = np.round(W * 2) / 2
            
            if err < 1e-12:
                break
            
            # Gradient-based update (simple)
            # dT/dU[i,j,c] = V[j,k,c] * W[k,i,c] summed over k
            # This gets complex; use simple coordinate descent
            
            # Update U
            for i in range(n):
                for j in range(m):
                    for c in range(rank):
                        # Gradient of err w.r.t. U[i*m+j, c]
                        # T[i,k,j] = sum_c' U[i,j,c']*V[j,k,c']*W[k,i,c']
                        # dT[i,k,j]/dU[i,j,c] = V[j,k,c]*W[k,i,c]
                        # derr/dU = sum_{k} 2*(T[i,k,j]-1)*V[j,k,c]*W[k,i,c]
                        grad = 0.0
                        for k in range(p):
                            grad += 2 * (T[i, k, j] - 1.0) * V[j * p + k, c] * W[k * n + i, c]
                        U[i * m + j, c] -= 0.01 * grad
            
            # Update V
            for j in range(m):
                for k in range(p):
                    for c in range(rank):
                        grad = 0.0
                        for i in range(n):
                            grad += 2 * (T[i, k, j] - 1.0) * U[i * m + j, c] * W[k * n + i, c]
                        V[j * p + k, c] -= 0.01 * grad
            
            # Update W
            for k in range(p):
                for i in range(n):
                    for c in range(rank):
                        grad = 0.0
                        for j in range(m):
                            grad += 2 * (T[i, k, j] - 1.0) * U[i * m + j, c] * V[j * p + k, c]
                        W[k * n + i, c] -= 0.01 * grad
        
        return None
    
    # Try ranks below schoolbook
    for rank in range(rank_sb - 1, max(rank_sb - 20, 1), -1):
        if time.time() > deadline - 30:
            break
        
        found = False
        for seed in range(10):
            if time.time() > deadline - 30:
                break
            
            result = try_random_decomposition(rank, max_iters=200, seed=seed + rank)
            if result is not None:
                U, V, W = result
                if verify_fast(U, V, W):
                    best_U, best_V, best_W, best_rank = U, V, W, rank
                    found = True
                    break
        
        if found:
            # Try to go even lower
            continue
    
    # Final verification
    if not verify_fast(best_U, best_V, best_W):
        # Fall back to schoolbook
        best_U, best_V, best_W, best_rank = U_sb, V_sb, W_sb, rank_sb
    
    # Convert to lists with integer/half-integer entries
    def to_clean_list(arr):
        """Convert numpy array to list with integer or half-integer entries."""
        result = []
        for row in arr:
            clean_row = []
            for val in row:
                # Round to nearest half-integer
                h = round(val * 2) / 2
                if abs(val - h) < 1e-9:
                    clean_row.append(h)
                else:
                    # Use the value as-is (shouldn't happen if verified)
                    clean_row.append(val)
            result.append(clean_row)
        return result
    
    return {"U": to_clean_list(best_U), "V": to_clean_list(best_V), "W": to_clean_list(best_W)}
# EVOLVE-BLOCK-END
