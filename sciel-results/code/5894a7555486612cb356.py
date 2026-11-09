"""Find a rank-minimal exact tensor decomposition for 4x4 * 4x8 matrix multiplication."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
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


def _verify(U, V, W, n, m, p, tol=1e-8):
    """Check that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    # The tensor T[i, j, k] = 1 if we're computing (AB)[i,k] = sum_j A[i,j]B[j,k]
    # Decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We check by verifying the matrix multiplication identity:
    # For random A (n x m) and B (m x p), C = A @ B should equal sum_c (U[:,c] outer V[:,c]) applied appropriately
    # More directly: C[i,k] = sum_j A[i,j] B[j,k] = sum_c (sum_j U[i*m+j,c] * A[i,j]) * (sum_k' ... )
    # Actually the standard check: for all i,k: C[i,k] = sum_j A[i,j]B[j,k]
    # And the decomposition says: C[i,k] = sum_c (sum_j U[i*m+j,c] A[i,j]) * (sum_j' V[j'*p+k,c] B[j',k])
    # Wait, that's not quite right. Let me re-derive.
    # The tensor for matrix mult: T_{i,j,k} where (AB)_{ik} = sum_j A_{ij} B_{jk}
    # A rank-r decomposition: T_{i,j,k} = sum_{c=1}^r U_{i,j}^{(c)} V_{j,k}^{(c)} W_{k,i}^{(c)}
    # But the convention here uses:
    # U has shape (n*m, rank), indexed by (i*m+j)
    # V has shape (m*p, rank), indexed by (j*p+k)  
    # W has shape (p*n, rank), indexed by (k*n+i)
    # The decomposition is: sum_j A[i,j] B[j,k] = sum_c U[i*m+j, c] * ... 
    # Hmm, let me think about this differently.
    # The standard formulation: (AB)_{ik} = sum_j A_{ij} B_{jk}
    # Tensor T[i,j,k] = 1 (the multiplication tensor has all-ones in the appropriate slice)
    # Actually T[i,j,k] represents the coefficient: the matrix multiplication tensor has T[i,j,k]=1 for all i,j,k
    # and (AB)_{ik} = sum_j A_{ij} T[i,j,k] B_{jk}
    # Rank-r decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # So we need: for all i,j,k: 1 = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]
    # Let's verify this.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(val - 1.0) > tol:
                    return False
    return True


def _verify_fast(U, V, W, n, m, p, tol=1e-8):
    """Faster verification using random test matrices."""
    rank = U.shape[1]
    # Use the identity: sum_j A[i,j] B[j,k] = sum_c (U^T A)[i,c] * (B V)[c,k] * ... 
    # Actually let's use: C = A @ B, and check C[i,k] = sum_c sum_j U[i*m+j,c] A[i,j] * sum_k' V[j'*p+k,c] B[j',k] * W[k*n+i,c]
    # This is getting complicated. Let me just use the direct tensor check but vectorized.
    # T[i,j,k] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] should equal 1 for all i,j,k.
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    # T[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    # We can compute this as: for each c, the contribution is U3[:,:,c] * V3[:,:,c] * W3[:,:,c]
    # But the indices don't align simply. Let's do it element-wise with broadcasting.
    # Create arrays of shape (n, m, p)
    # U3[i,j,c] depends on i,j,c
    # V3[j,k,c] depends on j,k,c  
    # W3[k,i,c] depends on k,i,c
    # T[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    # Let's compute using einsum or manual loops over c
    T = np.zeros((n, m, p))
    for c in range(rank):
        # U3[:, :, c] has shape (n, m)
        # V3[:, :, c] has shape (m, p)
        # W3[:, :, c] has shape (p, n)
        # We need T[i,j,k] += U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
        # This is: for each i,j,k: U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
        # = U3[i,j,c] * (V3[j,k,c] * W3[k,i,c])
        # V3[j,k,c] * W3[k,i,c] summed over... no, not summed, just multiplied element-wise
        # For fixed c, this is a 3D array: A[i,j,k] = U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
        # We can compute this as: 
        # temp[j,k] = V3[j,k,c]  (shape m x p)
        # Then for each i: temp2[k] = temp[:, i] * W3[:, i, c]  ... no
        # Let me just use np.einsum or explicit broadcasting
        # U3[i,j,c] -> shape (n, m)
        # V3[j,k,c] -> shape (m, p) 
        # W3[k,i,c] -> shape (p, n)
        # Result[i,j,k] = U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
        # = U3[i,j,c] * (V3[j,k,c] * W3[k,i,c])
        # The term V3[j,k,c] * W3[k,i,c] for fixed i is: V3[:, :, c] * W3[:, i, c] (broadcasting)
        # V3[:, :, c] is (m, p), W3[:, i, c] is (p,) -> broadcast to (m, p)
        # Then multiply by U3[i, :, c] which is (m,) -> broadcast to (m, p)
        # Result for fixed i: U3[i, :, c] * (V3[:, :, c] * W3[:, i, c])
        # Shape: (m, p)
        for i in range(n):
            # U3[i, :, c] has shape (m,)
            # V3[:, :, c] has shape (m, p)
            # W3[:, i, c] has shape (p,)
            contrib = U3[i, :, c][:, np.newaxis] * V3[:, :, c] * W3[:, i, c][np.newaxis, :]
            T[i, :, :] += contrib
    if np.max(np.abs(T - 1.0)) > tol:
        return False
    return True


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # Leave margin before 150s limit
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to find lower rank decompositions
    # Known lower bounds: rank >= max(n*m, m*p, p*n) = max(16, 32, 32) = 32
    # Upper bound from schoolbook: 128
    # Try various ranks from low to high
    
    # For a 4x4 * 4x8 matrix multiplication, we're looking for the tensor rank
    # of the matmul tensor T_{4,4,8}
    
    # Strategy: use random initialization and gradient-free optimization
    # or structured approaches
    
    # Try rank from a reasonable lower bound up
    # Lower bound: max(n*m, m*p, n*p) = max(16, 32, 32) = 32
    # But tighter bounds exist. For matmul, rank >= n*m*p / max(n,m,p) is not standard.
    # The trivial lower bound is max(n*m, m*p, n*p) = 32 here.
    
    # Let's try a few ranks. Given time constraints, we'll try a limited search.
    
    # Actually, for this specific case (4x4 * 4x8), let's try a structured approach.
    # The matrix multiplication tensor for m x n * n x p has rank at most m*n*p.
    # We want to find if there's a rank < 128 decomposition.
    
    # Try random + optimization approach for various ranks
    rng = np.random.RandomState(42)
    
    ranks_to_try = list(range(32, 129))
    # Sort by likelihood of success: try moderate ranks first
    # Actually, let's be strategic. Try ranks that are "nice" numbers first
    # and use a time budget per rank.
    
    time_per_rank = 10  # seconds per rank attempt
    attempts_made = 0
    
    for rank in sorted(ranks_to_try, key=lambda r: abs(r - 64)):  # Try middle ranks first
        if time.time() > deadline - 5:
            break
        
        start_time = time.time()
        max_iter = 5000
        
        # Try multiple random initializations
        for init_trial in range(5):
            if time.time() > deadline - 5:
                break
            if time.time() - start_time > time_per_rank:
                break
            
            # Random initialization with structure
            U = rng.randn(n * m, rank)
            V = rng.randn(m * p, rank)
            W = rng.randn(p * n, rank)
            
            # Scale to make initial tensor close to all-ones
            # T[i,j,k] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]
            # We want this to be ~1
            # Normalize each column
            for c in range(rank):
                norm = np.sqrt(np.sum(U[:, c]**2) * np.sum(V[:, c]**2) * np.sum(W[:, c]**2))
                if norm > 1e-10:
                    scale = 1.0 / (norm ** (1/3))
                    U[:, c] *= scale
                    V[:, c] *= scale
                    W[:, c] *= scale
            
            # Simple gradient descent on the loss: sum_{i,j,k} (T[i,j,k] - 1)^2
            lr = 0.01
            for it in range(max_iter):
                if time.time() > deadline - 5:
                    break
                
                # Compute T and gradient
                U3 = U.reshape(n, m, rank)
                V3 = V.reshape(m, p, rank)
                W3 = W.reshape(p, n, rank)
                
                # T[i,j,k] = sum_c U3[i,j,c] V3[j,k,c] W3[k,i,c]
                # grad_U[i,j,c] = sum_k V3[j,k,c] W3[k,i,c] * (T[i,j,k] - 1)
                # grad_V[j,k,c] = sum_i U3[i,j,c] W3[k,i,c] * (T[i,j,k] - 1)
                # grad_W[k,i,c] = sum_j U3[i,j,c] V3[j,k,c] * (T[i,j,k] - 1)
                
                T = np.zeros((n, m, p))
                for c in range(rank):
                    for i in range(n):
                        contrib = U3[i, :, c][:, np.newaxis] * V3[:, :, c] * W3[:, i, c][np.newaxis, :]
                        T[i, :, :] += contrib
                
                err = T - 1.0
                loss = np.sum(err**2)
                
                if loss < 1e-12:
                    break
                
                # Compute gradients
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                for c in range(rank):
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = err[i, j, k]
                                dU[i * m + j, c] += val * V3[j, k, c] * W3[k, i, c]
                                dV[j * p + k, c] += val * U3[i, j, c] * W3[k, i, c]
                                dW[k * n + i, c] += val * U3[i, j, c] * V3[j, k, c]
                
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                # Adaptive learning rate
                if it % 100 == 0:
                    new_loss = np.sum(err**2)
                    if new_loss < loss * 0.99:
                        lr *= 1.1
                    else:
                        lr *= 0.9
                    
                    if lr < 1e-6:
                        break
                    if lr > 0.1:
                        lr = 0.1
                
                if it % 500 == 0 and loss < 1e-6:
                    # Check if we've converged
                    if _verify_fast(U, V, W, n, m, p, tol=1e-8):
                        break
            
            # Final check
            if _verify_fast(U, V, W, n, m, p, tol=1e-8):
                # Snap to nearest integer/half-integer
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                
                if _verify_fast(U_snap, V_snap, W_snap, n, m, p, tol=1e-8):
                    if rank < best_rank:
                        best_rank = rank
                        best_U, best_V, best_W = U_snap.copy(), V_snap.copy(), W_snap.copy()
                        break
                else:
                    # Use the non-snapped version if it verifies
                    if rank < best_rank:
                        best_rank = rank
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                        break
        
        if rank < best_rank:
            break  # Found a better rank, no need to try higher
    
    # Return the best decomposition found
    if best_rank < n * m * p:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        # Fall back to schoolbook
        U, V, W = _schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
