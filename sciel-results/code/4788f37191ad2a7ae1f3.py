"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _verify(U, V, W):
    """Check if U, V, W form a valid decomposition of the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    # The tensor T[i,j,k] should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We verify a subset of entries for speed
    # Full check: for all i,j,k, T[i,j,k] = 1 if this is the standard MM tensor
    # Actually the MM tensor has T[i,j,k] = 1 for all i,j,k (it's the identity in the tensor sense)
    # Wait, no. The matrix multiplication tensor T satisfies:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So T[i,j,k] = 1 for all valid i,j,k.
    
    # Let's verify by checking that the decomposition reproduces the tensor.
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # For MM, T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    
    # Efficient check using tensor contraction
    # Build the tensor from U, V, W and compare with ones
    # T_decomp[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Use einsum for efficiency
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U_reshaped = U.reshape(n, m, -1)
    V_reshaped = V.reshape(m, p, -1)
    W_reshaped = W.reshape(p, n, -1)
    
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # Use einsum: 'ijc,jkc,kic->ijk'
    try:
        T_decomp = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped, optimize=True)
    except Exception:
        return False
    
    # The MM tensor is all ones
    T_expected = np.ones((n, m, p))
    
    return np.allclose(T_decomp, T_expected, atol=1e-10)


def construct():
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s margin
    
    # Start with schoolbook
    U, V, W = _schoolbook()
    best_rank = U.shape[1]
    best_result = (U.copy(), V.copy(), W.copy())
    
    # Try to find a lower rank decomposition
    # Strategy: Use a structured approach based on block matrix multiplication
    # For 4x5 * 5x6, we can try various block partitions
    
    n, m, p = DIMENSIONS
    
    # Attempt 1: Try a rank based on Strassen-like ideas
    # For matrices of size a x b * b x c, the minimum rank is at most a*b*c
    # But we can do better with clever decompositions
    
    # Let's try a direct optimization approach
    # We'll use a gradient-based method to find a low-rank decomposition
    
    def _try_rank(rank):
        """Try to find a decomposition of given rank."""
        if rank >= best_rank:
            return None
        
        # Initialize with random values
        np.random.seed(rank)  # Deterministic
        U_try = np.random.randn(n * m, rank) * 0.1
        V_try = np.random.randn(m * p, rank) * 0.1
        W_try = np.random.randn(p * n, rank) * 0.1
        
        # Use Adam-like optimization
        lr = 0.01
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        
        mU = np.zeros_like(U_try)
        mV = np.zeros_like(V_try)
        mW = np.zeros_like(W_try)
        vU = np.zeros_like(U_try)
        vV = np.zeros_like(V_try)
        vW = np.zeros_like(W_try)
        
        t = 0
        best_loss = np.inf
        best_U, best_V, best_W = U_try.copy(), V_try.copy(), W_try.copy()
        
        # Precompute for efficiency
        U_reshaped = U_try.reshape(n, m, rank)
        V_reshaped = V_try.reshape(m, p, rank)
        W_reshaped = W_try.reshape(p, n, rank)
        
        max_iters = 5000
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            t += 1
            t1 = t / (1 + beta1 ** t)
            t2 = t / (1 + beta2 ** t)
            
            # Compute T_decomp
            T_decomp = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped, optimize=True)
            loss = np.sum((T_decomp - 1.0) ** 2)
            
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U_try.copy(), V_try.copy(), W_try.copy()
            
            if loss < 1e-10:
                break
            
            # Compute gradients
            # dT/dU[i,j,c] = sum_{k'} V[j,k',c] * W[k',i,c] * 2*(T_decomp[i,j,k'] - 1)
            # This is expensive, so we use a simplified gradient
            
            # dU[i*m+j, c] = 2 * sum_k (T_decomp[i,j,k] - 1) * V[j*p+k, c] * W[k*n+i, c]
            # dV[j*p+k, c] = 2 * sum_i (T_decomp[i,j,k] - 1) * U[i*m+j, c] * W[k*n+i, c]
            # dW[k*n+i, c] = 2 * sum_j (T_decomp[i,j,k] - 1) * U[i*m+j, c] * V[j*p+k, c]
            
            # Use einsum for gradients
            err = T_decomp - 1.0  # (n, m, p)
            
            # dU[i,j,c] = 2 * sum_k err[i,j,k] * V[j,k,c] * W[k,i,c]
            dU = 2 * np.einsum('ijk,jkc,kic->ijc', err, V_reshaped, W_reshaped, optimize=True)
            # dV[j,k,c] = 2 * sum_i err[i,j,k] * U[i,j,c] * W[k,i,c]
            dV = 2 * np.einsum('ijk,ijc,kic->jkc', err, U_reshaped, W_reshaped, optimize=True)
            # dW[k,i,c] = 2 * sum_j err[i,j,k] * U[i,j,c] * V[j,k,c]
            dW = 2 * np.einsum('ijk,ijc,jkc->kic', err, U_reshaped, V_reshaped, optimize=True)
            
            dU_flat = dU.reshape(-1, rank)
            dV_flat = dV.reshape(-1, rank)
            dW_flat = dW.reshape(-1, rank)
            
            # Adam update
            mU = beta1 * mU + (1 - beta1) * dU_flat
            vU = beta2 * vU + (1 - beta2) * (dU_flat ** 2)
            U_try -= lr * (mU / t1) / (np.sqrt(vU / t2) + eps)
            
            mV = beta1 * mV + (1 - beta1) * dV_flat
            vV = beta2 * vV + (1 - beta2) * (dV_flat ** 2)
            V_try -= lr * (mV / t1) / (np.sqrt(vV / t2) + eps)
            
            mW = beta1 * mW + (1 - beta1) * dW_flat
            vW = beta2 * vW + (1 - beta2) * (dW_flat ** 2)
            W_try -= lr * (mW / t1) / (np.sqrt(vW / t2) + eps)
            
            U_reshaped = U_try.reshape(n, m, rank)
            V_reshaped = V_try.reshape(m, p, rank)
            W_reshaped = W_try.reshape(p, n, rank)
        
        if best_loss < 1e-8:
            return best_U, best_V, best_W
        return None
    
    # Try various ranks
    # Known lower bounds: rank >= max(n*m, m*p, p*n) = max(20, 30, 24) = 30
    # Also rank >= n*p = 24 (from the output size)
    # For 4x5x6, the minimum rank is not known exactly, but we can try to find it
    
    # Try ranks from a reasonable lower bound up
    lower_bound = max(n * m, m * p, p * n, n * p)
    # Actually, a better lower bound: the tensor rank of MM for n x m * m x p
    # is at least max(n*m, m*p, p*n) but can be lower
    
    # Let's try some specific ranks
    test_ranks = []
    for r in range(lower_bound, best_rank + 1, 2):
        test_ranks.append(r)
    
    # Also try some specific values
    test_ranks.extend([25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40])
    test_ranks = sorted(set(test_ranks))
    
    for rank in test_ranks:
        if time.time() > deadline - 10:
            break
        if rank >= best_rank:
            continue
        
        result = _try_rank(rank)
        if result is not None:
            U_try, V_try, W_try = result
            if _verify(U_try, V_try, W_try):
                best_rank = rank
                best_result = (U_try.copy(), V_try.copy(), W_try.copy())
                # Try to go lower
                break
    
    U_best, V_best, W_best = best_result
    
    # Convert to lists
    U_list = U_best.tolist()
    V_list = V_best.tolist()
    W_list = W_best.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
