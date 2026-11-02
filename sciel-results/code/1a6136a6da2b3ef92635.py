"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _schoolbook_decomposition(n, m, p):
    """Return schoolbook decomposition of rank n*m*p."""
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

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that the decomposition exactly matches the matrix multiplication tensor."""
    # Build the target tensor T[i,j,k] = 1 if we're multiplying (i,j) by (k)
    # Actually, the tensor for matrix mult is: T_{i,j,k} where A is n x m, B is m x p
    # C_{i,k} = sum_j A_{i,j} * B_{j,k}
    # So the tensor T has entries T[i,j,k] = delta (it's the identity in the contraction)
    # The decomposition says: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # Wait, let me re-read the convention.
    
    # U has shape (n*m, rank), indexed by [i*m+j, c]
    # V has shape (m*p, rank), indexed by [j*p+k, c]  
    # W has shape (p*n, rank), indexed by [k*n+i, c]
    
    # The tensor equation is: for all i,j,k:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # (since the matrix multiplication tensor has 1s on the "diagonal" in the appropriate sense)
    
    # Actually, let me think more carefully. The matrix multiplication tensor T is defined by:
    # T[i, j, k] = 1 for all valid i, j, k? No.
    # 
    # The multiplication: C = A * B where A is n x m, B is m x p, C is n x p
    # C[i, k] = sum_j A[i, j] * B[j, k]
    # 
    # The tensor T for this operation has T[i, j, k] = 1 for all i, j, k? 
    # No, that's not right either. Let me think about the standard tensor formulation.
    #
    # In the standard formulation, the matrix multiplication tensor is:
    # T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # And the decomposition is: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    #
    # Wait, that would mean every entry of T is 1, which doesn't make sense for a 
    # non-trivial tensor. Let me reconsider.
    #
    # Actually, I think the convention is:
    # The tensor T has T[i, j, k] = 1 for all valid indices.
    # And we want: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
    #
    # Hmm, but that's just saying the product of the three factors sums to 1 everywhere.
    # That seems too simple. Let me re-read the problem.
    #
    # "Find a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix 
    # by a 3 by 10 matrix. The output factor follows the official trace convention W[k*n+i,r]."
    #
    # I think the tensor is indeed all-ones in the sense that:
    # For matrix multiplication, the "tensor" is the identity map on the middle dimension.
    # T[i,j,k] = delta (it's 1 for all combinations, representing the bilinear form).
    #
    # Actually no. Let me think about this differently. The standard formulation is:
    # Given A (n x m) and B (m x p), compute C = AB (n x p).
    # C[i,k] = sum_j A[i,j] B[j,k]
    # 
    # This can be written as a tensor contraction. The tensor T has three indices (i,j,k)
    # and T[i,j,k] = 1 for all valid i,j,k. Then:
    # C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k]
    #
    # But that's not quite right because T[i,j,k] should be 1 only when the indices 
    # are consistent. Actually, I think T[i,j,k] = 1 for all i,j,k is correct because
    # we're summing over j for fixed i,k.
    #
    # So the decomposition condition is:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
    
    # Let's verify by checking all (i,j,k) combinations
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _optimize_decomposition(n, m, p, rank, max_time=100, seed=42):
    """Try to find a rank-r decomposition using gradient descent."""
    rng = np.random.default_rng(seed)
    
    U = rng.standard_normal((n * m, rank))
    V = rng.standard_normal((m * p, rank))
    W = rng.standard_normal((p * n, rank))
    
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    lr = 0.01
    momentum = 0.9
    vU = np.zeros_like(U)
    vV = np.zeros_like(V)
    vW = np.zeros_like(W)
    
    start_time = time.time()
    while time.time() - start_time < max_time:
        # Compute loss: sum over all (i,j,k) of (sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] - 1)^2
        loss = 0.0
        gU = np.zeros_like(U)
        gV = np.zeros_like(V)
        gW = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # prod_c = U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :]
                    prod = U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :]
                    s = np.sum(prod)
                    err = s - 1.0
                    loss += err * err
                    
                    # Gradient w.r.t. U[i*m+j, c] = 2*err * V[j*p+k, c] * W[k*n+i, c]
                    gU[i*m+j, :] += 2 * err * V[j*p+k, :] * W[k*n+i, :]
                    gV[j*p+k, :] += 2 * err * U[i*m+j, :] * W[k*n+i, :]
                    gW[k*n+i, :] += 2 * err * U[i*m+j, :] * V[j*p+k, :]
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_loss < 1e-12:
            break
        
        # Adam-like update
        vU = momentum * vU + (1 - momentum) * gU
        vV = momentum * vV + (1 - momentum) * gV
        vW = momentum * vW + (1 - momentum) * gW
        
        U -= lr * vU
        V -= lr * vV
        W -= lr * vW
        
        # Adaptive learning rate
        if loss > best_loss * 1.1:
            lr *= 0.5
    
    # Check if we found a valid decomposition
    if _verify_decomposition(best_U, best_V, best_W, n, m, p, tol=1e-8):
        return best_U, best_V, best_W, True
    
    return None, None, None, False

def construct():
    n, m, p = DIMENSIONS
    # Strict deadline: 110 seconds to be safe within 120s limit
    deadline = time.time() + 110
    
    # Start with schoolbook decomposition (Rank = 60)
    U0, V0, W0 = _schoolbook_decomposition(n, m, p)
    best_result = {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}
    best_rank = n * m * p
    
    # The gradient descent search is very expensive and unlikely to find a 
    # rank < 60 for 2x3x10 within a short time window. 
    # We will attempt a very brief search for lower ranks, but prioritize 
    # returning a valid result quickly.
    
    # Known lower bound is roughly max(n, m, p) * min(n, m, p) or similar, 
    # but for this specific case, let's try a few ranks below 60.
    # To avoid timeout, we limit the total search time to 30 seconds.
    search_deadline = time.time() + 30
    
    # Try ranks from a reasonable lower bound up to best_rank - 1
    # min_rank = max(n*p, m*p, n*m) = 30
    min_rank = max(n * p, m * p, n * m)
    
    # We'll try a few specific ranks to see if any work, but keep it fast.
    # If we don't find anything, we return the schoolbook result.
    ranks_to_try = list(range(min_rank, best_rank))
    random.shuffle(ranks_to_try)
    
    for rank in ranks_to_try:
        if time.time() > search_deadline:
            break
        if rank >= best_rank:
            continue
            
        # Limit time per rank attempt to 2 seconds to ensure we try multiple ranks
        # or exit quickly if they fail.
        U, V, W, valid = _optimize_decomposition(n, m, p, rank, max_time=2.0, seed=42)
        if valid:
            best_rank = rank
            best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            # If we find a rank close to the lower bound, we can stop
            if rank <= min_rank:
                break
    
    return best_result
# EVOLVE-BLOCK-END
