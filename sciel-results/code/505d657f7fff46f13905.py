"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _schoolbook():
    n, m, p = DIMENSIONS
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

def _verify(U, V, W):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    # The tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # For matrix multiplication, T[i,j,k] should be 1 if we're computing (AB)[i,k] from A[i,j]*B[j,k]
    # Actually, the standard decomposition is: (AB)[i,k] = sum_j sum_c U[i,j,c]*V[j,k,c]*W[c]
    # But here the convention is different. Let me check the schoolbook:
    # U[i*m+j, c]=1, V[j*p+k, c]=1, W[k*n+i, c]=1 for the term where c corresponds to (i,j,k)
    # So the product is sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # This should equal 1 for the "correct" (i,j,k) combination and 0 otherwise?
    # No, for matrix multiplication we need: sum_j A[i,j] B[j,k] = sum_c (sum_j A[i,j] U[i,j,c]) (sum_k B[j,k] V[j,k,c]) W... 
    # Let me think again. The tensor decomposition of matrix multiplication:
    # T_{i,j,k} = delta_{i,k} in some sense? No.
    # The matrix multiplication tensor has T[i,j,k] = 1 for all i,j,k (it's the identity-like tensor)
    # Actually no. The tensor is defined such that:
    # (A*B)[i,k] = sum_j A[i,j] B[j,k]
    # This can be written as sum_c (sum_j A[i,j] U[i,j,c]) (sum_k B[j,k] V[j,k,c]) * W[i,k,c]
    # Wait, that's not right either. Let me look at the schoolbook more carefully.
    # In the schoolbook: for each (i,j,k), we have a rank-1 term.
    # U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1
    # So the contribution to position (i,j,k) in the "tensor" is 1*1*1 = 1
    # And for other positions it's 0.
    # So the tensor T[i,j,k] = 1 for all i,j,k? That doesn't make sense for matrix multiplication.
    # 
    # Let me re-read: "Find a rank-minimal exact tensor decomposition for multiplying a 3 by 3 matrix by a 3 by 4 matrix"
    # The matrix multiplication tensor T satisfies: T[i,j,k] = 1 if we're computing element (i,k) of the product
    # using A[i,j] and B[j,k]. But actually the tensor is:
    # T_{i,j,k} = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # No wait, that's not right. The matrix multiplication map is:
    # (A, B) -> C where C[i,k] = sum_j A[i,j] B[j,k]
    # This is a bilinear map. The tensor associated with this is:
    # T[i,j,k] = 1 if we consider the "standard" basis, but actually:
    # The tensor is T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k where the bilinear form is <T, A⊗B> = sum_{i,j,k} T[i,j,k] A[i,j] B[j,k]
    # For matrix multiplication, <T, A⊗B> = sum_{i,k} C[i,k] = sum_{i,k,j} A[i,j] B[j,k]
    # So T[i,j,k] = 1 for all i,j,k.
    # 
    # So we need: sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1 for all i,j,k
    # 
    # Let me verify with schoolbook: for a given (i,j,k), only the term c corresponding to (i,j,k) contributes,
    # and it contributes 1*1*1 = 1. For any other (i',j',k'), the term c for (i,j,k) has
    # U[i'*m+j', c] = 0 unless i'=i and j'=j, V[j'*p+k', c] = 0 unless j'=j and k'=k,
    # W[k'*n+i', c] = 0 unless k'=k and i'=i.
    # So indeed T[i,j,k] = 1 for all i,j,k.
    
    # Build the tensor and verify
    rank = U.shape[1]
    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _random_decomposition(rank, seed=None):
    """Generate a random decomposition with the given rank."""
    if seed is not None:
        random.seed(seed)
    n, m, p = DIMENSIONS
    U = np.random.randint(-2, 3, size=(n*m, rank))
    V = np.random.randint(-2, 3, size=(m*p, rank))
    W = np.random.randint(-2, 3, size=(p*n, rank))
    return U, V, W

def _try_reduce_rank(U, V, W, max_iterations=100000):
    """Try to reduce the rank by zeroing out entries."""
    rank = U.shape[1]
    n, m, p = DIMENSIONS
    
    # Try to remove columns (reduce rank)
    for _ in range(max_iterations):
        # Pick a random column to try to eliminate
        c = random.randint(0, rank - 1)
        
        # Try to set this column to zero and see if we can compensate
        # by adjusting other columns
        # For now, just try zeroing and checking
        U_save = U[:, c].copy()
        V_save = V[:, c].copy()
        W_save = W[:, c].copy()
        
        U[:, c] = 0
        V[:, c] = 0
        W[:, c] = 0
        
        if _verify(U, V, W):
            # Successfully removed this column
            # Remove it from all matrices
            U = np.delete(U, c, axis=1)
            V = np.delete(V, c, axis=1)
            W = np.delete(W, c, axis=1)
            rank -= 1
        else:
            # Restore
            U[:, c] = U_save
            V[:, c] = V_save
            W[:, c] = W_save
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120 second limit, keep safety margin
    
    # Start with schoolbook
    U, V, W = _schoolbook()
    best_rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to reduce rank using local search
    U, V, W = _try_reduce_rank(U, V, W, max_iterations=50000)
    if U.shape[1] < best_rank:
        best_rank = U.shape[1]
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try random restarts
    for seed in range(200):
        if time.time() > deadline - 10:
            break
        
        U, V, W = _random_decomposition(30, seed=seed)
        
        # Try to verify and reduce
        if _verify(U, V, W):
            U, V, W = _try_reduce_rank(U, V, W, max_iterations=10000)
            if U.shape[1] < best_rank:
                best_rank = U.shape[1]
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Convert to lists
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
