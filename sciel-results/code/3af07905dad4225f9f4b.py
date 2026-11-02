"""Tensor decomposition search for 2x3x10 matrix multiplication."""
import numpy as np
import time
import itertools

DIMENSIONS = [2, 3, 10]
DEADLINE = 120  # seconds

def _build_tensor(n, m, p):
    """Build the n x m x p matrix multiplication tensor (flattened)."""
    tensor = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # A index: i*m+j, B index: j*p+k, C index: k*n+i
                tensor[i*m+j, j*p+k, k*n+i] = 1.0
    return tensor

def _verify_decomposition(U, V, W, tensor, tol=1e-9):
    """Verify that U, V, W decompose the tensor."""
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    # Reconstruct
    T = np.zeros_like(tensor)
    for r in range(rank):
        T += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    if not np.allclose(T, tensor, atol=tol):
        return False
    return True

def _schoolbook_decomposition(n, m, p):
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
    return U, V, W

def _try_winograd_like(n, m, p):
    """
    Try to find a lower-rank decomposition using Winograd-like techniques.
    
    For 2x3x10: C[i,k] = sum_j A[i,j] * B[j,k]
    
    We can group the k dimension. For each j, B[j,:] is a row of B.
    
    Key insight: If we can express the sum over j with fewer terms,
    we reduce rank.
    
    Strategy: Try to find a rank-r decomposition where r < 60.
    We'll use a systematic search over structured decompositions.
    """
    start_time = time.time()
    best_rank = n * m * p
    best_U, best_V, best_W = _schoolbook_decomposition(n, m, p)
    tensor = _build_tensor(n, m, p)
    
    # Try ranks from low to high
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(6, 30, 20) = 30
    # But with the structure, we might do better than 60.
    
    # Try a simple approach: use the fact that for fixed i,
    # C[i,:] = A[i,:] @ B
    # This is a linear map from B to C[i,:]
    
    # Let's try to find decompositions with rank between 30 and 60
    # using integer/half-integer entries
    
    # Approach: Systematic search over small perturbations of schoolbook
    # Or use known results for similar dimensions
    
    # For 2x3x10, let's try to exploit the structure:
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k]
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k]
    
    # Each output C[i,k] depends on all 3 rows of B and 2 entries of A.
    
    # Total number of (i,k) pairs: 2*10 = 20
    # Total number of (i,j) pairs in A: 2*3 = 6
    # Total number of (j,k) pairs in B: 3*10 = 30
    
    # The tensor has rank at most 60 (schoolbook).
    
    # Let's try to find a rank-50 decomposition or lower.
    
    # Use a randomized search with integer/half-integer entries
    np.random.seed(42)
    
    for target_rank in range(30, min(60, best_rank)):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        # Try to find a decomposition with this rank
        # Use a structured approach: assume U, V, W have sparse structure
        
        # For simplicity, try a few random structured decompositions
        for trial in range(100):
            if time.time() - start_time > DEADLINE - 10:
                break
            
            # Generate random sparse matrices with integer/half-integer entries
            U = np.zeros((n*m, target_rank))
            V = np.zeros((m*p, target_rank))
            W = np.zeros((p*n, target_rank))
            
            # Each column should have a few non-zero entries
            for r in range(target_rank):
                # Choose which positions are non-zero
                u_idx = np.random.choice(n*m, size=np.random.randint(1, 4), replace=False)
                v_idx = np.random.choice(m*p, size=np.random.randint(1, 4), replace=False)
                w_idx = np.random.choice(p*n, size=np.random.randint(1, 4), replace=False)
                
                # Assign integer/half-integer values
                U[u_idx, r] = np.random.choice([-2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2])
                V[v_idx, r] = np.random.choice([-2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2])
                W[w_idx, r] = np.random.choice([-2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2])
            
            # Verify
            if _verify_decomposition(U, V, W, tensor, tol=1e-9):
                best_rank = target_rank
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                break
    
    return best_U, best_V, best_W, best_rank

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Start with schoolbook
    U, V, W = _schoolbook_decomposition(n, m, p)
    best_rank = n * m * p
    
    # Try to find a lower-rank decomposition
    if time.time() - start_time < DEADLINE - 5:
        U_try, V_try, W_try, rank_try = _try_winograd_like(n, m, p)
        if rank_try < best_rank:
            U, V, W = U_try, V_try, W_try
            best_rank = rank_try
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
