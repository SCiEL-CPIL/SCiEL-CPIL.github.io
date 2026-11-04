"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying
    a 3x3 matrix by a 3x6 matrix.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 20s safety margin before 160s limit
    
    # Build the target tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U_fb = np.zeros((n*m, rank_schoolbook), dtype=np.int64)
    V_fb = np.zeros((m*p, rank_schoolbook), dtype=np.int64)
    W_fb = np.zeros((p*n, rank_schoolbook), dtype=np.int64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j, col] = 1
                V_fb[j*p+k, col] = 1
                W_fb[k*n+i, col] = 1
                col += 1
    
    def validate(U, V, W):
        """Check if U, V, W exactly decompose T."""
        U = np.asarray(U, dtype=np.float64)
        V = np.asarray(V, dtype=np.float64)
        W = np.asarray(W, dtype=np.float64)
        rank = U.shape[1]
        # Compute the decomposition
        # T_reconstructed[a, b, c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # This is expensive for large rank, so use einsum
        T_rec = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        # Check if it matches T
        if not np.allclose(T_rec, T, atol=1e-9):
            return False
        # Check entries are integer or half-integer
        for mat in [U, V, W]:
            if not np.all(np.abs(mat * 2 - np.round(mat * 2)) < 1e-9):
                return False
        return True
    
    def to_int_or_half(arr):
        """Round to nearest integer or half-integer."""
        arr = np.asarray(arr, dtype=np.float64)
        return np.round(arr * 2) / 2.0
    
    # Try to find a rank-46 decomposition using two rank-23 decompositions for 3x3x3
    # First, try to find a rank-23 decomposition for 3x3x3
    n2, m2, p2 = 3, 3, 3
    T2 = np.zeros((n2*m2, m2*p2, p2*n2), dtype=np.float64)
    for i in range(n2):
        for j in range(m2):
            for k in range(p2):
                T2[i*m2+j, j*p2+k, k*n2+i] = 1.0
    
    def find_3x3x3_decomp(rank_target=23, max_iter=5000, seed=42):
        """Try to find a rank-rank_target decomposition for 3x3x3."""
        np.random.seed(seed)
        best_err = np.inf
        best_U, best_V, best_W = None, None, None
        
        for attempt in range(20):
            if time.time() > deadline - 30:
                break
                
            U = np.random.randn(n2*m2, rank_target) * 0.5
            V = np.random.randn(m2*p2, rank_target) * 0.5
            W = np.random.randn(p2*n2, rank_target) * 0.5
            
            # Use gradient descent with Adam-like updates
            lr = 0.01
            for it in range(max_iter):
                if time.time() > deadline - 30:
                    break
                
                # Compute gradient
                T_rec = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
                err = T_rec - T2
                err_sq = np.sum(err**2)
                
                if err_sq < best_err:
                    best_err = err_sq
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                
                if err_sq < 1e-12:
                    break
                
                # Gradient of sum(err^2) w.r.t. U[a,r]
                # d/dU[a,r] sum_{a',b',c'} (T_rec[a',b',c'] - T2[a',b',c'])^2
                # = 2 * sum_{b',c'} err[a',b',c'] * V[b',r] * W[c',r] for a'=a
                gU = 2 * np.einsum('abc,br,cr->ar', err, V, W, optimize=True)
                gV = 2 * np.einsum('abc,ar,cr->br', err, U, W, optimize=True)
                gW = 2 * np.einsum('abc,ar,br->cr', err, U, V, optimize=True)
                
                U -= lr * gU
                V -= lr * gV
                W -= lr * gW
            
            if best_err < 1e-12:
                break
        
        if best_err < 1e-9:
            # Round to integers/half-integers and verify
            U_r = to_int_or_half(best_U)
            V_r = to_int_or_half(best_V)
            W_r = to_int_or_half(best_W)
            T_rec = np.einsum('ar,br,cr->abc', U_r, V_r, W_r, optimize=True)
            if np.allclose(T_rec, T2, atol=1e-9):
                return U_r, V_r, W_r
        
        return None, None, None
    
    # Try to find 3x3x3 decomposition
    U2, V2, W2 = find_3x3x3_decomp()
    
    if U2 is not None:
        # Use it for both blocks of the 3x3x6 problem
        # Block 1: columns 0-2 of B (k=0,1,2)
        # Block 2: columns 3-5 of B (k=3,4,5)
        rank = 2 * 23  # 46
        U = np.zeros((n*m, rank), dtype=np.float64)
        V = np.zeros((m*p, rank), dtype=np.float64)
        W = np.zeros((p*n, rank), dtype=np.float64)
        
        # Copy block 1 (k=0,1,2)
        U[:, 0:23] = U2
        V[:, 0:23] = V2
        W[:, 0:23] = W2  # This maps to k*n+i for k=0,1,2
        
        # Copy block 2 (k=3,4,5) - need to shift the W indices
        # For block 2, the output index is (k+3)*n+i for k=0,1,2 in the local 3x3x3
        # So W2[k*n+i] should go to W[(k+3)*n+i]
        for local_k in range(3):
            for local_i in range(3):
                W[(local_k+3)*n+local_i, 23:46] = W2[local_k*n+local_i, :]
        
        # But wait, V also needs adjustment for block 2
        # For block 2, B[j, k+3] for k=0,1,2
        # So V[j*p + (k+3)] for k=0,1,2
        for j in range(3):
            for local_k in range(3):
                V[j*p + (local_k+3), 23:46] = V2[j*p2 + local_k, :]
        
        # Verify
        if validate(U, V, W):
            # Convert to list format
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
# EVOLVE-BLOCK-END
