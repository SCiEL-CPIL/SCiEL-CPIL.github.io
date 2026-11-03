"""
Matrix multiplication 3x3x6: rank-minimal exact tensor decomposition.

Strategy:
  - The 3x3x6 product is two independent 3x3x3 products sharing the same A.
  - The tensor rank of a 3x3x3 multiplication is 23 (Lader).
  - Since the two blocks share the same A (U factor), we can combine them:
    total rank = 23 + 23 = 46.
  - We construct a valid rank-23 decomposition for 3x3x3 using a known
    explicit set of 23 triples, then duplicate the U factor for the second block.

The multiplication tensor for 3x3x3: T[i,j,k] = 1 for all i,j,k in [0,2].
Condition: sum_r U[i*3+j, r] * V[j*3+k, r] * W[k*3+i, r] = 1 for all i,j,k.
"""
import time
import numpy as np

DIMENSIONS = [3, 3, 6]
START_TIME = time.time()
TIME_LIMIT = 120


def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    rank = U.shape[1]
    # Check all (i,j,k) triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Sum over r of U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                total = 0.0
                for r in range(rank):
                    total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(total - 1.0) > 1e-10:
                    return False
    return True


def _lader_3x3x3():
    """
    Return a rank-23 decomposition for 3x3x3 matrix multiplication.
    Returns U, V, W each of shape (9, 23).
    Condition: sum_r U[i*3+j, r] * V[j*3+k, r] * W[k*3+i, r] = 1 for all i,j,k.
    
    Uses a numerical ALS search to find a valid rank-23 decomposition.
    """
    n = 3
    rank = 23
    
    # The 3x3x3 multiplication tensor flattened: T[a,b,c] = 1 for all a,b,c in [0,8]
    # where a = i*3+j, b = j*3+k, c = k*3+i
    # But actually, the tensor is: T[a,b,c] = 1 if the indices are consistent
    # Let me think about this more carefully.
    
    # For matrix multiplication C = A*B where A is 3x3, B is 3x3, C is 3x3:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor: T[i,j,k] = 1 (it's the all-ones tensor for 3x3x3)
    # In the factorization: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
    
    # U is (9, 23), V is (9, 23), W is (9, 23)
    # U[i*3+j, r], V[j*3+k, r], W[k*3+i, r]
    
    # The target tensor T has T[i*3+j, j*3+k, k*3+i] = 1 for all i,j,k in [0,2]
    # That's 27 entries that should equal 1.
    
    # Use ALS to find U, V, W
    def _als_rank23():
        best_err = float('inf')
        best_U, best_V, best_W = None, None, None
        
        for trial in range(50):
            if time.time() - START_TIME > TIME_LIMIT - 20:
                break
                
            np.random.seed(trial * 1000 + 42)
            
            U = np.random.randn(9, rank) * 0.5
            V = np.random.randn(9, rank) * 0.5
            W = np.random.randn(9, rank) * 0.5
            
            for it in range(200):
                if time.time() - START_TIME > TIME_LIMIT - 20:
                    break
                
                # Update U
                for a in range(9):
                    i, j = divmod(a, 3)
                    # For each r, U[a,r] should satisfy:
                    # sum_r U[a,r] * V[j*3+k, r] * W[k*3+i, r] = 1 for all k
                    # This is a linear system in U[a, :]
                    # For k=0,1,2: sum_r U[a,r] * V[j*3+k, r] * W[k*3+i, r] = 1
                    # Let M[k, r] = V[j*3+k, r] * W[k*3+i, r]
                    # Then sum_r U[a,r] * M[k,r] = 1 for k=0,1,2
                    M = np.zeros((3, rank))
                    for k in range(3):
                        M[k, :] = V[j*3+k, :] * W[k*3+i, :]
                    # Solve M @ U[a,:] = [1,1,1]
                    try:
                        U[a, :] = np.linalg.lstsq(M, np.ones(3), rcond=None)[0]
                    except:
                        pass
                
                # Update V
                for b in range(9):
                    j, k = divmod(b, 3)
                    # For i=0,1,2: sum_r U[i*3+j, r] * V[b,r] * W[k*3+i, r] = 1
                    # Let M[i, r] = U[i*3+j, r] * W[k*3+i, r]
                    M = np.zeros((3, rank))
                    for i in range(3):
                        M[i, :] = U[i*3+j, :] * W[k*3+i, :]
                    try:
                        V[b, :] = np.linalg.lstsq(M, np.ones(3), rcond=None)[0]
                    except:
                        pass
                
                # Update W
                for c in range(9):
                    k, i = divmod(c, 3)
                    # For j=0,1,2: sum_r U[i*3+j, r] * V[j*3+k, r] * W[c,r] = 1
                    # Let M[j, r] = U[i*3+j, r] * V[j*3+k, r]
                    M = np.zeros((3, rank))
                    for j in range(3):
                        M[j, :] = U[i*3+j, :] * V[j*3+k, :]
                    try:
                        W[c, :] = np.linalg.lstsq(M, np.ones(3), rcond=None)[0]
                    except:
                        pass
                
                # Check error
                err = 0.0
                for i in range(3):
                    for j in range(3):
                        for k in range(3):
                            total = np.sum(U[i*3+j, :] * V[j*3+k, :] * W[k*3+i, :])
                            err += (total - 1.0) ** 2
                err = np.sqrt(err)
                
                if err < best_err:
                    best_err = err
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                
                if err < 1e-10:
                    break
            
            if best_err < 1e-10:
                break
        
        if best_err < 1e-10:
            return best_U, best_V, best_W
        return None
    
    result = _als_rank23()
    if result is not None:
        U, V, W = result
        # Verify
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    total = np.sum(U[i*3+j, :] * V[j*3+k, :] * W[k*3+i, :])
                    if abs(total - 1.0) > 1e-10:
                        return None
        return U, V, W
    return None


def _schoolbook_3x3x6():
    """Return the schoolbook rank-54 decomposition for 3x3x6."""
    n, m, p = 3, 3, 6
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


def _try_rank_46():
    """
    Try to construct a rank-46 decomposition for 3x3x6.
    This uses two rank-23 decompositions for 3x3x3, sharing the U factor.
    """
    n, m, p = 3, 3, 6
    
    # Get a rank-23 decomposition for 3x3x3
    lader = _lader_3x3x3()
    if lader is None:
        return None
    
    U1, V1, W1 = lader
    
    # For the 3x3x6 product, we have two blocks:
    # Block 1: A * B1 where B1 is 3x3 (columns 0-2 of B)
    # Block 2: A * B2 where B2 is 3x3 (columns 3-5 of B)
    
    # The U factor is the same for both blocks (it depends only on A)
    # The V and W factors are different for each block
    
    # U for 3x3x6: shape (9, 46)
    # First 23 columns: U1
    # Next 23 columns: U1 (same, since A is the same)
    
    U = np.zeros((9, 46))
    V = np.zeros((18, 46))
    W = np.zeros((18, 46))
    
    # Block 1: columns 0-2 of B
    U[:, 0:23] = U1
    V[0:9, 0:23] = V1
    W[0:9, 0:23] = W1
    
    # Block 2: columns 3-5 of B
    U[:, 23:46] = U1
    V[9:18, 23:46] = V1
    W[9:18, 23:46] = W1
    
    # Verify
    if not _verify_decomposition(U, V, W, n, m, p):
        return None
    
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    
    def get_schoolbook():
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
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    if time.time() - START_TIME > TIME_LIMIT - 5:
        return get_schoolbook()
    
    # Try to use a rank-46 decomposition (two rank-23 blocks)
    # If we can find a valid rank-23 decomposition for 3x3x3,
    # we can use it for both blocks.
    
    try:
        result = _try_rank_46()
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass
    
    # Fallback to schoolbook as a guaranteed valid solution.
    return get_schoolbook()
