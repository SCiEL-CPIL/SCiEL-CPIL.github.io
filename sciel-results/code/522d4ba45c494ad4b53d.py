"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n=3, m=3, p=6):
    """Verify that the decomposition is exact: for all (i,j,k),
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    and for all (i,j,k) with i'!=i or k'!=k (same j) the sum is 0.
    Actually the condition is:
    T_{i,j,k} = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
    should equal 1 if the tensor entry (i,j,k) is 1 (which it always is for matrix mult),
    and 0 otherwise. But for matrix multiplication, T_{i,j,k} = 1 for all valid (i,j,k).
    
    Wait, let me re-read. The matrix multiplication tensor has entries:
    T_{i,j,k} = A[i,j] * B[j,k] * C[k,i] ... no.
    
    The tensor for multiplying (n x m) by (m x p) giving (n x p) is:
    T[i, j, k, l] = delta_{j,k}  (the index j from A and k from B must match)
    Then C[i,l] = sum_j A[i,j] * B[j,l].
    
    In the tensor decomposition convention for matrix multiplication:
    The multiplication tensor T has indices (i, j, k) where i is row of A, j is col of A = row of B, k is col of B.
    T[i,j,k] = 1 (it's just the identity in the middle index).
    
    The decomposition is: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    So we need: for all i in range(n), j in range(m), k in range(p):
        sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    And that's it! There are no other constraints because the tensor only has these entries.
    """
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    rank = U.shape[1]
    
    # Check all (i, j, k) triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _try_rank(rank, n=3, m=3, p=6, max_iter=1000, seed=42):
    """Try to find a rank-`rank` decomposition using ALS."""
    np.random.seed(seed)
    
    # Initialize U, V, W randomly
    U = np.random.randn(n*m, rank)
    V = np.random.randn(m*p, rank)
    W = np.random.randn(p*n, rank)
    
    # Normalize each column to have unit norm (for stability)
    for c in range(rank):
        norms = [np.linalg.norm(U[:, c]), np.linalg.norm(V[:, c]), np.linalg.norm(W[:, c])]
        if any(norm == 0 for norm in norms):
            continue
        scale = (norms[0] * norms[1] * norms[2]) ** (1.0/3.0)
        if scale > 1e-15:
            U[:, c] /= scale
            V[:, c] /= scale
            W[:, c] /= scale
    
    for iteration in range(max_iter):
        # ALS step: update U given V, W
        # For each row (i*m+j) of U, we need to solve for U[i*m+j, :] such that
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all k
        # This is a linear system: U[i*m+j, :] @ (V[j*p+k, :] * W[k*n+i, :])^T = 1 for all k
        # Let M be the matrix where M[k, c] = V[j*p+k, c] * W[k*n+i, c]
        # Then U[i*m+j, :] @ M^T = ones(p)
        # So U[i*m+j, :] = (M M^T)^{-1} M ones(p)  (least squares)
        
        for i in range(n):
            for j in range(m):
                # Build M: p x rank
                M = np.zeros((p, rank))
                for k in range(p):
                    M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                # Solve U[i*m+j, :] @ M^T = ones(p)
                # i.e., M @ U[i*m+j, :]^T = ones(p)
                try:
                    sol, _, _, _ = np.linalg.lstsq(M.T, np.ones(p), rcond=None)
                    U[i*m+j, :] = sol
                except:
                    pass
        
        # ALS step: update V given U, W
        # For each row (j*p+k) of V:
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i
        # M[i, c] = U[i*m+j, c] * W[k*n+i, c]
        # V[j*p+k, :] @ M^T = ones(n)
        for j in range(m):
            for k in range(p):
                M = np.zeros((n, rank))
                for i in range(n):
                    M[i, :] = U[i*m+j, :] * W[k*n+i, :]
                try:
                    sol, _, _, _ = np.linalg.lstsq(M.T, np.ones(n), rcond=None)
                    V[j*p+k, :] = sol
                except:
                    pass
        
        # ALS step: update W given U, V
        # For each row (k*n+i) of W:
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all j
        # M[j, c] = U[i*m+j, c] * V[j*p+k, c]
        # W[k*n+i, :] @ M^T = ones(m)
        for k in range(p):
            for i in range(n):
                M = np.zeros((m, rank))
                for j in range(m):
                    M[j, :] = U[i*m+j, :] * V[j*p+k, :]
                try:
                    sol, _, _, _ = np.linalg.lstsq(M.T, np.ones(m), rcond=None)
                    W[k*n+i, :] = sol
                except:
                    pass
        
        # Check convergence
        if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
            return U, V, W, True
    
    return None, None, None, False

def construct():
    """Construct the rank-minimal exact tensor decomposition."""
    start_time = time.time()
    n, m, p = DIMENSIONS
    deadline = start_time + 140  # 10s margin before 150s limit
    
    # Schoolbook rank as fallback
    schoolbook_rank = n * m * p
    
    # Try to find lower rank decompositions
    # The theoretical lower bound is max(n*m, m*p, p*n) = max(9, 18, 18) = 18
    # For 3x3x3, the best known is 23 (for 3x3x3, rank 23 is known)
    # For 3x3x6, we can split into two 3x3x3 problems, giving rank 46
    # But maybe we can do better.
    
    # First, try the block construction: two independent 3x3x3 problems
    # Each 3x3x3 has rank 23 (known optimal), so total rank = 46
    # But we need the actual 3x3x3 rank 23 decomposition.
    
    # Since finding the exact rank-23 3x3x3 decomposition is non-trivial without lookup,
    # let's try to find a good decomposition via search.
    
    # Strategy: try ranks from a reasonable lower bound up to schoolbook
    # Lower bound for 3x3x6: max(9, 18, 18) = 18
    # But realistically, for matrix multiplication, the rank is at least n*m + m*p + p*n - ... 
    # The Strassen lower bound etc. For 3x3x3, rank 23 is known.
    
    # Let's try a few ranks with multiple random seeds and ALS iterations
    best_result = None
    best_rank = schoolbook_rank
    
    # Return schoolbook first to have a valid result
    def _schoolbook():
        U = [[0] * schoolbook_rank for _ in range(n * m)]
        V = [[0] * schoolbook_rank for _ in range(m * p)]
        W = [[0] * schoolbook_rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    schoolbook = _schoolbook()
    
    # Try to improve with various ranks
    # Try ranks that are multiples of smaller structures
    candidate_ranks = []
    for r in range(18, schoolbook_rank):
        candidate_ranks.append(r)
    
    # Sort by rank (try smaller first)
    # But ALS may not converge for very small ranks, so try a few
    # Try a subset of ranks to save time
    ranks_to_try = list(range(20, min(50, schoolbook_rank)))
    
    for rank in ranks_to_try:
        if time.time() > deadline:
            break
        
        for seed in [42, 123, 777, 2024]:
            if time.time() > deadline:
                break
            
            U, V, W, success = _try_rank(rank, n, m, p, max_iter=200, seed=seed)
            if success:
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = rank
                # If we found a very low rank, we can stop
                if rank <= 25:
                    return best_result
    
    if best_result is not None:
        return best_result
    
    return schoolbook
# EVOLVE-BLOCK-END
