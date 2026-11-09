"""Matrix multiplication 3x4x8: rank-minimal exact tensor decomposition search."""
import time
import itertools
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _verify(U, V, W, n, m, p):
    """Check if the factorization exactly reproduces the matrix multiplication tensor."""
    rank = len(U[0])
    # Build the multiplication tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For matrix multiplication: (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # Tensor form: T[i,j,k] should equal delta (i.e., the multiplication tensor has T[i,j,k]=1 for all i,j,k
    # when we think of it as the trilinear form <U_r, V_r, W_r> giving the coefficient of A[i,j]*B[j,k] in C[i,k])
    
    # Actually, the standard tensor decomposition for matmul:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] * A[i,j] * B[j,k]... 
    # No wait. The tensor T has indices (i,j,k) and T[i,j,k] = 1 for all valid i,j,k.
    # The decomposition says: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Let's verify: for each (i,j,k), check that sum_r U[i*m+j][r]*V[j*p+k][r]*W[k*n+i][r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1) > 1e-9:
                    return False
    return True


def _try_random_search(n, m, p, max_rank, n_trials, deadline):
    """Try to find a low-rank decomposition using random initialization and local search."""
    best = None
    best_rank = n * m * p  # schoolbook rank
    
    start = time.time()
    for trial in range(n_trials):
        if time.time() > deadline - 5:
            break
        
        # Try ranks from low to high
        for rank in range(max_rank, best_rank):
            if time.time() > deadline - 5:
                break
            
            # Random initialization with small integers/half-integers
            np.random.seed(trial * 1000 + rank)
            U = np.random.choice([-2, -1, -0.5, 0, 0.5, 1, 2], size=(n*m, rank))
            V = np.random.choice([-2, -1, -0.5, 0, 0.5, 1, 2], size=(m*p, rank))
            W = np.random.choice([-2, -1, -0.5, 0, 0.5, 1, 2], size=(p*n, rank))
            
            # Check
            if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = rank
                break
            elif rank < best_rank:
                # Keep as candidate for local refinement
                pass
    
    return best, best_rank


def _structured_decomposition(n, m, p):
    """
    Try structured decompositions based on known techniques:
    1. Kronecker product structure
    2. Block diagonal structure
    3. Known small matrix multiplication algorithms (Strassen-like)
    """
    # For 3x4x8, let's think about what's possible.
    # The schoolbook rank is 3*4*8 = 96.
    # Lower bound: max(n*m, m*p, p*n) = max(12, 32, 24) = 32
    # Also, the rank must be at least the tensor rank lower bound.
    
    # Let's try a block-structured approach:
    # Partition the 4x8 matrix multiplication into blocks.
    # A is 3x4, B is 4x8, C is 3x8.
    
    # Try: decompose B into two 4x4 blocks: B = [B1 | B2]
    # C = A * B1 + A * B2
    # Each A*B_i is 3x4, which has schoolbook rank 12.
    # Total: 24. But this doesn't help because A is shared.
    
    # Better: use the fact that 3x4x8 can be viewed as 3x4x4 + 3x4x4
    # but with the same A. This is just schoolbook.
    
    # Let's try a different approach: use the structure of the problem.
    # The tensor is T[i,j,k] = 1 for i in [0,3), j in [0,4), k in [0,8).
    
    # One idea: factor the k dimension. k = 2*k' + k'' where k' in [0,4), k'' in {0,1}
    # T[i,j,2k'+k''] = 1
    # This doesn't immediately help.
    
    # Another idea: use the identity that for any a, b, c:
    # (a+b)(b+c) = a*b + a*c + b*b + b*c... no this is getting complicated.
    
    # Let's just return schoolbook as the structured decomposition for now.
    # The search will try to find something better.
    return _schoolbook(n, m, p)


def _local_refine(U, V, W, n, m, p, max_iters, deadline):
    """
    Local search: try to reduce rank by zeroing out columns or 
    finding redundant columns that can be eliminated.
    """
    rank = len(U[0])
    U = [row[:] for row in U]
    V = [row[:] for row in V]
    W = [row[:] for row in W]
    
    # Try to eliminate columns that are linearly dependent or redundant
    for _ in range(max_iters):
        if time.time() > deadline - 2:
            break
        
        improved = False
        for r in range(rank - 1, -1, -1):
            if time.time() > deadline - 2:
                break
            
            # Try removing column r: check if the remaining columns still span the tensor
            # This is expensive, so only try for the last few columns
            if r < rank - 5:
                break
            
            # Check if column r is "redundant" by seeing if we can express it
            # in terms of others (approximately)
            # For now, just check if setting it to 0 still works
            U_save = [U[i][r] for i in range(n*m)]
            V_save = [V[i][r] for i in range(m*p)]
            W_save = [W[i][r] for i in range(p*n)]
            
            # Zero out column r
            for i in range(n*m):
                U[i][r] = 0
            for i in range(m*p):
                V[i][r] = 0
            for i in range(p*n):
                W[i][r] = 0
            
            if _verify(U, V, W, n, m, p):
                # Remove column r
                for i in range(n*m):
                    U[i].pop(r)
                for i in range(m*p):
                    V[i].pop(r)
                for i in range(p*n):
                    W[i].pop(r)
                rank -= 1
                improved = True
            else:
                # Restore
                for i in range(n*m):
                    U[i][r] = U_save[i]
                for i in range(m*p):
                    V[i][r] = V_save[i]
                for i in range(p*n):
                    W[i][r] = W_save[i]
            
            if improved:
                break
        
        if not improved:
            break
    
    return {"U": U, "V": V, "W": W}, rank


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s safety margin
    
    # Start with schoolbook
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Verify schoolbook
    assert _verify(best_result["U"], best_result["V"], best_result["W"], n, m, p), "Schoolbook verification failed"
    
    # Try structured decompositions
    if time.time() < deadline - 10:
        structured = _structured_decomposition(n, m, p)
        if _verify(structured["U"], structured["V"], structured["W"], n, m, p):
            s_rank = len(structured["U"][0])
            if s_rank < best_rank:
                best_result = structured
                best_rank = s_rank
    
    # Try random search with various rank targets
    if time.time() < deadline - 5:
        # Try ranks from 40 up to best_rank - 1
        max_search_rank = min(best_rank - 1, 60)
        result, found_rank = _try_random_search(n, m, p, max_search_rank, 200, deadline)
        if result is not None:
            best_result = result
            best_rank = found_rank
    
    # Local refinement on the best result found
    if time.time() < deadline - 2:
        refined, refined_rank = _local_refine(
            best_result["U"], best_result["V"], best_result["W"], 
            n, m, p, max_iters=50, deadline=deadline
        )
        if _verify(refined["U"], refined["V"], refined["W"], n, m, p):
            if refined_rank < best_rank:
                best_result = refined
                best_rank = refined_rank
    
    # Final verification
    if not _verify(best_result["U"], best_result["V"], best_result["W"], n, m, p):
        # Fallback to schoolbook
        best_result = _schoolbook(n, m, p)
    
    return best_result
# EVOLVE-BLOCK-END
