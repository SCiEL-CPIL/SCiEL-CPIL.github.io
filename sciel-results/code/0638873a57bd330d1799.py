import time
import random
import numpy as np

DIMENSIONS = [2, 4, 8]


def _schoolbook(n=2, m=4, p=8):
    """Return the schoolbook (rank n*m*p) decomposition as (U, V, W) lists of lists."""
    r = n * m * p
    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W


def _verify(U, V, W, n=2, m=4, p=8, trials=30, tol=1e-9):
    """
    Verify the decomposition.
    U: (n*m, r), V: (m*p, r), W: (p*n, r)
    C[i,k] = sum_c (sum_j A[i,j]*U[i*m+j,c]) * (sum_j B[j,k]*V[j*p+k,c]) * W[k*n+i,c]
    """
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    r = U.shape[1]
    for _ in range(trials):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        # factor_A[i] = sum_j A[i,j] * U[i*m+j, c] for each c
        # Reshape U to (n, m, r)
        U3 = U.reshape(n, m, r)
        V3 = V.reshape(m, p, r)
        W3 = W.reshape(p, n, r)
        # For each c: factorA[i,c] = sum_j A[i,j]*U3[i,j,c]
        # factorA shape: (n, r)
        factorA = np.einsum('ij,ijc->ic', A, U3)  # (n, r)
        # factorB[k,c] = sum_j B[j,k]*V3[j,k,c]
        factorB = np.einsum('jk,jkc->kc', B, V3)  # (p, r)
        # C[i,k] = sum_c factorA[i,c] * factorB[k,c] * W3[k,i,c]
        C_computed = np.einsum('ic,kc,kic->ik', factorA, factorB, W3)
        if np.max(np.abs(C_computed - C_expected)) > tol:
            return False
    return True


def _construct_structured():
    """
    Build a structured decomposition that exploits the small dimensions.
    
    Key insight: The matrix multiplication tensor for (n,m,p) = (2,4,8) has
    rank at most n*m*p = 64 (schoolbook). But we can potentially do better
    by exploiting structure.
    
    One approach: Use the fact that we're multiplying a 2x4 by 4x8 matrix.
    The result is 2x8. Each entry C[i,k] = sum_{j=0}^{3} A[i,j]*B[j,k].
    
    A rank-1 term in the bilinear decomposition computes:
    C[i,k] += (sum_j A[i,j]*u_{i,j}) * (sum_j B[j,k]*v_{j,k}) * w_{i,k}
    
    For the schoolbook, each (i,j,k) triple gives one rank-1 term with
    u_{i,j}=1 for that specific (i,j), v_{j,k}=1 for that specific (j,k),
    w_{i,k}=1.
    
    To reduce rank, we need to "merge" some of these terms.
    
    Strategy: Try a parametrized family with shared structure.
    
    For n=2, m=4, p=8:
    - We have 2 rows of A, each with 4 columns
    - We have 4 rows of B, each with 8 columns
    
    Let's try: for each j (column of A / row of B, 0..3), we have 2*8=16
    output entries to compute. But we can share the "j-th" summation across
    all (i,k) pairs.
    
    Actually, the minimal rank is an open problem in general, but for small
    cases we can try exhaustive or smart search.
    
    Let me try a different approach: start from schoolbook and try to merge
    pairs of rank-1 terms by finding linear dependencies.
    """
    n, m, p = 2, 4, 8
    U, V, W = _schoolbook(n, m, p)
    rank = n * m * p
    
    # Try to reduce by checking if any two columns can be merged
    # Two columns c1, c2 can be merged if there exist alpha, beta such that
    # the sum alpha*col1 + beta*col2 still works (i.e., the combined term
    # is still a valid rank-1 term)
    
    # Actually, a cleaner approach: try all possible rank reductions by
    # solving for whether a column is redundant given the others.
    
    # For each column c, check if it can be expressed as a linear combination
    # of the other columns in a way that preserves the decomposition.
    # This is hard in general. Let's try a simpler heuristic.
    
    # Alternative: Try rank = n*m*p - 1, n*m*p - 2, ... by random search
    # with the constraint that entries are small integers or half-integers.
    
    return U, V, W, rank


def _try_rank_reduction(U, V, W, n=2, m=4, p=8, max_rounds=200):
    """
    Iteratively try to remove a column if the decomposition still holds
    after removing it. Use verification to check.
    """
    U = [list(row) for row in U]
    V = [list(row) for row in V]
    W = [list(row) for row in W]
    rank = len(U[0]) if U else 0
    
    for _ in range(max_rounds):
        removed = False
        for c in range(rank):
            U_test = [row[:c] + row[c+1:] for row in U]
            V_test = [row[:c] + row[c+1:] for row in V]
            W_test = [row[:c] + row[c+1:] for row in W]
            if _verify(U_test, V_test, W_test, n, m, p, trials=20, tol=1e-9):
                U, V, W = U_test, V_test, W_test
                rank -= 1
                removed = True
                break
        if not removed:
            break
    return U, V, W, rank


def construct():
    start_time = time.time()
    deadline = start_time + 120  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
    best_rank = schoolbook_rank
    
    # Verify schoolbook first
    if not _verify(U, V, W, n, m, p, trials=20, tol=1e-9):
        raise RuntimeError("Schoolbook verification failed")
    
    # Try rank reduction on schoolbook
    U_r, V_r, W_r, rank_r = _try_rank_reduction(U, V, W, n, m, p, max_rounds=100)
    if rank_r < best_rank:
        best_U, best_V, best_W = U_r, V_r, W_r
        best_rank = rank_r
    
    if time.time() > deadline - 5:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # Try structured lower-rank constructions
    # For (2,4,8), the tensor has 2*4*8 = 64 "natural" terms.
    # Known: rank of matrix mult for (2,4,8) is at most 64, at least 32 (trivially).
    # Let's try to find a decomposition with rank < 64.
    
    # Approach: Use the fact that for fixed i, C[i,k] = sum_j A[i,j]*B[j,k]
    # is a bilinear form. We can try to share structure across i.
    
    # Try: rank = 48, 40, 36, 32
    for target_rank in [56, 48, 40, 36, 32, 28, 24]:
        if time.time() > deadline - 10:
            break
        
        found = False
        for seed in range(10):
            if time.time() > deadline - 10:
                break
            
            random.seed(seed + target_rank * 100)
            np.random.seed(seed + target_rank * 100)
            
            # Try a structured initialization
            U_try = np.zeros((n * m, target_rank))
            V_try = np.zeros((m * p, target_rank))
            W_try = np.zeros((p * n, target_rank))
            
            # Initialize with small random values
            U_try = np.random.randn(n * m, target_rank) * 0.3
            V_try = np.random.randn(m * p, target_rank) * 0.3
            W_try = np.random.randn(p * n, target_rank) * 0.3
            
            if _verify(U_try.tolist(), V_try.tolist(), W_try.tolist(), n, m, p, trials=20, tol=1e-8):
                U_l = U_try.tolist()
                V_l = V_try.tolist()
                W_l = W_try.tolist()
                U_re, V_re, W_re, r_re = _try_rank_reduction(U_l, V_l, W_l, n, m, p, max_rounds=50)
                if r_re < best_rank:
                    best_U, best_V, best_W = U_re, V_re, W_re
                    best_rank = r_re
                    found = True
                    break
        
        if found:
            break
    
    if time.time() > deadline - 2:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # Final: try to reduce the best found so far
    U_f, V_f, W_f, rank_f = _try_rank_reduction(best_U, best_V, best_W, n, m, p, max_rounds=100)
    if rank_f < best_rank:
        best_U, best_V, best_W = U_f, V_f, W_f
        best_rank = rank_f
    
    return {"U": best_U, "V": best_V, "W": best_W}
