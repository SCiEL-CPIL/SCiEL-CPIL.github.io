"""Tensor decomposition search for 3x4 * 4x6 matrix multiplication."""
import time
import random
import numpy as np

DIMENSIONS = [3, 4, 6]
N, M, P = DIMENSIONS
RANK_SCHOOLBOOK = N * M * P  # 72
DEADLINE = 130.0  # safety margin before 150s kill


def _build_tensors(rank):
    """Return (A, B, C) as (n*m*p, rank) arrays for the matrix mult tensor."""
    A = np.zeros((N * M * P, rank))
    B = np.zeros((N * M * P, rank))
    C = np.zeros((N * M * P, rank))
    for i in range(N):
        for j in range(M):
            for k in range(P):
                idx = i * M * P + j * P + k
                A[idx, 0] = 1.0
                B[idx, 0] = 1.0
                C[idx, 0] = 1.0
    # Actually we need A[i,j,c], B[j,k,c], C[k,i,c] but flattened differently.
    # Let me redo: the tensor T[i,j,k] = e_i e_j^T e_k^T.
    # Decomposition: T = sum_c u_c ⊗ v_c ⊗ w_c
    # A[i*m+j, c] = u_c[i*m+j], B[j*p+k, c] = v_c[j*p+k], C[k*n+i, c] = w_c[k*n+i]
    # T[i,j,k] = sum_c A[i*m+j,c]*B[j*p+k,c]*C[k*n+i,c]
    # For schoolbook: A[i*m+j, i*m*p+j*p+k]=1, B[j*p+k, ...]=1, C[k*n+i,...]=1
    # So T[i,j,k] = 1 for the matching (i,j,k) term.
    return None  # placeholder, we'll verify differently


def verify(U, V, W, rank):
    """Check if decomposition is exact. Returns True if valid."""
    n, m, p = N, M, P
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # T[i,j,k] should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # and T[i,j,k] = 1 for all (i,j,k) in the multiplication tensor
    # Wait - the matrix multiplication tensor is T[i,j,k] = 1 if the (i,j) entry of A and (j,k) of B
    # contribute to (i,k) of C. So T[i,j,k] = 1 for all valid (i,j,k).
    # Actually the tensor for matrix mult is: T(i,j,k) = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # because C[i,k] = sum_j A[i,j]*B[j,k], so each (i,j,k) triple has coefficient 1.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True


def verify_fast(U, V, W, rank):
    """Faster verification using numpy."""
    n, m, p = N, M, P
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # Build the full tensor T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # This is a 3-way contraction.
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    Ur = U.reshape(n, m, rank)
    Vr = V.reshape(m, p, rank)
    Wr = W.reshape(p, n, rank)
    # T[i,j,k] = sum_c Ur[i,j,c] * Vr[j,k,c] * Wr[k,i,c]
    # Use einsum
    T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
    # T should be all ones
    return np.allclose(T, 1.0, atol=1e-9)


def schoolbook():
    """Return the schoolbook decomposition of rank n*m*p."""
    n, m, p = N, M, P
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


def try_random_search(max_rank, time_budget):
    """Try random search for a decomposition of rank max_rank."""
    n, m, p = N, M, P
    start = time.time()
    rng = np.random.RandomState(42)
    
    # Use iterative refinement: start with random matrices and try to project
    # onto the constraint manifold.
    
    for attempt in range(200):
        if time.time() - start > time_budget:
            break
        if time.time() > DEADLINE:
            break
        
        # Random initialization
        U = rng.randn(n * m, max_rank)
        V = rng.randn(m * p, max_rank)
        W = rng.randn(p * n, max_rank)
        
        # Try a few iterations of gradient-like updates
        # The constraint is: for all (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
        # We can try to use the fact that if we fix two, we can solve for the third.
        
        # Alternative: use the approach of trying to find rank-r decomposition
        # by solving a system of equations.
        
        # For small ranks, try brute force with structured entries.
        # Try entries from {-1, 0, 1} or {-1, 0, 1, 2}
        
        if max_rank <= 40:
            # Try random integer entries
            for trial in range(50):
                if time.time() > DEADLINE:
                    break
                U = rng.randint(-2, 3, size=(n * m, max_rank))
                V = rng.randint(-2, 3, size=(m * p, max_rank))
                W = rng.randint(-2, 3, size=(p * n, max_rank))
                if verify_fast(U, V, W, max_rank):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None


def try_structured_search(time_budget):
    """Try structured/analytic approaches."""
    n, m, p = N, M, P
    start = time.time()
    
    # Known bounds: For 3x4*4x6, the rank is between 3*4*6=72 (trivial upper)
    # and some lower bound. Let's try to find improvements.
    
    # One approach: use the fact that we can share computations.
    # For example, if we group rows or columns.
    
    # Try rank 60, 55, 50, 45, 40
    for target_rank in [60, 55, 50, 45, 42, 40]:
        if time.time() - start > time_budget or time.time() > DEADLINE:
            break
        result = try_random_search(target_rank, 5)
        if result is not None:
            return result
    
    return None


def construct():
    start_time = time.time()
    
    # Start with schoolbook as fallback
    best = schoolbook()
    
    # Try to find better decompositions
    # Strategy 1: Random search for lower rank
    remaining = min(60, DEADLINE - (time.time() - start_time) - 10)
    if remaining > 5:
        result = try_structured_search(remaining)
        if result is not None:
            best = result
    
    # Strategy 2: If we still have time, try more aggressive search
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 15:
        for target_rank in [70, 68, 66, 64, 62]:
            if time.time() > DEADLINE - 10:
                break
            result = try_random_search(target_rank, 8)
            if result is not None:
                best = result
                break
    
    # Verify the best result
    if not verify_fast(best["U"], best["V"], best["W"], len(best["U"][0])):
        # Fallback to schoolbook
        best = schoolbook()
    
    return best
