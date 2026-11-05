"""Search for a low-rank exact decomposition of 3x3x7 matrix multiplication."""
import time
import itertools
import random
import numpy as np

DIMENSIONS = [3, 3, 7]
DEADLINE = 120.0  # wall-clock seconds


def _time_left():
    return DEADLINE - time.time()


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check exact decomposition: sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] == 1."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    r = U.shape[1]
    # Vectorized verification: compute all (i,j,k) sums at once
    # Reshape to 3D tensors for efficient computation
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    
    # Compute sum over r of U[i,j,c] * V[j,k,c] * W[k,i,c]
    # This is a tensor contraction
    # First compute U*V: shape (n, m, p, r) -> sum over m and r
    # Better: use einsum
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # We can do: temp[j,k,c] = sum_i U[i,j,c] * W[k,i,c], then T[i,j,k] = sum_c temp[j,k,c] * V[j,k,c]
    # Actually, let's just do it directly with broadcasting
    # U[i,j,c] * V[j,k,c] * W[k,i,c]
    # Shape: (n,m,r) * (m,p,r) * (p,n,r)
    # Need to align indices: i from U and W, j from U and V, k from V and W, c summed
    
    # Use einsum for clarity
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return np.allclose(T, 1.0, atol=tol)


def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}


def _try_strassen_style(n, m, p, max_rank):
    """Try a structured low-rank decomposition with small integer/half-integer entries.

    We search over rank r and use a parametric family where each factor entry is
    drawn from a small set of values {-1, -0.5, 0, 0.5, 1}. We use a greedy /
    randomized approach to fill the tensor, then verify.
    
    Key insight: Use the structure of matrix multiplication. For each (i,j,k),
    we need sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1.
    
    Strategy: Start from schoolbook and try to merge terms by finding linear
    dependencies. Also try structured families based on known algorithms.
    """
    values = [-1.0, -0.5, 0.0, 0.0, 0.5, 1.0]
    rng = random.Random(12345)

    best = None
    # Try ranks from a reasonable lower bound upward
    # Lower bound: max(n*m, m*p, p*n) = max(9, 21, 21) = 21
    for r in range(21, max_rank + 1):
        if _time_left() < 10:
            break
        
        # Multiple random attempts for this rank
        attempts = 0
        max_attempts = 500
        while attempts < max_attempts and _time_left() > 10:
            attempts += 1
            
            U = np.zeros((n * m, r))
            V = np.zeros((m * p, r))
            W = np.zeros((p * n, r))
            
            # Use a sparse initialization: each column has only a few non-zero entries
            for c in range(r):
                # Pick random positions to be non-zero
                n_nonzero_U = rng.randint(1, 4)
                n_nonzero_V = rng.randint(1, 4)
                n_nonzero_W = rng.randint(1, 4)
                
                U_indices = rng.sample(range(n * m), n_nonzero_U)
                V_indices = rng.sample(range(m * p), n_nonzero_V)
                W_indices = rng.sample(range(p * n), n_nonzero_W)
                
                for idx in U_indices:
                    U[idx, c] = rng.choice(values)
                for idx in V_indices:
                    V[idx, c] = rng.choice(values)
                for idx in W_indices:
                    W[idx, c] = rng.choice(values)

            # Verify
            if _verify(U, V, W, n, m, p):
                best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                return best
    return best


def _try_constructive_low_rank(n, m, p):
    """Constructive approach: use a known small-rank scheme for the 3x3 core
    and extend to p=7 by sharing structure. We attempt rank = 7 * something.

    Strategy: Decompose the 3x7 multiplication into overlapping 3x3 blocks.
    For 3x3x3, Strassen's algorithm gives rank 7. We can tile this approach.
    
    For 3x3 by 3x7, we can view the 3x7 matrix as having columns grouped.
    One approach: use Strassen-like decomposition on 3x3 subproblems and
    combine them.
    
    Alternative: Use the fact that matrix multiplication is bilinear. We can
    try to find a decomposition where the rank is less than n*m*p.
    
    Let's try a more systematic approach: parameterize U, V, W with a small
    number of free parameters and solve the resulting system.
    """
    rng = random.Random(999)
    
    # Try a structured family: assume U, V, W have entries in {-1, 0, 1}
    # and use a more guided search.
    
    # Key idea: For each rank r, we have r terms. Each term (c) contributes
    # U[i,j,c] * V[j,k,c] * W[k,i,c] to T[i,j,k].
    # We need sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all i,j,k.
    
    # This is a system of 63 equations (3*3*7) with 3*r*9 + 3*r*21 + 3*r*21 = 9r*35 variables.
    # For r=21, we have 63 equations and 2205 variables - highly underdetermined.
    
    # But the equations are nonlinear (products of three variables per term).
    
    # Strategy: Use a greedy approach. Start with r=21 and try to satisfy
    # equations one at a time, adjusting the factors.
    
    values = [-1.0, 0.0, 1.0]
    
    for r in range(21, 35):
        if _time_left() < 15:
            break
        
        # Try multiple random initializations with sparse structure
        for attempt in range(100):
            if _time_left() < 15:
                break
            
            U = np.zeros((n * m, r))
            V = np.zeros((m * p, r))
            W = np.zeros((p * n, r))
            
            # Initialize with a pattern: for each (i,j,k), try to assign
            # a unique term that covers it, with some sharing.
            
            # Simple initialization: for the first n*m*p terms, use schoolbook
            # but only if r >= n*m*p. Otherwise, use random sparse init.
            
            if r >= n * m * p:
                # Can use schoolbook directly
                for idx in range(n * m * p):
                    i = idx // (m * p)
                    rem = idx % (m * p)
                    j = rem // p
                    k = rem % p
                    U[i * m + j, idx] = 1.0
                    V[j * p + k, idx] = 1.0
                    W[k * n + i, idx] = 1.0
                if _verify(U, V, W, n, m, p):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            else:
                # Random sparse initialization
                for c in range(r):
                    # Each term should cover a "block" of the tensor
                    # Randomly assign non-zero entries
                    n_U = rng.randint(1, 3)
                    n_V = rng.randint(1, 3)
                    n_W = rng.randint(1, 3)
                    
                    U_idx = rng.sample(range(n * m), n_U)
                    V_idx = rng.sample(range(m * p), n_V)
                    W_idx = rng.sample(range(p * n), n_W)
                    
                    for idx in U_idx:
                        U[idx, c] = rng.choice(values)
                    for idx in V_idx:
                        V[idx, c] = rng.choice(values)
                    for idx in W_idx:
                        W[idx, c] = rng.choice(values)
                
                if _verify(U, V, W, n, m, p):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None


def construct():
    n, m, p = DIMENSIONS
    start = time.time()

    # Always have a valid fallback: schoolbook
    fallback = _schoolbook(n, m, p)

    best = fallback
    best_rank = len(fallback["U"][0])

    # Try constructive low-rank search first (more structured)
    if _time_left() > 30:
        res = _try_constructive_low_rank(n, m, p)
        if res is not None:
            r = len(res["U"][0])
            if r < best_rank:
                best = res
                best_rank = r

    # Try strassen-style if time remains
    if _time_left() > 30 and best_rank > 21:
        res = _try_strassen_style(n, m, p, max_rank=min(best_rank, 40))
        if res is not None:
            r = len(res["U"][0])
            if r < best_rank:
                best = res
                best_rank = r

    # Final verification of the best result
    if not _verify(best["U"], best["V"], best["W"], n, m, p):
        # Fallback to schoolbook if best is invalid
        best = fallback

    return best
