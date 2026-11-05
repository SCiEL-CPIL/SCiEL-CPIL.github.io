"""Search for a lower-rank exact tensor decomposition of 3x5 * 5x7 matrix multiplication."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 5, 7]
N, M, P = DIMENSIONS
RANK_SCHOOLBOOK = N * M * P  # 105

# The matrix multiplication tensor T_{i,j,k} = delta (output index k*n+i, input U index i*m+j, V index j*p+k)
# We need: for all (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 if (i,j,k) is a valid triple, else 0
# Actually the tensor is: T[a, b, c] where a = i*m+j (U row), b = j*p+k (V row), c = k*n+i (W row)
# T[a,b,c] = 1 if there exist i,j,k with a=i*m+j, b=j*p+k, c=k*n+i, else 0.
# But note: a determines (i,j), b determines (j,k), c determines (k,i). They are consistent iff the j's and k's and i's match.
# So T[a,b,c] = 1 iff the i from a matches the i from c, the j from a matches the j from b, and the k from b matches the k from c.

def _build_tensor():
    """Build the matrix multiplication tensor as a sparse representation."""
    # T[a, b, c] where a in [0, N*M), b in [0, M*P), c in [0, P*N)
    # a = i*M + j, b = j*P + k, c = k*N + i
    # T[a,b,c] = 1 if consistent, else 0
    # We store as a dict for non-zero entries
    T = {}
    for i in range(N):
        for j in range(M):
            for k in range(P):
                a = i * M + j
                b = j * P + k
                c = k * N + i
                T[(a, b, c)] = 1
    return T

def _verify(U, V, W, rank):
    """Verify that U, V, W exactly decompose the tensor.
    U: (N*M, rank), V: (M*P, rank), W: (P*N, rank)
    For each (a,b,c) in the support: sum_r U[a,r]*V[b,r]*W[c,r] == 1
    For each (a,b,c) NOT in support but with a,b,c valid: sum_r U[a,r]*V[b,r]*W[c,r] == 0
    """
    T = _build_tensor()
    U_arr = np.array(U, dtype=object) if any(isinstance(x, Fraction) for row in U for x in row) else np.array(U, dtype=float)
    V_arr = np.array(V, dtype=object) if any(isinstance(x, Fraction) for row in V for x in row) else np.array(V, dtype=float)
    W_arr = np.array(W, dtype=object) if any(isinstance(x, Fraction) for row in W for x in row) else np.array(W, dtype=float)
    
    # Check all non-zero entries
    for (a, b, c) in T:
        val = 0
        for r in range(rank):
            val += U_arr[a, r] * V_arr[b, r] * W_arr[c, r]
        if isinstance(val, Fraction):
            if val != 1:
                return False
        else:
            if abs(val - 1) > 1e-9:
                return False
    
    # Check all zero entries
    for a in range(N * M):
        for b in range(M * P):
            for c in range(P * N):
                if (a, b, c) not in T:
                    val = 0
                    for r in range(rank):
                        val += U_arr[a, r] * V_arr[b, r] * W_arr[c, r]
                    if isinstance(val, Fraction):
                        if val != 0:
                            return False
                    else:
                        if abs(val) > 1e-9:
                            return False
    return True

def _schoolbook():
    """Return the schoolbook decomposition."""
    rank = RANK_SCHOOLBOOK
    U = [[0] * rank for _ in range(N * M)]
    V = [[0] * rank for _ in range(M * P)]
    W = [[0] * rank for _ in range(P * N)]
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j][col] = 1
                V[j * P + k][col] = 1
                W[k * N + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _try_rank(r, timeout=120):
    """Try to find a rank-r decomposition using random initialization and iterative refinement.
    Returns dict if found, None otherwise."""
    if r >= RANK_SCHOOLBOOK:
        return _schoolbook()
    
    deadline = time.time() + timeout
    T = _build_tensor()
    
    # Use random rational/integer initialization and try to satisfy constraints
    # This is a hard problem; for small ranks, we can try specific constructions.
    
    # Known bounds: for 3x5*5x7, the rank is at most 105 (schoolbook).
    # Lower bound: max(N*M, M*P, P*N) = max(15, 35, 21) = 35.
    # Actually, the tensor rank of matrix multiplication C_{n,m,p} is at least max(n*m, m*p, p*n) = 35.
    # Let's try to find a rank-35 or better decomposition.
    
    # For now, let's try a structured approach: use the fact that we can group terms.
    # A common technique: use bilinear forms that share indices.
    
    # Try rank = 35 (the lower bound)
    # This is very hard to find randomly. Let's try a few random attempts with small rank.
    
    best_result = None
    
    # Try a few random initializations for small ranks
    for attempt in range(50):
        if time.time() > deadline - 10:
            break
        
        # Random integer entries in {-1, 0, 1}
        rng = random.Random(attempt * 12345 + 67890)
        U = [[rng.choice([-1, 0, 1]) for _ in range(r)] for _ in range(N * M)]
        V = [[rng.choice([-1, 0, 1]) for _ in range(r)] for _ in range(M * P)]
        W = [[rng.choice([-1, 0, 1]) for _ in range(r)] for _ in range(P * N)]
        
        if _verify(U, V, W, r):
            return {"U": U, "V": V, "W": W}
    
    return None

def construct():
    """Find a rank-minimal exact tensor decomposition for 3x5 * 5x7 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Start with schoolbook
    best = _schoolbook()
    best_rank = RANK_SCHOOLBOOK
    
    # Try to improve from lower bound upward
    # Lower bound is max(N*M, M*P, P*N) = 35
    lower_bound = max(N * M, M * P, P * N)
    
    for r in range(lower_bound, best_rank):
        if time.time() > deadline:
            break
        remaining = deadline - time.time()
        if remaining < 5:
            break
        result = _try_rank(r, timeout=remaining - 2)
        if result is not None:
            best = result
            best_rank = r
            break
    
    return best
