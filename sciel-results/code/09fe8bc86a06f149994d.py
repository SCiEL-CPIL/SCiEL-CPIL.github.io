"""Matrix multiplication 2x3x10 tensor decomposition search."""
import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [2, 3, 10]

def _build_tensor():
    """Build the 60x60x100 multiplication tensor T[a,b,c] = 1 if a=ip+j, b=jp+k, c=kn+i for some i,j,k."""
    n, m, p = DIMENSIONS
    T = np.zeros((n*m, m*p, p*n), dtype=np.int8)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    return T

def _verify(U, V, W, T):
    """Check if sum_r U[:,r] V[:,r]^T W[:,r]^T == T exactly."""
    # U: (A, r), V: (B, r), W: (C, r)
    # Reconstruction: R[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # Use einsum for efficiency
    R = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
    return np.array_equal(R, T)

def _random_factors(A, B, C, r, rng):
    """Generate random U, V, W with integer/half-integer entries in [-1, 1]."""
    # Use values from {0, 0.5, 1, -0.5, -1} to keep it simple and exact
    vals = [0, 0.5, 1, -0.5, -1]
    U = np.zeros((A, r))
    V = np.zeros((B, r))
    W = np.zeros((C, r))
    for col in range(r):
        # Randomly choose non-zero entries
        for i in range(A):
            if rng.random() < 0.3:
                U[i, col] = rng.choice(vals)
        for j in range(B):
            if rng.random() < 0.3:
                V[j, col] = rng.choice(vals)
        for k in range(C):
            if rng.random() < 0.3:
                W[k, col] = rng.choice(vals)
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    A, B, C = n * m, m * p, p * n
    T = _build_tensor()

    # Start with schoolbook rank
    best_rank = A * B * C
    best_U = np.zeros((A, best_rank))
    best_V = np.zeros((B, best_rank))
    best_W = np.zeros((C, best_rank))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m * p + j * p + k
                a = i * m + j
                b = j * p + k
                c = k * n + i
                best_U[a, r] = 1
                best_V[b, r] = 1
                best_W[c, r] = 1

    # Try to find lower ranks via random search
    # We know Strassen gives 55. Let's try to beat that or at least match it.
    # Search ranks from 50 up to best_rank - 1
    rng = np.random.default_rng(42)
    random.seed(42)

    # We'll try a few random attempts for each rank
    # Since exact verification is expensive, we limit the number of trials per rank
    for rank in range(50, best_rank):
        if time.time() > deadline - 5:
            break
        
        # Number of trials depends on time budget
        # Let's try a small number of trials per rank
        trials = 5
        found = False
        for t in range(trials):
            if time.time() > deadline - 5:
                break
            U, V, W = _random_factors(A, B, C, rank, rng)
            if _verify(U, V, W, T):
                best_rank = rank
                best_U, best_V, best_W = U, V, W
                found = True
                break
        if found:
            # If we found a rank, we might try lower ones, but random search is unlikely to find much lower than 55 easily
            # Continue loop to check lower ranks? No, loop goes 50, 51... so if we found 55, we already checked 50-54.
            # Actually the loop is `for rank in range(50, best_rank)`. If best_rank was 60, it checks 50..59.
            # If we find rank 55, best_rank becomes 55. The loop condition `range(50, 55)` is already exhausted?
            # No, `range` is evaluated once at the start of the loop in Python 3? No, `range(50, best_rank)` uses the current value of best_rank at loop init?
            # In Python, `range` object is created once. So if I change best_rank inside, the loop doesn't shrink.
            # I should restructure or just break if I find a good one?
            # Let's just break if we find rank <= 55, as going lower is very hard with this random method.
            if rank <= 55:
                break

    # Convert to lists of lists for return
    def to_list(mat):
        return [[float(x) for x in row] for row in mat]

    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
