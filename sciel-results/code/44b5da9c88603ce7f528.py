"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

import time
import random

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Return the standard schoolbook rank n*m*p decomposition."""
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

def _validate(n, m, p, U, V, W, tolerance=1e-9):
    """
    Check whether (U, V, W) exactly decomposes the matrix multiplication tensor.
    Tensor entry T[i, j, k] = 1 if the product element (i,k) depends on A[i,j]*B[j,k].
    The decomposition computes: for each column c, contribution U[im,j, c] * V[jp,k, c] * W[kn,i, c]
    Actually the convention is:
      (A*B)[i,k] = sum_c  (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k])
    Wait, the standard trace convention:
      C[i,k] = sum_{i',j',k'} T[i,j,k] A[i,j] B[j,k]
    For the decomposition to be valid, we need:
      T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    for all i in [0,n), j in [0,m), k in [0,p).
    """
    rank = len(U[0])
    # Build tensor T[i,j,k] = 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > tolerance:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120  # 120 second limit, leave margin
    best = _schoolbook(n, m, p)
    best_rank = n * m * p

    # Try to find a lower rank decomposition via random search
    # We'll try ranks from best_rank-1 down to some minimum
    # For each rank, try a few random initializations
    for rank in range(best_rank - 1, 0, -1):
        if time.time() > deadline - 10:  # 10 seconds left
            break
        # Try a few random attempts at this rank
        max_attempts = 5 if rank > 50 else 10
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            # Random initialization
            U = [[random.uniform(-1, 1) for _ in range(rank)] for _ in range(n * m)]
            V = [[random.uniform(-1, 1) for _ in range(rank)] for _ in range(m * p)]
            W = [[random.uniform(-1, 1) for _ in range(rank)] for _ in range(p * n)]
            if _validate(n, m, p, U, V, W, tolerance=1e-6):
                best = {"U": U, "V": V, "W": W}
                best_rank = rank
                break  # Found a valid one at this rank, go to lower rank
        if best_rank < best_rank:  # This won't happen, but for clarity
            pass

    return best
# EVOLVE-BLOCK-END
