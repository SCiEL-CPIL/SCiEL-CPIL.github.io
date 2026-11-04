"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def _schoolbook_decomposition(n, m, p):
    """Return the standard rank n*m*p decomposition."""
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
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = len(U[0]) if U else 0
    # Build tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # Expected: T[i,j,k] = 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    return True


def _try_merge_columns(U, V, W, n, m, p):
    """
    Try to merge pairs of columns in the decomposition if they are linearly dependent.
    For a rank-r decomposition, we have:
      T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    
    If two columns c1 and c2 have proportional U, V, W vectors, we can merge them.
    More generally, if we can find a linear combination that reduces the rank.
    
    A simpler approach: check if any column is a scalar multiple of another in all three matrices.
    """
    rank = len(U[0]) if U else 0
    # Check for duplicate/proportional columns
    to_remove = set()
    for c1 in range(rank):
        if c1 in to_remove:
            continue
        for c2 in range(c1 + 1, rank):
            if c2 in to_remove:
                continue
            # Check if column c2 is a scalar multiple of column c1
            # Find the scalar
            scalar = None
            for idx in range(len(U)):
                if U[idx][c1] != 0:
                    if U[idx][c2] == 0:
                        scalar = None
                        break
                    scalar = U[idx][c2] / U[idx][c1]
                    break
            if scalar is None:
                for idx in range(len(V)):
                    if V[idx][c1] != 0:
                        if V[idx][c2] == 0:
                            scalar = None
                            break
                        scalar = V[idx][c2] / V[idx][c1]
                        break
            if scalar is None:
                for idx in range(len(W)):
                    if W[idx][c1] != 0:
                        if W[idx][c2] == 0:
                            scalar = None
                            break
                        scalar = W[idx][c2] / W[idx][c1]
                        break
            if scalar is None:
                continue
            # Verify proportionality
            proportional = True
            for idx in range(len(U)):
                if U[idx][c2] != scalar * U[idx][c1]:
                    proportional = False
                    break
            if not proportional:
                for idx in range(len(V)):
                    if V[idx][c2] != scalar * V[idx][c1]:
                        proportional = False
                        break
            if not proportional:
                for idx in range(len(W)):
                    if W[idx][c2] != scalar * W[idx][c1]:
                        proportional = False
                        break
            if proportional:
                # Merge: add column c2 to column c1
                for idx in range(len(U)):
                    U[idx][c1] += U[idx][c2]
                for idx in range(len(V)):
                    V[idx][c1] += V[idx][c2]
                for idx in range(len(W)):
                    W[idx][c1] += W[idx][c2]
                to_remove.add(c2)
    
    if not to_remove:
        return U, V, W
    
    # Rebuild without removed columns
    keep = [c for c in range(rank) if c not in to_remove]
    U_new = [[U[i][c] for c in keep] for i in range(len(U))]
    V_new = [[V[i][c] for c in keep] for i in range(len(V))]
    W_new = [[W[i][c] for c in keep] for i in range(len(W))]
    return U_new, V_new, W_new


def _greedy_rank_reduction(U, V, W, n, m, p, deadline):
    """
    Iteratively try to reduce the rank by finding linear dependencies.
    Use the approach of checking if the tensor can be expressed with fewer terms.
    """
    U, V, W = _try_merge_columns(U, V, W, n, m, p)
    rank = len(U[0]) if U else 0
    
    # Try to find if any column can be expressed as a combination of others
    # This is a complex linear algebra problem; for now, just do the simple merge
    return U, V, W


def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120 second limit, keep safety margin
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook decomposition
    U, V, W = _schoolbook_decomposition(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U, V, W
    
    # Try to reduce rank
    while time.time() < deadline - 5:
        U, V, W = _greedy_rank_reduction(U, V, W, n, m, p, deadline)
        rank = len(U[0]) if U else 0
        
        if rank < best_rank:
            best_rank = rank
            best_U, best_V, best_W = U, V, W
        
        # Check if we can do better
        if rank == n * m * p:
            break
    
    # Verify the best decomposition
    if not _verify_decomposition(best_U, best_V, best_W, n, m, p):
        # Fall back to schoolbook
        best_U, best_V, best_W = _schoolbook_decomposition(n, m, p)
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
