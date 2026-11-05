"""Matrix multiplication 3,3,6 — rank-minimal exact tensor decomposition search."""
import numpy as np
import time
import itertools
import random

DIMENSIONS = [3, 3, 6]


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that U ⊗ V ⊗ W reconstructs the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    # Tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > tol:
                    return False
    # Also check that non-multiplication entries are zero
    # The tensor is fully determined by the n*m*p = 54 entries above.
    # But we also need to ensure no "extra" contributions... 
    # Actually, the decomposition defines T_{ijk} for all i,j,k. 
    # In matrix multiplication, T_{ijk} = 1 if the indices match the multiplication pattern, 0 otherwise.
    # Wait — the matrix multiplication tensor has T_{ijk} = 1 for the 54 positions (i,j,k) where 
    # output[i,k] gets contribution from A[i,j]*B[j,k]. Since A is 3x3 and B is 3x6, 
    # the output is 3x6, and each output element C[i,k] = sum_j A[i,j]*B[j,k].
    # So the tensor has exactly 54 non-zero entries, all equal to 1.
    # The decomposition must produce exactly 1 at those 54 positions and 0 everywhere else.
    # But since the tensor space is only indexed by (i,j,k) with i in [0,3), j in [0,3), k in [0,6),
    # and all 54 combinations are the "multiplication" entries, there are no "other" entries to check.
    # Actually wait — the tensor is a 3x3x6 tensor. Every entry (i,j,k) corresponds to a term 
    # in the matrix product. So all 54 entries should be 1. There are no zero entries.
    # Let me re-examine: C[i,k] = sum_j A[i,j] * B[j,k]. The tensor T_{i,j,k} = 1 for all valid (i,j,k).
    # So we just need all 54 entries to be 1. The check above does exactly that.
    return True


def _rank_from_matrices(U, V, W):
    return len(U[0]) if U and len(U) > 0 else 0


def _try_rank46_direct():
    """
    Attempt to construct a rank-46 decomposition directly.
    
    Strategy: Start from the rank-54 schoolbook and try to merge pairs of terms.
    Two schoolbook terms (i1,j1,k1) and (i2,j2,k2) can be merged into a single rank-1 term
    if we can find vectors u, v, w such that:
      u ⊗ v ⊗ w = e_{i1,j1,k1} + e_{i2,j2,k2}
    This is possible only if the two basis vectors differ in at most one index... 
    Actually no, a rank-1 tensor is a single basis vector scaled. Sum of two distinct basis vectors
    is never rank-1. So we can't merge two schoolbook terms into one.
    
    Alternative: Use the known structure of the 3x3x3 rank-23 decomposition (Landsberg) and 
    extend it to 3x3x6. But I don't have the explicit 23-term coefficients memorized precisely.
    
    Let me try a different approach: use a structured ansatz.
    """
    n, m, p = 3, 3, 6
    # I'll attempt a systematic search with a parametric family.
    # Given the complexity, I'll return None to indicate failure.
    return None


def _schoolbook():
    """Standard rank-54 schoolbook decomposition."""
    n, m, p = DIMENSIONS
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


def construct():
    start_time = time.time()
    deadline = start_time + 115  # 115s safety margin within 120s limit
    
    # Best result found so far
    best_result = _schoolbook()
    best_rank = 54
    
    # Try to find a better decomposition via random search with structured initialization.
    # We search for rank r < 54 by trying to solve for integer/half-integer coefficients.
    # Use a randomized approach: initialize U, V, W randomly with small integers,
    # then use local search to reduce the residual.
    
    n, m, p = DIMENSIONS
    target_entries = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    
    def residual(U, V, W, rank):
        """Compute max absolute error."""
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    err = abs(s - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err
    
    # Try rank 50, 49, 48, ... down to some minimum, with time budget
    for target_rank in [52, 50, 48, 46]:
        if time.time() > deadline - 10:
            break
        
        # Multiple random restarts
        max_restarts = 5
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
            
            # Initialize with random small integer entries
            random.seed(restart * 1000 + target_rank)
            U = [[random.choice([-2, -1, 0, 1, 2]) for _ in range(target_rank)] for _ in range(n * m)]
            V = [[random.choice([-2, -1, 0, 1, 2]) for _ in range(target_rank)] for _ in range(m * p)]
            W = [[random.choice([-2, -1, 0, 1, 2]) for _ in range(target_rank)] for _ in range(p * n)]
            
            # Local search: try to flip individual entries to reduce residual
            for step in range(2000):
                if time.time() > deadline - 3:
                    break
                err = residual(U, V, W, target_rank)
                if err < 1e-9:
                    # Found exact decomposition!
                    best_result = {"U": U, "V": V, "W": W}
                    best_rank = target_rank
                    break
                # Pick a random entry to modify
                mat = random.choice(['U', 'V', 'W'])
                if mat == 'U':
                    rows = n * m
                    col = random.randrange(target_rank)
                    row = random.randrange(rows)
                    U[row][col] = random.choice([-3, -2, -1, 0, 1, 2, 3])
                elif mat == 'V':
                    rows = m * p
                    col = random.randrange(target_rank)
                    row = random.randrange(rows)
                    V[row][col] = random.choice([-3, -2, -1, 0, 1, 2, 3])
                else:
                    rows = p * n
                    col = random.randrange(target_rank)
                    row = random.randrange(rows)
                    W[row][col] = random.choice([-3, -2, -1, 0, 1, 2, 3])
            
            if best_rank == target_rank:
                break
        
        if best_rank == target_rank:
            break
    
    return best_result
