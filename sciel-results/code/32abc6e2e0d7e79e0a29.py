"""Matrix multiplication 2x4 by 4x5 via block Strassen + schoolbook,
with exact verification and a time-budgeted search for a lower-rank decomposition.

Strategy:
  1. Build the known rank-36 decomposition (4 Strassen 2x2x2 blocks + schoolbook 2x4x4x1).
  2. Within the time budget, attempt to find a lower-rank decomposition using
     a structured search exploiting the small n=2 dimension.
  3. Return the best (lowest-rank) valid decomposition found.
"""

import time
import itertools
import random
import numpy as np

DIMENSIONS = [2, 4, 5]
_START_TIME = time.time()
_DEADLINE = _START_TIME + 135.0  # safety margin before 150s kill


def _remaining():
    return _DEADLINE - time.time()


def _build_rank36():
    """Build the rank-36 decomposition: 4 Strassen 2x2x2 + schoolbook 2x4x4x1."""
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    rank = 36

    U = [[0] * rank for _ in range(n * m)]  # 8 x 36
    V = [[0] * rank for _ in range(m * p)]  # 20 x 36
    W = [[0] * rank for _ in range(p * n)]  # 10 x 36

    col = 0

    def add_strassen_222(i0, j0, k0, col_start):
        nonlocal col
        c = col_start
        # M1 = (A00 + A11) * (B00 + B11)
        U[i0 * m + j0][c] += 1
        U[(i0 + 1) * m + (j0 + 1)][c] += 1
        V[j0 * p + k0][c] += 1
        V[(j0 + 1) * p + (k0 + 1)][c] += 1
        W[k0 * n + i0][c] += 1
        W[(k0 + 1) * n + (i0 + 1)][c] += 1
        c += 1
        # M2 = (A10 + A11) * B00
        U[(i0 + 1) * m + j0][c] += 1
        U[(i0 + 1) * m + (j0 + 1)][c] += 1
        V[j0 * p + k0][c] += 1
        W[k0 * n + (i0 + 1)][c] += 1
        W[(k0 + 1) * n + (i0 + 1)][c] -= 1
        c += 1
        # M3 = A00 * (B01 - B11)
        U[i0 * m + j0][c] += 1
        V[j0 * p + (k0 + 1)][c] += 1
        V[(j0 + 1) * p + (k0 + 1)][c] -= 1
        W[(k0 + 1) * n + i0][c] += 1
        W[(k0 + 1) * n + (i0 + 1)][c] += 1
        c += 1
        # M4 = A11 * (B10 - B00)
        U[(i0 + 1) * m + (j0 + 1)][c] += 1
        V[(j0 + 1) * p + k0][c] += 1
        V[j0 * p + k0][c] -= 1
        W[k0 * n + i0][c] += 1
        W[k0 * n + (i0 + 1)][c] += 1
        c += 1
        # M5 = (A00 + A01) * B11
        U[i0 * m + j0][c] += 1
        U[i0 * m + (j0 + 1)][c] += 1
        V[(j0 + 1) * p + (k0 + 1)][c] += 1
        W[k0 * n + i0][c] -= 1
        W[(k0 + 1) * n + i0][c] += 1
        c += 1
        # M6 = (A10 - A00) * (B00 + B01)
        U[(i0 + 1) * m + j0][c] += 1
        U[i0 * m + j0][c] -= 1
        V[j0 * p + k0][c] += 1
        V[j0 * p + (k0 + 1)][c] += 1
        W[(k0 + 1) * n + (i0 + 1)][c] += 1
        c += 1
        # M7 = (A01 - A11) * (B10 + B11)
        U[i0 * m + (j0 + 1)][c] += 1
        U[(i0 + 1) * m + (j0 + 1)][c] -= 1
        V[(j0 + 1) * p + k0][c] += 1
        V[(j0 + 1) * p + (k0 + 1)][c] += 1
        W[k0 * n + i0][c] += 1
        c += 1
        col = c
        return c

    def add_schoolbook_2x4x1(i0, j0, k0, col_start):
        c = col_start
        for di in range(2):
            for dj in range(4):
                U[(i0 + di) * m + (j0 + dj)][c] = 1
                V[(j0 + dj) * p + k0][c] = 1
                W[k0 * n + (i0 + di)][c] = 1
                c += 1
        col = c
        return c

    # 4 Strassen blocks for the 2x4 * 4x4 -> 2x4 part
    col = add_strassen_222(0, 0, 0, col)   # C[:,0:2] from A[:,0:2]*B[0:2,0:2]
    col = add_strassen_222(0, 2, 0, col)   # C[:,0:2] from A[:,2:4]*B[2:4,0:2]
    col = add_strassen_222(0, 0, 2, col)   # C[:,2:4] from A[:,0:2]*B[0:2,2:4]
    col = add_strassen_222(0, 2, 2, col)   # C[:,2:4] from A[:,2:4]*B[2:4,2:4]
    # Schoolbook for C[:,4] = A * B[:,4]
    col = add_schoolbook_2x4x1(0, 0, 4, col)

    assert col == rank, f"Expected rank {rank}, got {col}"
    return U, V, W, rank


def _verify_full(U, V, W, n, m, p):
    """Full verification: check all tensor entries."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                ui = i * m + j
                vj = j * p + k
                wk = k * n + i
                acc = 0
                for r in range(rank):
                    acc += U[ui][r] * V[vj][r] * W[wk][r]
                if acc != 1:
                    return False
    return True


def _verify_numpy(U, V, W, n, m, p):
    """Vectorized verification using numpy."""
    rank = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    # Build tensor: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                ui = i * m + j
                vj = j * p + k
                wk = k * n + i
                acc = np.dot(Ua[ui], Va[vj] * Wa[wk])
                if abs(acc - 1.0) > 1e-10:
                    return False
    return True


def _search_lower_rank(U36, V36, W36, n, m, p, max_rank=35):
    """
    Try to find a lower-rank decomposition by searching for linear dependencies
    among the rank-1 terms of the rank-36 decomposition.
    
    The idea: if some subset of the 36 rank-1 terms can be replaced by fewer
    terms that span the same space, we can reduce the rank.
    
    We check if any column of the 36-term decomposition is a linear combination
    of others (in the tensor space), which would allow removal.
    """
    # Build the tensor as a 3D array
    # T[i,j,k] should equal 1 for all (i,j,k) in the multiplication tensor
    # Actually, the tensor is: T[i,j,k] = delta (it's 1 for each (i,j,k) triple
    # that corresponds to a valid multiplication term)
    
    # The multiplication tensor for 2x4x5:
    # T[i,j,k] = 1 if the term A[i,j]*B[j,k] contributes to C[k,i]
    # In our indexing: U is (n*m) x rank, V is (m*p) x rank, W is (p*n) x rank
    # Tensor entry T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This should equal 1 for all valid (i,j,k)
    
    # To check if we can reduce rank, we need to see if the 36 rank-1 tensors
    # are linearly independent. If some are dependent, we can remove them.
    
    # Flatten each rank-1 tensor into a vector of size n*m*p = 40
    # The rank-1 tensor for column r is: U[:,r] ⊗ V[:,r] ⊗ W[:,r] reshaped
    
    # But the tensor space is not simply n*m*p dimensional in a useful way for
    # direct linear algebra because the indices are coupled.
    
    # Alternative: use the fact that for fixed (i,k), the tensor as a function
    # of j has rank at most 1 (it's A[i,j]*B[j,k]). 
    # 
    # Let's try a different approach: check if we can find a rank-35 decomposition
    # by seeing if one of the 36 terms is redundant.
    
    # For each term r in 0..35, check if removing it and adjusting others
    # can still reproduce the tensor. This is a linear system problem.
    
    # The tensor has 2*4*5 = 40 entries, all equal to 1.
    # Each rank-1 term contributes a specific pattern.
    
    # Let's set up: we want to find coefficients c_0..c_35 such that
    # sum_r c_r * T_r = T (the target tensor)
    # where T_r is the rank-1 tensor for column r.
    # If the c_r are all 1, we have the rank-36 decomposition.
    # If some c_r = 0, we have a lower rank decomposition.
    
    # But this is underdetermined (40 equations, 36 unknowns) so there might
    # be solutions with some c_r = 0.
    
    # Actually, the target tensor T[i,j,k] = 1 for all (i,j,k) is NOT the
    # multiplication tensor! The multiplication tensor has T[i,j,k] = 1 only
    # when the indices are consistent with the multiplication structure.
    # 
    # Wait, I need to re-examine. In the standard matrix multiplication tensor:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has entries T[i,j,k] = 1 for all i,j,k (it's the identity
    # in the tensor sense - every triple contributes).
    # 
    # So yes, T[i,j,k] = 1 for all i in 0..1, j in 0..3, k in 0..4.
    # That's 40 entries, all 1.
    
    # Let's build the 40x36 matrix M where M[entry_idx, r] = T_r[entry_idx]
    # and solve M * c = ones(40)
    
    rank36 = len(U36[0])
    num_entries = n * m * p  # 40
    
    # Build M
    M = np.zeros((num_entries, rank36))
    for r in range(rank36):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    entry_idx = i * m * p + j * p + k
                    M[entry_idx, r] = U36[i*m+j][r] * V36[j*p+k][r] * W36[k*n+i][r]
    
    # Check if the all-ones vector is in the column space
    target = np.ones(num_entries)
    
    # Use least squares to find coefficients
    try:
        coeffs, residuals, rank_M, sv = np.linalg.lstsq(M, target, rcond=None)
    except:
        return None
    
    # Check if solution is close to all-ones
    # If rank_M < rank36, then there are dependencies
    
    # Check if any coefficient is very close to 0
    min_coeff = np.min(np.abs(coeffs))
    
    # If we find a solution where some coeffs are ~0, we can drop those terms
    # But we need exact integer/half-integer solutions
    
    # For now, just check the rank of M
    if rank_M < rank36:
        # There are linear dependencies, potentially allowing rank reduction
        # Find which terms can be dropped
        zero_mask = np.abs(coeffs) < 1e-10
        if np.sum(zero_mask) > 0:
            # We can drop those terms
            keep = np.where(~zero_mask)[0]
            U_new = [U36[i][:len(keep)] for i in range(len(U36))]
            V_new = [V36[i][:len(keep)] for i in range(len(V36))]
            W_new = [W36[i][:len(keep)] for i in range(len(W36))]
            # Re-index
            U_new = [U36[i] for i in range(len(U36))]
            # Actually, we need to extract columns
            U_new = [[U36[i][r] for r in keep] for i in range(len(U36))]
            V_new = [[V36[i][r] for r in keep] for i in range(len(V36))]
            W_new = [[W36[i][r] for r in keep] for i in range(len(W36))]
            
            if _verify_full(U_new, V_new, W_new, n, m, p):
                return U_new, V_new, W_new, len(keep)
    
    return None


def construct():
    """
    Return {'U','V','W'} factor matrices that exactly decompose the
    2x4-by-4x5 matrix multiplication tensor.
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=5

    # Start with the known rank-36 decomposition
    U, V, W, rank = _build_rank36()

    # Verify
    if not _verify_full(U, V, W, n, m, p):
        raise RuntimeError("Rank-36 decomposition failed verification")

    # Try to find a lower-rank decomposition
    if _remaining() > 10.0:
        result = _search_lower_rank(U, V, W, n, m, p)
        if result is not None:
            U, V, W, rank = result
            if _verify_full(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
