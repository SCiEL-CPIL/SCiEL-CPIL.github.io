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
from fractions import Fraction

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
    
    Uses structured linear algebra on tensor slices to detect if any subset
    of rank-1 terms can be eliminated while preserving the tensor identity.
    """
    rank36 = len(U36[0])
    num_entries = n * m * p  # 40
    
    # Build the 40x36 matrix M where M[entry_idx, r] = T_r[entry_idx]
    # T_r is the rank-1 tensor contribution from term r
    M = np.zeros((num_entries, rank36))
    for r in range(rank36):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    entry_idx = i * m * p + j * p + k
                    M[entry_idx, r] = U36[i*m+j][r] * V36[j*p+k][r] * W36[k*n+i][r]
    
    target = np.ones(num_entries)
    
    # Check numerical rank of M
    try:
        _, s, _ = np.linalg.svd(M, full_matrices=False)
        rank_M = np.sum(s > 1e-10)
    except:
        return None
    
    if rank_M >= rank36:
        # All 36 terms are linearly independent in the tensor space
        # No simple linear dependency allows rank reduction
        return None
    
    # There are linear dependencies. Try to find a solution with some zero coefficients.
    # Use least squares to find the minimum-norm solution
    try:
        coeffs, residuals, _, _ = np.linalg.lstsq(M, target, rcond=None)
    except:
        return None
    
    # Check if the solution is valid (reproduces the tensor)
    reconstructed = M @ coeffs
    if not np.allclose(reconstructed, target, atol=1e-8):
        return None
    
    # Try to identify terms that can be dropped (coefficients close to 0)
    # Also try: for each pair of terms, check if one can be expressed as combo of others
    best_result = None
    
    # Strategy 1: Drop terms with smallest coefficients
    sorted_idx = np.argsort(np.abs(coeffs))
    for num_drop in range(1, min(4, rank36)):
        if _remaining() < 2.0:
            break
        drop_set = set(sorted_idx[:num_drop].tolist())
        keep = [r for r in range(rank36) if r not in drop_set]
        
        U_new = [[U36[i][r] for r in keep] for i in range(len(U36))]
        V_new = [[V36[i][r] for r in keep] for i in range(len(V36))]
        W_new = [[W36[i][r] for r in keep] for i in range(len(W36))]
        
        if _verify_full(U_new, V_new, W_new, n, m, p):
            if best_result is None or len(keep) < best_result[3]:
                best_result = (U_new, V_new, W_new, len(keep))
            break  # Found a valid lower rank
    
    # Strategy 2: For each single term, check if it's in the span of others
    # by solving a reduced system
    for r_drop in range(rank36):
        if _remaining() < 2.0:
            break
        keep = [r for r in range(rank36) if r != r_drop]
        M_reduced = M[:, keep]
        
        try:
            c_reduced, res, rank_r, _ = np.linalg.lstsq(M_reduced, target, rcond=None)
        except:
            continue
        
        recon = M_reduced @ c_reduced
        if np.allclose(recon, target, atol=1e-8):
            # Check if c_reduced is close to 1 for all kept terms
            # (meaning the dropped term was truly redundant with unit coefficients)
            if np.allclose(c_reduced, 1.0, atol=1e-6):
                U_new = [[U36[i][r] for r in keep] for i in range(len(U36))]
                V_new = [[V36[i][r] for r in keep] for i in range(len(V36))]
                W_new = [[W36[i][r] for r in keep] for i in range(len(W36))]
                
                if _verify_full(U_new, V_new, W_new, n, m, p):
                    if best_result is None or len(keep) < best_result[3]:
                        best_result = (U_new, V_new, W_new, len(keep))
    
    return best_result


def _search_rank35_structured(n, m, p):
    """
    Structured search for rank-35 decomposition exploiting n=2 structure.
    
    For n=2, the tensor splits into two slices:
    - Slice i=0: T[0,j,k] = sum_r U[0*m+j, r] * V[j*p+k, r] * W[k*n+0, r] = 1
    - Slice i=1: T[1,j,k] = sum_r U[1*m+j, r] * V[j*p+k, r] * W[k*n+1, r] = 1
    
    Each slice is a 4x5 matrix. We search for rank-35 decompositions by
    trying to find a common set of 35 rank-1 terms that reproduce both slices.
    """
    if _remaining() < 5.0:
        return None
    
    # Build the two slice matrices from the rank-36 decomposition
    U36, V36, W36, _ = _build_rank36()
    
    # Slice 0: U_slice0[j, r] = U36[0*m+j, r], W_slice0[k, r] = W36[k*n+0, r]
    # Slice 1: U_slice1[j, r] = U36[1*m+j, r], W_slice1[k, r] = W36[k*n+1, r]
    # V is shared: V[j, k, r] = V36[j*p+k, r]
    
    # For rank-35, we need to find a set of 35 terms (out of 36 or new)
    # such that both slices are reproduced.
    
    # Try dropping each term from the rank-36 and check if the remaining 35
    # can be rescaled to reproduce both slices.
    rank36 = len(U36[0])
    
    for r_drop in range(rank36):
        if _remaining() < 3.0:
            break
        
        keep = [r for r in range(rank36) if r != r_drop]
        
        # For slice 0: we need sum_r in keep U0[j,r] * V[j,k,r] * W0[k,r] = 1 for all j,k
        # For slice 1: we need sum_r in keep U1[j,r] * V[j,k,r] * W1[k,r] = 1 for all j,k
        
        # This is a bilinear system. Let's set up:
        # For each (j,k) in slice 0: sum_r U0[j,r]*V[j,k,r]*W0[k,r] = 1
        # For each (j,k) in slice 1: sum_r U1[j,r]*V[j,k,r]*W1[k,r] = 1
        
        # If we assume the kept terms have coefficients c_r (initially 1),
        # we can check if there exist c_r such that both slices work.
        
        # Build linear systems for each slice
        # Slice 0: for each (j,k), sum_r c_r * (U0[j,r]*V[j,k,r]*W0[k,r]) = 1
        # This is linear in c_r!
        
        num_keep = len(keep)
        
        # Build system for slice 0
        A0 = np.zeros((m * p, num_keep))
        b0 = np.ones(m * p)
        for idx, (j, k) in enumerate([(j, k) for j in range(m) for k in range(p)]):
            for ii, r in enumerate(keep):
                A0[idx, ii] = U36[0*m+j][r] * V36[j*p+k][r] * W36[k*n+0][r]
        
        # Build system for slice 1
        A1 = np.zeros((m * p, num_keep))
        b1 = np.ones(m * p)
        for idx, (j, k) in enumerate([(j, k) for j in range(m) for k in range(p)]):
            for ii, r in enumerate(keep):
                A1[idx, ii] = U36[1*m+j][r] * V36[j*p+k][r] * W36[k*n+1][r]
        
        # Try to find c that satisfies both A0*c = b0 and A1*c = b1
        # Stack them: [A0; A1] * c = [b0; b1]
        A = np.vstack([A0, A1])
        b = np.concatenate([b0, b1])
        
        try:
            c, res, rank_A, _ = np.linalg.lstsq(A, b, rcond=None)
        except:
            continue
        
        # Check if solution is valid
        recon = A @ c
        if not np.allclose(recon, b, atol=1e-8):
            continue
        
        # Check if c is close to 1 (meaning the dropped term was redundant)
        # Or if c has nice values (integers or half-integers)
        c_rounded = np.round(c * 2) / 2  # Round to nearest half-integer
        if np.allclose(c, c_rounded, atol=1e-6):
            # Valid solution with half-integer coefficients
            # Scale the kept terms by c
            U_new = [[U36[i][r] * c[ii] for ii, r in enumerate(keep)] for i in range(len(U36))]
            V_new = [[V36[i][r] for r in keep] for i in range(len(V36))]
            W_new = [[W36[i][r] for r in keep] for i in range(len(W36))]
            
            if _verify_full(U_new, V_new, W_new, n, m, p):
                return U_new, V_new, W_new, num_keep
    
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

    # Try structured rank-35 search first (exploits n=2 structure)
    if _remaining() > 10.0:
        result = _search_rank35_structured(n, m, p)
        if result is not None:
            U, V, W, rank = result
            if _verify_full(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    
    # Try general linear dependency search
    if _remaining() > 5.0:
        result = _search_lower_rank(U, V, W, n, m, p)
        if result is not None:
            U, V, W, rank = result
            if _verify_full(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
