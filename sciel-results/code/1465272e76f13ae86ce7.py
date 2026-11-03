"""
Program to find a rank-minimal exact tensor decomposition for matrix multiplication 2x5 by 5x6.
The output factor follows the official trace convention W[k*n+i,r].
"""

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

_START_TIME = time.time()
_TIME_LIMIT = 115.0  # Safety margin before 150s kill

def _elapsed():
    return time.time() - _START_TIME

def _time_left():
    return _TIME_LIMIT - _elapsed()

def _build_54_rank(n, m, p):
    """
    Build 2x5x6 decomposition using block Strassen. Rank 54.
    Split j into {0,1}, {2,3}, {4}. Split k into {0,1}, {2,3}, {4,5}.
    """
    rank = 54
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    # Pairs of j indices for Strassen blocks
    j_pairs = [(0, 1), (2, 3)]
    # Pairs of k indices for Strassen blocks
    k_pairs = [(0, 1), (2, 3), (4, 5)]
    
    for j0, j1 in j_pairs:
        for k0, k1 in k_pairs:
            # Standard Strassen 2x2x2 block
            # M1 = (A00 + A11)(B00 + B11)
            U[0*m + j0, col] += 1
            U[1*m + j1, col] += 1
            V[j0*p + k0, col] += 1
            V[j1*p + k1, col] += 1
            W[k0*n + 0, col] += 1
            W[k1*n + 1, col] += 1
            col += 1
            
            # M2 = (A10 + A11)B00
            U[1*m + j0, col] += 1
            U[1*m + j1, col] += 1
            V[j0*p + k0, col] += 1
            W[k0*n + 1, col] += 1
            W[k1*n + 1, col] -= 1
            col += 1
            
            # M3 = A00(B10 - B00)
            U[0*m + j0, col] += 1
            V[j0*p + k1, col] += 1
            V[j1*p + k1, col] -= 1
            W[k1*n + 0, col] += 1
            W[k1*n + 1, col] += 1
            col += 1
            
            # M4 = A11(B01 - B11)
            U[1*m + j1, col] += 1
            V[j1*p + k0, col] += 1
            V[j0*p + k0, col] -= 1
            W[k0*n + 0, col] += 1
            W[k0*n + 1, col] += 1
            col += 1
            
            # M5 = (A00 + A01)B11
            U[0*m + j0, col] += 1
            U[0*m + j1, col] += 1
            V[j1*p + k1, col] += 1
            W[k0*n + 0, col] -= 1
            W[k1*n + 0, col] += 1
            col += 1
            
            # M6 = (A10 - A00)(B00 + B01)
            U[1*m + j0, col] += 1
            U[0*m + j0, col] -= 1
            V[j0*p + k0, col] += 1
            V[j0*p + k1, col] += 1
            W[k1*n + 1, col] += 1
            col += 1
            
            # M7 = (A01 - A11)(B10 + B11)
            U[0*m + j1, col] += 1
            U[1*m + j1, col] -= 1
            V[j1*p + k0, col] += 1
            V[j1*p + k1, col] += 1
            W[k0*n + 0, col] += 1
            col += 1
    
    # Remaining j=4: standard products
    for k in range(p):
        U[0*m + 4, col] += 1
        V[4*p + k, col] += 1
        W[k*n + 0, col] += 1
        col += 1
        
        U[1*m + 4, col] += 1
        V[4*p + k, col] += 1
        W[k*n + 1, col] += 1
        col += 1
    
    assert col == rank, f"Expected col={rank}, got {col}"
    return U, V, W

def _verify_full(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that the decomposition is exact using vectorized numpy operations.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    """
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    
    # For each valid (i,j,k) triple, check sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = U[i*m+j, :]
                v_row = V[j*p+k, :]
                w_row = W[k*n+i, :]
                s = np.dot(u_row, v_row * w_row)
                if abs(s - 1.0) > tol:
                    return False
    
    # Check invalid entries: for all (x,y,z) where the indices don't form a valid triple
    # x = k*n+i, y = i2*m+j, z = j2*p+k2
    # Valid if i==i2, j==j2, k==k2
    for x in range(p*n):
        i = x % n
        k = x // n
        for y in range(n*m):
            j = y % m
            i2 = y // m
            for z in range(m*p):
                j2 = z // p
                k2 = z % p
                if i == i2 and j == j2 and k == k2:
                    continue
                s = np.dot(U[y, :], V[z, :] * W[x, :])
                if abs(s) > tol:
                    return False
    return True

def _verify_skip_term(U, V, W, n, m, p, skip_term, tol=1e-9):
    """
    Verify that the decomposition remains valid if term skip_term is set to zero.
    """
    rank = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = U[i*m+j, :]
                v_row = V[j*p+k, :]
                w_row = W[k*n+i, :]
                s_full = np.dot(u_row, v_row * w_row)
                s_skip = u_row[skip_term] * v_row[skip_term] * w_row[skip_term]
                if abs(s_full - s_skip - 1.0) > tol:
                    return False
    
    # Check invalid entries
    for x in range(p*n):
        i = x % n
        k = x // n
        for y in range(n*m):
            j = y % m
            i2 = y // m
            for z in range(m*p):
                j2 = z // p
                k2 = z % p
                if i == i2 and j == j2 and k == k2:
                    continue
                s_full = np.dot(U[y, :], V[z, :] * W[x, :])
                s_skip = U[y, skip_term] * V[z, skip_term] * W[x, skip_term]
                if abs(s_full - s_skip) > tol:
                    return False
    return True

def _try_eliminate_terms(U, V, W, n, m, p, max_eliminations=20):
    """
    Try to eliminate redundant terms one by one.
    Returns (U, V, W, new_rank).
    """
    current_U = U.copy()
    current_V = V.copy()
    current_W = W.copy()
    current_rank = U.shape[1]
    eliminated = []
    
    # Try eliminating each term, starting with those that might be redundant
    # Sort by norm (smaller norm terms are more likely to be redundant)
    norms = []
    for r in range(current_rank):
        norm_sq = np.sum(current_U[:, r]**2) + np.sum(current_V[:, r]**2) + np.sum(current_W[:, r]**2)
        norms.append(norm_sq)
    
    # Try terms in order of increasing norm
    order = np.argsort(norms)
    
    for r in order:
        if _time_left() < 30:
            break
        if len(eliminated) >= max_eliminations:
            break
        
        # Check if we can eliminate term r
        if _verify_skip_term(current_U, current_V, current_W, n, m, p, r):
            # Eliminate this term
            mask = np.ones(current_rank, dtype=bool)
            mask[r] = False
            current_U = current_U[:, mask]
            current_V = current_V[:, mask]
            current_W = current_W[:, mask]
            current_rank -= 1
            eliminated.append(r)
    
    return current_U, current_V, current_W, current_rank

def _try_pairwise_elimination(U, V, W, n, m, p):
    """
    Try to eliminate pairs of terms simultaneously. This can sometimes reduce rank
    further than single term elimination.
    """
    current_U = U.copy()
    current_V = V.copy()
    current_W = W.copy()
    current_rank = U.shape[1]
    
    # We need to check if terms r1 and r2 can be removed simultaneously.
    # This means:
    # 1. For all valid (i,j,k): sum_{r != r1, r2} ... = 1
    # 2. For all invalid (x,y,z): sum_{r != r1, r2} ... = 0
    
    # This is equivalent to checking if the tensor T' = T - T_{r1} - T_{r2} is still the multiplication tensor.
    # We can check this by verifying the full condition on the reduced matrices.
    
    # To be efficient, we only try pairs where individual terms couldn't be removed,
    # or we just try a subset of pairs.
    
    # Let's try all pairs if time permits.
    # Number of pairs is R*(R-1)/2. For R=54, that's 1431 pairs.
    # Each check is O(nmp + pnm * nm * mp) which is fast.
    
    # First, identify candidates. Terms with small norms are more likely to be part of a canceling pair.
    norms = []
    for r in range(current_rank):
        norm_sq = np.sum(current_U[:, r]**2) + np.sum(current_V[:, r]**2) + np.sum(current_W[:, r]**2)
        norms.append(norm_sq)
    
    order = np.argsort(norms)
    # Only consider top 20 smallest norms to limit search space
    candidates = order[:20]
    
    for idx1 in range(len(candidates)):
        if _time_left() < 20:
            break
        r1 = candidates[idx1]
        for idx2 in range(idx1 + 1, len(candidates)):
            r2 = candidates[idx2]
            
            # Create masks excluding r1 and r2
            mask = np.ones(current_rank, dtype=bool)
            mask[r1] = False
            mask[r2] = False
            
            U_test = current_U[:, mask]
            V_test = current_V[:, mask]
            W_test = current_W[:, mask]
            
            if _verify_full(U_test, V_test, W_test, n, m, p):
                # Success! Remove both
                current_U = U_test
                current_V = V_test
                current_W = W_test
                current_rank -= 2
                # Update candidates? No, just return the reduced set.
                # We should probably re-run the single elimination on the new set.
                # But for now, just return.
                return current_U, current_V, current_W, current_rank
    
    return current_U, current_V, current_W, current_rank

def _try_linear_combination_reduction(U, V, W, n, m, p):
    """
    Try to reduce rank by finding linear dependencies among the terms.
    This is a heuristic approach using the tensor structure.
    """
    current_U = U.copy()
    current_V = V.copy()
    current_W = W.copy()
    current_rank = U.shape[1]
    removed = []
    
    for r1 in range(current_rank):
        if r1 in removed:
            continue
        if _time_left() < 20:
            break
        for r2 in range(r1+1, current_rank):
            if r2 in removed:
                continue
            
            # Check if term r2 is proportional to term r1 in all three matrices
            u1 = current_U[:, r1]
            u2 = current_U[:, r2]
            v1 = current_V[:, r1]
            v2 = current_V[:, r2]
            w1 = current_W[:, r1]
            w2 = current_W[:, r2]
            
            if np.allclose(u1, 0) and np.allclose(u2, 0):
                continue
            
            mask = np.abs(u2) > 1e-10
            if not np.any(mask):
                continue
            
            alpha = u1[mask] / u2[mask]
            if not np.allclose(alpha, alpha[0], atol=1e-8):
                continue
            alpha = alpha[0]
            
            if np.allclose(u1, alpha * u2, atol=1e-8) and \
               np.allclose(v1, alpha * v2, atol=1e-8) and \
               np.allclose(w1, alpha * w2, atol=1e-8):
                # Term r1 is proportional to term r2
                # We can't simply remove one; we need to check if the combination
                # with other terms still works. For now, skip this complex case.
                pass
    
    return current_U, current_V, current_W, current_rank

def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    """
    n, m, p = DIMENSIONS
    
    # Start with the known 54-rank decomposition
    U, V, W = _build_54_rank(n, m, p)
    best_U, best_V, best_W = U, V, W
    best_rank = 54
    
    # Verify the initial decomposition
    if not _verify_full(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        best_U, best_V, best_W = U, V, W
        best_rank = rank
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    
    # Try to eliminate redundant terms (single)
    if _time_left() > 60:
        U_reduced, V_reduced, W_reduced, new_rank = _try_eliminate_terms(U, V, W, n, m, p)
        if new_rank < best_rank:
            # Verify the reduced decomposition
            if _verify_full(U_reduced, V_reduced, W_reduced, n, m, p):
                best_U, best_V, best_W = U_reduced, V_reduced, W_reduced
                best_rank = new_rank
                U, V, W = U_reduced, V_reduced, W_reduced
    
    # Try to eliminate pairs of terms
    if _time_left() > 30:
        U_reduced, V_reduced, W_reduced, new_rank = _try_pairwise_elimination(U, V, W, n, m, p)
        if new_rank < best_rank:
            # Verify the reduced decomposition
            if _verify_full(U_reduced, V_reduced, W_reduced, n, m, p):
                best_U, best_V, best_W = U_reduced, V_reduced, W_reduced
                best_rank = new_rank
    
    # Try linear combination reduction
    if _time_left() > 15:
        U_reduced, V_reduced, W_reduced, new_rank = _try_linear_combination_reduction(U, V, W, n, m, p)
        if new_rank < best_rank:
            # Verify the reduced decomposition
            if _verify_full(U_reduced, V_reduced, W_reduced, n, m, p):
                best_U, best_V, best_W = U_reduced, V_reduced, W_reduced
                best_rank = new_rank

    # Return the best result found
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
