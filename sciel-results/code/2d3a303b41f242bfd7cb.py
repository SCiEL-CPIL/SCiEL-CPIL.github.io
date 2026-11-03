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
    """
    rank = 54
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    j_pairs = [(0, 1), (2, 3)]
    k_pairs = [(0, 1), (2, 3), (4, 5)]
    
    for j0, j1 in j_pairs:
        for k0, k1 in k_pairs:
            # M1
            U[0*m + j0, col] += 1
            U[1*m + j1, col] += 1
            V[j0*p + k0, col] += 1
            V[j1*p + k1, col] += 1
            W[k0*n + 0, col] += 1
            W[k1*n + 1, col] += 1
            col += 1
            # M2
            U[1*m + j0, col] += 1
            U[1*m + j1, col] += 1
            V[j0*p + k0, col] += 1
            W[k0*n + 1, col] += 1
            W[k1*n + 1, col] -= 1
            col += 1
            # M3
            U[0*m + j0, col] += 1
            V[j0*p + k1, col] += 1
            V[j1*p + k1, col] -= 1
            W[k1*n + 0, col] += 1
            W[k1*n + 1, col] += 1
            col += 1
            # M4
            U[1*m + j1, col] += 1
            V[j1*p + k0, col] += 1
            V[j0*p + k0, col] -= 1
            W[k0*n + 0, col] += 1
            W[k0*n + 1, col] += 1
            col += 1
            # M5
            U[0*m + j0, col] += 1
            U[0*m + j1, col] += 1
            V[j1*p + k1, col] += 1
            W[k0*n + 0, col] -= 1
            W[k1*n + 0, col] += 1
            col += 1
            # M6
            U[1*m + j0, col] += 1
            U[0*m + j0, col] -= 1
            V[j0*p + k0, col] += 1
            V[j0*p + k1, col] += 1
            W[k1*n + 1, col] += 1
            col += 1
            # M7
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

def _verify_full_fast(U, V, W, n, m, p, tol=1e-9):
    """
    Fast vectorized verification.
    """
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
        
    V_T = V.T  # (R, mp)
    
    for x in range(p * n):
        w = W[x, :]  # (R,)
        S = (U * w[None, :]) @ V_T
        
        i = x % n
        k = x // n
        
        # Check valid entries
        for j in range(m):
            y = i * m + j
            z = j * p + k
            if abs(S[y, z] - 1.0) > tol:
                return False
        
        # Check invalid entries
        valid_mask = np.zeros((n*m, m*p), dtype=bool)
        for j in range(m):
            valid_mask[i*m+j, j*p+k] = True
        
        if np.any(np.abs(S[~valid_mask]) > tol):
            return False
            
    return True

def _try_eliminate_terms_fast(U, V, W, n, m, p, max_eliminations=20):
    """
    Try to eliminate redundant terms one by one using fast verification.
    """
    current_U = U.copy()
    current_V = V.copy()
    current_W = W.copy()
    current_rank = U.shape[1]
    
    norms = []
    for r in range(current_rank):
        norm_sq = np.sum(current_U[:, r]**2) + np.sum(current_V[:, r]**2) + np.sum(current_W[:, r]**2)
        norms.append(norm_sq)
        
    order = np.argsort(norms)
    
    for r in order:
        if _time_left() < 30:
            break
        if current_rank <= 1:
            break
            
        mask = np.ones(current_rank, dtype=bool)
        mask[r] = False
        
        U_test = current_U[:, mask]
        V_test = current_V[:, mask]
        W_test = current_W[:, mask]
        
        if _verify_full_fast(U_test, V_test, W_test, n, m, p):
            current_U = U_test
            current_V = V_test
            current_W = W_test
            current_rank -= 1
            
    return current_U, current_V, current_W, current_rank

def _try_pairwise_elimination_fast(U, V, W, n, m, p):
    """
    Try to eliminate pairs of terms simultaneously.
    """
    current_U = U.copy()
    current_V = V.copy()
    current_W = W.copy()
    current_rank = U.shape[1]
    
    norms = []
    for r in range(current_rank):
        norm_sq = np.sum(current_U[:, r]**2) + np.sum(current_V[:, r]**2) + np.sum(current_W[:, r]**2)
        norms.append(norm_sq)
        
    order = np.argsort(norms)
    candidates = order[:30]  # Consider more candidates
    
    for idx1 in range(len(candidates)):
        if _time_left() < 20:
            break
        r1 = candidates[idx1]
        for idx2 in range(idx1 + 1, len(candidates)):
            r2 = candidates[idx2]
            
            mask = np.ones(current_rank, dtype=bool)
            mask[r1] = False
            mask[r2] = False
            
            U_test = current_U[:, mask]
            V_test = current_V[:, mask]
            W_test = current_W[:, mask]
            
            if _verify_full_fast(U_test, V_test, W_test, n, m, p):
                return U_test, V_test, W_test, current_rank - 2
    
    return current_U, current_V, current_W, current_rank

def _try_triple_elimination_fast(U, V, W, n, m, p):
    """
    Try to eliminate triples of terms simultaneously.
    """
    current_U = U.copy()
    current_V = V.copy()
    current_W = W.copy()
    current_rank = U.shape[1]
    
    norms = []
    for r in range(current_rank):
        norm_sq = np.sum(current_U[:, r]**2) + np.sum(current_V[:, r]**2) + np.sum(current_W[:, r]**2)
        norms.append(norm_sq)
        
    order = np.argsort(norms)
    candidates = order[:20]  # Fewer candidates for triples
    
    for idx1 in range(len(candidates)):
        if _time_left() < 15:
            break
        r1 = candidates[idx1]
        for idx2 in range(idx1 + 1, len(candidates)):
            r2 = candidates[idx2]
            for idx3 in range(idx2 + 1, len(candidates)):
                r3 = candidates[idx3]
                
                mask = np.ones(current_rank, dtype=bool)
                mask[r1] = False
                mask[r2] = False
                mask[r3] = False
                
                U_test = current_U[:, mask]
                V_test = current_V[:, mask]
                W_test = current_W[:, mask]
                
                if _verify_full_fast(U_test, V_test, W_test, n, m, p):
                    return U_test, V_test, W_test, current_rank - 3
    
    return current_U, current_V, current_W, current_rank

def _try_linear_combination_reduction(U, V, W, n, m, p):
    """
    Try to find a linear combination of terms that can be removed.
    This is a more sophisticated approach that looks for dependencies.
    """
    if _time_left() < 40:
        return None, None, None, U.shape[1]
    
    current_U = U.copy()
    current_V = V.copy()
    current_W = W.copy()
    current_rank = U.shape[1]
    
    # Try to find if there's a subset of terms that sums to zero
    # This is a linear algebra problem on the tensor space
    
    # For each pair of terms, check if their difference can be expressed
    # as a linear combination of other terms
    
    # This is complex, so let's try a simpler heuristic:
    # Look for terms that have the same U and V vectors but different W vectors
    # If so, we might be able to combine them
    
    # Group terms by their (U, V) patterns
    from collections import defaultdict
    
    groups = defaultdict(list)
    for r in range(current_rank):
        u_key = tuple(current_U[:, r].tolist())
        v_key = tuple(current_V[:, r].tolist())
        groups[(u_key, v_key)].append(r)
    
    for (u_key, v_key), indices in groups.items():
        if len(indices) > 1:
            # Multiple terms with same U and V - check if W vectors are linearly dependent
            W_sub = current_W[indices, :]
            # If W vectors are linearly dependent, we can eliminate one
            rank_w = np.linalg.matrix_rank(W_sub)
            if rank_w < len(indices):
                # We can eliminate (len(indices) - rank_w) terms
                # Keep only rank_w of them
                # Find which ones to keep using QR decomposition
                Q, R = np.linalg.qr(W_sub.T)
                # The first rank_w columns of Q span the space
                keep_indices = indices[:rank_w]
                remove_indices = indices[rank_w:]
                
                mask = np.ones(current_rank, dtype=bool)
                for ri in remove_indices:
                    mask[ri] = False
                    
                U_test = current_U[:, mask]
                V_test = current_V[:, mask]
                W_test = current_W[:, mask]
                
                if _verify_full_fast(U_test, V_test, W_test, n, m, p):
                    return U_test, V_test, W_test, current_rank - len(remove_indices)
    
    return current_U, current_V, current_W, current_rank

def construct():
    n, m, p = DIMENSIONS
    
    U, V, W = _build_54_rank(n, m, p)
    best_U, best_V, best_W = U, V, W
    best_rank = 54
    
    if not _verify_full_fast(U, V, W, n, m, p):
        # Fallback
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
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Single elimination
    if _time_left() > 50:
        U_reduced, V_reduced, W_reduced, new_rank = _try_eliminate_terms_fast(U, V, W, n, m, p)
        if new_rank < best_rank:
            best_U, best_V, best_W = U_reduced, V_reduced, W_reduced
            best_rank = new_rank
            U, V, W = U_reduced, V_reduced, W_reduced
    
    # Pairwise elimination
    if _time_left() > 25:
        U_reduced, V_reduced, W_reduced, new_rank = _try_pairwise_elimination_fast(U, V, W, n, m, p)
        if new_rank < best_rank:
            best_U, best_V, best_W = U_reduced, V_reduced, W_reduced
            best_rank = new_rank
            U, V, W = U_reduced, V_reduced, W_reduced
    
    # Triple elimination
    if _time_left() > 15:
        U_reduced, V_reduced, W_reduced, new_rank = _try_triple_elimination_fast(U, V, W, n, m, p)
        if new_rank < best_rank:
            best_U, best_V, best_W = U_reduced, V_reduced, W_reduced
            best_rank = new_rank
    
    # Linear combination reduction
    if _time_left() > 20:
        U_red, V_red, W_red, new_rank = _try_linear_combination_reduction(U, V, W, n, m, p)
        if new_rank < best_rank:
            best_U, best_V, best_W = U_red, V_red, W_red
            best_rank = new_rank
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
