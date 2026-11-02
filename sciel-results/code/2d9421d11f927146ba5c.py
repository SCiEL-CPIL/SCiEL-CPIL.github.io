"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START

def _strassen_2x2_terms():
    """Return the 7 terms of Strassen's 2x2x2 matrix multiplication.
    Each term has U, V, W as dicts mapping (row, col) -> coeff.
    U is 2x2 (A), V is 2x2 (B), W is 2x2 (C).
    """
    terms = [
        {'U': {(0,0):1,(1,1):1}, 'V': {(0,0):1,(1,1):1}, 'W': {(0,0):1,(1,1):1}},
        {'U': {(1,0):1,(1,1):1}, 'V': {(0,0):1}, 'W': {(1,0):1}},
        {'U': {(0,0):1}, 'V': {(1,0):1,(1,1):-1}, 'W': {(0,1):1,(1,1):-1}},
        {'U': {(1,1):1}, 'V': {(0,1):1,(0,0):-1}, 'W': {(0,0):-1,(1,0):1}},
        {'U': {(0,0):1,(0,1):1}, 'V': {(1,1):1}, 'W': {(0,1):1,(1,1):1}},
        {'U': {(1,0):1,(0,0):-1}, 'V': {(0,0):1,(0,1):1}, 'W': {(0,0):1,(1,0):1}},
        {'U': {(0,1):1,(1,1):-1}, 'V': {(1,0):1,(1,1):1}, 'W': {(0,0):1,(1,1):1}},
    ]
    return terms


def _rank9_2x2x3_terms():
    """Rank-9 decomposition for 2x2 @ 2x3.
    Strategy: 2x2x2 Strassen (7 terms) + 2 extra terms for the 3rd column.
    Total rank = 9.
    U is 2x2 (A), V is 2x3 (B), W is 3x2 (C).
    """
    strassen = _strassen_2x2_terms()
    terms = []
    # 7 Strassen terms: V only uses cols 0,1 of B; W only uses rows 0,1 of C
    for t in strassen:
        V = {k: v for k, v in t['V'].items()}
        W = {k: v for k, v in t['W'].items()}
        terms.append({'U': dict(t['U']), 'V': V, 'W': W})
    # 2 extra terms for column 2 of B, rows 0,1 of C
    # Term 7: A[:,0] * B[0,2] * C[0,:]  -> U: (0,0):1,(1,0):1; V: (0,2):1; W: (0,0):1,(1,0):1
    terms.append({'U': {(0,0):1,(1,0):1}, 'V': {(0,2):1}, 'W': {(0,0):1,(1,0):1}})
    # Term 8: A[:,1] * B[1,2] * C[1,:]  -> U: (0,1):1,(1,1):1; V: (1,2):1; W: (1,0):1,(1,1):1
    terms.append({'U': {(0,1):1,(1,1):1}, 'V': {(1,2):1}, 'W': {(1,0):1,(1,1):1}})
    return terms


def _build_block_decomposition(n, m, p):
    """Build a block decomposition for n x m @ m x p.
    Decompose into 2x2 blocks. For each (bi,bj,bk) block:
    - If bk is a full 2-block: use 2x2x2 Strassen (rank 7)
    - If bk is a partial block (odd p): use 2x2x1 (rank 2)
    Total rank = sum over blocks of block_rank.
    """
    # For 4x4x5: blocks are (bi=0,1) x (bj=0,1) x (bk=0,1) with bk=1 being 2x2x3
    # Actually let's think more carefully.
    # A is n x m = 4x4, B is m x p = 4x5
    # We partition A into 2x2 blocks: 2x2 blocks
    # We partition B into 2x2 or 2x3 blocks depending on p
    # C is p x n = 5x4, partition into 2x2 or 3x2 blocks
    
    # For 4x4x5:
    # A: 2x2 blocks (bi, bj) for bi,bj in {0,1}
    # B: 2x2 blocks for bk in {0}, and 2x3 block for bk in {1} (since p=5)
    # C: 2x2 blocks for bk in {0}, and 3x2 block for bk in {1}
    
    U = np.zeros((n*m, 0))
    V = np.zeros((m*p, 0))
    W = np.zeros((p*n, 0))
    
    rank = 0
    for bi in range(0, n, 2):
        for bj in range(0, m, 2):
            for bk in range(0, p, 2):
                # Determine block sizes
                ai = min(2, n - bi)
                aj = min(2, m - bj)
                bk_size = min(2, p - bk)
                
                if bk_size == 2:
                    block_terms = _strassen_2x2_terms()
                else:  # bk_size == 1
                    # 2x2x1: rank 2
                    block_terms = [
                        {'U': {(0,0):1,(1,0):1}, 'V': {(0,0):1}, 'W': {(0,0):1}},
                        {'U': {(0,1):1,(1,1):1}, 'V': {(1,0):1}, 'W': {(0,0):1}},
                    ]
                    # Wait, for 2x2x1: A is 2x2, B is 2x1, C is 1x2
                    # C[0,0] = A[0,0]*B[0,0] + A[1,0]*B[1,0]
                    # C[0,1] = A[0,1]*B[0,0] + A[1,1]*B[1,0]
                    # Term 0: A[:,0] * B[0,0] * [1,0] -> U: (0,0):1,(1,0):1; V: (0,0):1; W: (0,0):1
                    # Term 1: A[:,1] * B[1,0] * [0,1] -> U: (0,1):1,(1,1):1; V: (1,0):1; W: (0,1):1
                    block_terms = [
                        {'U': {(0,0):1,(1,0):1}, 'V': {(0,0):1}, 'W': {(0,0):1}},
                        {'U': {(0,1):1,(1,1):1}, 'V': {(1,0):1}, 'W': {(0,1):1}},
                    ]
                
                n_terms = len(block_terms)
                for t_idx, term in enumerate(block_terms):
                    r = rank + t_idx
                    # U: A block (ai x aj) at position (bi, bj)
                    for (li, lj), coeff in term['U'].items():
                        i = bi + li
                        j = bj + lj
                        if i < n and j < m:
                            U[i*m + j, r] = coeff
                    # V: B block (aj x bk_size) at position (bj, bk)
                    for (lj, lk), coeff in term['V'].items():
                        j = bj + lj
                        k = bk + lk
                        if j < m and k < p:
                            V[j*p + k, r] = coeff
                    # W: C block (bk_size x ai) at position (bk, bi)
                    for (lk, li), coeff in term['W'].items():
                        k = bk + lk
                        i = bi + li
                        if k < p and i < n:
                            W[k*n + i, r] = coeff
                rank += n_terms
    
    U = U[:, :rank]
    V = V[:, :rank]
    W = W[:, :rank]
    return {"U": U, "V": V, "W": W}, rank


def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check if sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for r in range(rank):
                    total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(total - 1.0) > tol:
                    return False
    return True


def _verify_integer_half_integer(U, V, W, tol=1e-9):
    """Check that all entries are integer or half-integer (multiple of 0.5)."""
    for M in [U, V, W]:
        for row in M:
            for val in row:
                if abs(val * 2 - round(val * 2)) > tol:
                    return False
    return True


def _snap_to_half_integer(M, tol=1e-6):
    """Snap values to nearest half-integer if close enough."""
    M = M.copy()
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            snapped = round(M[i,j] * 2) / 2.0
            if abs(M[i,j] - snapped) < tol:
                M[i,j] = snapped
    return M


def _build_rank64_baseline():
    """Build the rank-64 block decomposition for 4x4x5.
    4x4x5 = 2x2x2 blocks (rank 7 each) + 2x2x1 blocks (rank 2 each)
    For 4x4x5: 4 full 2x2x2 blocks (rank 28) + 2 half 2x2x1 blocks (rank 4) = 32?
    Wait, let me recalculate.
    n=4, m=4, p=5
    bi in {0,2}, bj in {0,2}, bk in {0,2,3} (since p=5, bk goes 0,2,4 but 4+1=5)
    Actually bk in range(0, 5, 2) = {0, 2, 4}
    bk=0: bk_size=2, 2x2x2 Strassen, rank 7
    bk=2: bk_size=2, 2x2x2 Strassen, rank 7
    bk=4: bk_size=1, 2x2x1, rank 2
    Total per (bi,bj): 7+7+2 = 16
    Total: 4 * 16 = 64
    """
    n, m, p = DIMENSIONS
    result, rank = _build_block_decomposition(n, m, p)
    return result, rank


def _try_merge_terms(U, V, W, n, m, p, max_time=30.0):
    """Try to merge linearly dependent terms to reduce rank.
    Two terms r1, r2 can be merged if one is a scalar multiple of the other
    in the tensor sense. We check by looking at the outer product structure.
    """
    start = time.time()
    rank = U.shape[1]
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    merged = True
    while merged:
        merged = False
        if time.time() - start > max_time:
            break
        for r1 in range(rank):
            if time.time() - start > max_time:
                break
            for r2 in range(r1+1, rank):
                # Check if term r2 is a scalar multiple of term r1
                # In the tensor decomposition, term r contributes U[:,r] * V[:,r] * W[:,r]
                # Two terms are linearly dependent if one is a scalar multiple of the other
                # This is hard to check directly. Instead, we can try to eliminate r2
                # by expressing it as a combination of r1 and others.
                # Simple check: if U[:,r2] is proportional to U[:,r1] AND
                # V[:,r2] is proportional to V[:,r1] AND W[:,r2] is proportional to W[:,r1]
                u1 = U[:, r1]
                u2 = U[:, r2]
                v1 = V[:, r1]
                v2 = V[:, r2]
                w1 = W[:, r1]
                w2 = W[:, r2]
                
                # Check proportionality
                def is_proportional(a, b, tol=1e-9):
                    if np.allclose(a, 0, atol=tol) and np.allclose(b, 0, atol=tol):
                        return True, 0.0
                    if np.allclose(a, 0, atol=tol):
                        return False, 0.0
                    if np.allclose(b, 0, atol=tol):
                        return False, 0.0
                    # Find a non-zero entry in a
                    idx = np.argmax(np.abs(a))
                    if abs(a[idx]) < tol:
                        return False, 0.0
                    scale = b[idx] / a[idx]
                    if np.allclose(b, scale * a, atol=tol):
                        return True, scale
                    return False, 0.0
                
                prop_u, su = is_proportional(u1, u2)
                prop_v, sv = is_proportional(v1, v2)
                prop_w, sw = is_proportional(w1, w2)
                
                if prop_u and prop_v and prop_w:
                    # Merge: term r2 = su*sv*sw * term r1
                    # We can absorb r2 into r1 by scaling
                    # Actually, if term r2 = c * term r1, then the contribution is (1+c) * term r1
                    # But we can't just scale U,V,W because that changes the product.
                    # The product is U*r * V*r * W*r. If we have two terms that are
                    # proportional, we can't simply merge them because the product
                    # is trilinear, not bilinear.
                    # Actually, if U2 = su*U1, V2 = sv*V1, W2 = sw*W1, then
                    # U2*V2*W2 = su*sv*sw * U1*V1*W1
                    # So the total contribution is (1 + su*sv*sw) * U1*V1*W1
                    # We can scale U1 by (1 + su*sv*sw)^(1/3) but that may not be clean.
                    # Better: just remove r2 and scale r1's contribution.
                    # Actually, the simplest merge: if the terms are identical (su=sv=sw=1),
                    # we can remove one. If they're proportional, we can combine.
                    # For now, only merge if they're identical.
                    if abs(su - 1) < 1e-9 and abs(sv - 1) < 1e-9 and abs(sw - 1) < 1e-9:
                        # Remove term r2
                        U = np.delete(U, r2, axis=1)
                        V = np.delete(V, r2, axis=1)
                        W = np.delete(W, r2, axis=1)
                        rank -= 1
                        merged = True
                        break
        if merged:
            continue
    
    return U, V, W, rank


def _alternating_minimization(U, V, W, n, m, p, max_iter=1000, tol=1e-10, max_time=30.0):
    """Alternating minimization to polish the decomposition.
    Fix two of U,V,W and solve for the third.
    """
    start = time.time()
    U = U.astype(float).copy()
    V = V.astype(float).copy()
    W = W.astype(float).copy()
    rank = U.shape[1]
    
    for iteration in range(max_iter):
        if time.time() - start > max_time:
            break
        
        # Solve for U: for each (i,j), U[i*m+j, :] should satisfy
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k
        # This is a linear system in U[i*m+j, :]
        # For each (i,j), the constraint is:
        # sum_k (sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]) = 1 for all k
        # Let A_k[r] = V[j*p+k, r] * W[k*n+i, r]
        # Then sum_r U[i*m+j,r] * A_k[r] = 1 for all k
        # This is a system: A * U[i*m+j,:]^T = 1
        # where A is p x rank
        
        # Solve for U
        for i in range(n):
            for j in range(m):
                # Build the p x rank matrix
                A = np.zeros((p, rank))
                for k in range(p):
                    A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                # Solve A * x = 1
                try:
                    x, residuals, rank_s, sv = np.linalg.lstsq(A, np.ones(p), rcond=None)
                    U[i*m+j, :] = x
                except:
                    pass
        
        # Solve for V
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, rank))
                for i in range(n):
                    A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                try:
                    x, residuals, rank_s, sv = np.linalg.lstsq(A, np.ones(n), rcond=None)
                    V[j*p+k, :] = x
                except:
                    pass
        
        # Solve for W
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, rank))
                for j in range(m):
                    A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                try:
                    x, residuals, rank_s, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                    W[k*n+i, :] = x
                except:
                    pass
        
        # Check convergence
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    max_err = max(max_err, abs(s - 1.0))
        if max_err < tol:
            break
    
    return U, V, W, max_err


def _try_reduce_rank(U, V, W, n, m, p, target_rank, max_time=30.0):
    """Try to reduce rank to target_rank by setting some columns to zero and re-optimizing."""
    start = time.time()
    rank = U.shape[1]
    if target_rank >= rank:
        return None
    
    # Try to find which columns to zero out
    # Strategy: try all subsets of columns to keep (if rank-target_rank is small)
    # For larger reductions, use a greedy approach
    
    U0 = U.copy()
    V0 = V.copy()
    W0 = W.copy()
    
    # Try removing columns one at a time, keeping the best
    U_cur = U0.copy()
    V_cur = V0.copy()
    W_cur = W0.copy()
    cur_rank = rank
    
    while cur_rank > target_rank:
        if time.time() - start > max_time:
            break
        best_err = np.inf
        best_col = -1
        for col in range(cur_rank):
            if time.time() - start > max_time:
                break
            # Try removing column col
            mask = np.ones(cur_rank, dtype=bool)
            mask[col] = False
            U_test = U_cur[:, mask]
            V_test = V_cur[:, mask]
            W_test = W_cur[:, mask]
            
            # Polish
            U_pol, V_pol, W_pol, err = _alternating_minimization(
                U_test, V_test, W_test, n, m, p, 
                max_iter=200, tol=1e-10, max_time=max_time/10.0
            )
            if err < best_err:
                best_err = err
                best_col = col
                U_best = U_pol
                V_best = V_pol
                W_best = W_pol
        
        if best_col == -1 or best_err > 1e-6:
            break
        
        U_cur = U_best
        V_cur = V_best
        W_cur = W_best
        cur_rank -= 1
    
    if cur_rank <= target_rank and best_err < 1e-6:
        return U_cur, V_cur, W_cur, cur_rank
    return None


def _random_search(U0, V0, W0, n, m, p, target_rank, max_time=30.0, n_restarts=5):
    """Random search for lower rank decomposition."""
    start = time.time()
    best_result = None
    best_err = np.inf
    
    for restart in range(n_restarts):
        if time.time() - start > max_time:
            break
        
        # Random initialization
        np.random.seed(42 + restart)
        U = np.random.randn(n*m, target_rank) * 0.5
        V = np.random.randn(m*p, target_rank) * 0.5
        W = np.random.randn(p*n, target_rank) * 0.5
        
        # Alternating minimization
        U, V, W, err = _alternating_minimization(
            U, V, W, n, m, p,
            max_iter=500, tol=1e-10, max_time=max_time/n_restarts
        )
        
        if err < best_err:
            best_err = err
            best_result = (U, V, W, target_rank)
        
        if err < 1e-8:
            break
    
    return best_result


def _try_rank_revealing(U, V, W, n, m, p, max_time=30.0):
    """Try rank-revealing decomposition using tensor train / CP decomposition ideas.
    The tensor T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    We want to find the minimal rank decomposition.
    """
    start = time.time()
    rank = U.shape[1]
    
    # Build the full tensor
    T = np.zeros((n, m, p))
    for r in range(rank):
        T += np.einsum('ij,jk,ki->ijk', 
                       U.reshape(n, m, rank), 
                       V.reshape(m, p, rank), 
                       W.reshape(p, n, rank))
    
    # Try to find a lower rank CP decomposition
    # Use a simple iterative approach
    best_result = None
    for target_rank in range(rank, max(rank-20, 10), -1):
        if time.time() - start > max_time:
            break
        result = _random_search(U, V, W, n, m, p, target_rank, max_time=max_time/5.0, n_restarts=3)
        if result is not None:
            best_result = result
            break
    
    return best_result


def construct():
    """Find a rank-minimal exact tensor decomposition for 4x4 @ 4x5 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Build the rank-64 baseline
    try:
        baseline, rank64 = _build_rank64_baseline()
        U = baseline["U"]
        V = baseline["V"]
        W = baseline["W"]
        
        # Verify
        if not _verify_decomposition(U, V, W, n, m, p):
            raise ValueError("Baseline verification failed")
        
        # Snap to half-integers
        U = _snap_to_half_integer(U)
        V = _snap_to_half_integer(V)
        W = _snap_to_half_integer(W)
        
        # Verify after snapping
        if not _verify_decomposition(U, V, W, n, m, p):
            # Use original
            U = baseline["U"]
            V = baseline["V"]
            W = baseline["W"]
        
        best_U, best_V, best_W, best_rank = U, V, W, rank64
    except Exception:
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
        best_U, best_V, best_W, best_rank = U, V, W, rank
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Try to reduce rank
    # 1. Merge identical terms
    if time.time() < deadline - 20:
        try:
            U_m, V_m, W_m, rank_m = _try_merge_terms(U, V, W, n, m, p, max_time=10.0)
            if rank_m < best_rank and _verify_decomposition(U_m, V_m, W_m, n, m, p):
                U_m = _snap_to_half_integer(U_m)
                V_m = _snap_to_half_integer(V_m)
                W_m = _snap_to_half_integer(W_m)
                if _verify_decomposition(U_m, V_m, W_m, n, m, p):
                    best_U, best_V, best_W, best_rank = U_m, V_m, W_m, rank_m
        except Exception:
            pass
    
    # 2. Alternating minimization polish
    if time.time() < deadline - 15:
        try:
            U_p, V_p, W_p, err = _alternating_minimization(
                best_U, best_V, best_W, n, m, p,
                max_iter=500, tol=1e-12, max_time=15.0
            )
            if err < 1e-8:
                U_p = _snap_to_half_integer(U_p)
                V_p = _snap_to_half_integer(V_p)
                W_p = _snap_to_half_integer(W_p)
                if _verify_decomposition(U_p, V_p, W_p, n, m, p):
                    best_U, best_V, best_W = U_p, V_p, W_p
        except Exception:
            pass
    
    # 3. Try to reduce rank by removing columns
    if time.time() < deadline - 10:
        for target_rank in range(best_rank - 1, max(best_rank - 10, 30), -1):
            if time.time() > deadline - 10:
                break
            try:
                result = _try_reduce_rank(best_U, best_V, best_W, n, m, p, 
                                          target_rank, max_time=10.0)
                if result is not None:
                    U_r, V_r, W_r, rank_r = result
                    U_r = _snap_to_half_integer(U_r)
                    V_r = _snap_to_half_integer(V_r)
                    W_r = _snap_to_half_integer(W_r)
                    if _verify_decomposition(U_r, V_r, W_r, n, m, p):
                        best_U, best_V, best_W, best_rank = U_r, V_r, W_r, rank_r
                        break
            except Exception:
                pass
    
    # 4. Random search for lower rank
    if time.time() < deadline - 5:
        for target_rank in range(best_rank - 1, max(best_rank - 5, 40), -1):
            if time.time() > deadline - 5:
                break
            try:
                result = _random_search(best_U, best_V, best_W, n, m, p,
                                        target_rank, max_time=5.0, n_restarts=3)
                if result is not None:
                    U_r, V_r, W_r, rank_r = result
                    U_r = _snap_to_half_integer(U_r)
                    V_r = _snap_to_half_integer(V_r)
                    W_r = _snap_to_half_integer(W_r)
                    if _verify_decomposition(U_r, V_r, W_r, n, m, p):
                        best_U, best_V, best_W, best_rank = U_r, V_r, W_r, rank_r
                        break
            except Exception:
                pass
    
    # Final verification
    if not _verify_decomposition(best_U, best_V, best_W, n, m, p):
        # Fallback to baseline
        baseline, rank64 = _build_rank64_baseline()
        best_U, best_V, best_W = baseline["U"], baseline["V"], baseline["W"]
        best_rank = rank64
    
    # Ensure all values are plain floats
    def _clean(M):
        return [[float(x) for x in row] for row in M]
    
    return {"U": _clean(best_U), "V": _clean(best_V), "W": _clean(best_W)}

# EVOLVE-BLOCK-END
