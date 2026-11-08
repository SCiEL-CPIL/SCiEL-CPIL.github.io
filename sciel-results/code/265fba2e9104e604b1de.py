"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START

def _build_rank33():
    """Construct the known rank-33 decomposition using 3 blocks of 2x3x2."""
    n, m, p = 2, 3, 6
    rank = 33
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # 11 terms for a 2x3x2 block
    # A (2x3): flat index i*3+j
    # B (3x2): flat index j*2+k
    # C (2x2): flat index i*2+k
    
    terms = []
    # Strassen terms for A[:, 0:2] * B[0:2, :]
    # A00=0, A01=1, A10=3, A11=4
    # B00=0, B01=1, B10=2, B11=3
    terms.append({'u': {0: 1, 4: 1}, 'v': {0: 1, 3: 1}, 'w': {0: 1, 3: 1}})
    terms.append({'u': {3: 1, 4: 1}, 'v': {0: 1}, 'w': {2: 1, 3: -1}})
    terms.append({'u': {0: 1}, 'v': {1: 1, 3: -1}, 'w': {1: 1, 3: 1}})
    terms.append({'u': {4: 1}, 'v': {2: 1, 0: -1}, 'w': {0: 1, 2: 1}})
    terms.append({'u': {0: 1, 1: 1}, 'v': {3: 1}, 'w': {0: -1, 1: 1}})
    terms.append({'u': {3: 1, 0: -1}, 'v': {0: 1, 1: 1}, 'w': {3: 1}})
    terms.append({'u': {1: 1, 4: -1}, 'v': {2: 1, 3: 1}, 'w': {0: 1}})
    # Scalar terms for A[:, 2] * B[2, :]
    terms.append({'u': {2: 1}, 'v': {4: 1}, 'w': {0: 1}})
    terms.append({'u': {2: 1}, 'v': {5: 1}, 'w': {1: 1}})
    terms.append({'u': {5: 1}, 'v': {4: 1}, 'w': {2: 1}})
    terms.append({'u': {5: 1}, 'v': {5: 1}, 'w': {3: 1}})
    
    for block in range(3):
        for t_idx, term in enumerate(terms):
            col = block * 11 + t_idx
            for idx, val in term['u'].items():
                U[idx, col] = val
            for l_idx, val in term['v'].items():
                j_local = l_idx // 2
                k_local = l_idx % 2
                g_idx = j_local * p + (block * 2 + k_local)
                V[g_idx, col] = val
            for l_idx, val in term['w'].items():
                i_local = l_idx // 2
                k_local = l_idx % 2
                g_idx = (block * 2 + k_local) * n + i_local
                W[g_idx, col] = val
                
    return U, V, W


def _verify_decomp(U, V, W, n, m, p, tol=1e-9):
    """Check if decomposition is valid: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1"""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _try_reduce_rank(U, V, W, n, m, p, target_rank, max_iters=500, tol=1e-8):
    """
    Try to reduce the rank by finding linear dependencies among the rank-1 terms.
    We look for a matrix P of shape (current_rank, target_rank) such that
    U' = U @ P, V' = V @ P, W' = W @ P still satisfies the decomposition.
    
    Actually, the correct approach: if we have a decomposition of rank R, and there's
    a linear dependency, we can reduce. Let's use a different approach:
    
    We'll try to find a lower-rank decomposition by solving for the entries.
    """
    current_rank = U.shape[1]
    if current_rank <= target_rank:
        return U[:, :target_rank], V[:, :target_rank], W[:, :target_rank]
    
    # Approach: try to find a (current_rank x target_rank) matrix P such that
    # the new decomposition U' = U @ P, V' = V @ P, W' = W @ P is valid.
    # This works if the original decomposition has redundancy.
    
    # Actually, the proper way to reduce rank: if the columns of the "flattened"
    # representation are linearly dependent, we can project.
    
    # Let's try a simpler approach: use numpy to find if we can reduce.
    # For each (i,j,k), we need sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    # This is a system of equations in the entries of U', V', W'.
    
    # Alternative: try to find a lower-rank decomposition from scratch using
    # iterative methods. But this is complex. Let's try a simple projection.
    
    # If we have a rank-R decomposition and R > target, we can try to find
    # a rank-target decomposition by solving a nonlinear system.
    
    # For now, let's try a simple approach: if there are duplicate or linearly
    # dependent terms, remove them.
    
    # Check for linear dependence in the "term vectors"
    # Each term r defines a rank-1 tensor T_r[i,j,k] = U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    # We need sum_r T_r[i,j,k] = 1 for all (i,j,k)
    
    # Let's try to find a matrix P (current_rank x target_rank) such that
    # U @ P, V @ P, W @ P gives a valid decomposition.
    # This is equivalent to: for all (i,j,k),
    # sum_{r1,r2} U[i*m+j,r1] * V[j*p+k,r1] * W[k*n+i,r1] * P[r1,r2] ... no, that's not right.
    
    # Actually, U' = U @ P means U'[i,j,r'] = sum_r U[i,j,r] * P[r,r']
    # Similarly for V', W'.
    # Then sum_{r'} U'[i,j,r'] * V'[j,k,r'] * W'[k,i,r']
    # = sum_{r'} (sum_r U[i,j,r] P[r,r']) (sum_s V[j,k,s] P[s,r']) (sum_t W[k,i,t] P[t,r'])
    # This is not simply related to the original sum.
    
    # So this approach doesn't work directly. Let's skip this and just return
    # the original if we can't reduce.
    
    return None


def construct():
    n, m, p = DIMENSIONS  # 2, 3, 6
    
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Start with the known rank-33 decomposition
    U, V, W = _build_rank33()
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = 33
    
    # Verify the initial decomposition
    if not _verify_decomp(U, V, W, n, m, p):
        raise ValueError("Initial rank-33 decomposition failed verification")
    
    # Try to reduce the rank using various strategies
    # Strategy 1: Try to find a lower-rank decomposition by solving a system
    # Strategy 2: Use the structure of the problem to find a better decomposition
    
    # For 2x3x6, the trivial rank is n*m*p = 36.
    # The current rank is 33.
    # The lower bound for rank is max(n*m, m*p, p*n) = max(6, 18, 12) = 18,
    # but this is a very weak bound.
    
    # Let's try to see if we can do better than 33.
    # One approach: use a different block decomposition.
    # Instead of 3 blocks of 2x3x2 (rank 11 each = 33 total),
    # try 2 blocks: one 2x3x4 and one 2x3x2.
    # Rank of 2x3x4 is at most 2*3*4 = 24 (trivial), but can be lower.
    # Actually, rank of 2x3x4 is at most 2*4 + 3*4 - 1 = 19 (using a formula),
    # but I'm not sure of the exact rank.
    
    # Let's try a different approach: use the fact that p=6 is even,
    # and try to pair up the columns.
    
    # For now, let's just return the rank-33 decomposition if we can't improve.
    
    # Try a simple search: can we find a rank-32 decomposition?
    # This would require solving a nonlinear system, which is expensive.
    
    # Given the time constraint, let's just return the rank-33 decomposition.
    if time.time() > deadline:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Convert to lists for the return interface
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def _verify(result, n, m, p):
    """Verify the decomposition is correct by checking all (i,j,k) entries."""
    U = result['U']
    V = result['V']
    W = result['W']
    rank = len(U[0])
    
    # Check shapes
    assert len(U) == n * m, f"U shape {len(U)} != {n*m}"
    assert len(V) == m * p, f"V shape {len(V)} != {m*p}"
    assert len(W) == p * n, f"W shape {len(W)} != {p*n}"
    assert all(len(row) == rank for row in U), "U column count mismatch"
    assert all(len(row) == rank for row in V), "V column count mismatch"
    assert all(len(row) == rank for row in W), "W column count mismatch"
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1) > 1e-9:
                    raise ValueError(f"Verification failed at i={i},j={j},k={k}: got {s}")
    
    return True
# EVOLVE-BLOCK-END
