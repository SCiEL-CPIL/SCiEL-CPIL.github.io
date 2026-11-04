"""Matrix multiplication 3x4x8 rank-minimal decomposition search."""
import numpy as np
import time
import itertools

DIMENSIONS = [3, 4, 8]

def _verify_safe(U, V, W, n=3, m=4, p=8, tol=1e-6):
    """
    Safely verify the decomposition using vectorized operations.
    T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    Target: T[i,j,k] = 1 for all i,j,k.
    """
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    if U_arr.shape[1] != V_arr.shape[1] or U_arr.shape[1] != W_arr.shape[1]:
        return False
    rank = U_arr.shape[1]
    
    # Reshape for vectorized computation
    U_3d = U_arr.reshape(n, m, rank)
    V_3d = V_arr.reshape(m, p, rank)
    W_3d = W_arr.reshape(p, n, rank)
    
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    T_recon = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
    
    if not np.allclose(T_recon, 1.0, atol=tol):
        return False
    return True

def _construct_rank_88():
    """
    Construct a rank-88 decomposition.
    Split A (3x4) into A0 (1x4) and A12 (2x4).
    A0 x B (4x8): Schoolbook, rank 32.
    A12 x B (4x8): Split A12 into two 2x2 blocks (left/right).
                   Split B into four 2x2 blocks (top/bottom, left/right).
                   8 sub-problems of 2x2x2, each Strassen (rank 7).
                   Total rank 56.
    Grand total: 32 + 56 = 88.
    """
    n, m, p = 3, 4, 8
    rank = 88
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Part 1: i=0, schoolbook 1x4x8
    for j in range(4):
        for k in range(8):
            U[0 * m + j, col] = 1.0
            V[j * p + k, col] = 1.0
            W[k * n + 0, col] = 1.0
            col += 1
            
    # Part 2: i=1,2 using Strassen on 2x2 blocks
    # Strassen matrices for 2x2x2 (rank 7)
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]], C = [[C00, C01], [C10, C11]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    strassen_U = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]).T  # shape (4, 7)
    
    strassen_V = np.array([
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]).T  # shape (4, 7)
    
    strassen_W = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]).T  # shape (4, 7)
    
    for k_block in range(4):
        k_start = k_block * 2
        
        # Sub-problem 1: A_L (cols 0-1) * B_TL (rows 0-1, cols k_start..k_start+1)
        for t in range(7):
            U[4, col] += strassen_U[0, t]
            U[5, col] += strassen_U[1, t]
            U[8, col] += strassen_U[2, t]
            U[9, col] += strassen_U[3, t]
            
            V[k_start, col] += strassen_V[0, t]
            V[k_start+1, col] += strassen_V[1, t]
            V[p+k_start, col] += strassen_V[2, t]
            V[p+k_start+1, col] += strassen_V[3, t]
            
            W[k_start*n + 1, col] += strassen_W[0, t]
            W[(k_start+1)*n + 1, col] += strassen_W[1, t]
            W[k_start*n + 2, col] += strassen_W[2, t]
            W[(k_start+1)*n + 2, col] += strassen_W[3, t]
            
            col += 1
            
        # Sub-problem 2: A_R (cols 2-3) * B_BL (rows 2-3, cols k_start..k_start+1)
        for t in range(7):
            U[6, col] += strassen_U[0, t]
            U[7, col] += strassen_U[1, t]
            U[10, col] += strassen_U[2, t]
            U[11, col] += strassen_U[3, t]
            
            V[2*p+k_start, col] += strassen_V[0, t]
            V[2*p+k_start+1, col] += strassen_V[1, t]
            V[3*p+k_start, col] += strassen_V[2, t]
            V[3*p+k_start+1, col] += strassen_V[3, t]
            
            W[k_start*n + 1, col] += strassen_W[0, t]
            W[(k_start+1)*n + 1, col] += strassen_W[1, t]
            W[k_start*n + 2, col] += strassen_W[2, t]
            W[(k_start+1)*n + 2, col] += strassen_W[3, t]
            
            col += 1

    return U, V, W

def _try_rank_reduction(U, V, W, target_rank, time_budget=10):
    """
    Try to reduce the rank by removing terms.
    For each term, check if it can be removed while maintaining the decomposition.
    """
    start_time = time.time()
    n, m, p = 3, 4, 8
    current_rank = U.shape[1]
    
    # Work with copies
    U_work = U.copy()
    V_work = V.copy()
    W_work = W.copy()
    
    # Try removing terms greedily
    removed = []
    for c in range(current_rank - 1, target_rank - 1, -1):
        if time.time() - start_time > time_budget:
            break
        
        # Try removing term c
        U_test = np.delete(U_work, c, axis=1)
        V_test = np.delete(V_work, c, axis=1)
        W_test = np.delete(W_work, c, axis=1)
        
        if _verify_safe(U_test, V_test, W_test, n, m, p, tol=1e-5):
            # Remove term
            U_work = U_test
            V_work = V_test
            W_work = W_test
            removed.append(c)
    
    if removed:
        return U_work, V_work, W_work
    
    return None

def _try_pairwise_merge(U, V, W, target_rank, time_budget=15):
    """
    Try to merge pairs of terms.
    Two terms can be merged if their outer products span the same 1-dimensional subspace.
    """
    start_time = time.time()
    n, m, p = 3, 4, 8
    current_rank = U.shape[1]
    
    U_work = U.copy()
    V_work = V.copy()
    W_work = W.copy()
    
    # Try merging pairs
    merged = False
    for i in range(current_rank - 1):
        if time.time() - start_time > time_budget:
            break
        
        for j in range(i + 1, current_rank):
            # Check if U[:, i] and U[:, j] are linearly dependent
            # and V[:, i] and V[:, j] are linearly dependent
            # and W[:, i] and W[:, j] are linearly dependent
            
            # Normalize
            U_i = U_work[:, i]
            U_j = U_work[:, j]
            V_i = V_work[:, i]
            V_j = V_work[:, j]
            W_i = W_work[:, i]
            W_j = W_work[:, j]
            
            # Check if they are proportional
            try:
                # Try to find scalar alpha such that U_j ≈ alpha * U_i
                alpha = np.dot(U_i, U_j) / np.dot(U_i, U_i)
                if np.allclose(U_j, alpha * U_i, atol=1e-5):
                    # Check V
                    beta = np.dot(V_i, V_j) / np.dot(V_i, V_i)
                    if np.allclose(V_j, beta * V_i, atol=1e-5):
                        # Check W
                        gamma = np.dot(W_i, W_j) / np.dot(W_i, W_i)
                        if np.allclose(W_j, gamma * W_i, atol=1e-5):
                            # Can merge: new term is (U_i, V_i, W_i) scaled appropriately
                            # But this is complex; for now, just try removing one
                            # If U_j = alpha * U_i, V_j = beta * V_i, W_j = gamma * W_i
                            # Then the term j is redundant if alpha*beta*gamma = 1
                            if abs(alpha * beta * gamma - 1) < 1e-5:
                                # Remove term j
                                U_work = np.delete(U_work, j, axis=1)
                                V_work = np.delete(V_work, j, axis=1)
                                W_work = np.delete(W_work, j, axis=1)
                                merged = True
                                break
            except Exception:
                pass
        
        if merged:
            break
    
    if merged:
        return U_work, V_work, W_work
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Fallback: Schoolbook
    rank_schoolbook = n * m * p
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    best_rank = rank_schoolbook
    
    # Try Rank 88
    if time.time() < deadline:
        try:
            U_88, V_88, W_88 = _construct_rank_88()
            if _verify_safe(U_88, V_88, W_88):
                if U_88.shape[1] < best_rank:
                    best_result = {"U": U_88.tolist(), "V": V_88.tolist(), "W": W_88.tolist()}
                    best_rank = U_88.shape[1]
        except Exception:
            pass
    
    # Try to reduce rank further
    if time.time() < deadline and best_rank == 88:
        try:
            # Load best result
            U_best = np.array(best_result["U"])
            V_best = np.array(best_result["V"])
            W_best = np.array(best_result["W"])
            
            # Try greedy removal
            if time.time() < deadline:
                reduced = _try_rank_reduction(U_best, V_best, W_best, target_rank=80, time_budget=5)
                if reduced is not None:
                    U_red, V_red, W_red = reduced
                    if U_red.shape[1] < best_rank:
                        best_result = {"U": U_red.tolist(), "V": V_red.tolist(), "W": W_red.tolist()}
                        best_rank = U_red.shape[1]
            
            # Try pairwise merge
            if time.time() < deadline and best_rank == 88:
                U_best = np.array(best_result["U"])
                V_best = np.array(best_result["V"])
                W_best = np.array(best_result["W"])
                merged = _try_pairwise_merge(U_best, V_best, W_best, target_rank=80, time_budget=5)
                if merged is not None:
                    U_mrg, V_mrg, W_mrg = merged
                    if U_mrg.shape[1] < best_rank:
                        best_result = {"U": U_mrg.tolist(), "V": V_mrg.tolist(), "W": W_mrg.tolist()}
                        best_rank = U_mrg.shape[1]
        except Exception:
            pass
    
    return best_result
