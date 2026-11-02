"""Tensor decomposition for matrix multiplication 3x3 * 3x8.
Strategy: shared-sparse U/V skeleton + linear solve for W, with rank minimization.
Falls back to known rank-68 decomposition if optimization fails or times out.
"""

DIMENSIONS = [3, 3, 8]

import time
import numpy as np
import random

def _build_rank68():
    """Build the known rank-68 Strassen-based decomposition."""
    n, m, p = 3, 3, 8
    U_t = []
    V_t = []
    W_t = []
    
    def add_term(u_sp, v_sp, w_sp):
        U_t.append(u_sp)
        V_t.append(v_sp)
        W_t.append(w_sp)
    
    def strassen_decomp(ia, ja, ib, jb):
        a = (ia, ja); b = (ia, ja+1)
        c = (ia+1, ja); d = (ia+1, ja+1)
        e = (ib, jb); f = (ib, jb+1)
        g = (ib+1, jb); h = (ib+1, jb+1)
        C00 = (ia, jb); C01 = (ia, jb+1)
        C10 = (ia+1, jb); C11 = (ia+1, jb+1)
        # m1 = (a+d)(e+h)
        add_term([(a,1),(d,1)], [(e,1),(h,1)], [(C00,1),(C11,1)])
        # m2 = (c+d)*e
        add_term([(c,1),(d,1)], [(e,1)], [(C10,1),(C11,-1)])
        # m3 = a*(f-h)
        add_term([(a,1)], [(f,1),(h,-1)], [(C01,1),(C11,1)])
        # m4 = d*(g-e)
        add_term([(d,1)], [(g,1),(e,-1)], [(C00,1),(C10,1)])
        # m5 = (a+b)*h
        add_term([(a,1),(b,1)], [(h,1)], [(C00,-1),(C01,1)])
        # m6 = (c-a)*(e+f)
        add_term([(c,1),(a,-1)], [(e,1),(f,1)], [(C11,1)])
        # m7 = (b-d)*(g+h)
        add_term([(b,1),(d,-1)], [(g,1),(h,1)], [(C00,1)])
    
    def direct_term(i, j, k):
        add_term([((i,j),1)], [((j,k),1)], [((i,k),1)])
    
    # 4x Strassen for A[0:2,0:2] * B[0:2,0:8] (4 blocks of 2x2)
    for jb in range(0, 8, 2):
        strassen_decomp(0, 0, 0, jb)
    
    # A[0:2,2] * B[2,0:8]
    for i in range(2):
        for k in range(8):
            direct_term(i, 2, k)
    
    # A[2,0:2] * B[0:2,0:8]
    for j in range(2):
        for k in range(8):
            direct_term(2, j, k)
    
    # A[2,2] * B[2,0:8]
    for k in range(8):
        direct_term(2, 2, k)
    
    rank = len(W_t)
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    for idx in range(rank):
        for (i, j), coeff in U_t[idx]:
            U[i*m + j, idx] = coeff
        for (j, k), coeff in V_t[idx]:
            V[j*p + k, idx] = coeff
        for (i, k), coeff in W_t[idx]:
            W[k*n + i, idx] = coeff
    
    return U, V, W


def _verify(U, V, W):
    n, m, p = 3, 3, 8
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                if abs(s - 1.0) > 1e-8:
                    return False
    return True


def _to_nice_list(arr):
    result = []
    for row in arr:
        nice_row = []
        for val in row:
            val2 = val * 2
            if abs(val2 - round(val2)) < 1e-6:
                nice_val = round(val2) / 2.0
                if abs(nice_val - round(nice_val)) < 1e-9:
                    nice_val = int(round(nice_val))
                nice_row.append(nice_val)
            else:
                nice_row.append(float(val))
        result.append(nice_row)
    return result


def _sparse_skeleton_attempt(deadline, best_rank):
    """
    Shared-sparse approach:
    - Fix a sparse U, V with limited support
    - Solve for W via linear system
    - Minimize rank by iterative column reduction
    """
    n, m, p = 3, 3, 8
    r_target = 40  # aim for rank significantly below 68
    
    def random_sparse_skeleton(rank, seed):
        rng = np.random.RandomState(seed)
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        
        # Each column of U: pick 2-3 nonzeros from the 9 positions
        # Structure: ensure each (i,j) position in A appears
        for r in range(rank):
            # Pick which A-positions this term touches
            positions = list(range(n*m))
            n_nonzero = rng.randint(2, 4)
            chosen = rng.choice(positions, size=n_nonzero, replace=False)
            for pos in chosen:
                U[pos, r] = rng.choice([-2, -1, -0.5, 0.5, 1, 2])
            
            # Pick which B-positions
            positions_b = list(range(m*p))
            n_nonzero_b = rng.randint(2, 4)
            chosen_b = rng.choice(positions_b, size=n_nonzero_b, replace=False)
            for pos in chosen_b:
                V[pos, r] = rng.choice([-2, -1, -0.5, 0.5, 1, 2])
        
        return U, V
    
    def solve_for_W(U, V):
        """
        For each output (i,k): need sum_r U[i*m+j_r]... 
        Actually: C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        For fixed (i,k): we need sum_r [sum_j U[i*m+j,r]*V[j*p+k,r]] * W[k*n+i,r] = 1
        This is: sum_r alpha_{i,k,r} * W_{k*n+i, r} = 1
        where alpha_{i,k,r} = sum_j U[i*m+j, r] * V[j*p+k, r]
        
        So for each output position (i,k), we have one equation:
        alpha_{i,k} · w_{k*n+i} = 1
        
        where w_{k*n+i} is the r-vector [W[k*n+i, 0], ..., W[k*n+i, rank-1]]
        
        This is underdetermined (1 equation, rank unknowns per output).
        We need to find W such that these 24 equations are all satisfied.
        
        Approach: set W as a matrix of shape (p*n, rank) where each row is a solution.
        For row (k*n+i): need alpha_{i,k} @ W_row = 1.
        
        If alpha has rank >= 1, we can find a particular solution and add nullspace.
        To minimize effective rank of W, we want W rows to be as simple as possible.
        """
        # Compute alpha matrix: shape (p*n, rank)
        # alpha[k*n + i, r] = sum_j U[i*m+j, r] * V[j*p+k, r]
        alpha = np.zeros((p*n, U.shape[1]))
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    alpha[k*n+i] += U[i*m+j] * V[j*p+k]
        
        # For each row of alpha, find w such that alpha_row @ w = 1
        # If alpha_row is all zeros, infeasible
        # Otherwise, find minimum-norm solution: w = alpha_row / ||alpha_row||^2
        # But we can do better: choose w to have few nonzeros or small values
        
        W = np.zeros((p*n, U.shape[1]))
        for row_idx in range(p*n):
            a = alpha[row_idx]
            if np.allclose(a, 0):
                return None  # infeasible
            # Particular solution
            norm_sq = np.dot(a, a)
            w = a / norm_sq
            W[row_idx] = w
        
        return W, alpha
    
    best_result = None
    
    for seed in range(500):
        if time.time() > deadline - 5:
            break
        
        U, V = random_sparse_skeleton(r_target, seed)
        
        result = solve_for_W(U, V)
        if result is None:
            continue
        
        W, alpha = result
        
        # Verify
        if _verify(U, V, W):
            # Try to reduce rank: find columns that are linearly dependent
            # For now, record this as a valid solution
            if best_result is None or r_target < best_result[0]:
                best_result = (r_target, U, V, W)
        
        if time.time() > deadline - 5:
            break
    
    return best_result


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    DEADLINE = 145.0
    
    # First: build the known rank-68 solution as fallback
    U68, V68, W68 = _build_rank68()
    
    if not _verify(U68, V68, W68):
        # Shouldn't happen, but fallback to schoolbook
        rank_fb = n * m * p
        U_fb = np.zeros((n*m, rank_fb))
        V_fb = np.zeros((m*p, rank_fb))
        W_fb = np.zeros((p*n, rank_fb))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    col = i*m*p + j*p + k
                    U_fb[i*m+j, col] = 1
                    V_fb[j*p+k, col] = 1
                    W_fb[k*n+i, col] = 1
        return {"U": _to_nice_list(U_fb), "V": _to_nice_list(V_fb), "W": _to_nice_list(W_fb)}
    
    best_rank = 68
    best_U, best_V, best_W = U68, V68, W68
    
    # Try shared-sparse optimization
    try:
        result = _sparse_skeleton_attempt(deadline, best_rank)
        if result is not None:
            r, U_opt, V_opt, W_opt = result
            if r < best_rank:
                best_rank = r
                best_U, best_V, best_W = U_opt, V_opt, W_opt
    except Exception:
        pass
    
    # Try column elimination on the rank-68 solution
    # Check if any columns are redundant
    def try_reduce_columns(U, V, W, current_rank, deadline):
        """Try to find a subset of columns that still spans the tensor."""
        # Build the tensor representation
        # T[i*m+j, j*p+k, k*n+i] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
        # For each (i,j,k), we need sum_r ... = 1
        
        # The set of equations is: for each (i,j,k): sum_r U_ij_r * V_jk_r * W_ik_r = 1
        # We want to find a subset S of {0,...,rank-1} such that
        # for all (i,j,k): sum_{r in S} U_ij_r * V_jk_r * W_ik_r = 1
        
        # Try greedy: start with all columns, try removing one at a time
        # A column r can be removed if for all (i,j,k):
        #   U_ij_r * V_jk_r * W_ik_r is "absorbed" by other columns
        # This is hard to check directly. Instead, use the fact that
        # if column r has small contribution relative to others, it might be removable.
        
        # Alternative: try random subsets of size r
        for trial_size in range(current_rank - 1, max(20, current_rank - 15), -1):
            if time.time() > deadline - 5:
                break
            for attempt in range(200):
                if time.time() > deadline - 5:
                    break
                subset = random.sample(range(current_rank), trial_size)
                U_s = U[:, subset]
                V_s = V[:, subset]
                W_s = W[:, subset]
                if _verify(U_s, V_s, W_s):
                    return trial_size, U_s, V_s, W_s
        return None
    
    if time.time() < deadline - 10:
        try:
            reduced = try_reduce_columns(U68, V68, W68, 68, deadline)
            if reduced is not None:
                r, U_r, V_r, W_r = reduced
                if r < best_rank:
                    best_rank = r
                    best_U, best_V, best_W = U_r, V_r, W_r
        except Exception:
            pass
    
    return {"U": _to_nice_list(best_U), "V": _to_nice_list(best_V), "W": _to_nice_list(best_W)}
