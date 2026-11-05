"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition correctly computes matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Pure schoolbook decomposition, rank = n*m*p."""
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
    return {"U": U, "V": V, "W": W}

def _strassen_block(U, V, W, i0, j0, k0, col):
    """
    Apply Strassen decomposition to a 2x2x2 block starting at (i0, j0, k0).
    Returns the new column index.
    """
    n, m, p = DIMENSIONS  # global dims for indexing
    # Strassen terms: (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: 4 values for the 4 entries of the 2x2 A submatrix
    # v_coeffs: 4 values for the 4 entries of the 2x2 B submatrix
    # w_coeffs: 4 values for the 4 entries of the 2x2 C submatrix
    strassen_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 1, 0, 0], [0, 0, 1, -1]),
        ([1, 1, 0, 0], [0, 0, 1, 1], [0, 1, 0, 1]),
        ([0, 0, 0, 1], [0, -1, 0, 1], [1, 0, -1, 0]),
        ([1, 0, 0, 0], [1, 0, -1, 0], [-1, 1, 0, 0]),
        ([-1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [1, 0, 1, 0], [1, 0, 0, 0]),
    ]
    
    for t in range(7):
        u_c, v_c, w_c = strassen_terms[t]
        for idx in range(4):
            ai, aj = idx // 2, idx % 2
            U[(i0+ai)*m + (j0+aj), col] = u_c[idx]
            bj, bk = idx // 2, idx % 2
            V[(j0+bj)*p + (k0+bk), col] = v_c[idx]
            ck, ci = idx // 2, idx % 2
            W[(k0+ck)*n + (i0+ci), col] = w_c[idx]
        col += 1
    return col

def _block_rank_optimal(di, dj, dk):
    """
    Return the optimal known rank for a di x dj x dk block.
    Uses Strassen for 2x2x2, schoolbook otherwise.
    Note: For 1xkx1, 1x1xk, kx1x1, rank is 1 (if k=1) or k (if one dim >1).
    Actually:
    Rank of matrix mult tensor M_{n,m,p} is:
    - 1 if n=m=p=1
    - 7 if n=m=p=2
    - n*m*p otherwise (schoolbook is optimal for most small cases, 
      but we don't have a general formula. We use schoolbook as a safe upper bound,
      and Strassen where known to be better).
    """
    if di == 2 and dj == 2 and dk == 2:
        return 7
    return di * dj * dk

def _get_all_set_partitions(elements):
    """
    Generate all set partitions of the list `elements`.
    Returns a list of partitions, where each partition is a list of frozensets.
    """
    if not elements:
        return [[]]
    
    first = elements[0]
    rest = elements[1:]
    rest_partitions = _get_all_set_partitions(rest)
    
    result = []
    for partition in rest_partitions:
        # Add `first` to each existing block
        for i in range(len(partition)):
            new_partition = [frozenset(b) for b in partition]
            new_partition[i] = new_partition[i] | {first}
            result.append(new_partition)
        
        # Create a new block with just `first`
        new_partition = [frozenset([first])] + [frozenset(b) for b in partition]
        result.append(new_partition)
    
    return result

def _search_best_partition(n, m, p, deadline):
    """
    Exhaustively search over all set partitions of the three index sets.
    For each combination of partitions, compute the total rank.
    Keep track of the best (minimum rank) combination.
    """
    i_elements = list(range(n))
    j_elements = list(range(m))
    k_elements = list(range(p))
    
    i_partitions = _get_all_set_partitions(i_elements)
    j_partitions = _get_all_set_partitions(j_elements)
    k_partitions = _get_all_set_partitions(k_elements)
    
    best_rank = n * m * p
    best_combo = None
    
    # Precompute block ranks for all possible block sizes
    # Max block size is n, m, p
    rank_table = {}
    for di in range(1, n+1):
        for dj in range(1, m+1):
            for dk in range(1, p+1):
                rank_table[(di, dj, dk)] = _block_rank_optimal(di, dj, dk)
    
    for i_part in i_partitions:
        if time.time() > deadline:
            break
        for j_part in j_partitions:
            if time.time() > deadline:
                break
            for k_part in k_partitions:
                total_rank = 0
                for ib in i_part:
                    for jb in j_part:
                        for kb in k_part:
                            di = len(ib)
                            dj = len(jb)
                            dk = len(kb)
                            total_rank += rank_table[(di, dj, dk)]
                            if total_rank >= best_rank:
                                break
                        if total_rank >= best_rank:
                            break
                    if total_rank >= best_rank:
                        break
                if total_rank < best_rank:
                    best_rank = total_rank
                    best_combo = (i_part, j_part, k_part)
    
    return best_rank, best_combo

def _build_decomposition_from_partitions(i_part, j_part, k_part, n, m, p):
    """
    Build the U, V, W matrices from the given set partitions.
    """
    rank = 0
    # First, compute the total rank
    for ib in i_part:
        for jb in j_part:
            for kb in k_part:
                di = len(ib)
                dj = len(jb)
                dk = len(kb)
                rank += _block_rank_optimal(di, dj, dk)
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for ib in i_part:
        for jb in j_part:
            for kb in k_part:
                di = len(ib)
                dj = len(jb)
                dk = len(kb)
                
                # Sort the indices in each block to ensure consistent ordering
                ib_sorted = sorted(ib)
                jb_sorted = sorted(jb)
                kb_sorted = sorted(kb)
                
                # Map local indices to global indices
                i_map = {local: global_idx for local, global_idx in enumerate(ib_sorted)}
                j_map = {local: global_idx for local, global_idx in enumerate(jb_sorted)}
                k_map = {local: global_idx for local, global_idx in enumerate(kb_sorted)}
                
                # Get the minimum indices for the block
                i0 = min(ib)
                j0 = min(jb)
                k0 = min(kb)
                
                if di == 2 and dj == 2 and dk == 2:
                    # Strassen
                    # We need to map the 2x2 block to the standard Strassen layout
                    # The local indices are 0,1 for each dimension
                    # We need to fill U, V, W for the global indices in ib, jb, kb
                    strassen_terms = [
                        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
                        ([0, 0, 1, 1], [1, 1, 0, 0], [0, 0, 1, -1]),
                        ([1, 1, 0, 0], [0, 0, 1, 1], [0, 1, 0, 1]),
                        ([0, 0, 0, 1], [0, -1, 0, 1], [1, 0, -1, 0]),
                        ([1, 0, 0, 0], [1, 0, -1, 0], [-1, 1, 0, 0]),
                        ([-1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 0, 1]),
                        ([0, 1, 0, -1], [1, 0, 1, 0], [1, 0, 0, 0]),
                    ]
                    
                    for t in range(7):
                        u_c, v_c, w_c = strassen_terms[t]
                        for local_idx in range(4):
                            li, lj = local_idx // 2, local_idx % 2
                            gi = ib_sorted[li]
                            gj = jb_sorted[lj]
                            U[gi*m + gj, col] = u_c[local_idx]
                            
                            bj, bk = local_idx // 2, local_idx % 2
                            gj2 = jb_sorted[bj]
                            gk = kb_sorted[bk]
                            V[gj2*p + gk, col] = v_c[local_idx]
                            
                            ck, ci = local_idx // 2, local_idx % 2
                            gk2 = kb_sorted[ck]
                            gi2 = ib_sorted[ci]
                            W[gk2*n + gi2, col] = w_c[local_idx]
                        col += 1
                else:
                    # Schoolbook
                    for li in range(di):
                        for lj in range(dj):
                            for lk in range(dk):
                                gi = ib_sorted[li]
                                gj = jb_sorted[lj]
                                gk = kb_sorted[lk]
                                U[gi*m + gj, col] = 1
                                V[gj*p + gk, col] = 1
                                W[gk*n + gi, col] = 1
                                col += 1
    
    if col != rank:
        return None
        
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    if _verify(U_list, V_list, W_list, n, m, p):
        return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Try the systematic set partition search
    best_rank, best_combo = _search_best_partition(n, m, p, deadline)
    
    if best_combo is not None and best_rank < n * m * p:
        result = _build_decomposition_from_partitions(best_combo[0], best_combo[1], best_combo[2], n, m, p)
        if result is not None:
            return result
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
