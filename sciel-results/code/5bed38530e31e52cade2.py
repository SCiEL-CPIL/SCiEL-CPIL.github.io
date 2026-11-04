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

def _strassen_terms():
    """Return the 7 Strassen terms as (a_coeffs, b_coeffs, c_coeffs) each of length 4."""
    return [
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        ([0,0,1,1], [1,1,0,0], [0,0,1,-1]),
        ([1,1,0,0], [0,0,1,1], [0,1,0,1]),
        ([0,0,0,1], [0,-1,0,1], [1,0,-1,0]),
        ([1,0,0,0], [1,0,-1,0], [-1,1,0,0]),
        ([-1,0,1,0], [0,1,0,1], [0,0,0,1]),
        ([0,1,0,-1], [1,0,1,0], [1,0,0,0]),
    ]

def _build_decomposition(partitions, n, m, p):
    """
    Build a decomposition from a set of block assignments.
    Each block is (i_range, j_range, k_range, method) where method is 'strassen' or 'schoolbook'.
    For strassen: i_range, j_range, k_range must each be exactly size 2.
    For schoolbook: any sizes.
    Returns (U, V, W, rank) or None if invalid.
    """
    # Determine total rank
    rank = 0
    for (i0, i1, j0, j1, k0, k1, method) in partitions:
        di = i1 - i0
        dj = j1 - j0
        dk = k1 - k0
        if method == 'strassen':
            if di != 2 or dj != 2 or dk != 2:
                return None
            rank += 7
        else:  # schoolbook
            rank += di * dj * dk
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for (i0, i1, j0, j1, k0, k1, method) in partitions:
        di = i1 - i0
        dj = j1 - j0
        dk = k1 - k0
        
        if method == 'strassen':
            terms = _strassen_terms()
            for t in range(7):
                a_coeffs = terms[t][0]
                b_coeffs = terms[t][1]
                c_coeffs = terms[t][2]
                
                for di_idx in range(2):
                    for dj_idx in range(2):
                        U[(i0+di_idx)*m + (j0+dj_idx)][col] = a_coeffs[di_idx*2+dj_idx]
                
                for dj_idx in range(2):
                    for dk_idx in range(2):
                        V[(j0+dj_idx)*p + (k0+dk_idx)][col] = b_coeffs[dj_idx*2+dk_idx]
                
                for di_idx in range(2):
                    for dk_idx in range(2):
                        W[(k0+dk_idx)*n + (i0+di_idx)][col] = c_coeffs[di_idx*2+dk_idx]
                
                col += 1
        else:  # schoolbook
            for di_idx in range(di):
                for dj_idx in range(dj):
                    for dk_idx in range(dk):
                        U[(i0+di_idx)*m + (j0+dj_idx)][col] = 1
                        V[(j0+dj_idx)*p + (k0+dk_idx)][col] = 1
                        W[(k0+dk_idx)*n + (i0+di_idx)][col] = 1
                        col += 1
    
    if col != rank:
        return None
    
    return U, V, W, rank

def _all_partitions(n, m, p):
    """
    Generate all ways to partition [0,n), [0,m), [0,p) into contiguous segments,
    and for each combination of segment boundaries, try all assignments of
    'strassen' (only valid for 2x2x2 blocks) or 'schoolbook' to each block.
    Yields (partitions_list, total_rank) for each valid assignment.
    """
    # Generate all ways to split a range of size N into contiguous segments
    def segment_splits(N):
        """Return list of all ways to split [0,N) into contiguous segments.
        Each split is a list of (start, end) tuples."""
        if N == 0:
            yield []
            return
        if N == 1:
            yield [(0, 1)]
            return
        # Recursively: first segment ends at k (1 <= k <= N)
        for k in range(1, N + 1):
            for rest in segment_splits(N - k):
                yield [(0, k)] + [(s + k, e + k) for s, e in rest]
    
    i_splits = list(segment_splits(n))
    m_splits = list(segment_splits(m))
    p_splits = list(segment_splits(p))
    
    for isegs in i_splits:
        for mseg in m_splits:
            for pseg in p_splits:
                # For each block (i_seg, j_seg, k_seg), decide strassen or schoolbook
                # Strassen is only valid if all three segments have size exactly 2
                # We need to assign a method to each block independently
                
                # Build list of blocks with their sizes
                blocks = []
                for (i0, i1) in isegs:
                    for (j0, j1) in mseg:
                        for (k0, k1) in pseg:
                            di = i1 - i0
                            dj = j1 - j0
                            dk = k1 - k0
                            blocks.append((i0, i1, j0, j1, k0, k1, di, dj, dk))
                
                # For each block, determine possible methods
                # Strassen: only if di==2, dj==2, dk==2
                # Schoolbook: always valid
                # We want to minimize total rank, so for each block:
                #   if strassen is valid, compare 7 vs di*dj*dk
                #   choose the cheaper one
                # But wait - we should also consider that using strassen for one block
                # might interact with others. Actually no - the blocks are independent
                # in terms of the tensor decomposition (each block contributes to
                # disjoint (i,j,k) entries).
                
                # So the optimal choice per block is independent:
                # Use strassen if 7 < di*dj*dk and all sizes are 2
                # Otherwise use schoolbook
                
                partitions = []
                total_rank = 0
                for (i0, i1, j0, j1, k0, k1, di, dj, dk) in blocks:
                    if di == 2 and dj == 2 and dk == 2:
                        # Compare: strassen (7) vs schoolbook (8)
                        # Strassen is better
                        partitions.append((i0, i1, j0, j1, k0, k1, 'strassen'))
                        total_rank += 7
                    else:
                        partitions.append((i0, i1, j0, j1, k0, k1, 'schoolbook'))
                        total_rank += di * dj * dk
                
                yield partitions, total_rank

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    start_time = time.time()
    deadline = start_time + 110
    
    # Exhaustive search over all partitions
    best_rank = n * m * p  # schoolbook
    best_result = None
    
    # Fallback: pure schoolbook
    def schoolbook():
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
        return U, V, W, rank
    
    # Search all partitions
    count = 0
    for partitions, total_rank in _all_partitions(n, m, p):
        if time.time() > deadline - 10:
            break
        
        if total_rank >= best_rank:
            continue
        
        result = _build_decomposition(partitions, n, m, p)
        if result is None:
            continue
        
        U, V, W, rank = result
        count += 1
        
        # Verify
        if _verify(U, V, W, n, m, p):
            best_rank = rank
            best_result = (U, V, W, rank)
    
    if best_result is not None:
        U, V, W, rank = best_result
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    U, V, W, rank = schoolbook()
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
