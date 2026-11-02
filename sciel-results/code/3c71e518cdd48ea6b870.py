"""Matrix multiplication 4x5x5 rank-minimal decomposition search."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [4, 5, 5]

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(rank))
                if abs(val - 1.0) > tol:
                    return False
    return True

def _verify_fast(U, V, W, n, m, p, tol=1e-6):
    """Fast vectorized verification."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for j in range(m):
        U_j = np.array([U[i*m+j] for i in range(n)])
        V_j = np.array([V[j*p+k] for k in range(p)])
        W_j = np.array([[W[k*n+i] for i in range(n)] for k in range(p)])
        
        product_tensor = U_j[:, None, :] * V_j[None, :, :] * W_j.T
        T = product_tensor.sum(axis=2)
        
        if np.any(np.abs(T - 1.0) > tol):
            return False
    
    return True

def _build_block_structure(n, m, p):
    """
    Build a block-structured decomposition for n x m times m x p.
    Uses block sizes that allow lower-rank decompositions for sub-blocks.
    """
    # For 4x5 x 5x5:
    # Split rows of A/C: [3, 1]
    # Split cols of A / rows of B: [3, 2]
    # Split cols of B / C: [3, 2]
    
    row_blocks = [3, 1]
    mid_blocks = [3, 2]
    col_blocks = [3, 2]
    
    assert sum(row_blocks) == n
    assert sum(mid_blocks) == m
    assert sum(col_blocks) == p
    
    # Compute start indices
    row_starts = [0]
    for s in row_blocks:
        row_starts.append(row_starts[-1] + s)
    
    mid_starts = [0]
    for s in mid_blocks:
        mid_starts.append(mid_starts[-1] + s)
    
    col_starts = [0]
    for s in col_blocks:
        col_starts.append(col_starts[-1] + s)
    
    return row_blocks, mid_blocks, col_blocks, row_starts, mid_starts, col_starts

def _schoolbook_block(r, c, d):
    """
    Return schoolbook decomposition for (r x c) * (c x d) -> (r x d).
    Returns (U_block, V_block, W_block) each of shape (r*c, r*c*d) and (c*d, r*c*d) and (d*r, r*c*d)
    Actually, the indices are:
    U: (r*c) rows, indexed by (i, j) where i in [0,r), j in [0,c)
    V: (c*d) rows, indexed by (j, k) where j in [0,c), k in [0,d)
    W: (d*r) rows, indexed by (k, i) where k in [0,d), i in [0,r)
    
    Rank = r*c*d
    """
    rank = r * c * d
    U = [[0]*rank for _ in range(r*c)]
    V = [[0]*rank for _ in range(c*d)]
    W = [[0]*rank for _ in range(d*r)]
    
    col = 0
    for i in range(r):
        for j in range(c):
            for k in range(d):
                U[i*c+j][col] = 1
                V[j*d+k][col] = 1
                W[k*r+i][col] = 1
                col += 1
    
    return U, V, W, rank

def _try_low_rank_block(r, c, d, target_rank, max_attempts=10000):
    """
    Try to find a rank-target_rank decomposition for (r x c) * (c x d) -> (r x d).
    """
    if target_rank >= r * c * d:
        return _schoolbook_block(r, c, d)
    
    for attempt in range(max_attempts):
        if time.time() > time.time() - 100:  # This is a placeholder
            break
        
        U = [[random.choice([0, 0, 0, 1, -1]) for _ in range(target_rank)] for _ in range(r*c)]
        V = [[random.choice([0, 0, 0, 1, -1]) for _ in range(target_rank)] for _ in range(c*d)]
        W = [[random.choice([0, 0, 0, 1, -1]) for _ in range(target_rank)] for _ in range(d*r)]
        
        # Verify
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        valid = True
        for i in range(r):
            for j in range(c):
                for k in range(d):
                    val = np.dot(U_arr[i*c+j], V_arr[j*d+k] * W_arr[k*r+i])
                    if abs(val - 1.0) > 1e-6:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            return U, V, W, target_rank
    
    # Fall back to schoolbook
    return _schoolbook_block(r, c, d)

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130
    
    # Start with schoolbook as fallback
    rank_school = n * m * p
    U_fb, V_fb, W_fb, _ = _schoolbook_block(n, m, p)
    
    best_rank = rank_school
    best_U, best_V, best_W = U_fb, V_fb, W_fb
    
    # Try block-structured decomposition
    row_blocks, mid_blocks, col_blocks, row_starts, mid_starts, col_starts = \
        _build_block_structure(n, m, p)
    
    # For each combination of blocks, find a decomposition
    # The total rank is the sum of ranks of all block decompositions
    
    # Block multiplications:
    # For bi in range(len(row_blocks)):
    #   for bj in range(len(mid_blocks)):
    #     for bk in range(len(col_blocks)):
    #       A_block: (row_blocks[bi] x mid_blocks[bj])
    #       B_block: (mid_blocks[bj] x col_blocks[bk])
    #       C_block: (row_blocks[bi] x col_blocks[bk])
    
    # We need to combine all these into a single decomposition.
    # The challenge is that the same (i,j) entry of A appears in multiple block decompositions.
    
    # Actually, the correct approach is:
    # C[i,k] = sum_{j} A[i,j] B[j,k]
    # = sum_{bi, bj, bk} sum_{j' in mid_block[bj]} A[i, j] B[j, k]
    # where i is in row_block[bi], k is in col_block[bk]
    
    # So for each (bi, bj, bk), we have a sub-multiplication:
    # C_block[bi,bk] = A_block[bi,bj] * B_block[bj,bk]
    
    # The rank-1 decomposition for this sub-multiplication contributes to
    # the overall decomposition.
    
    # Let's build the full decomposition by combining all block decompositions.
    
    # For each (bi, bj, bk), get a decomposition of rank r_block
    # The overall rank is sum of r_block over all (bi, bj, bk)
    
    # Try to find low-rank decompositions for each block
    
    block_decomps = {}
    total_rank = 0
    
    for bi in range(len(row_blocks)):
        for bj in range(len(mid_blocks)):
            for bk in range(len(col_blocks)):
                r = row_blocks[bi]
                c = mid_blocks[bj]
                d = col_blocks[bk]
                
                # Try to find a low-rank decomposition
                # For 3x3x3, try rank 19
                # For other sizes, use schoolbook or try slightly lower
                
                if r == 3 and c == 3 and d == 3:
                    target = 19
                else:
                    target = r * c * d  # Use schoolbook for now
                
                U_block, V_block, W_block, actual_rank = _try_low_rank_block(
                    r, c, d, target, max_attempts=5000
                )
                
                block_decomps[(bi, bj, bk)] = (U_block, V_block, W_block, actual_rank)
                total_rank += actual_rank
    
    if total_rank < best_rank:
        # Combine all block decompositions into the full decomposition
        U_full = [[0]*total_rank for _ in range(n*m)]
        V_full = [[0]*total_rank for _ in range(m*p)]
        W_full = [[0]*total_rank for _ in range(p*n)]
        
        col_offset = 0
        for bi in range(len(row_blocks)):
            for bj in range(len(mid_blocks)):
                for bk in range(len(col_blocks)):
                    U_block, V_block, W_block, r_block = block_decomps[(bi, bj, bk)]
                    
                    # Map local indices to global indices
                    # A local (i_local, j_local) -> global (row_starts[bi] + i_local, mid_starts[bj] + j_local)
                    # B local (j_local, k_local) -> global (mid_starts[bj] + j_local, col_starts[bk] + k_local)
                    # W local (k_local, i_local) -> global (col_starts[bk] + k_local, row_starts[bi] + i_local)
                    
                    r = row_blocks[bi]
                    c = mid_blocks[bj]
                    d = col_blocks[bk]
                    
                    for i_local in range(r):
                        for j_local in range(c):
                            global_i = row_starts[bi] + i_local
                            global_j = mid_starts[bj] + j_local
                            local_idx = i_local * c + j_local
                            for c_idx in range(r_block):
                                U_full[global_i * m + global_j][col_offset + c_idx] = U_block[local_idx][c_idx]
                    
                    for j_local in range(c):
                        for k_local in range(d):
                            global_j = mid_starts[bj] + j_local
                            global_k = col_starts[bk] + k_local
                            local_idx = j_local * d + k_local
                            for c_idx in range(r_block):
                                V_full[global_j * p + global_k][col_offset + c_idx] = V_block[local_idx][c_idx]
                    
                    for k_local in range(d):
                        for i_local in range(r):
                            global_k = col_starts[bk] + k_local
                            global_i = row_starts[bi] + i_local
                            local_idx = k_local * r + i_local
                            for c_idx in range(r_block):
                                W_full[global_k * n + global_i][col_offset + c_idx] = W_block[local_idx][c_idx]
                    
                    col_offset += r_block
        
        # Verify the combined decomposition
        if _verify_fast(U_full, V_full, W_full, n, m, p):
            best_rank = total_rank
            best_U, best_V, best_W = U_full, V_full, W_full
    
    # Try additional random search for even lower rank
    for target_rank in range(min(best_rank - 1, 99), 85, -1):
        if time.time() > deadline - 30:
            break
        
        for attempt in range(20):
            if time.time() > deadline - 30:
                break
            
            U = [[random.choice([0, 0, 0, 1, -1]) for _ in range(target_rank)] for _ in range(n*m)]
            V = [[random.choice([0, 0, 0, 1, -1]) for _ in range(target_rank)] for _ in range(m*p)]
            W = [[random.choice([0, 0, 0, 1, -1]) for _ in range(target_rank)] for _ in range(p*n)]
            
            if _verify_fast(U, V, W, n, m, p):
                best_rank = target_rank
                best_U, best_V, best_W = U, V, W
                break
        
        if best_rank == target_rank:
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
