"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

_START_TIME = time.time()
_TIMEOUT = 130  # safety margin before 150s kill

# Strassen 2x2x2 decomposition (rank 7)
# Each entry: (u_matrix, v_matrix, w_matrix)
# U: 2x2 (i,j), V: 2x2 (j,k), W: 2x2 (k,i)
STRASSEN_222 = [
    ([[1,0],[0,1]], [[1,0],[0,1]], [[1,0],[0,1]]),
    ([[0,0],[1,1]], [[1,0],[0,0]], [[1,0],[0,-1]]),
    ([[1,0],[0,0]], [[0,1],[1,-1]], [[0,1],[1,1]]),
    ([[0,0],[0,1]], [[1,0],[-1,0]], [[1,0],[1,0]]),
    ([[1,0],[0,1]], [[0,0],[0,1]], [[-1,0],[0,1]]),
    ([[0,0],[1,-1]], [[1,0],[0,1]], [[0,0],[0,1]]),
    ([[0,1],[1,-1]], [[1,0],[0,1]], [[1,0],[0,0]]),
]

def build_j_split(n, m, p, j_groups):
    """
    Build U, V, W for a block decomposition of matrix multiplication.
    j_groups: list of tuples of j-indices. Size 1 = schoolbook, size 2 = Strassen-like.
    Returns (U, V, W, rank) or None if invalid.
    """
    rank = 0
    for grp in j_groups:
        sz = len(grp)
        if sz == 1:
            rank += n * p
        elif sz == 2:
            # For 2 j-indices, use 7 terms per k-pair if p is even, else handle carefully
            # Actually for 2x2x2 Strassen, we need p to be split into pairs
            # But here we have fixed p=6, so for each pair of j's, we do Strassen over k pairs
            # The 2x2x2 tensor has rank 7, so for each pair of j's we get 7 * (p//2) terms? No.
            # Let me reconsider.
            # Actually the Strassen decomposition is for the 2x2x2 tensor.
            # If we fix j to be a pair, the tensor is 2x2x2 (i, j_pair, k_pair) where k_pair = p//2 = 3
            # That doesn't work directly.
            
            # Let me think differently. The tensor M_{n,m,p} has entries for (i,j,k).
            # If we group j into pairs, for each pair (j0,j1), the sub-tensor is n x 2 x p.
            # For n=2, this is 2x2x6. We can decompose 2x2x6 as Kronecker of 2x2x2 (Strassen, rank 7) 
            # and 1x1x3 (schoolbook, rank 3)? No, that gives 7*3=21.
            # Or we can think of 2x2x6 = sum over k-pairs of 2x2x2 blocks, each rank 7, total 3*7=21.
            # Or we can use a different decomposition.
            
            # Actually, the standard approach: if we have j-group of size 2, the subproblem is 
            # multiplying a 2x2 matrix by a 2x6 matrix, giving 2x6 output.
            # This is M_{2,2,6}. We can decompose this as: for each k-pair (k0,k1), we have a 
            # 2x2x2 subproblem, which has rank 7 (Strassen). There are p/2 = 3 such pairs.
            # Total: 3 * 7 = 21.
            rank += 7 * (p // 2)
        else:
            return None
    return rank

def build_block_decomp(n, m, p, j_groups):
    """
    Build U, V, W for a block decomposition.
    """
    rank = build_j_split(n, m, p, j_groups)
    if rank is None:
        return None
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for grp in j_groups:
        sz = len(grp)
        if sz == 1:
            j0 = grp[0]
            for i in range(n):
                for k in range(p):
                    U[i * m + j0, col] = 1.0
                    V[j0 * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        elif sz == 2:
            j0, j1 = grp
            for k_pair in range(p // 2):
                k0 = k_pair * 2
                k1 = k_pair + 1
                for (u_mat, v_mat, w_mat) in STRASSEN_222:
                    # u_mat: 2x2 (i, j_local) where j_local 0->j0, 1->j1
                    # v_mat: 2x2 (j_local, k_local) where k_local 0->k0, 1->k1
                    # w_mat: 2x2 (k_local, i)
                    for li in range(2):
                        for lj in range(2):
                            if u_mat[li][lj] != 0:
                                U[li * m + (j0 + lj), col] += u_mat[li][lj]
                    for lj in range(2):
                        for lk in range(2):
                            if v_mat[lj][lk] != 0:
                                V[(j0 + lj) * p + (k0 + lk), col] += v_mat[lj][lk]
                    for lk in range(2):
                        for li in range(2):
                            if w_mat[lk][li] != 0:
                                W[(k0 + lk) * n + li, col] += w_mat[lk][li]
                    col += 1
    return U, V, W, col

def verify_decomp(n, m, p, U, V, W):
    """Check if the decomposition is exact."""
    Ua = U.reshape(n, m, -1)
    Va = V.reshape(m, p, -1)
    Wa = W.reshape(p, n, -1)
    S = np.einsum('ijt,jkt,kit->ijk', Ua, Va, Wa)
    if not np.allclose(S, 1.0, atol=1e-9):
        return False
    return True

def generate_j_splits(m):
    """Generate all partitions of {0,...,m-1} into groups of size 1 or 2."""
    indices = list(range(m))
    
    def gen(rest):
        if not rest:
            yield []
            return
        # Try each element as first in group
        for i in range(len(rest)):
            elem = rest[i]
            r = rest[:i] + rest[i+1:]
            # Single
            for sub in gen(r):
                yield [[elem]] + sub
            # Pair with next
            if i + 1 < len(rest):
                elem2 = rest[i+1]
                r2 = rest[:i] + rest[i+2:]
                for sub in gen(r2):
                    yield [[elem, elem2]] + sub
    
    seen = set()
    results = []
    for sp in gen(indices):
        # Normalize: sort elements within each group, sort groups
        key = tuple(tuple(sorted(g)) for g in sorted(sp, key=lambda x: x[0]))
        if key not in seen:
            seen.add(key)
            results.append([list(g) for g in key])
    return results

def construct():
    n, m, p = DIMENSIONS  # 2, 5, 6
    
    splits = generate_j_splits(m)
    
    best_rank = None
    best_U = None
    best_V = None
    best_W = None
    
    for sp in splits:
        if time.time() - _START_TIME > _TIMEOUT - 5:
            break
        result = build_block_decomp(n, m, p, sp)
        if result is None:
            continue
        U, V, W, col = result
        if verify_decomp(n, m, p, U, V, W):
            if best_rank is None or col < best_rank:
                best_rank = col
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
    
    if best_rank is None:
        # Fallback: naive
        best_rank = n * m * p
        best_U = np.zeros((n * m, best_rank), dtype=float)
        best_V = np.zeros((m * p, best_rank), dtype=float)
        best_W = np.zeros((p * n, best_rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    best_U[i * m + j, col] = 1.0
                    best_V[j * p + k, col] = 1.0
                    best_W[k * n + i, col] = 1.0
                    col += 1
    
    def to_clean(arr):
        out = []
        for row in arr:
            out.append([int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row])
        return out
    
    return {"U": to_clean(best_U), "V": to_clean(best_V), "W": to_clean(best_W)}
# EVOLVE-BLOCK-END
