import numpy as np
import time
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = 4, 5, 6
    nm, mp, pn = n*m, m*p, p*n
    
    # We'll use a recursive decomposition with optimal split order
    # Try all valid split orders and pick the one minimizing rank
    @lru_cache(maxsize=None)
    def compute_rank(ni, mj, pk):
        if ni == 2 and mj == 2 and pk == 2:
            return 7
        if ni <= 2 and mj <= 2 and pk <= 2:
            return ni * mj * pk
        # Try all possible splits and take the minimum
        best = ni * mj * pk  # schoolbook fallback
        # Try splitting ni
        if ni > 2:
            for split in range(1, ni):
                best = min(best, compute_rank(split, mj, pk) + compute_rank(ni-split, mj, pk))
        # Try splitting mj
        if mj > 2:
            for split in range(1, mj):
                best = min(best, compute_rank(ni, split, pk) + compute_rank(ni, mj-split, pk))
        # Try splitting pk
        if pk > 2:
            for split in range(1, pk):
                best = min(best, compute_rank(ni, mj, split) + compute_rank(ni, mj, pk-split))
        return best
    
    total_rank = compute_rank(n, m, p)
    
    U = np.zeros((nm, total_rank), dtype=float)
    V = np.zeros((mp, total_rank), dtype=float)
    W = np.zeros((pn, total_rank), dtype=float)
    
    # Strassen terms for 2x2x2
    strassen_terms = [
        ([[1,0],[0,1]], [[1,0],[0,1]], [[1,0],[0,1]]),
        ([[0,0],[1,1]], [[1,0],[0,0]], [[0,1],[0,-1]]),
        ([[1,0],[0,0]], [[0,1],[0,-1]], [[0,0],[1,1]]),
        ([[0,0],[0,1]], [[-1,0],[1,0]], [[1,1],[0,0]]),
        ([[1,1],[0,0]], [[0,0],[0,1]], [[-1,0],[1,0]]),
        ([[-1,0],[1,0]], [[1,1],[0,0]], [[0,0],[0,1]]),
        ([[0,1],[0,-1]], [[0,0],[1,1]], [[1,0],[0,0]]),
    ]
    
    def add_strassen(U, V, W, col, i_off, j_off, k_off):
        for u_mat, v_mat, w_mat in strassen_terms:
            for ii in range(2):
                for jj in range(2):
                    if u_mat[ii][jj] != 0:
                        U[(i_off+ii)*m + (j_off+jj), col] += u_mat[ii][jj]
            for jj in range(2):
                for kk in range(2):
                    if v_mat[jj][kk] != 0:
                        V[(j_off+jj)*p + (k_off+kk), col] += v_mat[jj][kk]
            for kk in range(2):
                for ii in range(2):
                    if w_mat[kk][ii] != 0:
                        W[(k_off+kk)*n + (i_off+ii), col] += w_mat[kk][ii]
            col += 1
        return col
    
    def add_schoolbook(U, V, W, col, i_off, j_off, k_off, ni, mj, pk):
        for ii in range(ni):
            for jj in range(mj):
                for kk in range(pk):
                    U[(i_off+ii)*m + (j_off+jj), col] = 1.0
                    V[(j_off+jj)*p + (k_off+kk), col] = 1.0
                    W[(k_off+kk)*n + (i_off+ii), col] = 1.0
                    col += 1
        return col
    
    @lru_cache(maxsize=None)
    def best_split(ni, mj, pk):
        """Return (dim_to_split, split_point) that minimizes total rank."""
        if ni == 2 and mj == 2 and pk == 2:
            return None
        if ni <= 2 and mj <= 2 and pk <= 2:
            return None
        best_rank = ni * mj * pk
        best_choice = None
        # Try splitting ni
        if ni > 2:
            for split in range(1, ni):
                r = compute_rank(split, mj, pk) + compute_rank(ni-split, mj, pk)
                if r < best_rank:
                    best_rank = r
                    best_choice = ('ni', split)
        # Try splitting mj
        if mj > 2:
            for split in range(1, mj):
                r = compute_rank(ni, split, pk) + compute_rank(ni, mj-split, pk)
                if r < best_rank:
                    best_rank = r
                    best_choice = ('mj', split)
        # Try splitting pk
        if pk > 2:
            for split in range(1, pk):
                r = compute_rank(ni, mj, split) + compute_rank(ni, mj, pk-split)
                if r < best_rank:
                    best_rank = r
                    best_choice = ('pk', split)
        return best_choice
    
    def decompose(ni, mj, pk, i_off, j_off, k_off, U, V, W, col):
        if time.time() > deadline:
            raise TimeoutError()
        if ni == 2 and mj == 2 and pk == 2:
            return add_strassen(U, V, W, col, i_off, j_off, k_off)
        if ni <= 2 and mj <= 2 and pk <= 2:
            return add_schoolbook(U, V, W, col, i_off, j_off, k_off, ni, mj, pk)
        choice = best_split(ni, mj, pk)
        if choice is None:
            return add_schoolbook(U, V, W, col, i_off, j_off, k_off, ni, mj, pk)
        dim, split = choice
        if dim == 'ni':
            col = decompose(split, mj, pk, i_off, j_off, k_off, U, V, W, col)
            col = decompose(ni-split, mj, pk, i_off+split, j_off, k_off, U, V, W, col)
        elif dim == 'mj':
            col = decompose(ni, split, pk, i_off, j_off, k_off, U, V, W, col)
            col = decompose(ni, mj-split, pk, i_off, j_off+split, k_off, U, V, W, col)
        else:  # dim == 'pk'
            col = decompose(ni, mj, split, i_off, j_off, k_off, U, V, W, col)
            col = decompose(ni, mj, pk-split, i_off, j_off, k_off+split, U, V, W, col)
        return col
    
    col = decompose(n, m, p, 0, 0, 0, U, V, W, 0)
    assert col == total_rank, f"Column mismatch: {col} != {total_rank}"
    
    # Verify
    error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(total_rank))
                if abs(val - 1.0) > 1e-9:
                    error += abs(val - 1.0)
    
    if error > 1e-6:
        rank_school = n * m * p
        U_s = np.zeros((nm, rank_school), dtype=float)
        V_s = np.zeros((mp, rank_school), dtype=float)
        W_s = np.zeros((pn, rank_school), dtype=float)
        col2 = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i*m+j, col2] = 1.0
                    V_s[j*p+k, col2] = 1.0
                    W_s[k*n+i, col2] = 1.0
                    col2 += 1
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
