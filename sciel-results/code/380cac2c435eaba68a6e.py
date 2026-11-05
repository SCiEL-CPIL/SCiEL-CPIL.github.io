import numpy as np
import time
from functools import lru_cache
import scipy.optimize as opt

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = 4, 5, 6
    nm, mp, pn = n*m, m*p, p*n
    
    # Cache for small block factors
    small_block_factors = {}
    
    def find_small_factors(ni, mj, pk, target_rank):
        """
        Attempts to find factors U, V, W for ni x mj x pk multiplication
        with the given target_rank.
        Returns (U, V, W) as lists of lists, or None if failed.
        """
        if (ni, mj, pk, target_rank) in small_block_factors:
            return small_block_factors[(ni, mj, pk, target_rank)]
            
        # Skip if not a "small" block we care about
        if ni > 3 or mj > 3 or pk > 3:
            return None
            
        # We need to solve for U (ni*mj x R), V (mj*pk x R), W (pk*ni x R)
        # such that sum_r U[i*mj+j, r] * V[j*pk+k, r] * W[k*ni+i, r] == 1 for all i,j,k
        # and 0 otherwise.
        
        # Flatten U, V, W into a single vector x
        # x = [U_flat, V_flat, W_flat]
        # This is a non-convex problem.
        
        # For very small cases, we can try a grid search or random restarts.
        # Given the low dimension, we can use a simple local optimization.
        
        def error_func(x):
            R = target_rank
            U_flat = x[:ni*mj*R]
            V_flat = x[ni*mj*R : ni*mj*R + mj*pk*R]
            W_flat = x[ni*mj*R + mj*pk*R : ]
            
            U = U_flat.reshape((ni*mj, R))
            V = V_flat.reshape((mj*pk, R))
            W = W_flat.reshape((pk*ni, R))
            
            err = 0.0
            for i in range(ni):
                for j in range(mj):
                    for k in range(pk):
                        val = np.sum(U[i*mj+j, :] * V[j*pk+k, :] * W[k*ni+i, :])
                        # Target is 1
                        err += (val - 1.0)**2
            return err
        
        # Initialize
        # Strategy: Random initialization with small values
        best_x = None
        best_err = float('inf')
        
        num_restarts = 20
        for _ in range(num_restarts):
            x0 = np.random.randn(ni*mj*R + mj*pk*R + pk*ni*R) * 0.5
            # Result of minimize
            res = opt.minimize(error_func, x0, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4})
            if res.fun < best_err:
                best_err = res.fun
                best_x = res.x
                
        if best_err < 1e-6:
            R = target_rank
            U_flat = best_x[:ni*mj*R]
            V_flat = best_x[ni*mj*R : ni*mj*R + mj*pk*R]
            W_flat = best_x[ni*mj*R + mj*pk*R : ]
            
            U = U_flat.reshape((ni*mj, R))
            V = V_flat.reshape((mj*pk, R))
            W = W_flat.reshape((pk*ni, R))
            
            # Round to nearest 0.5
            U = np.round(U * 2) / 2
            V = np.round(V * 2) / 2
            W = np.round(W * 2) / 2
            
            # Verify after rounding
            err = 0.0
            for i in range(ni):
                for j in range(mj):
                    for k in range(pk):
                        val = np.sum(U[i*mj+j, :] * V[j*pk+k, :] * W[k*ni+i, :])
                        err += (val - 1.0)**2
                        
            if err < 1e-6:
                small_block_factors[(ni, mj, pk, target_rank)] = (U.tolist(), V.tolist(), W.tolist())
                return small_block_factors[(ni, mj, pk, target_rank)]
                
        return None

    @lru_cache(maxsize=None)
    def compute_rank(ni, mj, pk):
        if ni == 2 and mj == 2 and pk == 2:
            return 7
        if ni <= 2 and mj <= 2 and pk <= 2:
            return ni * mj * pk
        
        # Check for known improved small blocks
        # 2x2x3 rank 10
        if (ni, mj, pk) == (2, 2, 3) or (ni, mj, pk) == (2, 3, 2) or (ni, mj, pk) == (3, 2, 2):
             # Note: Dimensions are ordered (n, m, p) in compute_rank
             # My find_small_factors expects (ni, mj, pk)
             if find_small_factors(ni, mj, pk, 10) is not None:
                 return 10
                 
        # 2x3x3 rank 12
        if (ni, mj, pk) == (2, 3, 3) or (ni, mj, pk) == (3, 2, 3) or (ni, mj, pk) == (3, 3, 2):
             if find_small_factors(ni, mj, pk, 12) is not None:
                 return 12
                 
        # 3x3x3 rank 23 (Winograd)
        if (ni, mj, pk) == (3, 3, 3):
             if find_small_factors(ni, mj, pk, 23) is not None:
                 return 23

        best = ni * mj * pk
        if ni > 2:
            for split in range(1, ni):
                best = min(best, compute_rank(split, mj, pk) + compute_rank(ni-split, mj, pk))
        if mj > 2:
            for split in range(1, mj):
                best = min(best, compute_rank(ni, split, pk) + compute_rank(ni, mj-split, pk))
        if pk > 2:
            for split in range(1, pk):
                best = min(best, compute_rank(ni, mj, split) + compute_rank(ni, mj, pk-split))
        return best
    
    total_rank = compute_rank(n, m, p)
    
    U = np.zeros((nm, total_rank), dtype=float)
    V = np.zeros((mp, total_rank), dtype=float)
    W = np.zeros((pn, total_rank), dtype=float)
    
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
        
    def add_small_block(U, V, W, col, i_off, j_off, k_off, ni, mj, pk, rank):
        factors = find_small_factors(ni, mj, pk, rank)
        if factors is None:
            return add_schoolbook(U, V, W, col, i_off, j_off, k_off, ni, mj, pk)
        U_f, V_f, W_f = factors
        for r in range(rank):
            for ii in range(ni):
                for jj in range(mj):
                    if U_f[ii*mj+jj][r] != 0:
                        U[(i_off+ii)*m + (j_off+jj), col+r] += U_f[ii*mj+jj][r]
            for jj in range(mj):
                for kk in range(pk):
                    if V_f[jj*pk+kk][r] != 0:
                        V[(j_off+jj)*p + (k_off+kk), col+r] += V_f[jj*pk+kk][r]
            for kk in range(pk):
                for ii in range(ni):
                    if W_f[kk*ni+ii][r] != 0:
                        W[(k_off+kk)*n + (i_off+ii), col+r] += W_f[kk*ni+ii][r]
        return col + rank

    @lru_cache(maxsize=None)
    def best_split(ni, mj, pk):
        if ni == 2 and mj == 2 and pk == 2:
            return None
        if ni <= 2 and mj <= 2 and pk <= 2:
            return None
            
        # Check for known improved small blocks
        if (ni, mj, pk) == (2, 2, 3) or (ni, mj, pk) == (2, 3, 2) or (ni, mj, pk) == (3, 2, 2):
             if find_small_factors(ni, mj, pk, 10) is not None:
                 return ('small', 10)
        if (ni, mj, pk) == (2, 3, 3) or (ni, mj, pk) == (3, 2, 3) or (ni, mj, pk) == (3, 3, 2):
             if find_small_factors(ni, mj, pk, 12) is not None:
                 return ('small', 12)
        if (ni, mj, pk) == (3, 3, 3):
             if find_small_factors(ni, mj, pk, 23) is not None:
                 return ('small', 23)

        best_rank = ni * mj * pk
        best_choice = None
        if ni > 2:
            for split in range(1, ni):
                r = compute_rank(split, mj, pk) + compute_rank(ni-split, mj, pk)
                if r < best_rank:
                    best_rank = r
                    best_choice = ('ni', split)
        if mj > 2:
            for split in range(1, mj):
                r = compute_rank(ni, split, pk) + compute_rank(ni, mj-split, pk)
                if r < best_rank:
                    best_rank = r
                    best_choice = ('mj', split)
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
            
        if choice[0] == 'small':
            return add_small_block(U, V, W, col, i_off, j_off, k_off, ni, mj, pk, choice[1])
            
        dim, split = choice
        if dim == 'ni':
            col = decompose(split, mj, pk, i_off, j_off, k_off, U, V, W, col)
            col = decompose(ni-split, mj, pk, i_off+split, j_off, k_off, U, V, W, col)
        elif dim == 'mj':
            col = decompose(ni, split, pk, i_off, j_off, k_off, U, V, W, col)
            col = decompose(ni, mj-split, pk, i_off, j_off+split, k_off, U, V, W, col)
        else:
            col = decompose(ni, mj, split, i_off, j_off, k_off, U, V, W, col)
            col = decompose(ni, mj, pk-split, i_off, j_off, k_off+split, U, V, W, col)
        return col
    
    col = decompose(n, m, p, 0, 0, 0, U, V, W, 0)
    assert col == total_rank, f"Column mismatch: {col} != {total_rank}"
    
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
