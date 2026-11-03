"""Slice-method decomposition for 3x4x7 matrix multiplication."""
import numpy as np
from itertools import product
from functools import lru_cache
import time

DIMENSIONS = [3, 4, 7]


def _check_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    T = np.zeros((n, m, p))
    for c in range(rank):
        Uc = U[:, c]
        Vc = V[:, c]
        Wc = W[:, c]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += Uc[i * m + j] * Vc[j * p + k] * Wc[k * n + i]
    target = np.ones((n, m, p))
    return np.allclose(T, target, atol=tol)


def _search_sub_decomposition(n, m, p, max_rank, coeff_set, timeout=30):
    """Search for rank-minimal decomposition of n x m x p tensor."""
    start_time = time.time()
    nm = n * m
    mp = m * p
    pn = p * n
    
    best = None
    
    for r in range(1, max_rank + 1):
        if time.time() - start_time > timeout:
            return best
        
        # Strategy: enumerate U columns from coeff_set, then for each U,
        # enumerate V columns, then solve for W via linear system.
        # But full enumeration is too large. Instead, use a structured approach.
        
        # For rank r, we have r columns in U, V, W.
        # Key insight: we can fix U to have a specific structure.
        # For the multiplication tensor T[i,j,k]=1, one natural approach:
        # Use "slice" type decompositions.
        
        # Alternative: use the fact that for small p, we can set up
        # the problem differently. Let's try:
        # Fix U to be identity-like or sparse, enumerate V, solve W.
        
        num_trials = 20000
        found = False
        
        for trial in range(num_trials):
            if time.time() - start_time > timeout:
                break
            
            # Random sparse U with entries from coeff_set
            U_rand = np.zeros((nm, r))
            V_rand = np.zeros((mp, r))
            
            # Make U sparse: each row has 1-2 non-zero entries
            for i in range(nm):
                nz = np.random.randint(1, 3)
                cols = np.random.choice(r, size=nz, replace=False)
                for c in cols:
                    U_rand[i, c] = np.random.choice(coeff_set)
            
            for j in range(mp):
                nz = np.random.randint(1, 3)
                cols = np.random.choice(r, size=nz, replace=False)
                for c in cols:
                    V_rand[j, c] = np.random.choice(coeff_set)
            
            # Set up linear system for W
            # Variables: W[k*n+i, c] for k in range(p), i in range(n), c in range(r)
            # Total variables: p * n * r
            # Equation for (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
            
            A = np.zeros((n * m * p, p * n * r))
            b = np.ones(n * m * p)
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for c in range(r):
                            col = (k * n + i) * r + c
                            A[eq_idx, col] = U_rand[i * m + j, c] * V_rand[j * p + k, c]
                        eq_idx += 1
            
            try:
                W_flat, residuals, rank_ls, sv = np.linalg.lstsq(A, b, rcond=None)
                if np.allclose(A @ W_flat, b, atol=1e-8):
                    W_rand = W_flat.reshape((p * n, r))
                    # Check if W entries are half-integers
                    W_rounded = np.round(W_rand * 2) / 2
                    if np.allclose(W_rounded, W_rand, atol=1e-8):
                        # Full verification
                        if _check_decomposition(U_rand, V_rand, W_rounded, n, m, p):
                            best = (r, U_rand.tolist(), V_rand.tolist(), W_rounded.tolist())
                            found = True
                            break
            except:
                pass
        
        if found:
            return best
    
    return best


def _construct_sub_decomp(n, m, p, max_rank, timeout=30):
    """Construct optimal decomposition for n x m x p."""
    coeff_set = [-2, -1, -0.5, 0, 0.5, 1, 2]
    result = _search_sub_decomposition(n, m, p, max_rank, coeff_set, timeout)
    if result:
        return result
    # Fallback: schoolbook
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
    return (rank, U, V, W)


def _combine_decompositions(blocks, sub_decomps, n, m, p):
    """Combine sub-decompositions into full decomposition."""
    total_rank = sum(r for r, _, _, _ in sub_decomps)
    
    U_full = [[0] * total_rank for _ in range(n * m)]
    V_full = [[0] * total_rank for _ in range(m * p)]
    W_full = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    for idx, (r, U_sub, V_sub, W_sub) in enumerate(sub_decomps):
        b = blocks[idx]
        k_off = sum(blocks[:idx])
        
        for i in range(n * m):
            for c in range(r):
                U_full[i][col_offset + c] = U_sub[i][c]
        
        for j in range(m):
            for kk in range(b):
                k_global = k_off + kk
                row_v = j * p + k_global
                for c in range(r):
                    V_full[row_v][col_offset + c] = V_sub[j * b + kk][c]
        
        for kk in range(b):
            k_global = k_off + kk
            for i in range(n):
                row_w = k_global * n + i
                for c in range(r):
                    W_full[row_w][col_offset + c] = W_sub[kk * n + i][c]
        
        col_offset += r
    
    return {"U": U_full, "V": V_full, "W": W_full}


def _get_block_partitions(p, min_block=1):
    """Get all partitions of p into blocks of size >= min_block."""
    partitions = []
    def _gen(remaining, current):
        if remaining == 0:
            partitions.append(list(current))
            return
        for size in range(min_block, remaining + 1):
            current.append(size)
            _gen(remaining - size, current)
            current.pop()
    _gen(p, [])
    return partitions


def construct():
    n, m, p = DIMENSIONS  # 3, 4, 7
    
    # Try multiple block partitions
    partitions = _get_block_partitions(p, min_block=1)
    # Sort by number of blocks (fewer blocks = potentially better)
    partitions.sort(key=lambda x: (len(x), -max(x)))
    
    # Limit to reasonable number of partitions to try
    partitions_to_try = partitions[:10]
    
    best_result = None
    best_total_rank = float('inf')
    
    start_time = time.time()
    time_per_partition = 20  # seconds per partition
    
    for blocks in partitions_to_try:
        if time.time() - start_time > 100:
            break
        
        sub_decomps = []
        valid = True
        for b in blocks:
            r, U, V, W = _construct_sub_decomp(n, m, b, max_rank=30, timeout=time_per_partition)
            sub_decomps.append((r, U, V, W))
        
        total_rank = sum(r for r, _, _, _ in sub_decomps)
        
        if total_rank < best_total_rank:
            result = _combine_decompositions(blocks, sub_decomps, n, m, p)
            best_result = result
            best_total_rank = total_rank
    
    if best_result is None:
        # Fallback: schoolbook
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
        best_result = {"U": U, "V": V, "W": W}
    
    return best_result
