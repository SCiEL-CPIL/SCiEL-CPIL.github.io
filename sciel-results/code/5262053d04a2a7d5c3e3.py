import time
import numpy as np

_DIMENSIONS = [3, 4, 7]
_DEADLINE = 140.0

def construct():
    _t0 = time.time()
    n, m, p = _DIMENSIONS
    
    # All possible 2x2x2 blocks: (i0,i1, j0,j1, k0,k1)
    # i in [0..n-1], j in [0..m-1], k in [0..p-1]
    # Block uses rows i0,i0+1; cols j0,j0+1; depth k0,k0+1
    blocks = []
    for i0 in range(n - 1):
        for j0 in range(m - 1):
            for k0 in range(p - 1):
                blocks.append((i0, i0 + 1, j0, j0 + 1, k0, k0 + 1))
    
    # All entries to cover: (i, j, k) for i in [0,n), j in [0,m), k in [0,p)
    all_entries = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                all_entries.add((i, j, k))
    
    # For each block, which entries it covers
    block_coverage = []
    for b in blocks:
        i0, i1, j0, j1, k0, k1 = b
        covered = set()
        for ii in (i0, i1):
            for jj in (j0, j1):
                for kk in (k0, k1):
                    covered.add((ii, jj, kk))
        block_coverage.append(covered)
    
    # Greedy set cover: pick block covering most uncovered entries
    uncovered = set(all_entries)
    selected_blocks = []
    remaining = list(range(len(blocks)))
    
    while uncovered and remaining:
        if time.time() - _t0 > _DEADLINE - 5:
            break
        best_idx = -1
        best_count = -1
        for idx in remaining:
            count = len(uncovered & block_coverage[idx])
            if count > best_count:
                best_count = count
                best_idx = idx
        if best_idx < 0 or best_count == 0:
            break
        selected_blocks.append(blocks[best_idx])
        uncovered -= block_coverage[best_idx]
        remaining.remove(best_idx)
    
    # Uncovered entries get singleton terms
    singleton_entries = list(uncovered)
    
    rank = 7 * len(selected_blocks) + len(singleton_entries)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    for b in selected_blocks:
        i0, i1, j0, j1, k0, k1 = b
        # Strassen 7 terms
        # M1 = (A11+A22)(B11+B22)
        U[i0 * m + j0][col] = 1
        U[i1 * m + j1][col] = 1
        V[j0 * p + k0][col] = 1
        V[j1 * p + k1][col] = 1
        W[k0 * n + i0][col] = 1
        W[k1 * n + i1][col] = 1
        col += 1
        
        # M2 = A22(B21-B11)
        U[i1 * m + j0][col] = 1
        U[i1 * m + j1][col] = 1
        V[j0 * p + k0][col] = 1
        W[k0 * n + i1][col] = 1
        W[k1 * n + i1][col] = -1
        col += 1
        
        # M3 = (A11-A22)(B22+B12)
        U[i0 * m + j0][col] = 1
        V[j0 * p + k1][col] = 1
        V[j1 * p + k1][col] = -1
        W[k1 * n + i0][col] = 1
        W[k1 * n + i1][col] = 1
        col += 1
        
        # M4 = (A21+A22)B11
        U[i1 * m + j1][col] = 1
        V[j1 * p + k0][col] = 1
        V[j0 * p + k0][col] = -1
        W[k0 * n + i0][col] = 1
        W[k0 * n + i1][col] = 1
        col += 1
        
        # M5 = A11(B12-B22)
        U[i0 * m + j0][col] = 1
        U[i0 * m + j1][col] = 1
        V[j1 * p + k1][col] = 1
        W[k0 * n + i0][col] = -1
        W[k1 * n + i0][col] = 1
        col += 1
        
        # M6 = (A12-A11)(B11+B21)
        U[i1 * m + j0][col] = 1
        U[i0 * m + j0][col] = -1
        V[j0 * p + k0][col] = 1
        V[j0 * p + k1][col] = 1
        W[k1 * n + i1][col] = 1
        col += 1
        
        # M7 = (A21-A12)(B12+B22)
        U[i0 * m + j1][col] = 1
        U[i1 * m + j1][col] = -1
        V[j1 * p + k0][col] = 1
        V[j1 * p + k1][col] = 1
        W[k0 * n + i0][col] = 1
        col += 1
    
    # Singleton entries
    for (i, j, k) in singleton_entries:
        U[i * m + j][col] = 1
        V[j * p + k][col] = 1
        W[k * n + i][col] = 1
        col += 1
    
    assert col == rank, f"Column count {col} != rank {rank}"
    
    # Verification
    for i in range(n):
        if time.time() - _t0 > _DEADLINE:
            break
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                total = 0
                for c in range(rank):
                    total += U[u_idx][c] * V[v_idx][c] * W[w_idx][c]
                if total != 1:
                    raise ValueError(f"Decomposition failed at ({i},{j},{k}): got {total}")
    
    return {"U": U, "V": V, "W": W}
