import time
import numpy as np
from fractions import Fraction
from itertools import combinations

DIMENSIONS = [3, 5, 5]

def construct():
    start = time.time()
    deadline = 140  # safety margin before 150s kill
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    # All possible 2x2x2 block positions
    all_blocks = [(i0, j0, k0) for i0 in range(n-1) for j0 in range(m-1) for k0 in range(p-1)]
    
    def block_cells(i0, j0, k0):
        return frozenset((i0+di, j0+dj, k0+dk) for di in range(2) for dj in range(2) for dk in range(2))
    
    # Precompute cells for each block
    block_cell_sets = [block_cells(*b) for b in all_blocks]
    
    def total_rank_for_blocks(block_indices):
        covered = set()
        for idx in block_indices:
            covered |= block_cell_sets[idx]
        return len(block_indices) * 7 + (n*m*p - len(covered))
    
    # Search for best block combination
    best_rank = n * m * p
    best_block_indices = []
    
    # Try all combinations of up to 6 blocks
    for size in range(1, 7):
        if time.time() > deadline:
            break
        for combo in combinations(range(len(all_blocks)), size):
            r = total_rank_for_blocks(combo)
            if r < best_rank:
                best_rank = r
                best_block_indices = list(combo)
    
    # Convert to block tuples
    best_blocks = [all_blocks[i] for i in best_block_indices]
    
    # If no improvement found, use original 4 blocks
    if not best_blocks:
        best_blocks = [(0,0,0), (0,2,0), (0,0,2), (0,2,2)]
        best_rank = 71
    
    # Build the decomposition
    total_rank = best_rank
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(i0, j0, k0, col_offset):
        c = col_offset + 0
        U[i0*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        c = col_offset + 1
        U[(i0+1)*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += -1
        
        c = col_offset + 2
        U[i0*m + j0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += -1
        W[(k0+1)*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        c = col_offset + 3
        U[(i0+1)*m + (j0+1)][c] += 1
        V[(j0+1)*p + k0][c] += 1
        V[j0*p + k0][c] += -1
        W[k0*n + i0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        
        c = col_offset + 4
        U[i0*m + j0][c] += 1
        U[i0*m + (j0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += -1
        W[(k0+1)*n + i0][c] += 1
        
        c = col_offset + 5
        U[(i0+1)*m + j0][c] += 1
        U[i0*m + j0][c] += -1
        V[j0*p + k0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        c = col_offset + 6
        U[i0*m + (j0+1)][c] += 1
        U[(i0+1)*m + (j0+1)][c] += -1
        V[(j0+1)*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1

    for i0, j0, k0 in best_blocks:
        add_strassen_block(i0, j0, k0, col)
        col += 7
    
    covered = set()
    for i0, j0, k0 in best_blocks:
        covered |= block_cells(i0, j0, k0)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U[i*m+j][col] += 1
                    V[j*p+k][col] += 1
                    W[k*n+i][col] += 1
                    col += 1
    
    # Verify
    try:
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        target = np.zeros((n*m, m*p, p*n), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    target[i*m+j, j*p+k, k*n+i] = 1
        T_decomp = np.einsum('ar,br,cr->abc', U_np, V_np, W_np)
        if not np.allclose(T_decomp, target, atol=1e-10):
            raise ValueError("Verification failed")
    except Exception:
        rank = n * m * p
        U_f = [[0]*rank for _ in range(n*m)]
        V_f = [[0]*rank for _ in range(m*p)]
        W_f = [[0]*rank for _ in range(p*n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_f[i*m+j][c] = 1
                    V_f[j*p+k][c] = 1
                    W_f[k*n+i][c] = 1
                    c += 1
        return {"U": U_f, "V": V_f, "W": W_f}
    
    if time.time() > deadline:
        rank = n * m * p
        U_f = [[0]*rank for _ in range(n*m)]
        V_f = [[0]*rank for _ in range(m*p)]
        W_f = [[0]*rank for _ in range(p*n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_f[i*m+j][c] = 1
                    V_f[j*p+k][c] = 1
                    W_f[k*n+i][c] = 1
                    c += 1
        return {"U": U_f, "V": V_f, "W": W_f}
    
    return {"U": U, "V": V, "W": W}
