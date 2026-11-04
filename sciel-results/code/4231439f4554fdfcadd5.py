import numpy as np
import time
from itertools import product

DIMENSIONS = [4, 5, 6]

def _strassen_2x2x2(U, V, W, i0, j0, k0, offset):
    n, m, p = 4, 5, 6
    U[i0*m + j0, offset+0] = 1
    U[(i0+1)*m + (j0+1), offset+0] = 1
    V[j0*p + k0, offset+0] = 1
    V[(j0+1)*p + (k0+1), offset+0] = 1
    W[k0*n + i0, offset+0] = 1
    W[(k0+1)*n + i0, offset+0] = 1
    W[k0*n + (i0+1), offset+0] = 1
    W[(k0+1)*n + (i0+1), offset+0] = 1
    
    U[(i0+1)*m + j0, offset+1] = 1
    U[(i0+1)*m + (j0+1), offset+1] = 1
    V[j0*p + k0, offset+1] = 1
    W[k0*n + (i0+1), offset+1] = 1
    W[(k0+1)*n + (i0+1), offset+1] = -1
    
    U[i0*m + j0, offset+2] = 1
    V[j0*p + (k0+1), offset+2] = 1
    V[(j0+1)*p + (k0+1), offset+2] = -1
    W[(k0+1)*n + (i0+1), offset+2] = 1
    
    U[(i0+1)*m + (j0+1), offset+3] = 1
    V[(j0+1)*p + k0, offset+3] = 1
    V[j0*p + k0, offset+3] = -1
    W[k0*n + i0, offset+3] = -1
    W[(k0+1)*n + i0, offset+3] = 1
    
    U[i0*m + j0, offset+4] = 1
    U[i0*m + (j0+1), offset+4] = 1
    V[(j0+1)*p + (k0+1), offset+4] = 1
    W[(k0+1)*n + i0, offset+4] = 1
    W[(k0+1)*n + (i0+1), offset+4] = 1
    
    U[(i0+1)*m + j0, offset+5] = 1
    U[i0*m + j0, offset+5] = -1
    V[j0*p + k0, offset+5] = 1
    V[j0*p + (k0+1), offset+5] = 1
    W[k0*n + i0, offset+5] = 1
    W[k0*n + (i0+1), offset+5] = 1
    
    U[i0*m + (j0+1), offset+6] = 1
    U[(i0+1)*m + (j0+1), offset+6] = -1
    V[(j0+1)*p + k0, offset+6] = 1
    V[(j0+1)*p + (k0+1), offset+6] = 1
    W[k0*n + i0, offset+6] = 1
    W[(k0+1)*n + (i0+1), offset+6] = 1
    
    return offset + 7

def _schoolbook_block(U, V, W, i0, j0, k0, ni, nj, nk, offset):
    n, m, p = 4, 5, 6
    for ii in range(ni):
        for jj in range(nj):
            for kk in range(nk):
                i = i0 + ii
                j = j0 + jj
                k = k0 + kk
                U[i*m + j, offset] = 1
                V[j*p + k, offset] = 1
                W[k*n + i, offset] = 1
                offset += 1
    return offset

def _build_block_decomp(n, m, p, i_blocks, j_blocks, k_blocks):
    total_rank = 0
    for i0, ni in i_blocks:
        for j0, nj in j_blocks:
            for k0, nk in k_blocks:
                if ni == 2 and nj == 2 and nk == 2:
                    total_rank += 7
                else:
                    total_rank += ni * nj * nk
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    offset = 0
    for i0, ni in i_blocks:
        for j0, nj in j_blocks:
            for k0, nk in k_blocks:
                if ni == 2 and nj == 2 and nk == 2:
                    offset = _strassen_2x2x2(U, V, W, i0, j0, k0, offset)
                else:
                    offset = _schoolbook_block(U, V, W, i0, j0, k0, ni, nj, nk, offset)
    
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-9):
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                if abs(val - 1.0) > tol:
                    return False
    return True

def _find_best_partition(n, m, p):
    """Try different block partitions to minimize rank."""
    best_rank = n * m * p
    best_blocks = None
    
    def generate_partitions(size, max_blocks=8):
        results = []
        def rec(remaining, current):
            if remaining == 0:
                results.append(list(current))
                return
            if len(current) >= max_blocks:
                return
            for bs in range(1, remaining + 1):
                current.append(bs)
                rec(remaining - bs, current)
                current.pop()
        rec(size, [])
        return results
    
    i_parts = generate_partitions(n)
    j_parts = generate_partitions(m)
    k_parts = generate_partitions(p)
    
    for i_sizes in i_parts:
        i_blocks = []
        pos = 0
        for s in i_sizes:
            i_blocks.append((pos, s))
            pos += s
        
        for j_sizes in j_parts:
            j_blocks = []
            pos = 0
            for s in j_sizes:
                j_blocks.append((pos, s))
                pos += s
            
            for k_sizes in k_parts:
                k_blocks = []
                pos = 0
                for s in k_sizes:
                    k_blocks.append((pos, s))
                    pos += s
                
                rank = 0
                for i0, ni in i_blocks:
                    for j0, nj in j_blocks:
                        for k0, nk in k_blocks:
                            if ni == 2 and nj == 2 and nk == 2:
                                rank += 7
                            else:
                                rank += ni * nj * nk
                
                if rank < best_rank:
                    best_rank = rank
                    best_blocks = (i_blocks, j_blocks, k_blocks)
    
    return best_rank, best_blocks

def _build_schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    best_rank, best_blocks = _find_best_partition(n, m, p)
    
    if best_blocks is None:
        U, V, W = _build_schoolbook(n, m, p)
    else:
        i_blocks, j_blocks, k_blocks = best_blocks
        U, V, W = _build_block_decomp(n, m, p, i_blocks, j_blocks, k_blocks)
    
    rank = U.shape[1]
    
    if time.time() < deadline - 5:
        if _verify(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    U2, V2, W2 = _build_schoolbook(n, m, p)
    return {"U": U2.tolist(), "V": V2.tolist(), "W": W2.tolist()}
