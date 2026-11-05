"""Task: rank-minimal exact tensor decomposition of 3x4 by 4x7 matrix multiplication.
Trace convention: out[i,k] = sum_j A[i,j]*B[j,k] == sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r].
"""
import time
import itertools

DIMENSIONS = [3, 4, 7]
_DEADLINE = 120.0  # safety margin before 150s kill


def _verify_decomposition(n, m, p, rank, U, V, W):
    """Verify that the decomposition is exact."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                total = 0
                for c in range(rank):
                    total += U[u_idx][c] * V[v_idx][c] * W[w_idx][c]
                if total != 1:
                    return False
    return True


def _try_greedy_strassen(n, m, p, t0):
    """
    Greedy search for a decomposition using Strassen-like rank-7 blocks.
    
    Each block covers a 2x2x2 subproblem (2 i's, 2 j's, 2 k's) with 7 rank-1 terms.
    We greedily select non-overlapping blocks to maximize coverage, then fill
    the rest with schoolbook (1 rank-1 term per uncovered (i,j,k) triple).
    """
    # All possible (i0,i1,j0,j1,k0,k1) blocks
    i_pairs = list(itertools.combinations(range(n), 2))
    j_pairs = list(itertools.combinations(range(m), 2))
    k_pairs = list(itertools.combinations(range(p), 2))
    
    all_blocks = []
    for i0, i1 in i_pairs:
        for j0, j1 in j_pairs:
            for k0, k1 in k_pairs:
                all_blocks.append((i0, i1, j0, j1, k0, k1))
    
    # Greedy: repeatedly pick the block that covers the most uncovered triples
    covered = set()
    selected_blocks = []
    uncovered_count = n * m * p
    
    while uncovered_count > 0:
        if time.time() - t0 > _DEADLINE:
            break
        
        best_block = None
        best_gain = 0
        for block in all_blocks:
            i0, i1, j0, j1, k0, k1 = block
            gain = 0
            for ii in (i0, i1):
                for jj in (j0, j1):
                    for kk in (k0, k1):
                        if (ii, jj, kk) not in covered:
                            gain += 1
            if gain > best_gain:
                best_gain = gain
                best_block = block
        
        if best_gain == 0:
            break
        
        i0, i1, j0, j1, k0, k1 = best_block
        selected_blocks.append(best_block)
        for ii in (i0, i1):
            for jj in (j0, j1):
                for kk in (k0, k1):
                    covered.add((ii, jj, kk))
        uncovered_count -= best_gain
    
    rank = 7 * len(selected_blocks) + (n * m * p - len(covered))
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for (i0, i1, j0, j1, k0, k1) in selected_blocks:
        # M1: (A[i0,j0]+A[i1,j1])*(B[j0,k0]+B[j1,k1])
        U[i0 * m + j0][col] = 1
        U[i1 * m + j1][col] = 1
        V[j0 * p + k0][col] = 1
        V[j1 * p + k1][col] = 1
        W[k0 * n + i0][col] = 1
        W[k1 * n + i1][col] = 1
        col += 1
        
        # M2: (A[i1,j0]+A[i1,j1])*B[j0,k0]
        U[i1 * m + j0][col] = 1
        U[i1 * m + j1][col] = 1
        V[j0 * p + k0][col] = 1
        W[k0 * n + i1][col] = 1
        W[k1 * n + i1][col] = -1
        col += 1
        
        # M3: A[i0,j0]*(B[j0,k1]-B[j1,k1])
        U[i0 * m + j0][col] = 1
        V[j0 * p + k1][col] = 1
        V[j1 * p + k1][col] = -1
        W[k1 * n + i0][col] = 1
        W[k1 * n + i1][col] = 1
        col += 1
        
        # M4: A[i1,j1]*(B[j1,k0]-B[j0,k0])
        U[i1 * m + j1][col] = 1
        V[j1 * p + k0][col] = 1
        V[j0 * p + k0][col] = -1
        W[k0 * n + i0][col] = 1
        W[k0 * n + i1][col] = 1
        col += 1
        
        # M5: (A[i0,j0]+A[i0,j1])*B[j1,k1]
        U[i0 * m + j0][col] = 1
        U[i0 * m + j1][col] = 1
        V[j1 * p + k1][col] = 1
        W[k0 * n + i0][col] = -1
        W[k1 * n + i0][col] = 1
        col += 1
        
        # M6: (A[i1,j0]-A[i0,j0])*(B[j0,k0]+B[j0,k1])
        U[i1 * m + j0][col] = 1
        U[i0 * m + j0][col] = -1
        V[j0 * p + k0][col] = 1
        V[j0 * p + k1][col] = 1
        W[k1 * n + i1][col] = 1
        col += 1
        
        # M7: (A[i0,j1]-A[i1,j1])*(B[j1,k0]+B[j1,k1])
        U[i0 * m + j1][col] = 1
        U[i1 * m + j1][col] = -1
        V[j1 * p + k0][col] = 1
        V[j1 * p + k1][col] = 1
        W[k0 * n + i0][col] = 1
        col += 1
    
    # Schoolbook for remaining
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    if not _verify_decomposition(n, m, p, rank, U, V, W):
        return None
    
    return {"U": U, "V": V, "W": W, "rank": rank}


def _deterministic_strassen(n, m, p):
    """
    Deterministic multi-Strassen decomposition using a fixed set of 2x2x2 blocks.
    Uses 6 blocks covering i in {0,1}, j in {0,1}x{2,3}, k in {0,1}x{2,3}x{4,5}.
    """
    blocks = [
        (0, 1, 0, 1, 0, 1),
        (0, 1, 0, 1, 2, 3),
        (0, 1, 0, 1, 4, 5),
        (0, 1, 2, 3, 0, 1),
        (0, 1, 2, 3, 2, 3),
        (0, 1, 2, 3, 4, 5),
    ]
    
    covered = set()
    for (i0, i1, j0, j1, k0, k1) in blocks:
        for ii in (i0, i1):
            for jj in (j0, j1):
                for kk in (k0, k1):
                    covered.add((ii, jj, kk))
    
    rank = 7 * len(blocks) + (n * m * p - len(covered))
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for (i0, i1, j0, j1, k0, k1) in blocks:
        U[i0 * m + j0][col] = 1
        U[i1 * m + j1][col] = 1
        V[j0 * p + k0][col] = 1
        V[j1 * p + k1][col] = 1
        W[k0 * n + i0][col] = 1
        W[k1 * n + i1][col] = 1
        col += 1
        
        U[i1 * m + j0][col] = 1
        U[i1 * m + j1][col] = 1
        V[j0 * p + k0][col] = 1
        W[k0 * n + i1][col] = 1
        W[k1 * n + i1][col] = -1
        col += 1
        
        U[i0 * m + j0][col] = 1
        V[j0 * p + k1][col] = 1
        V[j1 * p + k1][col] = -1
        W[k1 * n + i0][col] = 1
        W[k1 * n + i1][col] = 1
        col += 1
        
        U[i1 * m + j1][col] = 1
        V[j1 * p + k0][col] = 1
        V[j0 * p + k0][col] = -1
        W[k0 * n + i0][col] = 1
        W[k0 * n + i1][col] = 1
        col += 1
        
        U[i0 * m + j0][col] = 1
        U[i0 * m + j1][col] = 1
        V[j1 * p + k1][col] = 1
        W[k0 * n + i0][col] = -1
        W[k1 * n + i0][col] = 1
        col += 1
        
        U[i1 * m + j0][col] = 1
        U[i0 * m + j0][col] = -1
        V[j0 * p + k0][col] = 1
        V[j0 * p + k1][col] = 1
        W[k1 * n + i1][col] = 1
        col += 1
        
        U[i0 * m + j1][col] = 1
        U[i1 * m + j1][col] = -1
        V[j1 * p + k0][col] = 1
        V[j1 * p + k1][col] = 1
        W[k0 * n + i0][col] = 1
        col += 1
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
    
    assert col == rank
    if not _verify_decomposition(n, m, p, rank, U, V, W):
        return None
    
    return {"U": U, "V": V, "W": W, "rank": rank}


def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS
    
    # Try greedy first (may find better packing)
    greedy_result = _try_greedy_strassen(n, m, p, t0)
    
    if greedy_result and greedy_result["rank"] <= 78:
        del greedy_result["rank"]
        return greedy_result
    
    # Fallback to deterministic
    det_result = _deterministic_strassen(n, m, p)
    if det_result:
        del det_result["rank"]
        return det_result
    
    # Last resort: pure schoolbook
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
