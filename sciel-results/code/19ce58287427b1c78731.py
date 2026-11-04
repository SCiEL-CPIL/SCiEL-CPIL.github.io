"""Matrix multiplication 3x5x5 tensor decomposition - optimized block partitioning."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START

def _strassen_222():
    """
    Return (U, V, W) for 2x2x2 Strassen multiplication.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indexing: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # M3 = A00(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4 = A11(B10-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _schoolbook_factors(n, m, p):
    """Return (U, V, W) as lists of lists for schoolbook multiplication."""
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
    return U, V, W

def _all_partitions(dim):
    """Generate all ways to partition a dimension into contiguous blocks of size 1 or 2."""
    if dim == 1:
        return [((0,),)]
    if dim == 2:
        return [((0,), (1,)), ((0,1),)]
    if dim == 3:
        return [
            ((0,), (1,), (2,)),
            ((0,), (1,2)),
            ((0,1), (2,)),
            ((0,1,2),)  # Not useful but include for completeness
        ]
    if dim == 4:
        return [
            ((0,), (1,), (2,), (3,)),
            ((0,), (1,), (2,3)),
            ((0,), (1,2), (3,)),
            ((0,), (1,2,3)),
            ((0,1), (2,), (3,)),
            ((0,1), (2,3)),
            ((0,1,2), (3,)),
            ((0,1,2,3)),
            ((0,1), (2,), (3,)),  # duplicate, will dedupe
        ]
    if dim == 5:
        results = []
        # Generate all compositions of 5 into parts of size 1 or 2
        def gen_comps(remaining, current):
            if remaining == 0:
                results.append(tuple(current))
                return
            if remaining >= 1:
                gen_comps(remaining - 1, current + [1])
            if remaining >= 2:
                gen_comps(remaining - 2, current + [2])
        gen_comps(dim, [])
        # Convert compositions to block tuples
        partitions = []
        for comp in results:
            blocks = []
            offset = 0
            for size in comp:
                blocks.append(tuple(range(offset, offset + size)))
                offset += size
            partitions.append(tuple(blocks))
        return partitions
    # For larger dims, just return full block
    return [tuple(range(dim))]

def _compute_rank_for_partition(i_blocks, j_blocks, k_blocks, n, m, p):
    """Compute the total rank for a given partition."""
    rank = 0
    for i_set in i_blocks:
        for j_set in j_blocks:
            for k_set in k_blocks:
                if len(i_set) == 2 and len(j_set) == 2 and len(k_set) == 2:
                    rank += 7
                else:
                    rank += len(i_set) * len(j_set) * len(k_set)
    return rank

def _build_decomposition(i_blocks, j_blocks, k_blocks, n, m, p):
    """Build the full decomposition for a given partition."""
    su, sv, sw = _strassen_222()
    
    # First compute total rank
    total_rank = 0
    for i_set in i_blocks:
        for j_set in j_blocks:
            for k_set in k_blocks:
                if len(i_set) == 2 and len(j_set) == 2 and len(k_set) == 2:
                    total_rank += 7
                else:
                    total_rank += len(i_set) * len(j_set) * len(k_set)
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    for i_set in i_blocks:
        for j_set in j_blocks:
            for k_set in k_blocks:
                if len(i_set) == 2 and len(j_set) == 2 and len(k_set) == 2:
                    for c in range(7):
                        for li in range(2):
                            gi = i_set[li]
                            for lj in range(2):
                                gj = j_set[lj]
                                U[gi * m + gj][col + c] = su[li * 2 + lj][c]
                        for lj in range(2):
                            gj = j_set[lj]
                            for lk in range(2):
                                gk = k_set[lk]
                                V[gj * p + gk][col + c] = sv[lj * 2 + lk][c]
                        for lk in range(2):
                            gk = k_set[lk]
                            for li in range(2):
                                gi = i_set[li]
                                W[gk * n + gi][col + c] = sw[lk * 2 + li][c]
                    col += 7
                else:
                    for gi in i_set:
                        for gj in j_set:
                            for gk in k_set:
                                U[gi * m + gj][col] = 1
                                V[gj * p + gk][col] = 1
                                W[gk * n + gi][col] = 1
                                col += 1
                                
    return U, V, W, total_rank

def _verify_decomposition(U, V, W, n, m, p, rank):
    """Verify that the decomposition is exact."""
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(W_np[k*n + i] * U_np[i*m + j] * V_np[j*p + k])
                if not np.isclose(s, 1.0, atol=1e-10):
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Generate all valid partitions
    i_parts = _all_partitions(n)
    j_parts = _all_partitions(m)
    k_parts = _all_partitions(p)
    
    # Find the partition with minimum rank
    best_rank = n * m * p  # schoolbook
    best_partition = None
    
    for i_b in i_parts:
        for j_b in j_parts:
            for k_b in k_parts:
                rank = _compute_rank_for_partition(i_b, j_b, k_b, n, m, p)
                if rank < best_rank:
                    best_rank = rank
                    best_partition = (i_b, j_b, k_b)
    
    # If no improvement found, use schoolbook
    if best_partition is None:
        U, V, W = _schoolbook_factors(n, m, p)
        return {"U": U, "V": W if False else V, "W": W}
    
    i_b, j_b, k_b = best_partition
    U, V, W, rank = _build_decomposition(i_b, j_b, k_b, n, m, p)
    
    # Verify the decomposition
    if _verify_decomposition(U, V, W, n, m, p, rank):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    U, V, W = _schoolbook_factors(n, m, p)
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
