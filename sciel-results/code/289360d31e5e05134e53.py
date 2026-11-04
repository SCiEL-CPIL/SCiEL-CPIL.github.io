"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition correctly computes matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _strassen_terms():
    """Return the 7 Strassen terms as (a_coeffs, b_coeffs, c_coeffs) each of length 4."""
    return [
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        ([0,0,1,1], [1,1,0,0], [0,0,1,-1]),
        ([1,1,0,0], [0,0,1,1], [0,1,0,1]),
        ([0,0,0,1], [0,-1,0,1], [1,0,-1,0]),
        ([1,0,0,0], [1,0,-1,0], [-1,1,0,0]),
        ([-1,0,1,0], [0,1,0,1], [0,0,0,1]),
        ([0,1,0,-1], [1,0,1,0], [1,0,0,0]),
    ]

def _build_strassen_block(i0, j0, k0, U, V, W, col_offset):
    """Build a 2x2x2 Strassen block at position (i0,j0,k0). Returns new col_offset."""
    terms = _strassen_terms()
    col = col_offset
    for t in range(7):
        a_coeffs = terms[t][0]
        b_coeffs = terms[t][1]
        c_coeffs = terms[t][2]
        
        for di_idx in range(2):
            for dj_idx in range(2):
                U[(i0+di_idx)*4 + (j0+dj_idx)][col] = a_coeffs[di_idx*2+dj_idx]
        
        for dj_idx in range(2):
            for dk_idx in range(2):
                V[(j0+dj_idx)*8 + (k0+dk_idx)][col] = b_coeffs[dj_idx*2+dk_idx]
        
        for di_idx in range(2):
            for dk_idx in range(2):
                W[(k0+dk_idx)*3 + (i0+di_idx)][col] = c_coeffs[di_idx*2+dk_idx]
        
        col += 1
    return col

def _build_schoolbook_block(i0, i1, j0, j1, k0, k1, U, V, W, col_offset):
    """Build a schoolbook block. Returns new col_offset."""
    col = col_offset
    for i in range(i0, i1):
        for j in range(j0, j1):
            for k in range(k0, k1):
                U[i*4 + j][col] = 1
                V[j*8 + k][col] = 1
                W[k*3 + i][col] = 1
                col += 1
    return col

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    start_time = time.time()
    deadline = start_time + 110
    
    # Strategy: Try all ways to partition the 3x4x8 tensor into 2x2x2 Strassen blocks
    # and schoolbook blocks, then find the minimum rank combination.
    
    # The key insight: we can place 2x2x2 Strassen blocks at various positions.
    # A 2x2x2 block at (i0,j0,k0) covers i in [i0,i0+2), j in [j0,j0+2), k in [k0,k0+2).
    # It uses 7 rank terms instead of 8 (saving 1 per block).
    # Blocks must not overlap.
    
    # Possible positions for 2x2x2 blocks:
    # i0 in {0,1} (since n=3, i0+2 <= 3, so i0 in {0,1})
    # j0 in {0,1,2} (since m=4, j0+2 <= 4, so j0 in {0,1,2})
    # k0 in {0,1,2,3,4,5,6} (since p=8, k0+2 <= 8, so k0 in {0,...,6})
    
    # We need to select a set of non-overlapping 2x2x2 blocks to maximize the number
    # of Strassen blocks (each saves 1 rank compared to schoolbook).
    
    # This is a maximum independent set / matching problem on a 3D grid.
    # Let's use a simple approach: try all combinations of block placements.
    
    # First, let's enumerate all possible 2x2x2 block positions
    possible_blocks = []
    for i0 in range(n - 1):  # 0, 1
        for j0 in range(m - 1):  # 0, 1, 2
            for k0 in range(p - 1):  # 0, 1, 2, 3, 4, 5, 6
                possible_blocks.append((i0, j0, k0))
    
    # Two blocks overlap if their 2x2x2 regions intersect
    def blocks_overlap(b1, b2):
        i0_1, j0_1, k0_1 = b1
        i0_2, j0_2, k0_2 = b2
        # Check if [i0_1, i0_1+2) and [i0_2, i0_2+2) overlap
        if not (i0_1 + 2 <= i0_2 or i0_2 + 2 <= i0_1):
            if not (j0_1 + 2 <= j0_2 or j0_2 + 2 <= j0_1):
                if not (k0_1 + 2 <= k0_2 or k0_2 + 2 <= k0_1):
                    return True
    return False
    
    # Find maximum set of non-overlapping blocks using backtracking
    # But 2*3*7 = 42 possible blocks, so we need a smarter approach.
    
    # Actually, let's use a greedy + local search approach:
    # 1. Greedily place blocks
    # 2. Try to improve by swapping
    
    # Or better: since the grid is small, use dynamic programming or exhaustive search
    # with pruning.
    
    # Let's use a different approach: since n=3 is small, we can enumerate
    # all ways to cover the i-dimension with 2x2 blocks and 1-wide strips.
    
    # For the i-dimension (size 3):
    # Option A: One 2-block at i0=0, one 1-strip at i=2
    # Option B: One 2-block at i0=1, one 1-strip at i=0
    # Option C: Three 1-strips (no 2-blocks in i)
    
    # For each i-partition, do the same for j and k dimensions.
    
    # This gives a structured search space.
    
    def i_partitions():
        """Yield all ways to partition [0,3) into segments of size 1 or 2."""
        # (start, end, is_strassen)
        yield [(0, 2, True), (2, 3, False)]
        yield [(0, 1, False), (1, 3, True)]
        yield [(0, 1, False), (1, 2, False), (2, 3, False)]
        # Also: could have (0,2,True) and (2,3,False) which is same as first
        # And (0,1,False), (1,3,True) which is same as second
        # No other options for size 3
    
    def m_partitions():
        """Yield all ways to partition [0,4) into segments of size 1 or 2."""
        yield [(0, 2, True), (2, 4, True)]
        yield [(0, 2, True), (2, 3, False), (3, 4, False)]
        yield [(0, 1, False), (1, 3, True), (3, 4, False)]
        yield [(0, 1, False), (1, 2, False), (2, 4, True)]
        yield [(0, 1, False), (1, 3, True), (3, 4, False)]  # duplicate, skip
        yield [(0, 2, True), (2, 4, True)]  # duplicate
        # Let me be more careful
        # Partitions of 4 into 1s and 2s:
        # 2+2, 2+1+1, 1+2+1, 1+1+2, 1+1+1+1
        yield [(0, 2, True), (2, 4, True)]
        yield [(0, 2, True), (2, 3, False), (3, 4, False)]
        yield [(0, 1, False), (1, 3, True), (3, 4, False)]
        yield [(0, 1, False), (1, 2, False), (2, 4, True)]
        yield [(0, 1, False), (1, 2, False), (2, 3, False), (3, 4, False)]
    
    def p_partitions():
        """Yield all ways to partition [0,8) into segments of size 1 or 2."""
        # This is more complex. Number of compositions of 8 into 1s and 2s is F(9)=34.
        # Let's generate them recursively.
        def gen(remaining, start):
            if remaining == 0:
                yield []
                return
            if remaining == 1:
                yield [(start, start+1, False)]
                return
            # Use a 2-block
            yield from [(start, start+2, True)] + rest for rest in gen(remaining-2, start+2)
            # Use a 1-strip
            yield from [(start, start+1, False)] + rest for rest in gen(remaining-1, start+1)
        
        return list(gen(8, 0))
    
    # Now try all combinations
    best_rank = n * m * p  # 96 schoolbook
    best_result = None
    
    def schoolbook_solution():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
        return U, V, W, rank
    
    U, V, W, rank = schoolbook_solution()
    best_result = (U, V, W, rank)
    best_rank = rank
    
    for i_parts in i_partitions():
        for m_parts in m_partitions():
            for p_parts in p_partitions():
                if time.time() > deadline - 15:
                    break
                
                # Build the decomposition
                rank = 0
                U = [[0] * 200 for _ in range(n * m)]  # Pre-allocate max possible
                V = [[0] * 200 for _ in range(m * p)]
                W = [[0] * 200 for _ in range(p * n)]
                col = 0
                
                for (i0, i1, i_strassen) in i_parts:
                    for (j0, j1, j_strassen) in m_parts:
                        for (k0, k1, k_strassen) in p_parts:
                            di = i1 - i0
                            dj = j1 - j0
                            dk = k1 - k0
                            
                            if i_strassen and j_strassen and k_strassen and di == 2 and dj == 2 and dk == 2:
                                # Use Strassen
                                col = _build_strassen_block(i0, j0, k0, U, V, W, col)
                                rank += 7
                            else:
                                # Use schoolbook
                                for i in range(i0, i1):
                                    for j in range(j0, j1):
                                        for k in range(k0, k1):
                                            U[i*m + j][col] = 1
                                            V[j*p + k][col] = 1
                                            W[k*n + i][col] = 1
                                            col += 1
                                rank += di * dj * dk
                
                if rank >= best_rank:
                    continue
                
                # Trim and verify
                U = [row[:col] for row in U]
                V = [row[:col] for row in V]
                W = [row[:col] for row in W]
                
                if _verify(U, V, W, n, m, p):
                    best_rank = rank
                    best_result = (U, V, W, rank)
    
    if best_result is not None:
        U, V, W, rank = best_result
        return {"U": U, "V": V, "W": W}
    
    # Fallback
    U, V, W, rank = schoolbook_solution()
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
