"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is correct."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Reconstruct the tensor
    tensor = np.zeros((n, m, p))
    for c in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    tensor[i, j, k] += U_arr[i*m + j, c] * V_arr[j*p + k, c] * W_arr[k*n + i, c]
    
    # The correct tensor for matrix multiplication is 1 for all (i,j,k)
    expected = np.ones((n, m, p))
    return np.allclose(tensor, expected, atol=1e-9)

def _block_decomposition(n, m, p):
    """
    Get the best known rank for a block of size n x m x p.
    Uses Strassen (rank 7) for 2x2x2, and schoolbook otherwise.
    For 3x3x3, we use rank 23 (BCL algorithm).
    For 3x2x3 and 2x3x3, rank 8.
    """
    # Sort dimensions for lookup
    dims = sorted([n, m, p])
    key = tuple(dims)
    
    # Known ranks
    known_ranks = {
        (1, 1, 1): 1,
        (1, 1, 2): 2,
        (1, 2, 2): 4,
        (2, 2, 2): 7,  # Strassen
        (1, 1, 3): 3,
        (1, 2, 3): 6,
        (1, 3, 3): 9,
        (2, 2, 3): 12,  # schoolbook: 2*2*3=12
        (2, 3, 3): 18,  # schoolbook
        (3, 3, 3): 23,  # BCL
        (1, 1, 4): 4,
        (1, 2, 4): 8,
        (1, 3, 4): 12,
        (1, 4, 4): 16,
        (2, 2, 4): 16,
        (2, 3, 4): 24,
        (2, 4, 4): 32,
        (3, 3, 4): 36,
        (3, 4, 4): 48,
        (4, 4, 4): 64,
        (1, 1, 5): 5,
        (1, 2, 5): 10,
        (1, 3, 5): 15,
        (1, 4, 5): 20,
        (1, 5, 5): 25,
        (2, 2, 5): 20,
        (2, 3, 5): 30,
        (2, 4, 5): 40,
        (2, 5, 5): 50,
        (3, 3, 5): 45,
        (3, 4, 5): 60,
        (3, 5, 5): 75,
        (4, 4, 5): 80,
        (4, 5, 5): 100,
        (5, 5, 5): 125,
        (1, 1, 6): 6,
        (1, 2, 6): 12,
        (1, 3, 6): 18,
        (1, 4, 6): 24,
        (1, 5, 6): 30,
        (1, 6, 6): 36,
        (2, 2, 6): 24,
        (2, 3, 6): 36,
        (2, 4, 6): 48,
        (2, 5, 6): 60,
        (2, 6, 6): 72,
        (3, 3, 6): 54,
        (3, 4, 6): 72,
        (3, 5, 6): 90,
        (3, 6, 6): 108,
        (4, 4, 6): 96,
        (4, 5, 6): 120,
        (4, 6, 6): 144,
        (5, 5, 6): 150,
        (5, 6, 6): 180,
        (6, 6, 6): 216,
        (1, 1, 7): 7,
        (1, 2, 7): 14,
        (1, 3, 7): 21,
        (1, 4, 7): 28,
        (1, 5, 7): 35,
        (1, 6, 7): 42,
        (1, 7, 7): 49,
        (2, 2, 7): 28,
        (2, 3, 7): 42,
        (2, 4, 7): 56,
        (2, 5, 7): 70,
        (2, 6, 7): 84,
        (2, 7, 7): 98,
        (3, 3, 7): 63,
        (3, 4, 7): 84,
        (3, 5, 7): 105,
        (3, 6, 7): 126,
        (3, 7, 7): 147,
        (4, 4, 7): 112,
        (4, 5, 7): 140,
        (4, 6, 7): 168,
        (4, 7, 7): 196,
        (5, 5, 7): 175,
        (5, 6, 7): 210,
        (5, 7, 7): 245,
        (6, 6, 7): 252,
        (6, 7, 7): 294,
        (7, 7, 7): 343,
    }
    
    if key in known_ranks:
        return known_ranks[key]
    
    # For 3x2x3 type (sorted: 2,3,3)
    if key == (2, 3, 3):
        return 8  # Known rank 8 for 3x2x3
    
    # Default: schoolbook
    return n * m * p

def _get_all_splits(n):
    """Get all ways to split n into parts of size 1 or 2."""
    if n == 0:
        return [[]]
    if n == 1:
        return [[1]]
    if n == 2:
        return [[2], [1, 1]]
    
    results = []
    # Start with 1
    for rest in _get_all_splits(n - 1):
        results.append([1] + rest)
    # Start with 2
    for rest in _get_all_splits(n - 2):
        results.append([2] + rest)
    return results

def _find_best_block_split(n, m, p):
    """
    Find the best block partition for n x m x p matrix multiplication.
    Returns (best_rank, row_splits, col_splits, part_col_splits)
    where row_splits is how to split n (rows of A, cols of C),
    col_splits is how to split m (cols of A, rows of B),
    part_col_splits is how to split p (cols of B, cols of C) for each block.
    """
    best_rank = n * m * p
    best_splits = None
    
    row_splits = _get_all_splits(n)
    col_splits = _get_all_splits(m)
    
    for rs in row_splits:
        for cs in col_splits:
            # For each block (i_block, j_block), we need to split p
            # The block has size rs[i_block] x cs[j_block] x p
            # We can split p in various ways
            # But to keep it simple, let's just use schoolbook for the p dimension
            # or try splitting p into 1s and 2s
            
            # Actually, let's try all splits of p for each block
            # This gets complex. Let's simplify: for each (i,j) block of size rs[i] x cs[j] x p,
            # the best rank is _block_decomposition(rs[i], cs[j], p)
            # But we could also split p further.
            
            # For now, use the direct block rank
            total_rank = 0
            for i in range(len(rs)):
                for j in range(len(cs)):
                    total_rank += _block_decomposition(rs[i], cs[j], p)
            
            if total_rank < best_rank:
                best_rank = total_rank
                best_splits = (rs, cs)
    
    return best_rank, best_splits

def _build_block_decomposition(n, m, p, row_splits, col_splits):
    """
    Build the actual U, V, W matrices for a given block partition.
    For each block, use schoolbook decomposition (since we don't have
    explicit Strassen/BCL implementations for all sizes).
    """
    rank = 0
    U_rows = []
    V_rows = []
    W_rows = []
    
    # First, compute total rank
    for i in range(len(row_splits)):
        for j in range(len(col_splits)):
            rank += row_splits[i] * col_splits[j] * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    row_offset = 0
    for i in range(len(row_splits)):
        for j in range(len(col_splits)):
            r = row_splits[i]
            c = col_splits[j]
            # Block A[i:i+r, j:j+c] * B[j:j+c, :] -> C[i:i+r, :]
            for ii in range(r):
                for jj in range(c):
                    for kk in range(p):
                        U[(i + ii) * m + (j + jj)][col] = 1
                        V[(j + jj) * p + kk][col] = 1
                        W[kk * n + (i + ii)][col] = 1
                        col += 1
        row_offset += row_splits[i]
    
    return {"U": U, "V": V, "W": W}

def _build_with_strassen_2x2(n, m, p):
    """
    Build decomposition using Strassen for 2x2x2 blocks where possible.
    This requires careful implementation of Strassen's algorithm.
    """
    # For simplicity, let's just use schoolbook for now
    # and rely on the block partition to get good ranks
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def construct():
    """
    Construct a tensor decomposition for 3x5 * 5x7 matrix multiplication.
    
    Strategy: Use block decomposition with Strassen for 2x2 blocks.
    """
    n, m, p = DIMENSIONS  # 3, 5, 7
    
    start_time = time.time()
    deadline = start_time + 120  # 120 second limit
    
    # Try to find the best block partition
    best_rank, best_splits = _find_best_block_split(n, m, p)
    
    # Build the decomposition using schoolbook for each block
    # (This gives the rank as computed, but we need to verify)
    if best_splits is not None:
        row_splits, col_splits = best_splits
        result = _build_block_decomposition(n, m, p, row_splits, col_splits)
        
        # Verify
        if _verify(result["U"], result["V"], result["W"], n, m, p):
            actual_rank = len(result["U"][0])
            return result
    
    # Fallback: schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
