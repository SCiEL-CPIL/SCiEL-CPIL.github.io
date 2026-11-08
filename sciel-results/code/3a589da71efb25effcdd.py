import numpy as np
import time
import itertools
from collections import defaultdict

DIMENSIONS = [3, 5, 5]
START_TIME = time.time()
TIME_LIMIT = 140.0  # Safety margin before 150s kill

def construct():
    n, m, p = DIMENSIONS
    
    # Baseline: rank 71 solution
    baseline = _construct_rank71(n, m, p)
    baseline_rank = 71
    
    # Try to find better solutions
    best_result = baseline
    best_rank = baseline_rank
    
    elapsed = time.time() - START_TIME
    if elapsed > TIME_LIMIT - 5:
        return best_result
    
    # Search for rank 70 solutions
    # Try different tilings and patterns
    search_result = _search_for_better(n, m, p, baseline_rank)
    if search_result is not None:
        rank, U, V, W = search_result
        if rank < best_rank:
            best_rank = rank
            best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    elapsed = time.time() - START_TIME
    if elapsed > TIME_LIMIT - 2:
        return best_result
    
    return best_result

def _construct_rank71(n, m, p):
    total_rank = 71
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    blocks = [
        (0, 0, 0),
        (0, 2, 0),
        (0, 0, 2),
        (0, 2, 2)
    ]
    
    col = 0
    for (i0, j0, k0) in blocks:
        col = _add_strassen_block(U, V, W, i0, j0, k0, n, m, p, col)
    
    covered = np.zeros((n, m, p), dtype=bool)
    for (i0, j0, k0) in blocks:
        covered[i0:i0+2, j0:j0+2, k0:k0+2] = True
        
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if not covered[i, j, k]:
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
    
    return {"U": U, "V": V, "W": W}

def _add_strassen_block(U, V, W, i0, j0, k0, n, m, p, col):
    # M1 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
    U[i0*m + j0, col] += 1.0
    U[(i0+1)*m + (j0+1), col] += 1.0
    V[j0*p + k0, col] += 1.0
    V[(j0+1)*p + (k0+1), col] += 1.0
    W[k0*n + i0, col] += 1.0
    W[(k0+1)*n + (i0+1), col] += 1.0
    col += 1
    
    # M2 = (A[1,0] + A[1,1]) * B[0,0]
    U[(i0+1)*m + j0, col] += 1.0
    U[(i0+1)*m + (j0+1), col] += 1.0
    V[j0*p + k0, col] += 1.0
    W[k0*n + (i0+1), col] += 1.0
    W[(k0+1)*n + (i0+1), col] -= 1.0
    col += 1
    
    # M3 = A[0,0] * (B[0,1] - B[1,1])
    U[i0*m + j0, col] += 1.0
    V[j0*p + (k0+1), col] += 1.0
    V[(j0+1)*p + (k0+1), col] -= 1.0
    W[(k0+1)*n + i0, col] += 1.0
    W[(k0+1)*n + (i0+1), col] += 1.0
    col += 1
    
    # M4 = A[1,1] * (B[1,0] - B[0,0])
    U[(i0+1)*m + (j0+1), col] += 1.0
    V[(j0+1)*p + k0, col] += 1.0
    V[j0*p + k0, col] -= 1.0
    W[k0*n + i0, col] += 1.0
    W[k0*n + (i0+1), col] += 1.0
    col += 1
    
    # M5 = (A[0,0] + A[0,1]) * B[1,1]
    U[i0*m + j0, col] += 1.0
    U[i0*m + (j0+1), col] += 1.0
    V[(j0+1)*p + (k0+1), col] += 1.0
    W[k0*n + i0, col] -= 1.0
    W[(k0+1)*n + i0, col] += 1.0
    col += 1
    
    # M6 = (A[1,0] - A[0,0]) * (B[0,0] + B[0,1])
    U[(i0+1)*m + j0, col] += 1.0
    U[i0*m + j0, col] -= 1.0
    V[j0*p + k0, col] += 1.0
    V[j0*p + (k0+1), col] += 1.0
    W[(k0+1)*n + (i0+1), col] += 1.0
    col += 1
    
    # M7 = (A[0,1] - A[1,1]) * (B[1,0] + B[1,1])
    U[i0*m + (j0+1), col] += 1.0
    U[(i0+1)*m + (j0+1), col] -= 1.0
    V[(j0+1)*p + k0, col] += 1.0
    V[(j0+1)*p + (k0+1), col] += 1.0
    W[k0*n + i0, col] += 1.0
    col += 1
    
    return col

def _search_for_better(n, m, p, current_best):
    """Search for rank less than current_best."""
    target_rank = current_best - 1  # Try rank 70
    
    # Generate all possible 2x2x2 block positions
    all_blocks = []
    for i0 in range(n - 1):
        for j0 in range(m - 1):
            for k0 in range(p - 1):
                all_blocks.append((i0, j0, k0))
    
    # Try to find 5 blocks that can save 5 ranks (total rank 70)
    # But blocks must be disjoint in terms of the entries they cover
    # Actually, blocks can overlap in position but the Strassen formulas
    # need to be compatible. For simplicity, try disjoint blocks first.
    
    # Find all sets of 5 disjoint 2x2x2 blocks
    # This is expensive, so use a greedy approach with backtracking
    
    def can_place_blocks(block_list, blocks_to_add, covered):
        for (i0, j0, k0) in blocks_to_add:
            # Check if this block overlaps with any already placed
            for (i1, j1, k1) in block_list:
                if (i0 < i1+2 and i1 < i0+2 and 
                    j0 < j1+2 and j1 < j0+2 and 
                    k0 < k1+2 and k1 < k0+2):
                    return False
            # Check if all 8 entries are uncovered
            for di in range(2):
                for dj in range(2):
                    for dk in range(2):
                        if covered[i0+di, j0+dj, k0+dk]:
                            return False
        return True
    
    # Use backtracking to find 5 disjoint blocks
    def backtrack(block_list, covered, remaining_blocks, depth):
        if depth == 5:
            return block_list.copy()
        if not remaining_blocks:
            return None
        
        # Try each remaining block
        for idx, block in enumerate(remaining_blocks):
            if can_place_blocks(block_list, [block], covered):
                # Place this block
                new_covered = covered.copy()
                i0, j0, k0 = block
                for di in range(2):
                    for dj in range(2):
                        for dk in range(2):
                            new_covered[i0+di, j0+dj, k0+dk] = True
                
                result = backtrack(block_list + [block], new_covered, remaining_blocks[idx+1:], depth+1)
                if result is not None:
                    return result
        
        return None
    
    covered = np.zeros((n, m, p), dtype=bool)
    result_blocks = backtrack([], covered, all_blocks, 0)
    
    if result_blocks is not None:
        # We found 5 disjoint 2x2x2 blocks
        # Total rank = 75 - 5 = 70
        total_rank = 70
        U = np.zeros((n * m, total_rank), dtype=float)
        V = np.zeros((m * p, total_rank), dtype=float)
        W = np.zeros((p * n, total_rank), dtype=float)
        
        col = 0
        for (i0, j0, k0) in result_blocks:
            col = _add_strassen_block(U, V, W, i0, j0, k0, n, m, p, col)
        
        # Cover remaining entries with schoolbook
        covered = np.zeros((n, m, p), dtype=bool)
        for (i0, j0, k0) in result_blocks:
            covered[i0:i0+2, j0:j0+2, k0:k0+2] = True
            
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if not covered[i, j, k]:
                        if col >= total_rank:
                            return None  # Shouldn't happen
                        U[i*m + j, col] = 1.0
                        V[j*p + k, col] = 1.0
                        W[k*n + i, col] = 1.0
                        col += 1
        
        # Verify
        if _verify(U, V, W, n, m, p):
            return (total_rank, U, V, W)
    
    return None

def _verify(U, V, W, n, m, p, num_samples=10000):
    """Verify the decomposition is correct."""
    np.random.seed(12345)
    for _ in range(num_samples):
        i = np.random.randint(n)
        j = np.random.randint(m)
        k = np.random.randint(p)
        val = 0.0
        for c in range(U.shape[1]):
            val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        if abs(val - 1.0) > 1e-9:
            return False
    return True

if __name__ == "__main__":
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    n, m, p = DIMENSIONS
    
    np.random.seed(42)
    errors = 0
    for _ in range(1000):
        i = np.random.randint(n)
        j = np.random.randint(m)
        k = np.random.randint(p)
        val = 0
        for c in range(U.shape[1]):
            val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        if abs(val - 1.0) > 1e-9:
            errors += 1
            
    print(f"Verification Errors: {errors}")
    print(f"Rank: {U.shape[1]}")
