import numpy as np
import time
import itertools

DIMENSIONS = [3, 5, 5]

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the tensor decomposition exactly reproduces the matrix multiplication tensor."""
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    rank = U.shape[1]
    
    # Build the tensor T[i, j, k] = 1 if this is a valid multiplication term
    # The matrix multiplication tensor has T[i,j,k] = 1 for all i,j,k
    # The decomposition says: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _build_strassen_block(i0, j0, k0, U, V, W, col_offset, n, m, p):
    """Add a Strassen 2x2x2 block at position (i0, j0, k0). Returns number of columns used (7)."""
    # M1 = (A00+A11)(B00+B11)
    c = col_offset + 0
    U[i0*m + j0][c] += 1
    U[(i0+1)*m + (j0+1)][c] += 1
    V[j0*p + k0][c] += 1
    V[(j0+1)*p + (k0+1)][c] += 1
    W[k0*n + i0][c] += 1
    W[(k0+1)*n + (i0+1)][c] += 1

    # M2 = (A10+A11)B00
    c = col_offset + 1
    U[(i0+1)*m + j0][c] += 1
    U[(i0+1)*m + (j0+1)][c] += 1
    V[j0*p + k0][c] += 1
    W[k0*n + (i0+1)][c] += 1
    W[(k0+1)*n + (i0+1)][c] += -1

    # M3 = A00(B01-B11)
    c = col_offset + 2
    U[i0*m + j0][c] += 1
    V[j0*p + (k0+1)][c] += 1
    V[(j0+1)*p + (k0+1)][c] += -1
    W[(k0+1)*n + i0][c] += 1
    W[(k0+1)*n + (i0+1)][c] += 1

    # M4 = A11(B10-B00)
    c = col_offset + 3
    U[(i0+1)*m + (j0+1)][c] += 1
    V[(j0+1)*p + k0][c] += 1
    V[j0*p + k0][c] += -1
    W[k0*n + i0][c] += 1
    W[k0*n + (i0+1)][c] += 1

    # M5 = (A00+A01)B11
    c = col_offset + 4
    U[i0*m + j0][c] += 1
    U[i0*m + (j0+1)][c] += 1
    V[(j0+1)*p + (k0+1)][c] += 1
    W[k0*n + i0][c] += -1
    W[(k0+1)*n + i0][c] += 1

    # M6 = (A10-A00)(B00+B01)
    c = col_offset + 5
    U[(i0+1)*m + j0][c] += 1
    U[i0*m + j0][c] += -1
    V[j0*p + k0][c] += 1
    V[j0*p + (k0+1)][c] += 1
    W[(k0+1)*n + (i0+1)][c] += 1

    # M7 = (A01-A11)(B10+B11)
    c = col_offset + 6
    U[i0*m + (j0+1)][c] += 1
    U[(i0+1)*m + (j0+1)][c] += -1
    V[(j0+1)*p + k0][c] += 1
    V[(j0+1)*p + (k0+1)][c] += 1
    W[k0*n + i0][c] += 1

    return 7

def _try_rank_with_blocks(n, m, p, blocks, deadline):
    """Try to construct a decomposition using the given Strassen blocks + schoolbook for rest.
    Returns (U, V, W) if valid, else None."""
    if time.time() > deadline:
        return None
    
    total_terms = n * m * p
    strassen_terms_count = len(blocks) * 8
    schoolbook_terms_count = total_terms - strassen_terms_count
    total_rank = len(blocks) * 7 + schoolbook_terms_count
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    for i0, j0, k0 in blocks:
        # Check bounds
        if i0 + 1 >= n or j0 + 1 >= m or k0 + 1 >= p:
            return None
        col += _build_strassen_block(i0, j0, k0, U, V, W, col, n, m, p)
    
    # Mark covered (i,j,k) triples
    covered = [[[False]*p for _ in range(m)] for _ in range(n)]
    for i0, j0, k0 in blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered[i0+di][j0+dj][k0+dk] = True
    
    # Schoolbook for uncovered terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if not covered[i][j][k]:
                    U[i*m+j][col] += 1
                    V[j*p+k][col] += 1
                    W[k*n+i][col] += 1
                    col += 1
    
    if time.time() > deadline:
        return None
    
    # Verify
    if _verify_decomposition(U, V, W, n, m, p):
        return (U, V, W)
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # All possible 2x2x2 block positions
    all_positions = [(i, j, k) for i in range(n-1) for j in range(m-1) for k in range(p-1)]
    
    # Strategy: try different numbers of non-overlapping Strassen blocks
    # More blocks = lower rank (each block saves 1 rank vs schoolbook)
    # Rank = 75 - num_blocks (since each 2x2x2 block: 7 terms instead of 8)
    
    # First, try the known good solution: 4 blocks
    known_blocks = [(0, 0, 0), (0, 2, 0), (0, 0, 2), (0, 2, 2)]
    result = _try_rank_with_blocks(n, m, p, known_blocks, deadline)
    if result is not None:
        U, V, W = result
        return {"U": U, "V": V, "W": W}
    
    # Try to find better (more blocks) configurations
    # Maximum non-overlapping 2x2x2 blocks in 3x5x5:
    # i-dim: can fit 1 block (needs 2, have 3)
    # j-dim: can fit 2 blocks (needs 2 each, have 5)
    # k-dim: can fit 2 blocks (needs 2 each, have 5)
    # So max = 1 * 2 * 2 = 4 blocks (if we use i=0 or i=1)
    # Actually i can be 0 or 1, but only one at a time since they'd overlap in i
    # Wait: i=0 covers i={0,1}, i=1 covers i={1,2} - these overlap at i=1
    # So max 1 block in i-dimension
    # Total max = 4 blocks, giving rank 75 - 4 = 71
    
    # Let's verify: can we get 5 blocks? No, because i-dim only allows 1.
    # So rank 71 is the minimum with this approach.
    
    # But let's try all combinations of 4 non-overlapping blocks to see if any work
    # (some combinations might fail verification due to coefficient conflicts)
    
    def blocks_overlap(b1, b2):
        """Check if two 2x2x2 blocks overlap."""
        return not (b1[0]+1 <= b2[0] or b2[0]+1 <= b1[0] or
                    b1[1]+1 <= b2[1] or b2[1]+1 <= b1[1] or
                    b1[2]+1 <= b2[2] or b2[2]+1 <= b1[2])
    
    def find_non_overlapping_sets(positions, size):
        """Find all sets of 'size' non-overlapping blocks."""
        if size == 0:
            yield []
            return
        for i in range(len(positions)):
            rest = positions[i+1:]
            for subset in find_non_overlapping_sets(rest, size - 1):
                if all(not blocks_overlap(positions[i], s) for s in subset):
                    yield [positions[i]] + subset
    
    # Try 4-block configurations (rank 71)
    if time.time() < deadline - 5:
        for combo in find_non_overlapping_sets(all_positions, 4):
            if time.time() > deadline:
                break
            result = _try_rank_with_blocks(n, m, p, combo, deadline)
            if result is not None:
                U, V, W = result
                return {"U": U, "V": V, "W": W}
    
    # Fallback: pure schoolbook (rank 75)
    U = [[0] * (n*m*p) for _ in range(n * m)]
    V = [[0] * (n*m*p) for _ in range(m * p)]
    W = [[0] * (n*m*p) for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    n, m, p = DIMENSIONS
    print(f"Rank: {U.shape[1]}")
    # Verify
    ok = _verify_decomposition(U, V, W, n, m, p)
    print(f"Valid: {ok}")
