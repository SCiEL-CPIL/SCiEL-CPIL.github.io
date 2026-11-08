import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 5, 5]

def _construct_rank_71():
    """
    Returns a rank-71 decomposition for 3x5x5 matrix multiplication.
    Uses 4 disjoint 2x2x2 Strassen blocks and schoolbook for the rest.
    """
    n, m, p = 3, 5, 5
    
    blocks = [
        (0, 0, 0),
        (0, 2, 0),
        (0, 0, 2),
        (0, 2, 2)
    ]
    
    total_strassen_rank = 4 * 7
    total_terms = n * m * p
    strassen_terms_count = 4 * 8
    schoolbook_terms_count = total_terms - strassen_terms_count
    total_rank = total_strassen_rank + schoolbook_terms_count
    
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(i0, j0, k0, col_offset):
        c = col_offset + 0
        U[i0*m + j0][c] += 1.0
        U[(i0+1)*m + (j0+1)][c] += 1.0
        V[j0*p + k0][c] += 1.0
        V[(j0+1)*p + (k0+1)][c] += 1.0
        W[k0*n + i0][c] += 1.0
        W[(k0+1)*n + (i0+1)][c] += 1.0
        
        c = col_offset + 1
        U[(i0+1)*m + j0][c] += 1.0
        U[(i0+1)*m + (j0+1)][c] += 1.0
        V[j0*p + k0][c] += 1.0
        W[k0*n + (i0+1)][c] += 1.0
        W[(k0+1)*n + (i0+1)][c] += -1.0
        
        c = col_offset + 2
        U[i0*m + j0][c] += 1.0
        V[j0*p + (k0+1)][c] += 1.0
        V[(j0+1)*p + (k0+1)][c] += -1.0
        W[(k0+1)*n + i0][c] += 1.0
        W[(k0+1)*n + (i0+1)][c] += 1.0
        
        c = col_offset + 3
        U[(i0+1)*m + (j0+1)][c] += 1.0
        V[(j0+1)*p + k0][c] += 1.0
        V[j0*p + k0][c] += -1.0
        W[k0*n + i0][c] += 1.0
        W[k0*n + (i0+1)][c] += 1.0
        
        c = col_offset + 4
        U[i0*m + j0][c] += 1.0
        U[i0*m + (j0+1)][c] += 1.0
        V[(j0+1)*p + (k0+1)][c] += 1.0
        W[k0*n + i0][c] += -1.0
        W[(k0+1)*n + i0][c] += 1.0
        
        c = col_offset + 5
        U[(i0+1)*m + j0][c] += 1.0
        U[i0*m + j0][c] += -1.0
        V[j0*p + k0][c] += 1.0
        V[j0*p + (k0+1)][c] += 1.0
        W[(k0+1)*n + (i0+1)][c] += 1.0
        
        c = col_offset + 6
        U[i0*m + (j0+1)][c] += 1.0
        U[(i0+1)*m + (j0+1)][c] += -1.0
        V[(j0+1)*p + k0][c] += 1.0
        V[(j0+1)*p + (k0+1)][c] += 1.0
        W[k0*n + i0][c] += 1.0

    for i0, j0, k0 in blocks:
        add_strassen_block(i0, j0, k0, col)
        col += 7
    
    covered_3d = [[[False]*p for _ in range(m)] for _ in range(n)]
    for i0, j0, k0 in blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered_3d[i0+di][j0+dj][k0+dk] = True
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if not covered_3d[i][j][k]:
                    U[i*m+j][col] += 1.0
                    V[j*p+k][col] += 1.0
                    W[k*n+i][col] += 1.0
                    col += 1
                    
    return {"U": U, "V": V, "W": W}

def _blocks_overlap(b1, b2):
    """Check if two 2x2x2 blocks overlap in all three dimensions."""
    i01, j01, k01 = b1
    i02, j02, k02 = b2
    # i ranges: [i0, i0+1]
    i_overlap = not (i01+1 < i02 or i02+1 < i01)
    j_overlap = not (j01+1 < j02 or j02+1 < j01)
    k_overlap = not (k01+1 < k02 or k02+1 < k01)
    return i_overlap and j_overlap and k_overlap

def _find_best_block_config(num_blocks, n=3, m=5, p=5):
    """
    Find a configuration of num_blocks non-overlapping 2x2x2 blocks
    that maximizes coverage (minimizes schoolbook terms).
    Returns list of (i0,j0,k0) tuples or None if not found.
    """
    # Possible block positions
    i0_range = range(n - 1)  # 0 to n-2
    j0_range = range(m - 1)  # 0 to m-2
    k0_range = range(p - 1)  # 0 to p-2
    
    all_positions = list(product(i0_range, j0_range, k0_range))
    
    # For small num_blocks, we can do combinatorial search
    # But 32 choose 5 is 201376, which is manageable
    from itertools import combinations
    
    best_coverage = -1
    best_blocks = None
    
    # For efficiency, first filter out positions that are clearly bad
    # Actually, let's just enumerate
    for combo in combinations(all_positions, num_blocks):
        # Check if any two blocks overlap
        valid = True
        for i in range(num_blocks):
            for j in range(i+1, num_blocks):
                if _blocks_overlap(combo[i], combo[j]):
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            continue
        
        # Calculate coverage
        covered = set()
        for i0, j0, k0 in combo:
            for di in range(2):
                for dj in range(2):
                    for dk in range(2):
                        covered.add((i0+di, j0+dj, k0+dk))
        
        coverage = len(covered)
        if coverage > best_coverage:
            best_coverage = coverage
            best_blocks = combo
    
    return best_blocks

def _construct_from_blocks(blocks, n=3, m=5, p=5):
    """Construct decomposition from given Strassen blocks."""
    num_blocks = len(blocks)
    total_strassen_rank = num_blocks * 7
    total_terms = n * m * p
    strassen_terms_count = num_blocks * 8
    
    # Calculate actual coverage
    covered_3d = [[[False]*p for _ in range(m)] for _ in range(n)]
    for i0, j0, k0 in blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    if 0 <= i0+di < n and 0 <= j0+dj < m and 0 <= k0+dk < p:
                        covered_3d[i0+di][j0+dj][k0+dk] = True
    
    schoolbook_count = sum(1 for i in range(n) for j in range(m) for k in range(p) if not covered_3d[i][j][k])
    total_rank = total_strassen_rank + schoolbook_count
    
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(i0, j0, k0, col_offset):
        c = col_offset + 0
        U[i0*m + j0][c] += 1.0
        U[(i0+1)*m + (j0+1)][c] += 1.0
        V[j0*p + k0][c] += 1.0
        V[(j0+1)*p + (k0+1)][c] += 1.0
        W[k0*n + i0][c] += 1.0
        W[(k0+1)*n + (i0+1)][c] += 1.0
        
        c = col_offset + 1
        U[(i0+1)*m + j0][c] += 1.0
        U[(i0+1)*m + (j0+1)][c] += 1.0
        V[j0*p + k0][c] += 1.0
        W[k0*n + (i0+1)][c] += 1.0
        W[(k0+1)*n + (i0+1)][c] += -1.0
        
        c = col_offset + 2
        U[i0*m + j0][c] += 1.0
        V[j0*p + (k0+1)][c] += 1.0
        V[(j0+1)*p + (k0+1)][c] += -1.0
        W[(k0+1)*n + i0][c] += 1.0
        W[(k0+1)*n + (i0+1)][c] += 1.0
        
        c = col_offset + 3
        U[(i0+1)*m + (j0+1)][c] += 1.0
        V[(j0+1)*p + k0][c] += 1.0
        V[j0*p + k0][c] -= 1.0
        W[k0*n + i0][c] += 1.0
        W[k0*n + (i0+1)][c] += 1.0
        
        c = col_offset + 4
        U[i0*m + j0][c] += 1.0
        U[i0*m + (j0+1)][c] += 1.0
        V[(j0+1)*p + (k0+1)][c] += 1.0
        W[k0*n + i0][c] += -1.0
        W[(k0+1)*n + i0][c] += 1.0
        
        c = col_offset + 5
        U[(i0+1)*m + j0][c] += 1.0
        U[i0*m + j0][c] += -1.0
        V[j0*p + k0][c] += 1.0
        V[j0*p + (k0+1)][c] += 1.0
        W[(k0+1)*n + (i0+1)][c] += 1.0
        
        c = col_offset + 6
        U[i0*m + (j0+1)][c] += 1.0
        U[(i0+1)*m + (j0+1)][c] += -1.0
        V[(j0+1)*p + k0][c] += 1.0
        V[(j0+1)*p + (k0+1)][c] += 1.0
        W[k0*n + i0][c] += 1.0

    for i0, j0, k0 in blocks:
        add_strassen_block(i0, j0, k0, col)
        col += 7
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if not covered_3d[i][j][k]:
                    U[i*m+j][col] += 1.0
                    V[j*p+k][col] += 1.0
                    W[k*n+i][col] += 1.0
                    col += 1
                    
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(result, n=3, m=5, p=5, tol=1e-10):
    """Verify that the decomposition is correct."""
    U = np.array(result["U"])
    V = np.array(result["V"])
    W = np.array(result["W"])
    rank = U.shape[1]
    
    U_3d = U.reshape(n, m, rank)
    V_3d = V.reshape(m, p, rank)
    W_3d = W.reshape(p, n, rank)
    
    T = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d, optimize=True)
    
    max_err = np.max(np.abs(T - 1.0))
    return max_err < tol, max_err

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x5x5 matrix multiplication.
    Returns dict with 'U', 'V', 'W' factor matrices.
    """
    start_time = time.time()
    deadline = 115.0  # Leave 5s margin
    
    n, m, p = 3, 5, 5
    
    # Start with known rank-71 solution
    base_result = _construct_rank_71()
    best_result = base_result
    best_rank = 71
    
    # Try to find better configurations with more Strassen blocks
    # Try 5 blocks first (target rank 70)
    if time.time() < deadline:
        try:
            blocks5 = _find_best_block_config(5, n, m, p)
            if blocks5 is not None:
                result5 = _construct_from_blocks(blocks5, n, m, p)
                valid, err = _verify_decomposition(result5, n, m, p)
                if valid:
                    rank5 = len(np.array(result5["U"]).shape[1] if isinstance(result5["U"], list) else result5["U"].shape[1])
                    # Actually, let's get rank properly
                    U5 = np.array(result5["U"])
                    rank5 = U5.shape[1]
                    if rank5 < best_rank:
                        best_result = result5
                        best_rank = rank5
        except Exception:
            pass
    
    # Try 6 blocks (target rank 69)
    if time.time() < deadline:
        try:
            blocks6 = _find_best_block_config(6, n, m, p)
            if blocks6 is not None:
                result6 = _construct_from_blocks(blocks6, n, m, p)
                valid, err = _verify_decomposition(result6, n, m, p)
                if valid:
                    U6 = np.array(result6["U"])
                    rank6 = U6.shape[1]
                    if rank6 < best_rank:
                        best_result = result6
                        best_rank = rank6
        except Exception:
            pass
    
    # Try 7 blocks (target rank 68)
    if time.time() < deadline:
        try:
            blocks7 = _find_best_block_config(7, n, m, p)
            if blocks7 is not None:
                result7 = _construct_from_blocks(blocks7, n, m, p)
                valid, err = _verify_decomposition(result7, n, m, p)
                if valid:
                    U7 = np.array(result7["U"])
                    rank7 = U7.shape[1]
                    if rank7 < best_rank:
                        best_result = result7
                        best_rank = rank7
        except Exception:
            pass
    
    return best_result

if __name__ == "__main__":
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    n, m, p = DIMENSIONS
    print(f"Rank: {U.shape[1]}")
    
    # Verification
    U_3d = U.reshape(n, m, U.shape[1])
    V_3d = V.reshape(m, p, V.shape[1])
    W_3d = W.reshape(p, n, W.shape[1])
    T = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d, optimize=True)
    max_err = np.max(np.abs(T - 1.0))
    print(f"Max error: {max_err}")
