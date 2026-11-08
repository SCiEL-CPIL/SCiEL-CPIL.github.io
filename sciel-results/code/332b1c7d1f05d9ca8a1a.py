import numpy as np
import time
from itertools import combinations

DIMENSIONS = [3, 5, 5]

def _build_decomposition(blocks, n=3, m=5, p=5):
    """Build decomposition given a list of 2x2x2 Strassen block origins."""
    total_strassen_rank = len(blocks) * 7
    
    covered_3d = [[[False]*p for _ in range(m)] for _ in range(n)]
    for i0, j0, k0 in blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    if i0+di < n and j0+dj < m and k0+dk < p:
                        covered_3d[i0+di][j0+dj][k0+dk] = True
    
    schoolbook_count = sum(1 for i in range(n) for j in range(m) for k in range(p) 
                          if not covered_3d[i][j][k])
    total_rank = total_strassen_rank + schoolbook_count
    
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    col = 0
    for i0, j0, k0 in blocks:
        # M1 = (A00+A11)(B00+B11)
        c = col
        U[i0*m + j0][c] += 1; U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1; V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1; W[(k0+1)*n + (i0+1)][c] += 1
        col += 1
        
        # M2 = (A10+A11)B00
        c = col
        U[(i0+1)*m + j0][c] += 1; U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        W[k0*n + (i0+1)][c] += 1; W[(k0+1)*n + (i0+1)][c] += -1
        col += 1
        
        # M3 = A00(B01-B11)
        c = col
        U[i0*m + j0][c] += 1
        V[j0*p + (k0+1)][c] += 1; V[(j0+1)*p + (k0+1)][c] += -1
        W[(k0+1)*n + i0][c] += 1; W[(k0+1)*n + (i0+1)][c] += 1
        col += 1
        
        # M4 = A11(B10-B00)
        c = col
        U[(i0+1)*m + (j0+1)][c] += 1
        V[(j0+1)*p + k0][c] += 1; V[j0*p + k0][c] += -1
        W[k0*n + i0][c] += 1; W[k0*n + (i0+1)][c] += 1
        col += 1
        
        # M5 = (A00+A01)B11
        c = col
        U[i0*m + j0][c] += 1; U[i0*m + (j0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += -1; W[(k0+1)*n + i0][c] += 1
        col += 1
        
        # M6 = (A10-A00)(B00+B01)
        c = col
        U[(i0+1)*m + j0][c] += 1; U[i0*m + j0][c] += -1
        V[j0*p + k0][c] += 1; V[j0*p + (k0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        col += 1
        
        # M7 = (A01-A11)(B10+B11)
        c = col
        U[i0*m + (j0+1)][c] += 1; U[(i0+1)*m + (j0+1)][c] += -1
        V[(j0+1)*p + k0][c] += 1; V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
        col += 1
    
    # Schoolbook for uncovered entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if not covered_3d[i][j][k]:
                    U[i*m+j][col] += 1
                    V[j*p+k][col] += 1
                    W[k*n+i][col] += 1
                    col += 1
    
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(decomp, n=3, m=5, p=5, tol=1e-9):
    """Verify that the decomposition is exact."""
    U = np.array(decomp["U"])
    V = np.array(decomp["V"])
    W = np.array(decomp["W"])
    rank = U.shape[1]
    
    # T_ijk = sum_c U_{i*m+j, c} * V_{j*p+k, c} * W_{k*n+i, c}
    # Should equal 1 for all i,j,k (since matrix mult tensor has all 1s)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s
    
    n, m, p = 3, 5, 5
    
    # Start with known rank-71 solution
    default_blocks = [(0,0,0), (0,2,0), (0,0,2), (0,2,2)]
    best_decomp = _build_decomposition(default_blocks, n, m, p)
    best_rank = 71
    
    # Verify the default solution
    if not _verify_decomposition(best_decomp, n, m, p):
        raise RuntimeError("Default decomposition failed verification")
    
    # Search for better block configurations
    # Try different numbers of blocks and different placements
    i_choices = list(combinations(range(n), 2))  # (0,1), (0,2), (1,2)
    j_choices = list(combinations(range(m), 2))  # 10 choices
    k_choices = list(combinations(range(p), 2))  # 10 choices
    
    # For each block, the origin is (min_i, min_j, min_k)
    # Try all subsets of size 3, 4, 5, 6, 7, 8 from the possible blocks
    # But that's too many. Instead, try greedy: pick blocks that maximize coverage
    
    # Generate all possible block origins
    all_blocks = []
    for i0, i1 in i_choices:
        for j0, j1 in j_choices:
            for k0, k1 in k_choices:
                all_blocks.append((i0, j0, k0))
    
    # Try all subsets of size 4 (same as current) with different placements
    # C(30, 4) = 27405, manageable
    from itertools import combinations as comb
    
    tried = set()
    best_found_rank = best_rank
    
    # First try all size-4 subsets
    for block_set in comb(range(len(all_blocks)), 4):
        if time.time() - start_time > deadline:
            break
        blocks = [all_blocks[i] for i in block_set]
        # Check no overlapping in a way that breaks correctness
        # Actually, overlapping blocks are fine as long as we only use schoolbook for uncovered
        # But two blocks can't both claim the same (i,j,k) with different decompositions
        # So blocks must be disjoint in their 2x2x2 subcubes
        covered = [[[False]*p for _ in range(m)] for _ in range(n)]
        valid = True
        for i0, j0, k0 in blocks:
            for di in range(2):
                for dj in range(2):
                    for dk in range(2):
                        ii, jj, kk = i0+di, j0+dj, k0+dk
                        if ii < n and jj < m and kk < p:
                            if covered[ii][jj][kk]:
                                valid = False
                                break
                            covered[ii][jj][kk] = True
            if not valid:
                break
        if not valid:
            continue
        
        schoolbook = sum(1 for i in range(n) for j in range(m) for k in range(p) 
                        if not covered[i][j][k])
        total_rank = 4 * 7 + schoolbook
        if total_rank < best_found_rank:
            decomp = _build_decomposition(blocks, n, m, p)
            if _verify_decomposition(decomp, n, m, p):
                best_found_rank = total_rank
                best_decomp = decomp
    
    # Try size-5 subsets
    if time.time() - start_time < deadline:
        for block_set in comb(range(len(all_blocks)), 5):
            if time.time() - start_time > deadline:
                break
            blocks = [all_blocks[i] for i in block_set]
            covered = [[[False]*p for _ in range(m)] for _ in range(n)]
            valid = True
            for i0, j0, k0 in blocks:
                for di in range(2):
                    for dj in range(2):
                        for dk in range(2):
                            ii, jj, kk = i0+di, j0+dj, k0+dk
                            if ii < n and jj < m and kk < p:
                                if covered[ii][jj][kk]:
                                    valid = False
                                    break
                                covered[ii][jj][kk] = True
                if not valid:
                    break
            if not valid:
                continue
            
            schoolbook = sum(1 for i in range(n) for j in range(m) for k in range(p) 
                            if not covered[i][j][k])
            total_rank = 5 * 7 + schoolbook
            if total_rank < best_found_rank:
                decomp = _build_decomposition(blocks, n, m, p)
                if _verify_decomposition(decomp, n, m, p):
                    best_found_rank = total_rank
                    best_decomp = decomp
    
    # Try size-6 subsets
    if time.time() - start_time < deadline:
        for block_set in comb(range(len(all_blocks)), 6):
            if time.time() - start_time > deadline:
                break
            blocks = [all_blocks[i] for i in block_set]
            covered = [[[False]*p for _ in range(m)] for _ in range(n)]
            valid = True
            for i0, j0, k0 in blocks:
                for di in range(2):
                    for dj in range(2):
                        for dk in range(2):
                            ii, jj, kk = i0+di, j0+dj, k0+dk
                            if ii < n and jj < m and kk < p:
                                if covered[ii][jj][kk]:
                                    valid = False
                                    break
                                covered[ii][jj][kk] = True
                if not valid:
                    break
            if not valid:
                continue
            
            schoolbook = sum(1 for i in range(n) for j in range(m) for k in range(p) 
                            if not covered[i][j][k])
            total_rank = 6 * 7 + schoolbook
            if total_rank < best_found_rank:
                decomp = _build_decomposition(blocks, n, m, p)
                if _verify_decomposition(decomp, n, m, p):
                    best_found_rank = total_rank
                    best_decomp = decomp
    
    # Try size-7 subsets
    if time.time() - start_time < deadline:
        for block_set in comb(range(len(all_blocks)), 7):
            if time.time() - start_time > deadline:
                break
            blocks = [all_blocks[i] for i in block_set]
            covered = [[[False]*p for _ in range(m)] for _ in range(n)]
            valid = True
            for i0, j0, k0 in blocks:
                for di in range(2):
                    for dj in range(2):
                        for dk in range(2):
                            ii, jj, kk = i0+di, j0+dj, k0+dk
                            if ii < n and jj < m and kk < p:
                                if covered[ii][jj][kk]:
                                    valid = False
                                    break
                                covered[ii][jj][kk] = True
                if not valid:
                    break
            if not valid:
                continue
            
            schoolbook = sum(1 for i in range(n) for j in range(m) for k in range(p) 
                            if not covered[i][j][k])
            total_rank = 7 * 7 + schoolbook
            if total_rank < best_found_rank:
                decomp = _build_decomposition(blocks, n, m, p)
                if _verify_decomposition(decomp, n, m, p):
                    best_found_rank = total_rank
                    best_decomp = decomp
    
    # Try size-8 subsets
    if time.time() - start_time < deadline:
        for block_set in comb(range(len(all_blocks)), 8):
            if time.time() - start_time > deadline:
                break
            blocks = [all_blocks[i] for i in block_set]
            covered = [[[False]*p for _ in range(m)] for _ in range(n)]
            valid = True
            for i0, j0, k0 in blocks:
                for di in range(2):
                    for dj in range(2):
                        for dk in range(2):
                            ii, jj, kk = i0+di, j0+dj, k0+dk
                            if ii < n and jj < m and kk < p:
                                if covered[ii][jj][kk]:
                                    valid = False
                                    break
                                covered[ii][jj][kk] = True
                if not valid:
                    break
            if not valid:
                continue
            
            schoolbook = sum(1 for i in range(n) for j in range(m) for k in range(p) 
                            if not covered[i][j][k])
            total_rank = 8 * 7 + schoolbook
            if total_rank < best_found_rank:
                decomp = _build_decomposition(blocks, n, m, p)
                if _verify_decomposition(decomp, n, m, p):
                    best_found_rank = total_rank
                    best_decomp = decomp
    
    return best_decomp

if __name__ == "__main__":
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    n, m, p = DIMENSIONS
    print(f"Rank: {U.shape[1]}")
    # Verify
    rank = U.shape[1]
    ok = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(rank))
                if abs(val - 1.0) > 1e-9:
                    ok = False
                    print(f"FAIL at ({i},{j},{k}): {val}")
                    break
            if not ok:
                break
        if not ok:
            break
    if ok:
        print("Verification passed")
    else:
        print("Verification FAILED")
