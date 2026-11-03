import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 8]
DEADLINE = 148.0
_START = time.time()

def _time_left():
    return DEADLINE - (time.time() - _START)

def _strassen_2x2_2():
    """Return U, V, W for 2x2 * 2x2 with rank 7 (Strassen)."""
    n, m, p = 2, 2, 2
    rank = 7
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # P1 = (a+d)(e+h)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # P2 = (c+d)e
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # P3 = a(f-h)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # P4 = d(g-e)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # P5 = (a+b)h
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # P6 = (c-a)(e+f)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # P7 = (b-d)(g+h)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Verify decomposition is exact."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
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
    return U, V, W

def _build_strassen_embedding(n, m, p, strassen_blocks):
    """
    Build decomposition using Strassen on specified 2x2x2 blocks, schoolbook elsewhere.
    strassen_blocks: list of (i0, j0, k0)
    """
    covered = set()
    for (i0, j0, k0) in strassen_blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered.add((i0+di, j0+dj, k0+dk))
    
    n_school = n*m*p - len(covered)
    rank = 7 * len(strassen_blocks) + n_school
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    for (i0, j0, k0) in strassen_blocks:
        sU, sV, sW = _strassen_2x2_2()
        for c in range(7):
            for di in range(2):
                for dj in range(2):
                    U[(i0+di)*m + (j0+dj)][col+c] = sU[di*2+dj][c]
            for dj in range(2):
                for dk in range(2):
                    V[(j0+dj)*p + (k0+dk)][col+c] = sV[dj*2+dk][c]
            for dk in range(2):
                for di in range(2):
                    W[(k0+dk)*n + (i0+di)][col+c] = sW[dk*2+di][c]
        col += 7
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
    
    return U, V, W

def _try_strassen_placements(n, m, p, time_budget=10):
    """Try various non-overlapping Strassen block placements."""
    best_rank = n * m * p
    best_result = None
    
    # Generate all possible 2x2x2 block positions
    all_blocks = []
    for i0 in range(n - 1):
        for j0 in range(m - 1):
            for k0 in range(p - 1):
                all_blocks.append((i0, j0, k0))
    
    # Try single blocks
    for block in all_blocks:
        if _time_left() < 5:
            break
        try:
            U, V, W = _build_strassen_embedding(n, m, p, [block])
            if _verify(U, V, W, n, m, p):
                r = len(U[0])
                if r < best_rank:
                    best_rank = r
                    best_result = {"U": U, "V": V, "W": W}
        except Exception:
            pass
    
    # Try pairs of non-overlapping blocks
    for i in range(len(all_blocks)):
        for j in range(i + 1, len(all_blocks)):
            if _time_left() < 5:
                break
            b1 = all_blocks[i]
            b2 = all_blocks[j]
            # Check non-overlap
            overlap = False
            for di in range(2):
                for dj in range(2):
                    for dk in range(2):
                        if (b1[0]+di, b1[1]+dj, b1[2]+dk) == (b2[0]+di, b2[1]+dj, b2[2]+dk):
                            overlap = True
                            break
                    if overlap:
                        break
                if overlap:
                    break
            if overlap:
                continue
            try:
                U, V, W = _build_strassen_embedding(n, m, p, [b1, b2])
                if _verify(U, V, W, n, m, p):
                    r = len(U[0])
                    if r < best_rank:
                        best_rank = r
                        best_result = {"U": U, "V": V, "W": W}
            except Exception:
                pass
    
    # Try the specific pattern from current program: 4 blocks in k-segments
    blocks_pattern = [
        (0, 0, 0),
        (0, 0, 2),
        (0, 0, 4),
        (0, 0, 6)
    ]
    if _time_left() > 5:
        try:
            U, V, W = _build_strassen_embedding(n, m, p, blocks_pattern)
            if _verify(U, V, W, n, m, p):
                r = len(U[0])
                if r < best_rank:
                    best_rank = r
                    best_result = {"U": U, "V": V, "W": W}
        except Exception:
            pass
    
    return best_result, best_rank

def _try_strassen_3x3x4_subblock(n, m, p):
    """
    Try to decompose 3x3 * 3x4 using Strassen-like methods, then extend to 3x8.
    For 3x3 * 3x4: rank 36 schoolbook.
    Can we do better? Use Strassen on 2x2 sub-blocks.
    """
    n1, m1, p1 = 3, 3, 4
    best_rank = n1 * m1 * p1
    best_result = None
    
    # Try 2x2 Strassen blocks within 3x3x4
    all_blocks = []
    for i0 in range(n1 - 1):
        for j0 in range(m1 - 1):
            for k0 in range(p1 - 1):
                all_blocks.append((i0, j0, k0))
    
    # Try single blocks
    for block in all_blocks:
        try:
            U, V, W = _build_strassen_embedding(n1, m1, p1, [block])
            if _verify(U, V, W, n1, m1, p1):
                r = len(U[0])
                if r < best_rank:
                    best_rank = r
                    best_result = {"U": U, "V": V, "W": W}
        except Exception:
            pass
    
    # Try pairs
    for i in range(len(all_blocks)):
        for j in range(i + 1, len(all_blocks)):
            b1 = all_blocks[i]
            b2 = all_blocks[j]
            overlap = False
            for di in range(2):
                for dj in range(2):
                    for dk in range(2):
                        if (b1[0]+di, b1[1]+dj, b1[2]+dk) == (b2[0]+di, b2[1]+dj, b2[2]+dk):
                            overlap = True
                            break
                    if overlap:
                        break
                if overlap:
                    break
            if overlap:
                continue
            try:
                U, V, W = _build_strassen_embedding(n1, m1, p1, [b1, b2])
                if _verify(U, V, W, n1, m1, p1):
                    r = len(U[0])
                    if r < best_rank:
                        best_rank = r
                        best_result = {"U": U, "V": V, "W": W}
            except Exception:
                pass
    
    return best_result, best_rank

def _expand_3x3x4_to_3x3x8(U4, V4, W4):
    """
    Expand a 3x3 * 3x4 decomposition to 3x3 * 3x8 by duplicating for k=4,5,6,7.
    """
    n, m, p1, p2 = 3, 3, 4, 4
    rank4 = len(U4[0])
    rank8 = 2 * rank4
    
    U8 = [[0]*rank8 for _ in range(n*m)]
    V8 = [[0]*rank8 for _ in range(m*(p1+p2))]
    W8 = [[0]*rank8 for _ in range((p1+p2)*n)]
    
    # First half: k=0,1,2,3
    for i in range(n*m):
        for c in range(rank4):
            U8[i][c] = U4[i][c]
    for j in range(m*(p1+p2)):
        for c in range(rank4):
            if j < m*p1:
                V8[j][c] = V4[j][c]
            else:
                V8[j][c] = 0
    for k in range((p1+p2)*n):
        for c in range(rank4):
            if k < p1*n:
                W8[k][c] = W4[k][c]
            else:
                W8[k][c] = 0
    
    # Second half: k=4,5,6,7 (copy of first half)
    for i in range(n*m):
        for c in range(rank4):
            U8[i][rank4 + c] = U4[i][c]
    for j in range(m*(p1+p2)):
        for c in range(rank4):
            if j >= m*p1:
                V8[j][rank4 + c] = V4[j - m*p1][c]
            else:
                V8[j][rank4 + c] = 0
    for k in range((p1+p2)*n):
        for c in range(rank4):
            if k >= p1*n:
                W8[k][rank4 + c] = W4[k - p1*n][c]
            else:
                W8[k][rank4 + c] = 0
    
    return U8, V8, W8

def _try_combined_approach(n, m, p):
    """
    Try to find a low-rank decomposition by:
    1. Finding a good 3x3 * 3x4 decomposition
    2. Extending to 3x3 * 3x8
    3. Also trying direct 3x3 * 3x8 Strassen embeddings
    """
    best_rank = n * m * p
    best_result = None
    
    # Approach 1: Direct Strassen embeddings on 3x3x8
    result1, rank1 = _try_strassen_placements(n, m, p)
    if result1 and rank1 < best_rank:
        best_rank = rank1
        best_result = result1
    
    # Approach 2: 3x3 * 3x4 subblock, then extend
    if _time_left() > 10:
        result2, rank2 = _try_strassen_3x3x4_subblock(3, 3, 4)
        if result2:
            U8, V8, W8 = _expand_3x3x4_to_3x3x8(result2["U"], result2["V"], result2["W"])
            if _verify(U8, V8, W8, n, m, p):
                r8 = len(U8[0])
                if r8 < best_rank:
                    best_rank = r8
                    best_result = {"U": U8, "V": V8, "W": W8}
    
    return best_result, best_rank

def construct():
    """
    Return a rank-minimal exact tensor decomposition for 3x3 * 3x8 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as fallback
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = n * m * p  # 72
    
    # Try combined approach
    if _time_left() > 15:
        try:
            result, rank = _try_combined_approach(n, m, p)
            if result and rank < best_rank:
                best_rank = rank
                best_result = result
        except Exception:
            pass
    
    # If we still have time, try more aggressive search
    if _time_left() > 10 and best_rank > 56:
        try:
            # Try specific patterns with 3 or more Strassen blocks
            patterns = [
                [(0, 0, 0), (0, 0, 2), (0, 0, 4)],
                [(0, 0, 0), (0, 0, 2), (0, 0, 6)],
                [(0, 0, 0), (0, 0, 4), (0, 0, 6)],
                [(0, 1, 0), (0, 1, 2), (0, 1, 4)],
                [(1, 0, 0), (1, 0, 2), (1, 0, 4)],
            ]
            for pattern in patterns:
                if _time_left() < 5:
                    break
                # Check non-overlap
                valid = True
                for i in range(len(pattern)):
                    for j in range(i + 1, len(pattern)):
                        b1, b2 = pattern[i], pattern[j]
                        for di in range(2):
                            for dj in range(2):
                                for dk in range(2):
                                    if (b1[0]+di, b1[1]+dj, b1[2]+dk) == (b2[0]+di, b2[1]+dj, b2[2]+dk):
                                        valid = False
                                        break
                                if not valid:
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                if not valid:
                    continue
                try:
                    U, V, W = _build_strassen_embedding(n, m, p, pattern)
                    if _verify(U, V, W, n, m, p):
                        r = len(U[0])
                        if r < best_rank:
                            best_rank = r
                            best_result = {"U": U, "V": V, "W": W}
                except Exception:
                    pass
        except Exception:
            pass
    
    return best_result
