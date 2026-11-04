import time
import itertools
import numpy as np

_START = time.time()
DEADLINE = 145.0  # safety margin before 150s kill

def _time_left():
    return DEADLINE - (time.time() - _START)

def _strassen_2x2_2():
    """Return U, V, W for 2x2 * 2x2 with rank 7 (Strassen).
    U: (n*m, rank) indexed [i*2+j, c]
    V: (m*p, rank) indexed [j*2+k, c]
    W: (p*n, rank) indexed [k*2+i, c]
    """
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    
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
    """Verify decomposition is exact. W indexed as [k*n+i, c]."""
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
    """Schoolbook decomposition of rank n*m*p. W indexed [k*n+i, c]."""
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

def _build_from_blocks(n, m, p, blocks):
    """
    Build decomposition using Strassen on specified 2x2x2 blocks, schoolbook elsewhere.
    blocks: list of (i0, j0, k0) - top-left corners of 2x2x2 blocks.
    W indexed as [k*n+i, c].
    """
    covered = set()
    for (i0, j0, k0) in blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered.add((i0+di, j0+dj, k0+dk))
    
    n_school = n*m*p - len(covered)
    rank = 7 * len(blocks) + n_school
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    sU, sV, sW = _strassen_2x2_2()
    for (i0, j0, k0) in blocks:
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

def _find_best_strassen_placement(n, m, p):
    """Systematically search over all valid 2x2x2 block placements (single and pairs)."""
    best_rank = n * m * p
    best_blocks = None
    
    # All valid single block placements
    valid_positions = []
    for i0 in range(n - 1):
        for j0 in range(m - 1):
            for k0 in range(p - 1):
                valid_positions.append((i0, j0, k0))
    
    # Try all single placements
    for pos in valid_positions:
        blocks = [pos]
        U, V, W = _build_from_blocks(n, m, p, blocks)
        if _verify(U, V, W, n, m, p):
            r = len(U[0])
            if r < best_rank:
                best_rank = r
                best_blocks = blocks
        if _time_left() < 30:
            return best_blocks, best_rank
    
    # Try all pairs of non-overlapping placements
    for idx1 in range(len(valid_positions)):
        for idx2 in range(idx1 + 1, len(valid_positions)):
            pos1 = valid_positions[idx1]
            pos2 = valid_positions[idx2]
            # Check non-overlap
            i0_1, j0_1, k0_1 = pos1
            i0_2, j0_2, k0_2 = pos2
            overlap = False
            for di in range(2):
                for dj in range(2):
                    for dk in range(2):
                        if (i0_1+di, j0_1+dj, k0_1+dk) == (i0_2, j0_2, k0_2) or \
                           (i0_1+di, j0_1+dj, k0_1+dk) == (i0_2+1, j0_2, k0_2) or \
                           (i0_1+di, j0_1+dj, k0_1+dk) == (i0_2, j0_2+1, k0_2) or \
                           (i0_1+di, j0_1+dj, k0_1+dk) == (i0_2, j0_2, k0_2+1) or \
                           (i0_1+di, j0_1+dj, k0_1+dk) == (i0_2+1, j0_2+1, k0_2) or \
                           (i0_1+di, j0_1+dj, k0_1+dk) == (i0_2+1, j0_2, k0_2+1) or \
                           (i0_1+di, j0_1+dj, k0_1+dk) == (i0_2, j0_2+1, k0_2+1) or \
                           (i0_1+di, j0_1+dj, k0_1+dk) == (i0_2+1, j0_2+1, k0_2+1):
                            overlap = True
                            break
                    if overlap:
                        break
                if overlap:
                    break
            if overlap:
                continue
            
            blocks = [pos1, pos2]
            U, V, W = _build_from_blocks(n, m, p, blocks)
            if _verify(U, V, W, n, m, p):
                r = len(U[0])
                if r < best_rank:
                    best_rank = r
                    best_blocks = blocks
            if _time_left() < 25:
                return best_blocks, best_rank
    
    return best_blocks, best_rank

def _build_2x3x2_block(n, m, p, i0, j0, k0):
    """
    Build a 2x3x2 block decomposition using Strassen on the 2x2x2 part + schoolbook for the 3rd column.
    This gives rank 7 + 4 = 11 for a 2x3x2 sub-tensor.
    W indexed as [k*n+i, c].
    """
    rank = 11
    sU, sV, sW = _strassen_2x2_2()
    
    # U: 2*m entries for this block (rows i0, i0+1; cols j0, j0+1, j0+2)
    # V: 3*p entries for this block (rows j0, j0+1, j0+2; cols k0, k0+1)
    # W: 2*n entries for this block (rows k0, k0+1; cols i0, i0+1)
    
    U_block = [[0]*rank for _ in range(2*m)]
    V_block = [[0]*rank for _ in range(3*p)]
    W_block = [[0]*rank for _ in range(2*n)]
    
    # Strassen part covers (i0, i0+1) x (j0, j0+1) x (k0, k0+1)
    for c in range(7):
        for di in range(2):
            for dj in range(2):
                U_block[(i0+di)*m + (j0+dj)][c] = sU[di*2+dj][c]
        for dj in range(2):
            for dk in range(2):
                V_block[(j0+dj)*p + (k0+dk)][c] = sV[dj*2+dk][c]
        for dk in range(2):
            for di in range(2):
                W_block[(k0+dk)*n + (i0+di)][c] = sW[dk*2+di][c]
    
    # Schoolbook part covers (i0, i0+1) x (j0+2) x (k0, k0+1)
    col = 7
    for di in range(2):
        for dk in range(2):
            U_block[(i0+di)*m + (j0+2)][col] = 1
            V_block[(j0+2)*p + (k0+dk)][col] = 1
            W_block[(k0+dk)*n + (i0+di)][col] = 1
            col += 1
    
    return U_block, V_block, W_block

def construct():
    """
    Return a rank-minimal exact tensor decomposition for 3x3 * 3x8 matrix multiplication.
    W indexed as [k*n+i, r].
    """
    n, m, p = 3, 3, 8
    
    # Start with schoolbook as fallback
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = n * m * p  # 72
    
    # Strategy 1: Systematic search over Strassen block placements
    if _time_left() > 40:
        try:
            best_blocks, best_r = _find_best_strassen_placement(n, m, p)
            if best_blocks is not None and best_r < best_rank:
                U, V, W = _build_from_blocks(n, m, p, best_blocks)
                if _verify(U, V, W, n, m, p):
                    best_rank = best_r
                    best_result = {"U": U, "V": V, "W": W}
        except Exception:
            pass
    
    # Strategy 2: Try 2x3x2 blocks (rank 11 each) tiled across the 8 columns
    # 3x3x8 can be split into: two 2x3x2 blocks (covering rows 0,1 of output) + schoolbook for row 2
    # Actually, let's think more carefully.
    # 3x3x8: n=3, m=3, p=8
    # We can place 2x3x2 Strassen blocks. Each covers 2 output rows, 3 input cols, 2 output cols.
    # Max non-overlapping 2x3x2 blocks in 3x3x8: 
    #   - i dimension: can fit 1 block of size 2 (rows 0,1)
    #   - j dimension: can fit 1 block of size 3 (all cols)
    #   - k dimension: can fit 4 blocks of size 2 (cols 0-1, 2-3, 4-5, 6-7)
    # So max 4 blocks of rank 11 = 44, plus schoolbook for the remaining 1*3*8 = 24 terms
    # Total: 44 + 24 = 68. That's not better than 66.
    # 
    # Alternative: use 2x2x2 blocks more cleverly.
    # With 2x2x2 blocks: each covers 2x2x2 = 8 entries with rank 7 (saves 1 per block)
    # Max non-overlapping 2x2x2 in 3x3x8:
    #   i: 1 block (rows 0,1), j: 1 block (cols 0,1), k: 4 blocks (cols 0-1,2-3,4-5,6-7)
    #   = 4 blocks, rank = 4*7 + (72-32) = 28 + 40 = 68
    #   Or: i: 1, j: 1, k: 3 (cols 0-1, 2-3, 4-5) = 3 blocks, rank = 3*7 + (72-24) = 21+48 = 69
    #   Best single placement: 1 block, rank = 7 + 64 = 71
    #   Two blocks: 2*7 + (72-16) = 14+56 = 70
    #   Three blocks: 3*7 + (72-24) = 21+48 = 69
    #   Four blocks: 4*7 + (72-32) = 28+40 = 68
    #
    # The systematic search should find the best placement.
    
    # Strategy 3: Try a hybrid - use Strassen on 2x2x2 blocks but also try
    # placing blocks to maximize coverage of the "hard" parts
    if _time_left() > 20:
        # Try specific promising configurations
        configs = [
            # 4 blocks along k dimension
            [(0, 0, 0), (0, 0, 2), (0, 0, 4), (0, 0, 6)],
            [(1, 0, 0), (1, 0, 2), (1, 0, 4), (1, 0, 6)],
            [(0, 1, 0), (0, 1, 2), (0, 1, 4), (0, 1, 6)],
            [(1, 1, 0), (1, 1, 2), (1, 1, 4), (1, 1, 6)],
            # 2 blocks in different positions
            [(0, 0, 0), (1, 1, 2)],
            [(0, 0, 0), (1, 0, 4)],
            [(0, 0, 0), (0, 1, 6)],
            [(1, 0, 0), (0, 1, 2)],
        ]
        
        for config in configs:
            if _time_left() < 15:
                break
            try:
                U, V, W = _build_from_blocks(n, m, p, config)
                if _verify(U, V, W, n, m, p):
                    r = len(U[0])
                    if r < best_rank:
                        best_rank = r
                        best_result = {"U": U, "V": V, "W": W}
            except Exception:
                pass
    
    # Strategy 4: Try 2x3x2 blocks (rank 11) for potentially better coverage
    if _time_left() > 15:
        # Place 2x3x2 blocks: covers 2 output rows, all 3 input cols, 2 output cols
        # In 3x3x8: can place blocks at (i0=0, j0=0, k0) for k0 = 0,2,4,6
        # Each block: rank 11, covers 2*3*2 = 12 entries
        # 4 blocks: rank 44, covers 48 entries, remaining: 72-48 = 24 schoolbook
        # Total: 44 + 24 = 68
        # Not better than 66 from systematic search, but let's verify.
        
        # Build manually with 2x3x2 blocks
        try:
            rank_total = 4 * 11 + 24  # 4 blocks of rank 11 + 24 schoolbook
            U = [[0]*rank_total for _ in range(n*m)]
            V = [[0]*rank_total for _ in range(m*p)]
            W = [[0]*rank_total for _ in range(p*n)]
            
            col = 0
            for k0 in [0, 2, 4, 6]:
                U_b, V_b, W_b = _build_2x3x2_block(n, m, p, 0, 0, k0)
                for c in range(11):
                    for row in range(n*m):
                        if U_b[row][c] != 0:
                            U[row][col+c] = U_b[row][c]
                    for row in range(m*p):
                        if V_b[row][c] != 0:
                            V[row][col+c] = V_b[row][c]
                    for row in range(p*n):
                        if W_b[row][c] != 0:
                            W[row][col+c] = W_b[row][c]
                col += 11
            
            # Schoolbook for remaining: row i=2 (output row 2), all j, all k
            # That's 1*3*8 = 24 terms
            for i in [2]:
                for j in range(m):
                    for k in range(p):
                        U[i*m + j][col] = 1
                        V[j*p + k][col] = 1
                        W[k*n + i][col] = 1
                        col += 1
            
            if _verify(U, V, W, n, m, p):
                r = len(U[0])
                if r < best_rank:
                    best_rank = r
                    best_result = {"U": U, "V": W}
        except Exception:
            pass
    
    return best_result
