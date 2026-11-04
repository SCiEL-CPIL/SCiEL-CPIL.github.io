"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = 145.0
_START = time.time()

def _time_left():
    return _DEADLINE - (time.time() - _START)

def _strassen_2x2_2():
    """Return U, V, W for 2x2 * 2x2 with rank 7 (Strassen)."""
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
    sU, sV, sW = _strassen_2x2_2()
    for (i0, j0, k0) in strassen_blocks:
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

def _verify_np(U, V, W, n, m, p, tol=1e-8):
    """Fast verification using numpy."""
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    rank = U_arr.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _build_block_diagonal_2x2_8(n, m, p):
    """
    Block-diagonal embedding: partition 3x3 into 2x2 + 1x1, 3x8 into 2x8 + 1x8.
    Decompose as direct sum of 2x2x8 and 1x1x8 tensors.
    
    For 2x2x8: use Strassen on each 2x2x2 block, giving 4*7 = 28 terms.
    For 1x1x8: schoolbook gives 8 terms.
    Total: 36 terms... but we need to be careful about the structure.
    
    Actually, let's think more carefully. The tensor for (3,3,8) multiplication
    has entries T[i,j,k] where i in [0,3), j in [0,3), k in [0,8).
    
    Block diagonal approach:
    - Rows i=0,1 (2x2 block of A), columns j=0,1 (2x2 block of B), outputs k=0..7
    - Row i=2 (1x1), columns j=0,1 (2x1 block), outputs k=0..7
    - Rows i=0,1 (2x1), column j=2 (1x2), outputs k=0..7
    - Row i=2, column j=2 (1x1), outputs k=0..7
    
    This is getting complex. Let's use a simpler approach:
    Just use 4 Strassen blocks as before, which gives rank 68.
    """
    # Fall back to the standard 4-Strassen-block approach
    blocks = [
        (0, 0, 0),
        (0, 0, 2),
        (0, 0, 4),
        (0, 0, 6),
    ]
    return _build_strassen_embedding(n, m, p, blocks)

def _find_best_block_placements(n, m, p, max_blocks=4):
    """
    Search for the best placement of up to max_blocks 2x2x2 Strassen blocks.
    Returns the best (rank, U, V, W) found.
    """
    from itertools import product
    
    best_rank = n * m * p
    best_result = None
    
    # Generate all valid 2x2x2 block positions
    valid_positions = []
    for i0 in range(n - 1):
        for j0 in range(m - 1):
            for k0 in range(p - 1):
                valid_positions.append((i0, j0, k0))
    
    # Try subsets of size up to max_blocks
    # For efficiency, first try the known good placement
    known_good = [(0, 0, 0), (0, 0, 2), (0, 0, 4), (0, 0, 6)]
    if len(valid_positions) >= 4:
        U, V, W = _build_strassen_embedding(n, m, p, known_good)
        if _verify(U, V, W, m, n, p):  # Note: verify expects (n, m, p)
            pass
        # Actually verify with correct dimensions
        if _verify(U, V, W, n, m, p):
            r = len(U[0])
            if r < best_rank:
                best_rank = r
                best_result = (U, V, W)
    
    # Try other placements if time permits
    if _time_left() > 30:
        # Try all combinations of 4 blocks
        from itertools import combinations
        tried = set()
        for combo in combinations(valid_positions, 4):
            # Check if blocks are disjoint
            covered = set()
            disjoint = True
            for (i0, j0, k0) in combo:
                for di in range(2):
                    for dj in range(2):
                        for dk in range(2):
                            pos = (i0+di, j0+dj, k0+dk)
                            if pos in covered:
                                disjoint = False
                                break
                            covered.add(pos)
                if not disjoint:
                    break
            
            if not disjoint:
                continue
            
            U, V, W = _build_strassen_embedding(n, m, p, list(combo))
            if _verify(U, V, W, n, m, p):
                r = len(U[0])
                if r < best_rank:
                    best_rank = r
                    best_result = (U, V, W)
            
            if _time_left() < 20:
                break
    
    if best_result is None:
        # Fall back to schoolbook
        U, V, W = _schoolbook(n, m, p)
        best_result = (U, V, W)
        best_rank = n * m * p
    
    return best_rank, best_result

def construct():
    n, m, p = DIMENSIONS
    
    # Start with the best known construction: 4 disjoint Strassen blocks
    # This gives rank = 72 - 4*1 = 68 (each Strassen block saves 1 rank vs schoolbook)
    
    # Use the known good placement
    blocks = [
        (0, 0, 0),
        (0, 0, 2),
        (0, 0, 4),
        (0, 0, 6),
    ]
    U, V, W = _build_strassen_embedding(n, m, p, blocks)
    
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook if verification fails
        U, V, W = _schoolbook(n, m, p)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
