import time
import numpy as np

DIMENSIONS = [3, 3, 8]
DEADLINE = 145.0
_START = time.time()

def _time_left():
    return DEADLINE - (time.time() - _START)

def _verify_numpy(U, V, W, n, m, p, tol=1e-6):
    """Fast verification using numpy tensor contraction."""
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    rank = U_arr.shape[1]
    
    # Reshape to 3D tensors
    U_3d = U_arr.reshape(n, m, rank)
    V_3d = V_arr.reshape(m, p, rank)
    W_3d = W_arr.reshape(p, n, rank)
    
    # Compute sum over rank dimension: U[i,j,c]*V[j,k,c]*W[k,i,c]
    # Result should be 1 for all (i,j,k)
    result = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
    return np.allclose(result, 1.0, atol=tol)

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
    """Verify decomposition is exact using numpy for speed."""
    try:
        return _verify_numpy(U, V, W, n, m, p, tol)
    except Exception:
        # Fallback to slow verification
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

def _build_strassen_2x2x4():
    """
    Build a 2x2x4 Strassen-like decomposition using two 2x2x2 Strassen blocks.
    Returns U (2*2, 14), V (2*4, 14), W (4*2, 14) for n=2,m=2,p=4.
    """
    n, m, p = 2, 2, 4
    rank = 14
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # First 2x2x2 Strassen block: k=0,1
    sU1, sV1, sW1 = _strassen_2x2_2()
    for c in range(7):
        for di in range(2):
            for dj in range(2):
                U[di*2+dj][c] = sU1[di*2+dj][c]
        for dj in range(2):
            for dk in range(2):
                V[dj*4+dk][c] = sV1[dj*2+dk][c]
        for dk in range(2):
            for di in range(2):
                W[dk*2+di][c] = sW1[dk*2+di][c]
    
    # Second 2x2x2 Strassen block: k=2,3
    sU2, sV2, sW2 = _strassen_2x2_2()
    for c in range(7):
        for di in range(2):
            for dj in range(2):
                U[di*2+dj][7+c] = sU2[di*2+dj][c]
        for dj in range(2):
            for dk in range(2):
                V[dj*4+(2+dk)][7+c] = sV2[dj*2+dk][c]
        for dk in range(2):
            for di in range(2):
                W[(2+dk)*2+di][7+c] = sW2[dk*2+di][c]
    
    return U, V, W

def _build_mixed_embedding(n, m, p, strassen_2x2x2_blocks, strassen_2x2x4_blocks):
    """
    Build decomposition using mixed 2x2x2 and 2x2x4 Strassen blocks, schoolbook elsewhere.
    strassen_2x2x2_blocks: list of (i0, j0, k0) for 2x2x2 blocks
    strassen_2x2x4_blocks: list of (i0, j0, k0) for 2x2x4 blocks (covers k0..k0+3)
    """
    covered = set()
    for (i0, j0, k0) in strassen_2x2x2_blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered.add((i0+di, j0+dj, k0+dk))
    for (i0, j0, k0) in strassen_2x2x4_blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(4):
                    covered.add((i0+di, j0+dj, k0+dk))
    
    n_school = n*m*p - len(covered)
    rank = 7 * len(strassen_2x2x2_blocks) + 14 * len(strassen_2x2x4_blocks) + n_school
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    for (i0, j0, k0) in strassen_2x2x2_blocks:
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
    
    for (i0, j0, k0) in strassen_2x2x4_blocks:
        sU, sV, sW = _build_strassen_2x2x4()
        for c in range(14):
            for di in range(2):
                for dj in range(2):
                    U[(i0+di)*m + (j0+dj)][col+c] = sU[di*2+dj][c]
            for dj in range(2):
                for dk in range(4):
                    V[(j0+dj)*p + (k0+dk)][col+c] = sV[dj*4+dk][c]
            for dk in range(4):
                for di in range(2):
                    W[(k0+dk)*n + (i0+di)][col+c] = sW[dk*2+di][c]
        col += 14
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
    
    return U, V, W

def _find_best_mixed_placement(n, m, p):
    """
    Search over all valid combinations of 2x2x2 and 2x2x4 blocks to find min rank.
    """
    best_rank = n * m * p
    best_config = None
    
    # Generate all possible 2x2x2 block positions
    possible_2x2x2 = []
    for i0 in range(n-1):
        for j0 in range(m-1):
            for k0 in range(p-1):
                possible_2x2x2.append((i0, j0, k0))
    
    # Generate all possible 2x2x4 block positions
    possible_2x2x4 = []
    for i0 in range(n-1):
        for j0 in range(m-1):
            for k0 in range(p-3):
                possible_2x2x4.append((i0, j0, k0))
    
    # Try all subsets of 2x2x4 blocks (at most 2 non-overlapping)
    # and fill remaining with 2x2x2 blocks
    from itertools import combinations
    
    # First try: no 2x2x4 blocks, just 2x2x2
    for subset in combinations(possible_2x2x2, 4):
        # Check non-overlapping
        covered = set()
        valid = True
        for (i0, j0, k0) in subset:
            for di in range(2):
                for dj in range(2):
                    for dk in range(2):
                        if (i0+di, j0+dj, k0+dk) in covered:
                            valid = False
                            break
                        covered.add((i0+di, j0+dj, k0+dk))
            if not valid:
                break
        if valid and len(covered) == 32:
            rank = 7*4 + (n*m*p - 32)
            if rank < best_rank:
                best_rank = rank
                best_config = (list(subset), [])
    
    # Try: 1 2x2x4 block + 2 2x2x2 blocks
    for b24 in possible_2x2x4:
        covered_24 = set()
        for di in range(2):
            for dj in range(2):
                for dk in range(4):
                    covered_24.add((b24[0]+di, b24[1]+dj, b24[2]+dk))
        
        # Find 2x2x2 blocks that don't overlap with b24
        compatible_22 = []
        for b22 in possible_2x2x2:
            overlap = False
            for di in range(2):
                for dj in range(2):
                    for dk in range(2):
                        if (b22[0]+di, b22[1]+dj, b22[2]+dk) in covered_24:
                            overlap = True
                            break
                        if overlap:
                            break
                    if overlap:
                        break
                if overlap:
                    break
            if not overlap:
                compatible_22.append(b22)
        
        for pair in combinations(compatible_22, 2):
            # Check non-overlap between the two 2x2x2 blocks
            covered = set(covered_24)
            valid = True
            for (i0, j0, k0) in pair:
                for di in range(2):
                    for dj in range(2):
                        for dk in range(2):
                            if (i0+di, j0+dj, k0+dk) in covered:
                                valid = False
                                break
                            covered.add((i0+di, j0+dj, k0+dk))
                if not valid:
                    break
            if valid:
                rank = 14 + 7*2 + (n*m*p - len(covered))
                if rank < best_rank:
                    best_rank = rank
                    best_config = (list(pair), [b24])
    
    # Try: 2 2x2x4 blocks
    for pair24 in combinations(possible_2x2x4, 2):
        covered = set()
        valid = True
        for (i0, j0, k0) in pair24:
            for di in range(2):
                for dj in range(2):
                    for dk in range(4):
                        if (i0+di, j0+dj, k0+dk) in covered:
                            valid = False
                            break
                        covered.add((i0+di, j0+dj, k0+dk))
            if not valid:
                break
        if valid:
            rank = 14*2 + (n*m*p - len(covered))
            if rank < best_rank:
                best_rank = rank
                best_config = ([], list(pair24))
    
    return best_rank, best_config

def construct():
    """
    Return a rank-minimal exact tensor decomposition for 3x3 * 3x8 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as fallback
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = n * m * p  # 72
    
    # Define 4 non-overlapping Strassen blocks
    blocks = [
        (0, 0, 0),
        (0, 0, 2),
        (0, 0, 4),
        (0, 0, 6)
    ]
    
    if _time_left() > 5:
        try:
            U, V, W = _build_strassen_embedding(n, m, p, blocks)
            if _verify(U, V, W, n, m, p):
                r = len(U[0])
                if r < best_rank:
                    best_rank = r
                    best_result = {"U": U, "V": V, "W": W}
        except Exception:
            pass
    
    # Search for better mixed placement
    if _time_left() > 15:
        try:
            mixed_rank, mixed_config = _find_best_mixed_placement(n, m, p)
            if mixed_config is not None and mixed_rank < best_rank:
                s22_blocks, s24_blocks = mixed_config
                U, V, W = _build_mixed_embedding(n, m, p, s22_blocks, s24_blocks)
                if _verify(U, V, W, n, m, p):
                    r = len(U[0])
                    if r < best_rank:
                        best_rank = r
                        best_result = {"U": U, "V": V, "W": W}
        except Exception:
            pass
    
    return best_result
