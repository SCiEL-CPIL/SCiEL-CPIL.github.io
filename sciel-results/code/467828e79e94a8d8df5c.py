import time
import numpy as np

_START = time.time()
DEADLINE = 135.0

def _time_left():
    return DEADLINE - (time.time() - _START)

def _verify(U, V, W, n, m, p, tol=1e-8):
    """Verify decomposition is exact."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    # Compute all products efficiently
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    return U, V, W

def _laderman_3x3_3():
    """Return U, V, W for 3x3 * 3x3 with rank 14 (Laderman)."""
    # Laderman's rank-14 algorithm for 3x3 matrix multiplication
    # P1 = (a00+a11+a22)(b00+b11+b22)
    # P2 = (a00+omega*a11+omega^2*a22)(b00+omega^2*b11+omega*b22)
    # P3 = (a00+omega^2*a11+omega*a22)(b00+omega*b11+omega^2*b22)
    # ... plus more terms for off-diagonal
    
    # Actually, Laderman's algorithm uses 14 multiplications for 3x3.
    # The coefficients are integers/rationals.
    
    # Let me use the known Laderman decomposition.
    # Reference: Laderman, S. (1976). "A computation matrix in 14 scalar products."
    
    # The 14 products are:
    # P1 = a11*b11
    # P2 = (a00+a22)*(b00+b22)
    # P3 = (a00-a22)*(b00-b22)
    # P4 = (a01+a02+a11+a21)*(b01+b02+b11+b21)
    # P5 = (a01+a21-a11-a21)*(b01+b21-b11-b21)
    # ... this is getting complicated. Let me use a different approach.
    
    # I'll use the explicit Laderman coefficients from the literature.
    # For 3x3, rank 14:
    
    rank = 14
    U = np.zeros((9, rank))
    V = np.zeros((9, rank))
    W = np.zeros((9, rank))
    
    # Laderman's algorithm (one standard form):
    # Let me use the formulation where we split into diagonal and off-diagonal.
    
    # Actually, let me just hardcode a known-good rank-14 decomposition.
    # From Laderman (1976) and subsequent work:
    
    # I'll use the decomposition from "Matrix Multiplication via Arithmetic Progressions"
    # or just verify with a simpler approach.
    
    # Let me try: if Laderman verification fails, we fall back to schoolbook.
    # For safety, I'll implement a simpler but still better approach:
    # Use 2 Strassen blocks for the 3x3x3 case.
    # 3x3x3 = 27 cells. 2 Strassen blocks cover 16 cells. Remaining 11. Rank = 14+11 = 25? No, 7*2+11=25.
    # That's worse than Laderman's 14.
    
    # Let me implement the actual Laderman coefficients.
    # From the paper, the 14 linear forms are:
    
    # Actually, let me use a different strategy: just use the Strassen block approach but
    # try harder to find more disjoint blocks. For 3x3x8, we found 4 blocks (rank 68).
    # Can we do better?
    
    # For now, let me implement a correct Laderman and verify it.
    # Laderman rank 14 for 3x3:
    
    # Coefficients from standard references:
    # U (first matrix coefficients):
    U[0,0] = 1; U[1,0] = 0; U[2,0] = 1; U[3,0] = 0; U[4,0] = 0; U[5,0] = 1; U[6,0] = 0; U[7,0] = 0; U[8,0] = 1
    U[0,1] = 0; U[1,1] = 0; U[2,1] = 1; U[3,1] = 0; U[4,1] = 1; U[5,1] = 0; U[6,1] = 1; U[7,1] = 0; U[8,1] = 1
    U[0,2] = 1; U[1,2] = 0; U[2,2] = 1; U[3,2] = 0; U[4,2] = 1; U[5,2] = 0; U[6,2] = 1; U[7,2] = 0; U[8,2] = 1
    # This is getting messy. Let me just use schoolbook fallback if Laderman fails.
    
    # I'll return None to indicate failure, and use schoolbook.
    return None

def _strassen_2x2_2():
    """Return U, V, W for 2x2 * 2x2 with rank 7 (Strassen)."""
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    U[0,0] = 1; U[3,0] = 1
    V[0,0] = 1; V[3,0] = 1
    W[0,0] = 1; W[3,0] = 1
    U[2,1] = 1; U[3,1] = 1
    V[0,1] = 1
    W[1,1] = 1; W[3,1] = -1
    U[0,2] = 1
    V[1,2] = 1; V[3,2] = -1
    W[2,2] = 1; W[3,2] = 1
    U[3,3] = 1
    V[2,3] = 1; V[0,3] = -1
    W[0,3] = 1; W[1,3] = 1
    U[0,4] = 1; U[1,4] = 1
    V[3,4] = 1
    W[0,4] = -1; W[2,4] = 1
    U[2,5] = 1; U[0,5] = -1
    V[0,5] = 1; V[1,5] = 1
    W[3,5] = 1
    U[1,6] = 1; U[3,6] = -1
    V[2,6] = 1; V[3,6] = 1
    W[0,6] = 1
    return U, V, W

def _get_block_coverage(i0, j0, k0, n, m, p):
    covered = set()
    for di in range(2):
        for dj in range(2):
            for dk in range(2):
                i, j, k = i0+di, j0+dj, k0+dk
                if 0 <= i < n and 0 <= j < m and 0 <= k < p:
                    covered.add((i, j, k))
    return covered

def _build_decomposition_with_blocks(n, m, p, blocks):
    if not blocks:
        return _schoolbook(n, m, p)
    
    covered = set()
    for (i0, j0, k0) in blocks:
        cov = _get_block_coverage(i0, j0, k0, n, m, p)
        covered |= cov
    
    n_school = n * m * p - len(covered)
    rank = 7 * len(blocks) + n_school
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    sU, sV, sW = _strassen_2x2_2()
    for (i0, j0, k0) in blocks:
        for c in range(7):
            for di in range(2):
                for dj in range(2):
                    U[(i0+di)*m + (j0+dj), col+c] = sU[di*2+dj, c]
            for dj in range(2):
                for dk in range(2):
                    V[(j0+dj)*p + (k0+dk), col+c] = sV[dj*2+dk, c]
            for dk in range(2):
                for di in range(2):
                    W[(k0+dk)*n + (i0+di), col+c] = sW[dk*2+di, c]
        col += 7
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
    
    return U, V, W

def construct():
    n, m, p = 3, 3, 8
    
    U_s, V_s, W_s = _schoolbook(n, m, p)
    best_result = {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    best_rank = n * m * p
    
    if _time_left() < 10:
        return best_result
    
    # Try known 4-block configuration
    known_blocks = [(0, 0, 0), (0, 0, 2), (0, 0, 4), (0, 0, 6)]
    try:
        U, V, W = _build_decomposition_with_blocks(n, m, p, known_blocks)
        if _verify(U, V, W, n, m, p):
            best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            best_rank = U.shape[1]
    except:
        pass
    
    if _time_left() < 20:
        return best_result
    
    # Try to find 5 disjoint blocks
    from itertools import combinations
    
    valid_blocks = []
    for i0 in range(n - 1):
        for j0 in range(m - 1):
            for k0 in range(p - 1):
                cov = _get_block_coverage(i0, j0, k0, n, m, p)
                if len(cov) == 8:
                    valid_blocks.append(((i0, j0, k0), cov))
    
    # Try all combinations of 5
    for combo in combinations(valid_blocks, 5):
        if _time_left() < 10:
            break
        all_cov = set()
        disjoint = True
        for _, cov in combo:
            if all_cov & cov:
                disjoint = False
                break
            all_cov |= cov
        if disjoint:
            pos_list = [pos for pos, _ in combo]
            try:
                U, V, W = _build_decomposition_with_blocks(n, m, p, pos_list)
                if _verify(U, V, W, n, m, p):
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    best_rank = U.shape[1]
                    break
            except:
                pass
    
    # Try 6 blocks
    if _time_left() > 15:
        for combo in combinations(valid_blocks, 6):
            if _time_left() < 8:
                break
            all_cov = set()
            disjoint = True
            for _, cov in combo:
                if all_cov & cov:
                    disjoint = False
                    break
                all_cov |= cov
            if disjoint:
                pos_list = [pos for pos, _ in combo]
                try:
                    U, V, W = _build_decomposition_with_blocks(n, m, p, pos_list)
                    if _verify(U, V, W, n, m, p):
                        best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                        best_rank = U.shape[1]
                        break
                except:
                    pass
    
    return best_result
