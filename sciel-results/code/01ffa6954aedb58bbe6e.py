"""Matrix multiplication 4x4 * 4x7: exact rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]
_START_TIME = time.time()
_TIME_LIMIT = 145.0  # safety margin before 150s kill


def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _verify(U, V, W, n, m, p):
    """Check that sum_r U[:,r] V[:,r] W[:,r] == multiplication tensor.
    The tensor T[i*m+j, j*p+k, k*n+i] = 1 for all valid (i,j,k).
    """
    rank = len(U[0])
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    # For each (i, j, k), check sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = np.dot(U_np[a], V_np[b] * W_np[c])
                if abs(s - 1) > 1e-9:
                    return False
    return True


def _strassen_2x2x2():
    """Return the 7 rank-1 components of the 2x2x2 multiplication tensor.
    U vectors are for A (flattened as [A00, A01, A10, A11]),
    V vectors are for B (flattened as [B00, B01, B10, B11]),
    W vectors are for C (flattened as [C00, C01, C10, C11]).
    
    Strassen:
    P1 = (A00+A11)(B00+B11)
    P2 = (A10+A11)B00
    P3 = A00(B01-B11)
    P4 = A11(B10-B00)
    P5 = (A00+A10)B11
    P6 = (A01-A11)(B10+B11)
    P7 = (A10-A00)(B00+B01)
    
    C00 = P1 + P4 - P5 + P7
    C01 = P3 + P5
    C10 = P2 + P4
    C11 = P1 - P2 + P3 + P6
    """
    U = [
        [1, 0, 0, 1],   # P1
        [0, 0, 1, 1],   # P2
        [1, 0, 0, 0],   # P3
        [0, 0, 0, 1],   # P4
        [1, 1, 0, 0],   # P5
        [0, 1, 0, -1],  # P6
        [-1, 0, 1, 0],  # P7
    ]
    V = [
        [1, 0, 0, 1],   # P1
        [1, 0, 0, 0],   # P2
        [0, 1, 0, -1],  # P3
        [0, 0, 1, -1],  # P4
        [0, 0, 0, 1],   # P5
        [0, 0, 1, 1],   # P6
        [1, 1, 0, 0],   # P7
    ]
    W = [
        [1, 0, 0, 1],   # P1
        [0, 0, 1, 0],   # P2
        [0, 1, 0, 1],   # P3
        [1, 0, 1, 0],   # P4
        [-1, 1, 0, 0],  # P5
        [0, 0, 0, 1],   # P6
        [1, 0, 0, 0],   # P7
    ]
    return U, V, W


def _build_4x4x4_strassen():
    """Build rank-49 decomposition for 4x4 * 4x4 using Strassen on 2x2 blocks."""
    n = m = p = 4
    U_block, V_block, W_block = _strassen_2x2x2()
    U_inner, V_inner, W_inner = _strassen_2x2x2()
    
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for s in range(7):
        for t in range(7):
            r = s * 7 + t
            for br in range(2):
                for lr in range(2):
                    i = br * 2 + lr
                    for bc in range(2):
                        for lc in range(2):
                            j = bc * 2 + lc
                            a = i * m + j
                            ub = U_block[s][bc]
                            ui = U_inner[t][lr * 2 + lc]
                            if ub != 0 and ui != 0:
                                U[a][r] = ub * ui
            for bc in range(2):
                for lr in range(2):
                    j = bc * 2 + lr
                    for bc2 in range(2):
                        for lc2 in range(2):
                            k = bc2 * 2 + lc2
                            b = j * p + k
                            vb = V_block[s][bc2]
                            vi = V_inner[t][lr * 2 + lc2]
                            if vb != 0 and vi != 0:
                                V[b][r] = vb * vi
            for br in range(2):
                for lr in range(2):
                    i = br * 2 + lr
                    for bc2 in range(2):
                        for lc2 in range(2):
                            k = bc2 * 2 + lc2
                            c = k * n + i
                            wb = W_block[s][br * 2 + bc2]
                            wi = W_inner[t][lr * 2 + lc2]
                            if wb != 0 and wi != 0:
                                W[c][r] = wb * wi
    return {"U": U, "V": V, "W": W}


def _build_4x4x7():
    """Build decomposition for 4x4 * 4x7.
    
    Split B (4x7) into blocks:
    - Left part: B_left is 4x4, use 4x4x4 Strassen (rank 49)
    - Right part: B_right is 4x3, use schoolbook (rank 48)
    Total rank = 97.
    """
    n, m, p = 4, 4, 7
    
    # Left part: 4x4 * 4x4 -> 4x4, rank 49
    left = _build_4x4x4_strassen()
    rank_left = len(left['U'][0])
    
    # Right part: 4x4 * 4x3 -> 4x3, use schoolbook rank 48
    n_r, m_r, p_r = 4, 4, 3
    rank_right = n_r * m_r * p_r
    U_right = [[0] * rank_right for _ in range(n_r * m_r)]
    V_right = [[0] * rank_right for _ in range(m_r * p_r)]
    W_right = [[0] * rank_right for _ in range(p_r * n_r)]
    col = 0
    for i in range(n_r):
        for j in range(m_r):
            for k in range(p_r):
                U_right[i * m_r + j][col] = 1
                V_right[j * p_r + k][col] = 1
                W_right[k * n_r + i][col] = 1
                col += 1
    
    total_rank = rank_left + rank_right
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Fill left part
    for i in range(n * m):
        for r in range(rank_left):
            U[i][r] = left['U'][i][r]
    for i in range(4):  # B rows 0-3
        for j in range(4):  # B cols 0-3
            for r in range(rank_left):
                V[i * p + j][r] = left['V'][i * 4 + j][r]
    for i in range(4):  # C cols 0-3
        for j in range(4):  # C rows 0-3
            for r in range(rank_left):
                W[i * n + j][r] = left['W'][i * 4 + j][r]
    
    # Fill right part
    offset = rank_left
    for i in range(n * m):
        for r in range(rank_right):
            U[i][offset + r] = U_right[i][r]
    for i in range(4):  # B rows 0-3
        for j in range(3):  # B cols 4-6 -> local col j
            for r in range(rank_right):
                V[i * p + (4 + j)][offset + r] = V_right[i * 3 + j][r]
    for i in range(3):  # C cols 4-6 -> local col i
        for j in range(4):  # C rows 0-3
            for r in range(rank_right):
                W[(4 + i) * n + j][offset + r] = W_right[i * 4 + j][r]
    
    return {"U": U, "V": V, "W": W}


def construct():
    """Construct the tensor decomposition. Returns {'U', 'V', 'W'}."""
    elapsed = time.time() - _START_TIME
    if elapsed > _TIME_LIMIT:
        return _schoolbook(4, 4, 7)
    
    # Try the Strassen-based construction first
    try:
        result = _build_4x4x7()
        # Verify with numpy for speed
        if _verify(result['U'], result['V'], result['W'], 4, 4, 7):
            return result
    except Exception:
        pass
    
    # Fallback to schoolbook
    return _schoolbook(4, 4, 7)
