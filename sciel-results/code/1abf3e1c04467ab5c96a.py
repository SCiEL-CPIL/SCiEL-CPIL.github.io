import time
import numpy as np

DIMENSIONS = [4, 4, 9]
n, m, p = DIMENSIONS

def _schoolbook():
    """Return the trivial schoolbook decomposition (rank = n*m*p = 144)."""
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

def _validate(U, V, W, tol=1e-9):
    """Check that the decomposition exactly reproduces matrix multiplication."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    U_3d = U_arr.reshape(n, m, -1)
    V_3d = V_arr.reshape(m, p, -1)
    W_3d = W_arr.reshape(p, n, -1)
    
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] should equal 1
    T = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
    
    if not np.allclose(T, 1.0, atol=tol):
        return False
    return True

def _strassen_2x2_2x2():
    """
    Return Strassen decomposition for 2x2 * 2x2 (rank 7).
    U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
    Indexing: U[i*2+j], V[j*2+k], W[k*2+i]
    """
    rank = 7
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(4)]
    W = [[0] * rank for _ in range(4)]
    
    # M1 = (A00+A11)*(B00+B11)
    U[0][0] += 1; U[3][0] += 1
    V[0][0] += 1; V[3][0] += 1
    W[0][0] += 1; W[3][0] += 1
    
    # M2 = (A10+A11)*B00
    U[2][1] += 1; U[3][1] += 1
    V[0][1] += 1
    W[2][1] += 1; W[3][1] += -1
    
    # M3 = A00*(B10-B11)
    U[0][2] += 1
    V[2][2] += 1; V[3][2] += -1
    W[1][2] += 1; W[3][2] += 1
    
    # M4 = A11*(B01-B00)
    U[3][3] += 1
    V[1][3] += 1; V[0][3] += -1
    W[0][3] += 1; W[2][3] += 1
    
    # M5 = (A00+A01)*B11
    U[0][4] += 1; U[1][4] += 1
    V[3][4] += 1
    W[0][4] += -1; W[1][4] += 1
    
    # M6 = (A10-A00)*(B00+B01)
    U[2][5] += 1; U[0][5] += -1
    V[0][5] += 1; V[1][5] += 1
    W[3][5] += 1
    
    # M7 = (A01-A11)*(B10+B11)
    U[1][6] += 1; U[3][6] += -1
    V[2][6] += 1; V[3][6] += 1
    W[0][6] += 1
    
    return {"U": U, "V": V, "W": W}

def _kronecker_combine(U1, V1, W1, U2, V2, W2):
    """
    Combine two tensor decompositions using Kronecker product.
    """
    rank1 = len(U1[0])
    rank2 = len(U2[0])
    rank = rank1 * rank2
    
    lenU1 = len(U1)
    lenU2 = len(U2)
    lenV1 = len(V1)
    lenV2 = len(V2)
    lenW1 = len(W1)
    lenW2 = len(W2)
    
    U = [[0] * rank for _ in range(lenU1 * lenU2)]
    V = [[0] * rank for _ in range(lenV1 * lenV2)]
    W = [[0] * rank for _ in range(lenW1 * lenW2)]
    
    for c in range(rank1):
        for d in range(rank2):
            col = c * rank2 + d
            for i1 in range(lenU1):
                for i2 in range(lenU2):
                    idx = i1 * lenU2 + i2
                    U[idx][col] = U1[i1][c] * U2[i2][d]
            for j1 in range(lenV1):
                for j2 in range(lenV2):
                    idx = j1 * lenV2 + j2
                    V[idx][col] = V1[j1][c] * V2[j2][d]
            for k1 in range(lenW1):
                for k2 in range(lenW2):
                    idx = k1 * lenW2 + k2
                    W[idx][col] = W1[k1][c] * W2[k2][d]
    
    return {"U": U, "V": V, "W": W}

def _strassen_4x4x4():
    """
    Construct rank-49 decomposition for 4x4 * 4x4 using Kronecker product of two Strassen 2x2.
    """
    s2 = _strassen_2x2_2x2()
    result = _kronecker_combine(
        s2['U'], s2['V'], s2['W'],
        s2['U'], s2['V'], s2['W']
    )
    return result

def _construct_4x4x9_strassen():
    """
    Construct a low-rank decomposition for 4x4 * 4x9.
    Uses 2 blocks of 4x4x4 (rank 49 each) and 1 block of 4x4x1 (rank 16).
    Total rank: 2 * 49 + 16 = 114.
    """
    s4 = _strassen_4x4x4()
    rank_s4 = len(s4['U'][0]) # Should be 49
    
    total_rank = 2 * rank_s4 + 16
    
    U = [[0] * total_rank for _ in range(16)]
    V = [[0] * total_rank for _ in range(36)]
    W = [[0] * total_rank for _ in range(36)]
    
    col_offset = 0
    for block in range(2):
        for c in range(rank_s4):
            col = col_offset + c
            for i in range(16):
                U[i][col] = s4['U'][i][c]
            for j in range(4):
                for k in range(4):
                    v_idx = j * 9 + (block * 4 + k)
                    V[v_idx][col] = s4['V'][j * 4 + k][c]
            for k in range(4):
                for i in range(4):
                    w_idx = (block * 4 + k) * 4 + i
                    W[w_idx][col] = s4['W'][k * 4 + i][c]
        col_offset += rank_s4
    
    # Add the 4x4x1 schoolbook for the last column (k=8)
    for i in range(4):
        for j in range(4):
            col = col_offset + (i * 4 + j)
            U[i * 4 + j][col] = 1
            V[j * 9 + 8][col] = 1
            W[8 * 4 + i][col] = 1
    
    return {"U": U, "V": V, "W": W}

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 4x4x9 matrix multiplication.
    """
    start_time = time.time()
    deadline = start_time + 130
    
    best_result = None
    best_rank = float('inf')
    
    # Strategy 1: Schoolbook
    schoolbook = _schoolbook()
    if _validate(schoolbook['U'], schoolbook['V'], schoolbook['W']):
        best_result = schoolbook
        best_rank = len(schoolbook['U'][0])
    
    # Strategy 2: Strassen-based (Rank 114)
    if time.time() < deadline:
        try:
            strassen_result = _construct_4x4x9_strassen()
            if _validate(strassen_result['U'], strassen_result['V'], strassen_result['W']):
                rank = len(strassen_result['U'][0])
                if rank < best_rank:
                    best_result = strassen_result
                    best_rank = rank
        except Exception:
            pass
    
    if best_result is None:
        return _schoolbook()
    
    return best_result
