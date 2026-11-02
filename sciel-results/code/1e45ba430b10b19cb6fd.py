# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 105.0

def _time_remaining():
    return _DEADLINE - time.time()

def _build_rank25():
    """
    Rank-25 decomposition: 3 Strassen blocks (cols 0-1, 2-3, 4-5) + schoolbook (col 6).
    """
    n, m, p = 2, 2, 7
    rank = 25
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))

    def add_strassen(col_start, k_start):
        # M1 = (A00+A11)*(B0k+B1,k+1)
        U[0, col_start+0] += 1; U[3, col_start+0] += 1
        V[7*0+k_start, col_start+0] += 1; V[7*1+(k_start+1), col_start+0] += 1
        W[2*k_start+0, col_start+0] += 1; W[2*(k_start+1)+1, col_start+0] += 1
        # M2 = (A10+A11)*B0k
        U[2, col_start+1] += 1; U[3, col_start+1] += 1
        V[7*0+k_start, col_start+1] += 1
        W[2*k_start+1, col_start+1] += 1; W[2*(k_start+1)+1, col_start+1] -= 1
        # M3 = A00*(B0,k+1-B1,k+1)
        U[0, col_start+2] += 1
        V[7*0+(k_start+1), col_start+2] += 1; V[7*1+(k_start+1), col_start+2] -= 1
        W[2*(k_start+1)+0, col_start+2] += 1; W[2*(k_start+1)+1, col_start+2] += 1
        # M4 = A11*(B1k-B0k)
        U[3, col_start+3] += 1
        V[7*1+k_start, col_start+3] += 1; V[7*0+k_start, col_start+3] -= 1
        W[2*k_start+0, col_start+3] += 1; W[2*k_start+1, col_start+3] += 1
        # M5 = (A00+A01)*B1,k+1
        U[0, col_start+4] += 1; U[1, col_start+4] += 1
        V[7*1+(k_start+1), col_start+4] += 1
        W[2*k_start+0, col_start+4] -= 1; W[2*(k_start+1)+0, col_start+4] += 1
        # M6 = (A10-A00)*(B0k+B0,k+1)
        U[2, col_start+5] += 1; U[0, col_start+5] -= 1
        V[7*0+k_start, col_start+5] += 1; V[7*0+(k_start+1), col_start+5] += 1
        W[2*(k_start+1)+1, col_start+5] += 1
        # M7 = (A01-A11)*(B1k+B1,k+1)
        U[1, col_start+6] += 1; U[3, col_start+6] -= 1
        V[7*1+k_start, col_start+6] += 1; V[7*1+(k_start+1), col_start+6] += 1
        W[2*k_start+0, col_start+6] += 1

    add_strassen(0, 0)
    add_strassen(7, 2)
    add_strassen(14, 4)

    # Schoolbook for column 6
    cs = 21
    U[0, cs+0] += 1; V[7*0+6, cs+0] += 1; W[2*6+0, cs+0] += 1
    U[1, cs+1] += 1; V[7*1+6, cs+1] += 1; W[2*6+0, cs+1] += 1
    U[2, cs+2] += 1; V[7*0+6, cs+2] += 1; W[2*6+1, cs+2] += 1
    U[3, cs+3] += 1; V[7*1+6, cs+3] += 1; W[2*6+1, cs+3] += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _build_rank28_fallback():
    n, m, p = 2, 2, 7
    rank = 28
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, idx] = 1
                V[j*p+k, idx] = 1
                W[k*n+i, idx] = 1
                idx += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _verify(U, V, W, n=2, m=2, p=7, seed=42):
    rank = len(U[0]) if isinstance(U, list) else U.shape[1]
    np.random.seed(seed)
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C_expected = A @ B
    C_computed = np.zeros((n, p))
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    for r in range(rank):
        SA = np.dot(Ua[:, r], A.flatten())
        SB = np.dot(Va[:, r], B.flatten())
        for k in range(p):
            for i in range(n):
                C_computed[i, k] += Wa[k*n+i, r] * SA * SB
    return np.allclose(C_computed, C_expected, atol=1e-10)

def construct():
    if _time_remaining() < 2.0:
        return _build_rank28_fallback()
    
    result = _build_rank25()
    U = np.array(result['U'], dtype=float)
    V = np.array(result['V'], dtype=float)
    W = np.array(result['W'], dtype=float)
    
    if not _verify(U, V, W):
        return _build_rank28_fallback()
    
    return result

# EVOLVE-BLOCK-END
