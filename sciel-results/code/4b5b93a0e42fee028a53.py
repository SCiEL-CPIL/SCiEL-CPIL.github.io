"""Matrix multiplication 6,6,6 rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]
_DEADLINE = time.time() + 140.0  # safety margin before 150s kill


def _tensor_from_factors(U, V, W):
    """Reconstruct the multiplication tensor T[i,j,k] = sum_c U[i,c]*V[j,c]*W[k,c]."""
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    # T is a 3D array indexed [i, j, k]
    T = np.einsum('ic,jc,kc->ijk', U, V, W)
    return T


def _target_tensor(n, m, p):
    """
    Multiplication tensor for C = A @ B.
    C[i, k] = sum_j A[i, j] B[j, k].
    The tensor T[i, k, j] represents the coefficient of A[i, j] B[j, k] in C[i, k].
    Since every term A[i, j] B[j, k] contributes exactly 1 to C[i, k],
    T[i, k, j] = 1 for all i, k, j.
    """
    return np.ones((n, p, m), dtype=np.float64)


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check exact decomposition under convention W[k*n+i, r]."""
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    
    # Reshape to 3D for efficient einsum
    # U[i*m+j, r] -> U3[i, j, r]
    U3 = U.reshape(n, m, rank)
    # V[j*p+k, r] -> V3[j, k, r]
    V3 = V.reshape(m, p, rank)
    # W[k*n+i, r] -> W3[k, i, r]
    W3 = W.reshape(p, n, rank)
    
    # Compute T_recon[i, k, j] = sum_r U3[i, j, r] * V3[j, k, r] * W3[k, i, r]
    # Note the index mapping:
    # U3 indices: i, j, r
    # V3 indices: j, k, r
    # W3 indices: k, i, r
    # Result indices: i, k, j
    T_recon = np.einsum('ijr,jkr,kir->ikj', U3, V3, W3)
    
    Ttarget = _target_tensor(n, m, p)
    return bool(np.allclose(T_recon, Ttarget, atol=tol, rtol=0.0))


def _strassen_2x2_factors():
    """Return (U,V,W) for 2x2 matrix multiplication, rank 7, convention W[k*n+i,r].
    n=m=p=2. U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]."""
    n = m = p = 2
    U = np.zeros((n * m, 7), dtype=np.float64)
    V = np.zeros((m * p, 7), dtype=np.float64)
    W = np.zeros((p * n, 7), dtype=np.float64)
    
    def ui(i, j): return i * m + j
    def vi(j, k): return j * p + k
    def wi(k, i): return k * n + i

    # M1
    U[ui(0,0),0] += 1; U[ui(1,1),0] += 1
    V[vi(0,0),0] += 1; V[vi(1,1),0] += 1
    W[wi(0,0),0] += 1; W[wi(1,1),0] += 1
    # M2
    U[ui(1,0),1] += 1; U[ui(1,1),1] += 1
    V[vi(0,0),1] += 1
    W[wi(1,0),1] += 1; W[wi(1,1),1] += 1
    # M3
    U[ui(0,0),2] += 1
    V[vi(1,0),2] += 1; V[vi(1,1),2] -= 1
    W[wi(0,1),2] += 1; W[wi(1,0),2] += 1
    # M4
    U[ui(1,1),3] += 1
    V[vi(0,1),3] += 1; V[vi(1,0),3] -= 1
    W[wi(0,0),3] += 1; W[wi(1,0),3] += 1
    # M5
    U[ui(0,0),4] += 1; U[ui(0,1),4] += 1
    V[vi(1,1),4] += 1
    W[wi(0,0),4] -= 1; W[wi(0,1),4] += 1
    # M6
    U[ui(1,0),5] += 1; U[ui(0,0),5] -= 1
    V[vi(0,0),5] += 1; V[vi(1,0),5] += 1
    W[wi(1,1),5] += 1
    # M7
    U[ui(0,1),6] += 1; U[ui(1,1),6] -= 1
    V[vi(0,1),6] += 1; V[vi(1,1),6] += 1
    W[wi(0,0),6] += 1; W[wi(1,1),6] -= 1
    
    return U, V, W


def _kron_block_decomposition():
    """Build 6x6 decomposition as 3x3 block grid where each 2x2 block product
    uses Strassen (rank 7), with schoolbook over the 3x3 block indices.
    Rank = 9 (block positions) * 3 (sum index) * 7 (strassen) = 189.
    """
    n = m = p = 6
    B = 3   # number of blocks per side
    b = 2   # block size
    
    rank = B * B * B * 7
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)

    sU, sV, sW = _strassen_2x2_factors()
    
    col = 0
    for br in range(B):
        for bc in range(B):
            for bs in range(B):
                for r in range(7):
                    # Local Strassen factor r
                    # sU local: index li*b + lj
                    for li in range(b):
                        for lj in range(b):
                            val = sU[li * b + lj, r]
                            if val != 0:
                                gi = br * b + li
                                gj = bs * b + lj
                                U[gi * m + gj, col] += val
                    for lj in range(b):
                        for lk in range(b):
                            val = sV[lj * b + lk, r]
                            if val != 0:
                                gj = bs * b + lj
                                gk = bc * b + lk
                                V[gj * p + gk, col] += val
                    for lk in range(b):
                        for li in range(b):
                            val = sW[lk * b + li, r]
                            if val != 0:
                                gk = bc * b + lk
                                gi = br * b + li
                                W[gk * n + gi, col] += val
                    col += 1
    assert col == rank, (col, rank)
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    
    # Candidate 1: block-Strassen rank 189
    try:
        U, V, W = _kron_block_decomposition()
        if _verify(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass

    # Fallback: schoolbook rank 216
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
    return {"U": U, "V": V, "W": W}
