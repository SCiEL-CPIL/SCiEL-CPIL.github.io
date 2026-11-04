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
    """Multiplication tensor: T[i,j,k] = 1 iff i==k (C[i,k] = sum_j A[i,j] B[j,k])."""
    T = np.zeros((n, p, m), dtype=np.float64)
    for i in range(n):
        for k in range(p):
            for j in range(m):
                if i == k:
                    T[i, k, j] = 1.0
    return T


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
    # Reconstruct: (A @ B)[i,k] = sum_j A[i,j] B[j,k]
    # = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]  summed over j
    # Build full product via tensor contraction.
    # T_recon[i, k, j] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    T = np.zeros((n, p, m), dtype=np.float64)
    for r in range(rank):
        uc = U[:, r]
        vc = V[:, r]
        wc = W[:, r]
        # uc indexed by i*m+j, vc by j*p+k, wc by k*n+i
        uc3 = uc.reshape(n, m)
        vc3 = vc.reshape(m, p)
        wc3 = wc.reshape(p, n)
        # contribution to T[i,k,j] = uc3[i,j] * vc3[j,k] * wc3[k,i]
        for i in range(n):
            for k in range(p):
                wi = wc3[k, i]
                if wi == 0.0:
                    continue
                for j in range(m):
                    T[i, k, j] += uc3[i, j] * vc3[j, k] * wi
    Ttarget = _target_tensor(n, m, p)
    return bool(np.allclose(T, Ttarget, atol=tol, rtol=0.0))


def _strassen_2x2_factors():
    """Return (U,V,W) for 2x2 matrix multiplication, rank 7, convention W[k*n+i,r].
    n=m=p=2. U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]."""
    n = m = p = 2
    # Strassen 7 products for 2x2:
    # A = [[a00,a01],[a10,a11]], B = [[b00,b01],[b10,b11]]
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11) b00
    # M3 = a00 (b10-b11)
    # M4 = a11 (b01-b10)
    # M5 = (a00+a01) b11
    # M6 = (a10-a00)(b00+b10)
    # M7 = (a01-a11)(b01+b11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    U = np.zeros((n * m, 7), dtype=np.float64)
    V = np.zeros((m * p, 7), dtype=np.float64)
    W = np.zeros((p * n, 7), dtype=np.float64)
    # index helpers
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
    This gives rank = 9 (block positions) * 7 (strassen) = 63? No:
    Actually each C-block (r,c) = sum_s A-block(r,s) * B-block(s,c).
    We need to compute 9 output blocks, each is sum of 3 2x2 products.
    Total 2x2 products = 9*3 = 27, each rank 7 -> 27*7 = 189.
    This is the known rank-189 construction.
    """
    n = m = p = 6
    B = 3   # number of blocks per side
    b = 2   # block size
    # We'll index the 6x6 as 3x3 blocks of 2x2.
    # For each block position (br, bc) of output, and each block (bs) summed,
    # we do a 2x2 Strassen product of A-block(br,bs) and B-block(bs,bc).
    # Each such 2x2 product contributes 7 rank-1 terms.
    # Total rank = B*B (output positions) * B (sum index) * 7 = 9*3*7 = 189.
    rank = B * B * B * 7
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)

    # Strassen 2x2 local factors (local indices 0..1 for rows/cols within block)
    sU, sV, sW = _strassen_2x2_factors()  # shapes (4,7) each, local convention
    # local: sU[li*m_l + lj, r], sV[lj*p_l + lk, r], sW[lk*n_l + li, r], n_l=m_l=p_l=2

    col = 0
    for br in range(B):
        for bc in range(B):
            for bs in range(B):
                # A-block (br, bs): global rows br*b .. (br+1)*b-1, cols bs*b ..
                # B-block (bs, bc): global rows bs*b .., cols bc*b ..
                # Output C-block (br, bc): global rows br*b.., cols bc*b..
                for r in range(7):
                    # local strassen factor r
                    # sU local: index li*2 + lj
                    for li in range(b):
                        for lj in range(b):
                            val = sU[li * b + lj, r]
                            if val == 0:
                                continue
                            gi = br * b + li
                            gj = bs * b + lj
                            U[gi * m + gj, col] += val
                    for lj in range(b):
                        for lk in range(b):
                            val = sV[lj * b + lk, r]
                            if val == 0:
                                continue
                            gj = bs * b + lj
                            gk = bc * b + lk
                            V[gj * p + gk, col] += val
                    for lk in range(b):
                        for li in range(b):
                            val = sW[lk * b + li, r]
                            if val == 0:
                                continue
                            gk = bc * b + lk
                            gi = br * b + li
                            W[gk * n + gi, col] += val
                    col += 1
    assert col == rank, (col, rank)
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    best = None
    best_rank = None

    # Candidate 1: block-Strassen rank 189
    try:
        U, V, W = _kron_block_decomposition()
        if _verify(U, V, W, n, m, p):
            best_rank = U.shape[1]
            best = (U.tolist(), V.tolist(), W.tolist())
    except Exception:
        pass

    if best is not None:
        return {"U": best[0], "V": best[1], "W": best[2]}

    # Fallback: schoolbook rank 216 (guaranteed valid)
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
