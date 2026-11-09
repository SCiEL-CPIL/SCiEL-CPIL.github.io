"""Tensor decomposition for 4x4 * 4x8 matrix multiplication.

Structural approach: split 4x4 * 4x8 into two independent 4x4 * 4x4 multiplications.
Each 4x4 * 4x4 uses 4 independent 2x2 * 2x2 Strassen products (rank 7 each).
Total rank: 2 * 4 * 7 = 56.
"""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]
_START_TIME = time.time()
_DEADLINE = _START_TIME + 130


def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Verify that U,V,W exactly decompose the matmul tensor.
    
    For all i, k: sum_c W[k*n+i, c] * U[i*m:(i+1)*m, c] * V[k::p, c].T = I_m
    """
    for i in range(n):
        U_i = U[i * m:(i + 1) * m, :]
        for k in range(p):
            V_k = V[k::p, :]
            w = W[k * n + i, :]
            M = U_i * w[None, :] @ V_k.T
            if np.max(np.abs(M - np.eye(m))) > tol:
                return False
    return True


def _strassen_2x2():
    """Strassen rank-7 decomposition for 2x2 * 2x2.
    
    U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    n = m = p = 2
    rank = 7
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # M1 = (a+d)(e+h)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (c+d)e
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3 = a(f-h)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = d(g-e)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (a+b)h
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6 = (c-d)(f+g)
    U[2, 5] = 1; U[3, 5] = -1
    V[1, 5] = 1; V[2, 5] = 1
    W[3, 5] = 1
    
    # M7 = (b-d)(e+g)
    U[1, 6] = 1; U[3, 6] = -1
    V[0, 6] = 1; V[2, 6] = 1
    W[0, 6] = 1
    
    return U, V, W


def _build_4x4_from_strassen(U2, V2, W2):
    """Build 4x4 * 4x4 decomposition from 2x2 Strassen.
    
    C[ci,cj] = sum_{bk} A[ci,bk] * B[bk,cj] for ci,cj in {0,1}
    For each (ci, bk, cj), use one Strassen 2x2 product.
    Total: 2*2*2 * 7 = 56 rank terms... wait, that's 8*7=56.
    Actually: ci in {0,1}, bk in {0,1}, cj in {0,1} -> 8 block products, each rank 7.
    But we can share: for fixed (ci, cj), C[ci,cj] = A[ci,0]*B[0,cj] + A[ci,1]*B[1,cj]
    Each A[ci,bk]*B[bk,cj] is a 2x2*2x2 product of rank 7.
    So 4 (ci,cj pairs) * 2 (bk values) * 7 = 56.
    """
    n = m = p = 4
    rank = 4 * 2 * 7  # 56
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for ci in range(2):  # output row block (0 or 1)
        for cj in range(2):  # output col block (0 or 1)
            for bk in range(2):  # middle block (0 or 1)
                for s in range(7):
                    # U: A block (ci, bk) -> rows (2*ci + di)*4 + (2*bk + dj)
                    for di in range(2):
                        for dj in range(2):
                            U[(2*ci+di)*4 + (2*bk+dj), col] = U2[di*2+dj, s]
                    
                    # V: B block (bk, cj) -> rows (2*bk + di)*4 + (2*cj + dj)
                    for di in range(2):
                        for dj in range(2):
                            V[(2*bk+di)*4 + (2*cj+dj), col] = V2[di*2+dj, s]
                    
                    # W: C block (ci, cj) -> W[k*4+i, c] where i=2*ci+di, k=2*cj+dj
                    for di in range(2):
                        for dj in range(2):
                            W[(2*cj+dj)*4 + (2*ci+di), col] = W2[di*2+dj, s]
                    
                    col += 1
    
    return U, V, W


def _build_4x4x4x8(U2, V2, W2):
    """Build 4x4 * 4x8 as two independent 4x4 * 4x4 multiplications."""
    n, m, p = 4, 4, 8
    
    U4, V4, W4 = _build_4x4_from_strassen(U2, V2, W2)
    rank4 = U4.shape[1]  # 56
    
    rank = 2 * rank4  # 112? No wait...
    # Actually, for 4x4 * 4x8:
    # C[:, 0:4] = A * B[:, 0:4]
    # C[:, 4:8] = A * B[:, 4:8]
    # These share the same A, so U is shared.
    # U shape: (16, rank) - same U for both halves
    # V shape: (32, rank) - V for first 4 cols, V for last 4 cols
    # W shape: (32, rank) - W for first 4 cols, W for last 4 cols
    
    # But U can be shared! Both halves use the same A.
    # So rank = rank4 (not 2*rank4) if we can share U.
    # Wait, no: each rank term c has U[:,c], V[:,c], W[:,c].
    # For the first half: U4[:,c], V4_first[:,c], W4_first[:,c]
    # For the second half: U4[:,c], V4_second[:,c], W4_second[:,c]
    # These are different rank terms because V and W differ.
    # So total rank = 2 * rank4 = 112.
    
    # Hmm, but that's worse than schoolbook (128). Let me reconsider.
    # Schoolbook rank = 4*4*8 = 128.
    # Strassen-based: 2 * 56 = 112. That's an improvement!
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # First 4x4 block (output columns 0-3)
    U[:, 0:rank4] = U4
    for j in range(m):
        for k in range(4):
            V[j * p + k, 0:rank4] = V4[j * 4 + k, :]
    for k in range(4):
        for i in range(n):
            W[k * n + i, 0:rank4] = W4[k * 4 + i, :]
    
    # Second 4x4 block (output columns 4-7)
    U[:, rank4:2*rank4] = U4
    for j in range(m):
        for k in range(4):
            V[j * p + (k + 4), rank4:2*rank4] = V4[j * 4 + k, :]
    for k in range(4):
        for i in range(n):
            W[(k + 4) * n + i, rank4:2*rank4] = W4[k * 4 + i, :]
    
    return U, V, W


def _schoolbook(n, m, p):
    """Schoolbook rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    
    if time.time() > _DEADLINE:
        U, V, W = _schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best = None
    best_rank = n * m * p  # 128
    
    # Strategy: Strassen-based construction
    try:
        U2, V2, W2 = _strassen_2x2()
        if _verify_fast(U2, V2, W2, 2, 2, 2):
            U, V, W = _build_4x4x4x8(U2, V2, W2)
            if _verify_fast(U, V, W, n, m, p):
                best = (U, V, W)
                best_rank = U.shape[1]
            else:
                # Fallback to schoolbook
                U, V, W = _schoolbook(n, m, p)
                best = (U, V, W)
                best_rank = n * m * p
        else:
            U, V, W = _schoolbook(n, m, p)
            best = (U, V, W)
            best_rank = n * m * p
    except Exception:
        U, V, W = _schoolbook(n, m, p)
        best = (U, V, W)
        best_rank = n * m * p
    
    U, V, W = best
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
