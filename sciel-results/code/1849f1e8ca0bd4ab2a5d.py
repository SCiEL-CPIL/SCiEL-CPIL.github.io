"""Tensor decomposition for 2x2 x 2x9 matrix multiplication."""
DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
_TIMEOUT = 130  # safety margin before 150s kill


def _elapsed():
    return time.time() - _START_TIME


def _build_tensor(n, m, p):
    """Build the multiplication tensor T of shape (n*m, m*p, p*n)."""
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T


def _validate(U, V, W, n, m, p, atol=1e-9):
    """Check if U,V,W exactly decompose the multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    T = _build_tensor(n, m, p)
    T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
    return np.allclose(T, T_recon, atol=atol)


def _strassen_2x2x2():
    """
    Return Strassen (rank 7) decomposition for 2x2x2 matrix multiplication.
    Returns list of 7 (u, v, w) vectors, each of length 4.
    """
    # Strassen terms for 2x2x2:
    # P0=(A00+A11)(B00+B11)
    # P1=(A10+A11)B00
    # P2=A00(B01-B11)
    # P3=A11(B10-B00)
    # P4=(A00+A01)B11
    # P5=(A10-A00)(B00+B01)
    # P6=(A01-A11)(B10+B11)
    # C00=P0-P1-P4+P6, C01=P4+P6, C10=P1+P5, C11=P0-P2+P3-P5
    # U rows: 0=A00,1=A01,2=A10,3=A11
    # V rows: 0=B00,1=B01,2=B10,3=B11
    # W rows: 0=C00,1=C01,2=C10,3=C11
    terms = [
        # (U coeffs, V coeffs, W coeffs)
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, -1), (1, 1), (2, 1)]),
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        ([(0, 1)], [(1, 1), (3, -1)], [(1, -1), (3, 1)]),
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, -1)]),
        ([(0, 1), (1, 1)], [(3, 1)], [(1, 1), (3, 1)]),
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(0, -1), (2, 1)]),
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1), (1, 1)]),
    ]
    out = []
    for uc, vc, wc in terms:
        u = np.zeros(4)
        for idx, c in uc:
            u[idx] += c
        v = np.zeros(4)
        for idx, c in vc:
            v[idx] += c
        w = np.zeros(4)
        for idx, c in wc:
            w[idx] += c
        out.append((u, v, w))
    return out


def _kron_decomposition(n, m, p):
    """
    Build decomposition using Kronecker product structure.
    
    For 2x2 x 2x9: We can view this as (2x2 x 2x2) ⊗ (2x2 x 2x1)
    where the second factor handles the 9 columns via 4 blocks of 2 (with one partial).
    
    Actually, let's use a different approach:
    The tensor T[i*m+j, j*p+k, k*n+i] = 1
    
    We can group the k dimension into blocks. For p=9, we have 9 values of k.
    Let's use the fact that 9 = 2*4 + 1, so we can use 4 blocks of 2x2x2 (rank 7 each)
    plus 1 block of 2x2x1 (rank 2).
    
    Total rank: 4*7 + 1*2 = 30
    
    But we can do better by using the Kronecker structure more carefully.
    """
    # For now, use the block approach: 4 blocks of 2x2x2 + 1 block of 2x2x1
    U_terms = []
    V_terms = []
    W_terms = []
    
    strassen = _strassen_2x2x2()
    
    # Block 1: k in {0,1}
    for u, v, w in strassen:
        U_terms.append(u.copy())
        v_full = np.zeros(m * p)
        v_full[0] = v[0]  # B00
        v_full[1] = v[1]  # B01
        v_full[9] = v[2]  # B10
        v_full[10] = v[3]  # B11
        V_terms.append(v_full)
        w_full = np.zeros(p * n)
        w_full[0] = w[0]  # C00
        w_full[1] = w[1]  # C01
        w_full[2] = w[2]  # C10
        w_full[3] = w[3]  # C11
        W_terms.append(w_full)
    
    # Block 2: k in {2,3}
    for u, v, w in strassen:
        U_terms.append(u.copy())
        v_full = np.zeros(m * p)
        v_full[2] = v[0]  # B02
        v_full[3] = v[1]  # B03
        v_full[11] = v[2]  # B12
        v_full[12] = v[3]  # B13
        V_terms.append(v_full)
        w_full = np.zeros(p * n)
        w_full[4] = w[0]  # C20
        w_full[5] = w[1]  # C21
        w_full[6] = w[2]  # C30
        w_full[7] = w[3]  # C31
        W_terms.append(w_full)
    
    # Block 3: k in {4,5}
    for u, v, w in strassen:
        U_terms.append(u.copy())
        v_full = np.zeros(m * p)
        v_full[4] = v[0]  # B04
        v_full[5] = v[1]  # B05
        v_full[13] = v[2]  # B14
        v_full[14] = v[3]  # B15
        V_terms.append(v_full)
        w_full = np.zeros(p * n)
        w_full[8] = w[0]  # C40
        w_full[9] = w[1]  # C41
        w_full[10] = w[2]  # C50
        w_full[11] = w[3]  # C51
        W_terms.append(w_full)
    
    # Block 4: k in {6,7}
    for u, v, w in strassen:
        U_terms.append(u.copy())
        v_full = np.zeros(m * p)
        v_full[6] = v[0]  # B06
        v_full[7] = v[1]  # B07
        v_full[15] = v[2]  # B16
        v_full[16] = v[3]  # B17
        V_terms.append(v_full)
        w_full = np.zeros(p * n)
        w_full[12] = w[0]  # C60
        w_full[13] = w[1]  # C61
        w_full[14] = w[2]  # C70
        w_full[15] = w[3]  # C71
        W_terms.append(w_full)
    
    # Block 5: k=8 (single)
    # For 2x2x1, we need rank 2
    # C[i][j] = sum over k of A[i][k] * B[k][j], but here k is fixed at 8
    # Actually for 2x2x1: T[i*2+j, j*1+0, 0*2+i] = T[i*2+j, j, i] = 1
    # This means: for each (i,j), we have T[i*2+j, j, i] = 1
    # Rank 2 decomposition:
    # Term 1: u=[1,0,0,0], v=[1,0], w=[1,0] -> contributes to T[0,0,0]=1
    # Term 2: u=[0,1,0,0], v=[0,1], w=[0,1] -> contributes to T[1,1,1]=1
    # But we also need T[0,1,0]=1 and T[1,0,1]=1
    # So we need:
    # Term 1: u=[1,0,0,0], v=[1,1], w=[1,0] -> T[0,0,0]=1, T[0,1,0]=1
    # Term 2: u=[0,0,1,0], v=[1,1], w=[0,1] -> T[2,0,1]=1, T[2,1,1]=1
    # Wait, let me reconsider the indices.
    
    # For k=8:
    # U index: i*2+j where i in {0,1}, j in {0,1}
    # V index: j*9+8 where j in {0,1} -> indices 8, 17
    # W index: 8*2+i where i in {0,1} -> indices 16, 17
    
    # T[i*2+j, j*9+8, 16+i] = 1 for all i,j in {0,1}
    # So we have 4 entries: (0,8,16), (1,8,17), (2,17,16), (3,17,17)
    
    # Rank 2 decomposition:
    # Term 1: u=[1,0,1,0], v=[1,0], w=[1,1]
    #   Contributes: u[0]*v[8]*w[16]=1*1*1=1 at (0,8,16) ✓
    #                u[0]*v[8]*w[17]=1*1*1=1 at (0,8,17) ✗ (should be 0)
    # This doesn't work directly.
    
    # Let's use the standard 2x2x1 decomposition:
    # The tensor has entries: T[0,8,16]=1, T[1,8,17]=1, T[2,17,16]=1, T[3,17,17]=1
    # This is actually a rank 4 tensor if we look at it naively, but we can use structure.
    
    # Actually, for 2x2x1 matrix multiplication (A is 2x1, B is 1x2, C is 2x2):
    # C[i][j] = A[i][0] * B[0][j]
    # So C[0][0] = A[0][0]*B[0][0], C[0][1] = A[0][0]*B[0][1]
    # C[1][0] = A[1][0]*B[0][0], C[1][1] = A[1][0]*B[0][1]
    
    # This is rank 2:
    # Term 1: u=[1,0], v=[1,0], w=[1,1] -> C[0][0]=1, C[0][1]=1
    # Term 2: u=[0,1], v=[1,0], w=[1,1] -> C[1][0]=1, C[1][1]=1
    # Wait, that gives C[0][1]=1 and C[1][1]=1, but we need them to be separate.
    
    # Let me reconsider. The output C is 2x2, so W has 4 entries.
    # U has 2 entries (A has 2x1 = 2 elements)
    # V has 2 entries (B has 1x2 = 2 elements)
    
    # C[0][0] = A[0]*B[0]
    # C[0][1] = A[0]*B[1]
    # C[1][0] = A[1]*B[0]
    # C[1][1] = A[1]*B[1]
    
    # Rank 2:
    # Term 1: u=[1,0], v=[1,1], w=[1,0,1,0]
    #   C[0][0] += 1*1*1 = 1 ✓
    #   C[0][1] += 1*1*0 = 0 ✗ (should be 1)
    
    # I think I'm confusing myself. Let's just use rank 4 for the single k=8 block.
    # Or better, let's use a different grouping.
    
    # For now, let's just add 4 terms for k=8:
    for i in range(2):
        for j in range(2):
            u = np.zeros(4)
            u[i * 2 + j] = 1
            v = np.zeros(m * p)
            v[j * p + 8] = 1
            w = np.zeros(p * n)
            w[8 * n + i] = 1
            U_terms.append(u)
            V_terms.append(v)
            W_terms.append(w)
    
    if not U_terms:
        return None
    
    U = np.array(U_terms).T  # (4, rank)
    V = np.array(V_terms).T  # (18, rank)
    W = np.array(W_terms).T  # (18, rank)
    return U, V, W


def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return U, V, W


def construct():
    n, m, p = DIMENSIONS

    best = None
    best_rank = None

    # Try Kronecker decomposition
    try:
        res = _kron_decomposition(n, m, p)
        if res is not None:
            U, V, W = res
            if _validate(U, V, W, n, m, p):
                rank = U.shape[1]
                if best_rank is None or rank < best_rank:
                    best = (U, V, W)
                    best_rank = rank
    except Exception:
        pass

    if best is not None:
        U, V, W = best
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
