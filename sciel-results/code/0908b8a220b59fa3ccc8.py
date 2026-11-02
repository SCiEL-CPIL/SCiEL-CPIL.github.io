import time
import numpy as np

DEADLINE_S = 110.0
N, M, P = 4, 5, 5


def _validate(U, V, W, tol=1e-9):
    """Check sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] == 1 for all i,j,k."""
    n, m, p = N, M, P
    R = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(R))
                if abs(s - 1.0) > tol:
                    return False
    return True


def _strassen_2x2x2():
    """Return (U,V,W) for rank-7 Strassen decomposition of 2x2x2 matrix multiplication.
    U: (4,7), V: (4,7), W: (4,7) with W[k*2+i, r].
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1: (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[2, 0] = 1; W[1, 0] = -1; W[3, 0] = -1
    
    # M2: (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # M3: A00(B10-B11)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = -1
    
    # M4: A11(B01-B00)
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = -1; W[1, 3] = 1
    
    # M5: (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[3, 4] = -1
    
    # M6: (A10-A00)(B00+B10)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[2, 5] = 1
    W[3, 5] = 1
    
    # M7: (A01-A11)(B01+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[1, 6] = 1; V[3, 6] = 1
    W[0, 6] = -1; W[2, 6] = 1
    
    return U, V, W


def _schoolbook_block(n, m, p):
    """Return (U,V,W) for schoolbook decomposition of n x m * m x p.
    U: (n*m, n*m*p), V: (m*p, n*m*p), W: (p*n, n*m*p)
    """
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, c] = 1.0
                V[j*p+k, c] = 1.0
                W[k*n+i, c] = 1.0
                c += 1
    return U, V, W


def _combine_decompositions(decomps):
    """Combine a list of (U,V,W) decompositions into one.
    Each (U,V,W) has the same number of rows in U, V, W as the final result.
    Returns combined (U,V,W).
    """
    if not decomps:
        return None
    
    n, m, p = N, M, P
    total_rank = sum(U.shape[1] for U, V, W in decomps)
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    c = 0
    for U_i, V_i, W_i in decomps:
        r = U_i.shape[1]
        U[:, c:c+r] = U_i
        V[:, c:c+r] = V_i
        W[:, c:c+r] = W_i
        c += r
    
    return U, V, W


def _embed_block(U_block, V_block, W_block, i_off, j_off, k_off, n, m, p):
    """Embed a block decomposition into the full 4x5x5 space."""
    n_b = U_block.shape[0] // m_b if m_b > 0 else 0
    return None


def construct():
    t0 = time.time()
    n, m, p = N, M, P  # 4, 5, 5
    
    # Strategy: Split C (4x5) into 2x2 blocks:
    # C = [[C11 (2x2), C12 (2x3)],
    #      [C21 (2x2), C22 (2x3)]]
    #
    # C11 = A[0:2, 0:5] * B[0:5, 0:2]
    #   Split inner dim: A_top = [A_top1 (2x2) | A_top2 (2x3)], B_left = [B_left1 (2x2); B_left2 (3x2)]
    #   C11 = A_top1*B_left1 + A_top2*B_left2
    #   A_top1*B_left1: 2x2 * 2x2, Strassen rank 7
    #   A_top2*B_left2: 2x3 * 3x2, schoolbook rank 12
    #   Subtotal: 19
    #
    # C21 = A[2:4, 0:5] * B[0:5, 0:2]
    #   Similarly: 19
    #
    # C12 = A[0:2, 0:5] * B[0:5, 2:5]
    #   A_top = [A_top1 (2x2) | A_top2 (2x3)], B_right = [B_right1 (2x3); B_right2 (3x3)]
    #   C12 = A_top1*B_right1 + A_top2*B_right2
    #   A_top1*B_right1: 2x2 * 2x3, schoolbook rank 12
    #   A_top2*B_right2: 2x3 * 3x3, schoolbook rank 18
    #   Subtotal: 30
    #
    # C22 = A[2:4, 0:5] * B[0:5, 2:5]
    #   Similarly: 30
    #
    # Total: 19 + 19 + 30 + 30 = 98
    
    decomps = []
    
    # Helper to create a schoolbook decomposition embedded in the full space
    def make_schoolbook(n_b, m_b, p_b, i_off, j_off, k_off):
        """Create schoolbook for A[i_off:i_off+n_b, j_off:j_off+m_b] * B[j_off:j_off+m_b, k_off:k_off+p_b]
        contributing to C[i_off:i_off+n_b, k_off:k_off+p_b]
        """
        rank = n_b * m_b * p_b
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        c = 0
        for i in range(n_b):
            for j in range(m_b):
                for k in range(p_b):
                    U[(i_off+i)*m + (j_off+j), c] = 1.0
                    V[(j_off+j)*p + (k_off+k), c] = 1.0
                    W[(k_off+k)*n + (i_off+i), c] = 1.0
                    c += 1
        return U, V, W
    
    def make_strassen_2x2(i_off, j_off, k_off):
        """Create Strassen rank-7 for A[i_off:i_off+2, j_off:j_off+2] * B[j_off:j_off+2, k_off:k_off+2]
        contributing to C[i_off:i_off+2, k_off:k_off+2]
        """
        U_s, V_s, W_s = _strassen_2x2x2()
        rank = 7
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        for r in range(rank):
            for i in range(2):
                for j in range(2):
                    U[(i_off+i)*m + (j_off+j), r] = U_s[i*2+j, r]
            for j in range(2):
                for k in range(2):
                    V[(j_off+j)*p + (k_off+k), r] = V_s[j*2+k, r]
            for k in range(2):
                for i in range(2):
                    W[(k_off+k)*n + (i_off+i), r] = W_s[k*2+i, r]
        return U, V, W
    
    # C11: A[0:2, 0:5] * B[0:5, 0:2]
    # A_top1 (2x2) at A[0:2, 0:2], B_left1 (2x2) at B[0:2, 0:2] -> Strassen
    decomps.append(make_strassen_2x2(0, 0, 0))
    # A_top2 (2x3) at A[0:2, 2:5], B_left2 (3x2) at B[2:5, 0:2] -> schoolbook
    decomps.append(make_schoolbook(2, 3, 2, 0, 2, 0))
    
    # C21: A[2:4, 0:5] * B[0:5, 0:2]
    # A_bot1 (2x2) at A[2:4, 0:2], B_left1 (2x2) at B[0:2, 0:2] -> Strassen
    decomps.append(make_strassen_2x2(2, 0, 0))
    # A_bot2 (2x3) at A[2:4, 2:5], B_left2 (3x2) at B[2:5, 0:2] -> schoolbook
    decomps.append(make_schoolbook(2, 3, 2, 2, 2, 0))
    
    # C12: A[0:2, 0:5] * B[0:5, 2:5]
    # A_top1 (2x2) at A[0:2, 0:2], B_right1 (2x3) at B[0:2, 2:5] -> schoolbook
    decomps.append(make_schoolbook(2, 2, 3, 0, 0, 2))
    # A_top2 (2x3) at A[0:2, 2:5], B_right2 (3x3) at B[2:5, 2:5] -> schoolbook
    decomps.append(make_schoolbook(2, 3, 3, 0, 2, 2))
    
    # C22: A[2:4, 0:5] * B[0:5, 2:5]
    # A_bot1 (2x2) at A[2:4, 0:2], B_right1 (2x3) at B[0:2, 2:5] -> schoolbook
    decomps.append(make_schoolbook(2, 2, 3, 2, 0, 2))
    # A_bot2 (2x3) at A[2:4, 2:5], B_right2 (3x3) at B[2:5, 2:5] -> schoolbook
    decomps.append(make_schoolbook(2, 3, 3, 2, 2, 2))
    
    # Combine all decompositions
    U, V, W = _combine_decompositions(decomps)
    
    # Validate
    if not _validate(U, V, W):
        # Fallback to schoolbook rank 100
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, c] = 1.0
                    V[j*p+k, c] = 1.0
                    W[k*n+i, c] = 1.0
                    c += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
