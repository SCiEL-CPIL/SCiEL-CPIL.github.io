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


def _schoolbook_2x2x1():
    """Return (U,V,W) for rank-4 schoolbook of 2x2 * 2x1 -> 2x1.
    U: (4,4), V: (2,4), W: (2,4)
    """
    U = np.zeros((4, 4))
    V = np.zeros((2, 4))
    W = np.zeros((2, 4))
    c = 0
    for i in range(2):
        for j in range(2):
            for k in range(1):
                U[i*2+j, c] = 1.0
                V[j*1+k, c] = 1.0
                W[k*2+i, c] = 1.0
                c += 1
    return U, V, W

def _schoolbook_2x1x2():
    """Return (U,V,W) for rank-4 schoolbook of 2x1 * 1x2 -> 2x2.
    U: (2,4), V: (2,4), W: (4,4)
    """
    U = np.zeros((2, 4))
    V = np.zeros((2, 4))
    W = np.zeros((4, 4))
    c = 0
    for i in range(2):
        for j in range(1):
            for k in range(2):
                U[i*1+j, c] = 1.0
                V[j*2+k, c] = 1.0
                W[k*2+i, c] = 1.0
                c += 1
    return U, V, W

def _schoolbook_2x1x1():
    """Return (U,V,W) for rank-2 schoolbook of 2x1 * 1x1 -> 2x1.
    U: (2,2), V: (1,2), W: (2,2)
    """
    U = np.zeros((2, 2))
    V = np.zeros((1, 2))
    W = np.zeros((2, 2))
    c = 0
    for i in range(2):
        for j in range(1):
            for k in range(1):
                U[i*1+j, c] = 1.0
                V[j*1+k, c] = 1.0
                W[k*2+i, c] = 1.0
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


def construct():
    t0 = time.time()
    n, m, p = N, M, P  # 4, 5, 5
    
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

    def make_2x2x1(i_off, j_off, k_off):
        """Create rank-4 for A[i_off:i_off+2, j_off:j_off+2] * B[j_off:j_off+2, k_off:k_off+1]
        contributing to C[i_off:i_off+2, k_off:k_off+1]
        """
        U_s, V_s, W_s = _schoolbook_2x2x1()
        rank = 4
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        for r in range(rank):
            for i in range(2):
                for j in range(2):
                    U[(i_off+i)*m + (j_off+j), r] = U_s[i*2+j, r]
            for j in range(2):
                for k in range(1):
                    V[(j_off+j)*p + (k_off+k), r] = V_s[j*1+k, r]
            for k in range(1):
                for i in range(2):
                    W[(k_off+k)*n + (i_off+i), r] = W_s[k*2+i, r]
        return U, V, W

    def make_2x1x2(i_off, j_off, k_off):
        """Create rank-4 for A[i_off:i_off+2, j_off:j_off+1] * B[j_off:j_off+1, k_off:k_off+2]
        contributing to C[i_off:i_off+2, k_off:k_off+2]
        """
        U_s, V_s, W_s = _schoolbook_2x1x2()
        rank = 4
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        for r in range(rank):
            for i in range(2):
                for j in range(1):
                    U[(i_off+i)*m + (j_off+j), r] = U_s[i*1+j, r]
            for j in range(1):
                for k in range(2):
                    V[(j_off+j)*p + (k_off+k), r] = V_s[j*2+k, r]
            for k in range(2):
                for i in range(2):
                    W[(k_off+k)*n + (i_off+i), r] = W_s[k*2+i, r]
        return U, V, W

    def make_2x1x1(i_off, j_off, k_off):
        """Create rank-2 for A[i_off:i_off+2, j_off:j_off+1] * B[j_off:j_off+1, k_off:k_off+1]
        contributing to C[i_off:i_off+2, k_off:k_off+1]
        """
        U_s, V_s, W_s = _schoolbook_2x1x1()
        rank = 2
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        for r in range(rank):
            for i in range(2):
                for j in range(1):
                    U[(i_off+i)*m + (j_off+j), r] = U_s[i*1+j, r]
            for j in range(1):
                for k in range(1):
                    V[(j_off+j)*p + (k_off+k), r] = V_s[j*1+k, r]
            for k in range(1):
                for i in range(2):
                    W[(k_off+k)*n + (i_off+i), r] = W_s[k*2+i, r]
        return U, V, W

    # C11: A[0:2, 0:5] * B[0:5, 0:2]
    # Split A[0:2, 0:5] into A_L (2x2) at j=0:2 and A_R (2x1) at j=2:3? 
    # Wait, B is 5x2. Inner dim 5.
    # Let's stick to the previous split but optimize the sub-blocks.
    # Previous: A[0:2, 0:2] * B[0:2, 0:2] (2x2x2) and A[0:2, 2:5] * B[2:5, 0:2] (2x3x2).
    
    # Term 1: 2x2x2 Strassen
    decomps.append(make_strassen_2x2(0, 0, 0))
    
    # Term 2: 2x3x2. Split A_R (2x3) and B_B (3x2).
    # A_R is A[0:2, 2:5]. B_B is B[2:5, 0:2].
    # We can split A_R into A_RL (2x2) at j=2:4 and A_RR (2x1) at j=4:5.
    # And B_B into B_BT (2x2) at j=2:4 and B_BB (1x2) at j=4:5.
    # Term 2a: A_RL (2x2) * B_BT (2x2) -> 2x2x2 Strassen.
    decomps.append(make_strassen_2x2(0, 2, 0))
    
    # Term 2b: A_RR (2x1) * B_BB (1x2) -> 2x1x2 Rank 4.
    decomps.append(make_2x1x2(0, 4, 0))
    
    # C21: A[2:4, 0:5] * B[0:5, 0:2]
    # Same structure.
    decomps.append(make_strassen_2x2(2, 0, 0))
    decomps.append(make_strassen_2x2(2, 2, 0))
    decomps.append(make_2x1x2(2, 4, 0))
    
    # C12: A[0:2, 0:5] * B[0:5, 2:5]
    # Output 2x3.
    # Split B[0:5, 2:5] into B_L (5x2) at k=2:4 and B_R (5x1) at k=4:5.
    # C12 = A[0:2, 0:5] * B_L + A[0:2, 0:5] * B_R.
    
    # Part 1: A[0:2, 0:5] * B[0:5, 2:4] (2x5 * 5x2 -> 2x2)
    # Split A into A_L (2x2) j=0:2 and A_R (2x3) j=2:5.
    # Split B_L into B_LT (2x2) j=0:2 and B_LB (3x2) j=2:5.
    # Term 1a: A_L (2x2) * B_LT (2x2) -> 2x2x2 Strassen.
    decomps.append(make_strassen_2x2(0, 0, 2))
    
    # Term 1b: A_R (2x3) * B_LB (3x2) -> 2x3x2.
    # Split A_R into A_RL (2x2) j=2:4 and A_RR (2x1) j=4:5.
    # Split B_LB into B_LBT (2x2) j=2:4 and B_LBB (1x2) j=4:5.
    # Term 1b1: A_RL (2x2) * B_LBT (2x2) -> 2x2x2 Strassen.
    decomps.append(make_strassen_2x2(0, 2, 2))
    
    # Term 1b2: A_RR (2x1) * B_LBB (1x2) -> 2x1x2 Rank 4.
    decomps.append(make_2x1x2(0, 4, 2))
    
    # Part 2: A[0:2, 0:5] * B[0:5, 4:5] (2x5 * 5x1 -> 2x1)
    # Split A into A_L (2x2) j=0:2 and A_R (2x3) j=2:5.
    # Split B_R into B_RT (2x1) j=0:2 and B_RB (3x1) j=2:5.
    # Term 2a: A_L (2x2) * B_RT (2x1) -> 2x2x1 Rank 4.
    decomps.append(make_2x2x1(0, 0, 4))
    
    # Term 2b: A_R (2x3) * B_RB (3x1) -> 2x3x1.
    # Split A_R into A_RL (2x2) j=2:4 and A_RR (2x1) j=4:5.
    # Split B_RB into B_RBT (2x1) j=2:4 and B_RBB (1x1) j=4:5.
    # Term 2b1: A_RL (2x2) * B_RBT (2x1) -> 2x2x1 Rank 4.
    decomps.append(make_2x2x1(0, 2, 4))
    
    # Term 2b2: A_RR (2x1) * B_RBB (1x1) -> 2x1x1 Rank 2.
    decomps.append(make_2x1x1(0, 4, 4))
    
    # C22: A[2:4, 0:5] * B[0:5, 2:5]
    # Same as C12 but i_off=2.
    decomps.append(make_strassen_2x2(2, 0, 2))
    decomps.append(make_strassen_2x2(2, 2, 2))
    decomps.append(make_2x1x2(2, 4, 2))
    decomps.append(make_2x2x1(2, 0, 4))
    decomps.append(make_2x2x1(2, 2, 4))
    decomps.append(make_2x1x1(2, 4, 4))
    
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
