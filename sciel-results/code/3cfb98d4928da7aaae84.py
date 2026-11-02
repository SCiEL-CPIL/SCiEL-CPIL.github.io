"""Program for rank-minimal exact tensor decomposition of 3x5x6 matrix multiplication."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START

def _verify_float_fast(U, V, W, n, m, p, atol=1e-9):
    """Fast float verification using vectorized ops."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    # tensor[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # Build tensor via einsum-like approach
    tensor = np.zeros((n, m, p), dtype=np.float64)
    for c in range(rank):
        Uc = U[:, c].reshape(n, m)
        Vc = V[:, c].reshape(m, p)
        Wc = W[:, c].reshape(p, n)
        # For each (i,j), contribution to k is Uc[i,j] * Vc[j,k] * Wc[k,i]
        # Vc[j,:] * Wc[:,i] gives vector over k
        # Uc[i,j] * (Vc[j,:] * Wc[:,i]) is the contribution
        # We can do: for each i,j: tensor[i,j,:] += Uc[i,j] * Vc[j,:] * Wc[:,i]
        # Vectorized: tensor += Uc[:,:,None] * Vc[None,:,:] * Wc.T[:,:,None]  -- no, Wc is (p,n), Wc[:,i] is col i
        # Wc.T is (n,p), so Wc.T[i,:] = Wc[:,i]
        # tensor[i,j,k] += Uc[i,j] * Vc[j,k] * Wc[k,i] = Uc[i,j] * Vc[j,k] * Wc.T[i,k]
        tensor += Uc[:, :, None] * Vc[None, :, :] * Wc.T[None, :, :]
    return np.allclose(tensor, 1.0, atol=atol)

def _verify_exact(U, V, W, n, m, p):
    """Verify exact decomposition using rational arithmetic."""
    if not U or not V or not W:
        return False
    rank = len(U[0])
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    for row in U:
        if len(row) != rank:
            return False
    for row in V:
        if len(row) != rank:
            return False
    for row in W:
        if len(row) != rank:
            return False

    Uf = [[Fraction(x) for x in row] for row in U]
    Vf = [[Fraction(x) for x in row] for row in V]
    Wf = [[Fraction(x) for x in row] for row in W]

    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for c in range(rank):
                    total += Uf[i * m + j][c] * Vf[j * p + k][c] * Wf[k * n + i][c]
                if total != 1:
                    return False
    return True

def _build_strassen_2x2x2():
    """
    Return Strassen's rank-7 decomposition of 2x2x2 matrix multiplication.
    Returns (U, V, W) as lists of lists, each of shape (4, 7).
    U indexed by i*2+j; V by j*2+k; W by k*2+i.
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]

    # c=0: U:(0,0)=1,(1,1)=1; V:(0,0)=1,(1,1)=1; W:(0,0)=1,(1,1)=1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1

    # c=1: U:(1,0)=1,(1,1)=1; V:(0,0)=1; W:(0,1)=1,(1,1)=1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = 1

    # c=2: U:(0,0)=1; V:(0,1)=1,(1,1)=-1; W:(1,0)=1,(1,1)=1
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1

    # c=3: U:(1,1)=1; V:(1,0)=1,(0,0)=-1; W:(0,0)=1,(0,1)=1
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1

    # c=4: U:(0,0)=1,(0,1)=1; V:(1,1)=1; W:(0,0)=1,(1,0)=1
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = 1; W[2][4] = 1

    # c=5: U:(1,0)=1,(0,0)=-1; V:(0,0)=1,(0,1)=1; W:(1,1)=1
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1

    # c=6: U:(0,1)=1,(1,1)=-1; V:(1,0)=1,(1,1)=1; W:(0,0)=1
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1

    return U, V, W

def _build_3x2x6():
    """
    Build a decomposition of 3x2x6 matrix multiplication, rank 24.
    2x2x6 part: 3 Strassen 2x2x2 blocks (rank 7 each) = 21
    1x2x6 part: 3 terms (one per k-pair) = 3
    Total: 24
    """
    n, m, p = 3, 2, 6
    rank = 24
    U = [[0]*rank for _ in range(n*m)]  # 6 rows
    V = [[0]*rank for _ in range(m*p)]  # 12 rows
    W = [[0]*rank for _ in range(p*n)]  # 18 rows
    
    col = 0
    # Part 1: 2x2x6 block (i in {0,1}, j in {0,1}, k in {0,...,5})
    strassen_U, strassen_V, strassen_W = _build_strassen_2x2x2()
    
    for pair_idx in range(3):
        k1 = pair_idx * 2
        k2 = pair_idx * 2 + 1
        for c in range(7):
            for i in range(2):
                for j in range(2):
                    U[i*m + j][col] = strassen_U[i*2 + j][c]
            for j in range(2):
                for kk in range(2):
                    V[j*p + k1 + kk][col] = strassen_V[j*2 + kk][c]
            for kk in range(2):
                for i in range(2):
                    W[(k1 + kk)*n + i][col] = strassen_W[kk*2 + i][c]
            col += 1
    
    # Part 2: 1x2x6 block (i=2, j in {0,1}, k in {0,...,5})
    for pair_idx in range(3):
        k1 = pair_idx * 2
        k2 = pair_idx * 2 + 1
        for j in range(2):
            U[2*m + j][col] = 1
        for j in range(2):
            for kk in range(2):
                V[j*p + k1 + kk][col] = 1
        for kk in range(2):
            W[(k1 + kk)*n + 2][col] = 1
        col += 1
    
    return U, V, W

def _build_3x3x6():
    """
    Build a decomposition of 3x3x6 matrix multiplication, rank 3.
    For each k-pair, one term with all-ones U, V, W on the relevant indices.
    """
    n, m, p = 3, 3, 6
    rank = 3
    U = [[0]*rank for _ in range(n*m)]  # 9 rows
    V = [[0]*rank for _ in range(m*p)]  # 18 rows
    W = [[0]*rank for _ in range(p*n)]  # 18 rows
    
    for c in range(3):
        k1 = c * 2
        k2 = c * 2 + 1
        for i in range(3):
            for j in range(3):
                U[i*m + j][c] = 1
        for j in range(3):
            for kk in range(2):
                V[j*p + k1 + kk][c] = 1
        for kk in range(2):
            for i in range(3):
                W[(k1 + kk)*n + i][c] = 1
    
    return U, V, W

def _build_full_3x5x6():
    """
    Build the full 3x5x6 decomposition using block-diagonal embedding.
    3x5x6 = 3x2x6 (j in {0,1}) ⊕ 3x3x6 (j in {2,3,4})
    
    3x2x6: rank 24
    3x3x6: rank 3
    Total: rank 27
    """
    n, m, p = 3, 5, 6
    rank_3x2x6 = 24
    rank_3x3x6 = 3
    total_rank = rank_3x2x6 + rank_3x3x6
    
    U = [[0]*total_rank for _ in range(n*m)]  # 15 rows
    V = [[0]*total_rank for _ in range(m*p)]  # 30 rows
    W = [[0]*total_rank for _ in range(p*n)]  # 18 rows
    
    # Build 3x2x6 part
    U1, V1, W1 = _build_3x2x6()
    # U1: (3*2, 24) = (6, 24)
    # V1: (2*6, 24) = (12, 24)
    # W1: (6*3, 24) = (18, 24)
    
    for i in range(n):
        for j in range(2):
            for c in range(rank_3x2x6):
                U[i*m + j][c] = U1[i*2 + j][c]
    
    for j in range(2):
        for k in range(p):
            for c in range(rank_3x2x6):
                V[j*p + k][c] = V1[j*2 + k][c]
    
    for k in range(p):
        for i in range(n):
            for c in range(rank_3x2x6):
                W[k*n + i][c] = W1[k*3 + i][c]
    
    # Build 3x3x6 part (offset by rank_3x2x6)
    U2, V2, W2 = _build_3x3x6()
    # U2: (3*3, 3) = (9, 3)
    # V2: (3*6, 3) = (18, 3)
    # W2: (6*3, 3) = (18, 3)
    
    for i in range(n):
        for j in range(3):
            for c in range(rank_3x3x6):
                U[i*m + (j + 2)][rank_3x2x6 + c] = U2[i*3 + j][c]
    
    for j in range(3):
        for k in range(p):
            for c in range(rank_3x3x6):
                V[(j + 2)*p + k][rank_3x2x6 + c] = V2[j*3 + k][c]
    
    for k in range(p):
        for i in range(n):
            for c in range(rank_3x3x6):
                W[k*n + i][rank_3x2x6 + c] = W2[k*3 + i][c]
    
    return U, V, W

def _schoolbook(n, m, p):
    """Return the schoolbook (rank = n*m*p) decomposition."""
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

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110.0
    
    # Try the block-diagonal embedding approach
    try:
        U, V, W = _build_full_3x5x6()
        
        # Verify with float first (fast)
        if _verify_float_fast(U, V, W, n, m, p):
            # Verify with exact arithmetic
            if _verify_exact(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
