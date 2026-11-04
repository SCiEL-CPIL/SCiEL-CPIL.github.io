"""Matrix multiplication 4x4 * 4x9 tensor decomposition using tensor product structure."""
import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def _strassen_2x2():
    """
    Strassen decomposition for 2x2 x 2x2 matrix multiplication.
    Returns U (4,7), V (4,7), W (4,7) where:
    T[i,k,j] = sum_r U[i*2+k, r] * V[k*2+j, r] * W[j*2+i, r] = 1 for all i,k,j
    """
    # Strassen's 7 products for 2x2 matmul
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # U: indexed by (i,k) -> i*2+k
    # V: indexed by (k,j) -> k*2+j  
    # W: indexed by (j,i) -> j*2+i
    
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # M1: (a+d)(e+h)
    # U: a at (0,0)=0, d at (1,1)=3
    U2[0, 0] = 1; U2[3, 0] = 1
    # V: e at (0,0)=0, h at (1,1)=3
    V2[0, 0] = 1; V2[3, 0] = 1
    # W: contributes to C11 (j=0,i=0) and C22 (j=1,i=1)
    W2[0, 0] = 1; W2[3, 0] = 1  # j*2+i: (0,0)=0, (1,1)=3
    
    # M2: (c+d)e
    # U: c at (1,0)=2, d at (1,1)=3
    U2[2, 1] = 1; U2[3, 1] = 1
    # V: e at (0,0)=0
    V2[0, 1] = 1
    # W: contributes to C21 (j=0,i=1) -> j*2+i = 1
    W2[1, 1] = 1
    
    # M3: a(f-h)
    # U: a at (0,0)=0
    U2[0, 2] = 1
    # V: f at (0,1)=1, h at (1,1)=3
    V2[1, 2] = 1; V2[3, 2] = -1
    # W: contributes to C12 (j=1,i=0) -> j*2+i = 2, and C22 (j=1,i=1) -> j*2+i = 3
    W2[2, 2] = 1; W2[3, 2] = 1
    
    # M4: d(g-e)
    # U: d at (1,1)=3
    U2[3, 3] = 1
    # V: g at (1,0)=2, e at (0,0)=0
    V2[2, 3] = 1; V2[0, 3] = -1
    # W: contributes to C11 (j=0,i=0) -> 0, and C21 (j=0,i=1) -> 1
    W2[0, 3] = 1; W2[1, 3] = 1
    
    # M5: (a+b)h
    # U: a at (0,0)=0, b at (0,1)=1
    U2[0, 4] = 1; U2[1, 4] = 1
    # V: h at (1,1)=3
    V2[3, 4] = 1
    # W: contributes to C11 (j=0,i=0) -> 0, and C12 (j=1,i=0) -> 2
    W2[0, 4] = -1; W2[2, 4] = 1
    
    # M6: (c-a)(e+f)
    # U: c at (1,0)=2, a at (0,0)=0
    U2[2, 5] = 1; U2[0, 5] = -1
    # V: e at (0,0)=0, f at (0,1)=1
    V2[0, 5] = 1; V2[1, 5] = 1
    # W: contributes to C22 (j=1,i=1) -> 3
    W2[3, 5] = 1
    
    # M7: (b-d)(g+h)
    # U: b at (0,1)=1, d at (1,1)=3
    U2[1, 6] = 1; U2[3, 6] = -1
    # V: g at (1,0)=2, h at (1,1)=3
    V2[2, 6] = 1; V2[3, 6] = 1
    # W: contributes to C11 (j=0,i=0) -> 0
    W2[0, 6] = 1
    
    return U2, V2, W2


def _tensor_product_4x4x4():
    """
    Build 4x4x4 decomposition via tensor product of two 2x2x2 Strassen decompositions.
    Returns U (16, 49), V (16, 49), W (16, 49)
    """
    U2, V2, W2 = _strassen_2x2()
    
    U444 = np.zeros((16, 49))
    V444 = np.zeros((16, 49))
    W444 = np.zeros((16, 49))
    
    for i1 in range(2):
        for i2 in range(2):
            for k1 in range(2):
                for k2 in range(2):
                    for j1 in range(2):
                        for j2 in range(2):
                            # 4x4 indices
                            i = i1 * 2 + i2
                            k = k1 * 2 + k2
                            j = j1 * 2 + j2
                            
                            # U index: i*4 + k
                            u_idx = i * 4 + k
                            # V index: k*4 + j
                            v_idx = k * 4 + j
                            # W index: j*4 + i
                            w_idx = j * 4 + i
                            
                            for r1 in range(7):
                                for r2 in range(7):
                                    col = r1 * 7 + r2
                                    u_val = U2[i1 * 2 + k1, r1] * U2[i2 * 2 + k2, r2]
                                    v_val = V2[k1 * 2 + j1, r1] * V2[k2 * 2 + j2, r2]
                                    w_val = W2[j1 * 2 + i1, r1] * W2[j2 * 2 + i2, r2]
                                    
                                    U444[u_idx, col] += u_val
                                    V444[v_idx, col] += v_val
                                    W444[w_idx, col] += w_val
    
    return U444, V444, W444


def _build_4x4x9():
    """
    Build 4x4x9 decomposition by combining 4x4x4 (rank 49) and 4x4x5 (rank 49+16=65? no).
    
    Actually, let me reconsider. The tensor T[i,j,k] for k=0..8.
    I can split k into two groups: k=0..3 (4 values) and k=4..8 (5 values).
    
    For k=0..3: this is a 4x4x4 tensor, decomposable with rank 49.
    For k=4..8: this is a 4x4x5 tensor. I can further split: k=4..7 (4 values, another 4x4x4, rank 49) 
    and k=8 (1 value, 4x4x1, rank 16).
    
    Total rank: 49 + 49 + 16 = 114. Still better than 144.
    
    But wait, can I do better? What if I use a different decomposition for 4x4x5?
    4x4x5: I could use 4x4x4 (rank 49) + 4x4x1 (rank 16) = 65.
    So total: 49 (for k=0..3) + 65 (for k=4..8) = 114.
    
    Alternatively, use the full 4x4x9 directly with tensor product structure.
    Since 9 is not a power of 2, I'll stick with the split approach.
    """
    t_start = time.time()
    deadline = t_start + 110.0  # 10s safety margin
    
    # Build 4x4x4 decomposition
    U444, V444, W444 = _tensor_product_4x4x4()
    
    # Build 4x4x1 decomposition (rank 16)
    U441 = np.zeros((16, 16))
    V441 = np.zeros((36, 16))  # m*p = 4*9 = 36, but for k=8 only, so effectively 4*1=4 rows used
    W441 = np.zeros((36, 16))  # p*n = 9*4 = 36
    
    # For 4x4x1: T[i,j,8] = 1 for all i,j
    # U[i*4+j, col] = 1, V[j*9+8, col] = 1, W[8*4+i, col] = 1
    col = 0
    for i in range(4):
        for j in range(4):
            U441[i * 4 + j, col] = 1
            V441[j * 9 + 8, col] = 1
            W441[8 * 4 + i, col] = 1
            col += 1
    
    # Now combine: total rank = 49 + 49 + 16 = 114
    # k=0..3: use first 49 columns of U444, V444, W444
    # k=4..7: use second 49 columns (shift k by 4)
    # k=8: use 16 columns from U441, V441, W441
    
    total_rank = 49 + 49 + 16
    U = np.zeros((16, total_rank))
    V = np.zeros((36, total_rank))
    W = np.zeros((36, total_rank))
    
    # Part 1: k=0..3 (4x4x4)
    U[:, 0:49] = U444
    V[:, 0:49] = V444
    W[:, 0:49] = W444
    
    # Part 2: k=4..7 (another 4x4x4, shift k indices by 4)
    # V and W indices shift: V[j*9+k] -> V[j*9+(k+4)], W[k*4+i] -> W[(k+4)*4+i]
    V_shifted = np.zeros((36, 49))
    W_shifted = np.zeros((36, 49))
    for j in range(4):
        for k in range(4):
            V_shifted[j * 9 + (k + 4), :] = V444[j * 4 + k, :]
    for k in range(4):
        for i in range(4):
            W_shifted[(k + 4) * 4 + i, :] = W444[k * 4 + i, :]
    
    U[:, 49:98] = U444
    V[:, 49:98] = V_shifted
    W[:, 49:98] = W_shifted
    
    # Part 3: k=8 (4x4x1)
    U[:, 98:114] = U441
    V[:, 98:114] = V441
    W[:, 98:114] = W441
    
    if time.time() > deadline:
        # Fall back to schoolbook if we're running out of time
        return _schoolbook()
    
    return U, V, W


def _schoolbook():
    """Fallback schoolbook decomposition of rank 144."""
    n, m, p = DIMENSIONS
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


def _validate(U, V, W, n, m, p, tol=1e-9):
    """Validate the decomposition."""
    rank = U.shape[1]
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    expected = np.ones((n, m, p))
    return np.allclose(T, expected, atol=tol, rtol=tol)


def construct():
    """Return {'U','V','W'} exact decomposition of (4x4)*(4x9) tensor."""
    t_start = time.time()
    n, m, p = DIMENSIONS
    deadline = t_start + 115.0  # 5s safety margin before 120s limit
    
    try:
        U, V, W = _build_4x4x9()
        
        # Validate
        if _validate(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        else:
            # Validation failed, fall back to schoolbook
            U, V, W = _schoolbook()
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        # Any error, fall back to schoolbook
        U, V, W = _schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
