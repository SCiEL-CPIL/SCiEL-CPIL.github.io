"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _find_decomposition(n, m, p, rank, max_iter=3000, seed=42):
    """Find a rank-r decomposition of the (n,m,p) matrix multiplication tensor via ALS."""
    np.random.seed(seed)
    U = np.random.randn(n*m, rank) * 0.1
    V = np.random.randn(m*p, rank) * 0.1
    W = np.random.randn(p*n, rank) * 0.1
    
    def get_target():
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m+j, j*p+k, k*n+i] = 1.0
        return T
    
    target = get_target()
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for iteration in range(max_iter):
        # Update U
        for a in range(n*m):
            i, j = a // m, a % m
            M = np.array([V[j*p+k] * W[k*n+i] for k in range(p)])
            U[a] = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
        
        # Update V
        for b in range(m*p):
            j, k = b // p, b % p
            M = np.array([U[i*m+j] * W[k*n+i] for i in range(n)])
            V[b] = np.linalg.lstsq(M, np.ones(n), rcond=None)[0]
        
        # Update W
        for c in range(p*n):
            k, i = c // n, c % n
            M = np.array([U[i*m+j] * V[j*p+k] for j in range(m)])
            W[c] = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
        
        # Check error (sampled for speed)
        err_sum = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a, b, c = i*m+j, j*p+k, k*n+i
                    val = np.dot(U[a], V[b] * W[c])
                    err_sum += (val - 1.0)**2
        err = np.sqrt(err_sum)
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        if err < 1e-8:
            break
    
    return best_U, best_V, best_W, best_err

def _strassen_2x2x4():
    """Strassen-based rank-14 decomposition for 2x2 * 2x4."""
    n, m, p = 2, 2, 4
    rank = 14
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    products = [
        ([((0,0),1),((1,1),1)], [((0,0),1),((1,1),1)]),
        ([((1,0),1),((1,1),1)], [((0,0),1)]),
        ([((0,0),1)], [((0,1),1),((1,1),-1)]),
        ([((1,1),1)], [((1,0),1),((0,0),-1)]),
        ([((0,0),1),((0,1),1)], [((1,1),1)]),
        ([((1,0),1),((0,0),-1)], [((0,0),1),((0,1),1)]),
        ([((0,1),1),((1,1),-1)], [((1,0),1),((1,1),1)]),
    ]
    output_contribs = [
        [((0,0),1),((1,1),1)],
        [((1,0),1),((1,1),-1)],
        [((0,1),1),((1,1),1)],
        [((0,0),1),((1,0),1)],
        [((0,0),-1),((0,1),1)],
        [((1,1),1)],
        [((0,0),1)],
    ]
    
    col = 0
    for t in range(2):
        base_col = 2*t
        for s in range(7):
            for (ai,aj), coeff in products[s][0]:
                U[ai*m+aj, col] = coeff
            for (bj,bk), coeff in products[s][1]:
                V[bj*p+base_col+bk, col] = coeff
            for (ci,ck), coeff in output_contribs[s]:
                W[(base_col+ck)*n+ci, col] = coeff
            col += 1
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is correct."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a, b, c = i*m+j, j*p+k, k*n+i
                val = np.dot(U[a], V[b] * W[c])
                if abs(val - 1.0) > 1e-6:
                    return False
    # Check zeros
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                i, j = a//m, a%m
                jj, k = b//p, b%p
                kk, ii = c//n, c%n
                if j != jj or k != kk or i != ii:
                    val = np.dot(U[a], V[b] * W[c])
                    if abs(val) > 1e-6:
                        return False
    return True

def _strassen_2x2x4():
    """Rank-10 decomposition for 2x2 * 2x4 using Strassen-like scheme."""
    n, m, p = 2, 2, 4
    rank = 10
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen for 2x2x4: partition B into two 2x2 blocks
    # B = [B0 | B1] where B0, B1 are 2x2
    # C = [C0 | C1] where C0, C1 are 2x2
    # 
    # Using Strassen's 7 products on the 2x2 blocks:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B00)
    # M4 = A11(B11-B01)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # For B1 (second 2x2 block), similar with M8-M14 but we can reuse
    # Actually for 2x2x4, we need 10 products total.
    
    # Products for B0 (columns 0-1 of B):
    # P1: (A00+A11)(B00+B11)
    # P2: (A10+A11)B00  
    # P3: A00(B10-B00)
    # P4: A11(B11-B01)
    # P5: (A00+A01)B11
    # P6: (A10-A00)(B00+B01)
    # P7: (A01-A11)(B10+B11)
    
    # Products for B1 (columns 2-3 of B):
    # P8: (A00+A11)(B20+B31)
    # P9: (A10+A11)B20
    # P10: A00(B30-B20)
    
    # But wait, we need all 4 output blocks. Let me reconsider.
    # For 2x2x4, the minimum rank is known to be 10.
    # Let me use a known construction.
    
    # Actually, let me just use the standard approach:
    # The tensor for 2x2x4 can be decomposed with rank 10.
    # One construction uses 7 Strassen products for one half and 3 more for the other.
    
    # Let me define the 10 products explicitly.
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01, b02, b03], [b10, b11, b12, b13]]
    # C = [[c00, c01, c02, c03], [c10, c11, c12, c13]]
    
    # P1 = (a00+a11)(b00+b11)
    U[0, 0] = 1  # a00
    U[3, 0] = 1  # a11
    V[0, 0] = 1  # b00
    V[5, 0] = 1  # b11
    W[0, 0] = 1  # c00
    W[3, 0] = 1  # c11
    W[2, 0] = 1  # c02 (from second block)
    W[5, 0] = 1  # c13 (from second block)
    
    # P2 = (a10+a11)b00
    U[2, 1] = 1  # a10
    U[3, 1] = 1  # a11
    V[0, 1] = 1  # b00
    W[2, 1] = 1  # c10
    W[3, 1] = -1  # c11
    
    # P3 = a00(b10-b00)
    U[0, 2] = 1  # a00
    V[4, 2] = 1  # b10
    V[0, 2] = -1  # b00
    W[1, 2] = 1  # c01
    W[3, 2] = 1  # c11
    
    # P4 = a11(b11-b01)
    U[3, 3] = 1  # a11
    V[5, 3] = 1  # b11
    V[1, 3] = -1  # b01
    W[0, 3] = 1  # c00
    W[2, 3] = 1  # c02
    
    # P5 = (a00+a01)b11
    U[0, 4] = 1  # a00
    U[1, 4] = 1  # a01
    V[5, 4] = 1  # b11
    W[0, 4] = -1  # c00
    W[1, 4] = 1  # c01
    
    # P6 = (a10-a00)(b00+b01)
    U[2, 5] = 1  # a10
    U[0, 5] = -1  # a00
    V[0, 5] = 1  # b00
    V[1, 5] = 1  # b01
    W[3, 5] = 1  # c11
    
    # P7 = (a01-a11)(b10+b11)
    U[1, 6] = 1  # a01
    U[3, 6] = -1  # a11
    V[4, 6] = 1  # b10
    V[5, 6] = 1  # b11
    W[0, 6] = 1  # c00
    
    # P8 = (a00+a11)(b02+b13)
    U[0, 7] = 1  # a00
    U[3, 7] = 1  # a11
    V[2, 7] = 1  # b02
    V[7, 7] = 1  # b13
    W[2, 7] = 1  # c02
    W[5, 7] = 1  # c13
    
    # P9 = (a10+a11)b02
    U[2, 8] = 1  # a10
    U[3, 8] = 1  # a11
    V[2, 8] = 1  # b02
    W[4, 8] = 1  # c12
    W[5, 8] = -1  # c13
    
    # P10 = a00(b12-b02)
    U[0, 9] = 1  # a00
    V[6, 9] = 1  # b12
    V[2, 9] = -1  # b02
    W[2, 9] = 1  # c02
    W[4, 9] = 1  # c12
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is correct."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a, b, c = i*m+j, j*p+k, k*n+i
                val = np.dot(U[a], V[b] * W[c])
                if abs(val - 1.0) > 1e-6:
                    return False
    # Check zeros
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                i, j = a//m, a%m
                jj, k = b//p, b%p
                kk, ii = c//n, c%n
                if j != jj or k != kk or i != ii:
                    val = np.dot(U[a], V[b] * W[c])
                    if abs(val) > 1e-6:
                        return False
    return True

def construct():
    """Rank-minimal decomposition for 2x2 * 2x12 matrix multiplication.
    
    Strategy: Try to find a rank-30 decomposition by using a rank-10
    decomposition of 2x2x4, then tiling 3 times. If that fails, fall back to
    rank-42 (Strassen on 6 pairs).
    """
    start_time = time.time()
    deadline = start_time + 130.0
    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    
    # Try rank-10 decomposition for 2x2x4
    U4, V4, W4 = _strassen_2x2x4()
    if _verify_decomposition(U4, V4, W4, 2, 2, 4):
        # Tile 3 times for 2x2x12
        rank = 30
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        for t in range(3):
            base_col = 4*t
            for r in range(10):
                U[:, t*10+r] = U4[:, r]
                for j in range(2):
                    for k in range(4):
                        V[j*p + base_col + k, t*10+r] = V4[j*4+k, r]
                for k in range(4):
                    for i in range(2):
                        W[(base_col+k)*n+i, t*10+r] = W4[k*2+i, r]
        
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: rank-42 Strassen on 6 pairs
    rank = 42
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    products = [
        ([((0,0),1),((1,1),1)], [((0,0),1),((1,1),1)]),
        ([((1,0),1),((1,1),1)], [((0,0),1)]),
        ([((0,0),1)], [((0,1),1),((1,1),-1)]),
        ([((1,1),1)], [((1,0),1),((0,0),-1)]),
        ([((0,0),1),((0,1),1)], [((1,1),1)]),
        ([((1,0),1),((0,0),-1)], [((0,0),1),((0,1),1)]),
        ([((0,1),1),((1,1),-1)], [((1,0),1),((1,1),1)]),
    ]
    output_contribs = [
        [((0,0),1),((1,1),1)],
        [((1,0),1),((1,1),-1)],
        [((0,1),1),((1,1),1)],
        [((0,0),1),((1,0),1)],
        [((0,0),-1),((0,1),1)],
        [((1,1),1)],
        [((0,0),1)],
    ]
    
    col = 0
    for t in range(6):
        if time.time() > deadline:
            break
        base_col = 2*t
        for s in range(7):
            for (ai,aj), coeff in products[s][0]:
                U[ai*m+aj][col] = coeff
            for (bj,bk), coeff in products[s][1]:
                V[bj*p+base_col+bk][col] = coeff
            for (ci,ck), coeff in output_contribs[s]:
                W[(base_col+ck)*n+ci][col] = coeff
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
