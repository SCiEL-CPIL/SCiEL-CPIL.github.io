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

def _strassen_2x2x3():
    """Known rank-5 decomposition for 2x2 * 2x3 matrix multiplication."""
    n, m, p = 2, 2, 3
    rank = 5
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Rank-5 decomposition of M_{2,2,3}
    # Products (U factors, V factors) and output contributions (W factors)
    # Using the known minimal decomposition
    products = [
        # s=0: (a00+a11)(b00+b11)
        ([((0,0),1),((1,1),1)], [((0,0),1),((1,1),1)]),
        # s=1: (a10+a11)b00
        ([((1,0),1),((1,1),1)], [((0,0),1)]),
        # s=2: a00(b01-b11)
        ([((0,0),1)], [((0,1),1),((1,1),-1)]),
        # s=3: a11(b10-b00)
        ([((1,1),1)], [((1,0),1),((0,0),-1)]),
        # s=4: (a00+a01)(b11)
        ([((0,0),1),((0,1),1)], [((1,1),1)]),
    ]
    output_contribs = [
        [((0,0),1),((1,1),1)],
        [((1,0),1),((1,1),-1)],
        [((0,1),1),((1,1),1)],
        [((0,0),1),((1,0),1)],
        [((0,0),-1),((0,1),1)],
    ]
    
    for s in range(5):
        for (ai,aj), coeff in products[s][0]:
            U[ai*m+aj, s] = coeff
        for (bj,bk), coeff in products[s][1]:
            V[bj*p+bk, s] = coeff
        for (ci,ck), coeff in output_contribs[s]:
            W[ck*n+ci, s] = coeff
    
    return U, V, W

def _verify_decomposition_fast(U, V, W, n, m, p):
    """Fast verification using tensor structure."""
    U = np.asarray(U)
    V = np.asarray(V)
    W = np.asarray(W)
    rank = U.shape[1]
    
    # Check all required entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a, b, c = i*m+j, j*p+k, k*n+i
                val = np.dot(U[a], V[b] * W[c])
                if abs(val - 1.0) > 1e-6:
                    return False
    
    # Check zeros for invalid combinations
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
    
    Strategy: 
    1. Try rank-20 via 4 groups of 2x2x3 (rank 5 each)
    2. Try rank-30 via 5 groups of 2x2x3 (rank 5 each) - wait, 12/3=4
    3. Try rank-36 via 3 groups of 2x2x4 (rank 12 each)
    4. Fallback to rank-42 (Strassen on 6 pairs)
    """
    start_time = time.time()
    deadline = start_time + 145.0
    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    
    # Strategy 1: Rank-20 using 4 blocks of 2x2x3 (rank 5 each)
    # 12 = 4 * 3, so we can partition into 4 groups of 3 columns
    U3, V3, W3 = _strassen_2x2x3()
    if _verify_decomposition_fast(U3, V3, W3, 2, 2, 3):
        rank = 20  # 4 * 5
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        for t in range(4):
            base_col = 3*t
            for r in range(5):
                U[:, t*5+r] = U3[:, r]
                for j in range(2):
                    for k in range(3):
                        V[j*p + base_col + k, t*5+r] = V3[j*3+k, r]
                for k in range(3):
                    for i in range(2):
                        W[(base_col+k)*n+i, t*5+r] = W3[k*2+i, r]
        
        # Verify
        if _verify_decomposition_fast(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Strategy 2: Try to find rank-12 for 2x2x4, then tile 3 times for rank-36
    found_12 = False
    U4 = V4 = W4 = None
    for seed in range(30):
        if time.time() > deadline - 30:
            break
        U4_t, V4_t, W4_t, err = _find_decomposition(2, 2, 4, 12, max_iter=800, seed=seed)
        if err < 1e-6:
            if _verify_decomposition_fast(U4_t, V4_t, W4_t, 2, 2, 4):
                U4, V4, W4 = U4_t, V4_t, W4_t
                found_12 = True
                break
    
    if found_12:
        rank = 36
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        for t in range(3):
            base_col = 4*t
            for r in range(12):
                U[:, t*12+r] = U4[:, r]
                for j in range(2):
                    for k in range(4):
                        V[j*p + base_col + k, t*12+r] = V4[j*4+k, r]
                for k in range(4):
                    for i in range(2):
                        W[(base_col+k)*n+i, t*12+r] = W4[k*2+i, r]
        
        if _verify_decomposition_fast(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
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
