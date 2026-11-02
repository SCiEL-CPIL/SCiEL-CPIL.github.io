import time
import numpy as np

DIMENSIONS = [2, 2, 12]

def _find_decomposition(n, m, p, rank, max_iter=5000, seed=42):
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
            # M shape: (p, rank)
            # We want U[a] such that M @ U[a] approx ones(p)
            # This is a linear system M x = y
            try:
                U[a] = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
            except:
                pass
        
        # Update V
        for b in range(m*p):
            j, k = b // p, b % p
            M = np.array([U[i*m+j] * W[k*n+i] for i in range(n)])
            try:
                V[b] = np.linalg.lstsq(M, np.ones(n), rcond=None)[0]
            except:
                pass
        
        # Update W
        for c in range(p*n):
            k, i = c // n, c % n
            M = np.array([U[i*m+j] * V[j*p+k] for j in range(m)])
            try:
                W[c] = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
            except:
                pass
        
        # Check error
        err_sum = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a, b, c = i*m+j, j*p+k, k*n+i
                    val = np.dot(U[a], V[b] * W[c])
                    err_sum += (val - 1.0)**2
                    # Also check zeros
                    for a2 in range(n*m):
                        for b2 in range(m*p):
                            for c2 in range(p*n):
                                if (a2,b2,c2) == (a,b,c): continue
                                i2, j2 = a2//m, a2%m
                                jj2, k2 = b2//p, b2%p
                                kk2, ii2 = c2//n, c2%n
                                if j2 != jj2 or k2 != kk2 or i2 != ii2:
                                    val2 = np.dot(U[a2], V[b2] * W[c2])
                                    err_sum += val2**2
        err = np.sqrt(err_sum)
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        if err < 1e-8:
            break
    
    return best_U, best_V, best_W, best_err

def _verify_decomposition_fast(U, V, W, n, m, p):
    """Fast verification using tensor structure."""
    U = np.asarray(U)
    V = np.asarray(V)
    W = np.asarray(W)
    
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
    start_time = time.time()
    deadline = start_time + 145.0
    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    
    # Strategy 1: Find rank-5 decomposition for 2x2x3, tile 4 times -> Rank 20
    U3 = V3 = W3 = None
    for seed in range(50):
        if time.time() > deadline - 10:
            break
        U3_t, V3_t, W3_t, err = _find_decomposition(2, 2, 3, 5, max_iter=2000, seed=seed)
        if err < 1e-8:
            if _verify_decomposition_fast(U3_t, V3_t, W3_t, 2, 2, 3):
                U3, V3, W3 = U3_t, V3_t, W3_t
                break
    
    if U3 is not None:
        rank = 20
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
        
        if _verify_decomposition_fast(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Strategy 2: Find rank-9 decomposition for 2x2x4, tile 3 times -> Rank 27
    U4 = V4 = W4 = None
    for seed in range(20):
        if time.time() > deadline - 10:
            break
        U4_t, V4_t, W4_t, err = _find_decomposition(2, 2, 4, 9, max_iter=2000, seed=seed)
        if err < 1e-8:
            if _verify_decomposition_fast(U4_t, V4_t, W4_t, 2, 2, 4):
                U4, V4, W4 = U4_t, V4_t, W4_t
                break
    
    if U4 is not None:
        rank = 27
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        for t in range(3):
            base_col = 4*t
            for r in range(9):
                U[:, t*9+r] = U4[:, r]
                for j in range(2):
                    for k in range(4):
                        V[j*p + base_col + k, t*9+r] = V4[j*4+k, r]
                for k in range(4):
                    for i in range(2):
                        W[(base_col+k)*n+i, t*9+r] = W4[k*2+i, r]
        
        if _verify_decomposition_fast(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: Rank 42 Strassen on 6 pairs
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
