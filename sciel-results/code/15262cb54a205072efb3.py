"""Matrix multiplication 2x3x8 - rank-minimal tensor decomposition search."""
import time
import numpy as np

DIMENSIONS = [2, 3, 8]

def _verify(U, V, W, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    T = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T += np.einsum('a,b,c->abc', U_arr[:,r], V_arr[:,r], W_arr[:,r])
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                err = abs(T[a,b,c] - 1.0)
                if err > max_err:
                    max_err = err
    
    target = np.zeros((n*m, m*p, p*n), dtype=bool)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = True
    
    non_target_err = np.max(np.abs(T[~target])) if (~target).any() else 0.0
    
    return max_err <= tol and non_target_err <= tol

def _round_and_verify(U, V, W, tol=1e-5):
    """Try to round U, V, W to integers or half-integers and verify."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Try half-integers
    U_half = np.round(U_arr * 2) / 2
    V_half = np.round(V_arr * 2) / 2
    W_half = np.round(W_arr * 2) / 2
    if _verify(U_half.tolist(), V_half.tolist(), W_half.tolist(), tol=tol):
        return U_half.tolist(), V_half.tolist(), W_half.tolist()
        
    # Try integers
    U_int = np.round(U_arr)
    V_int = np.round(V_arr)
    W_int = np.round(W_arr)
    if _verify(U_int.tolist(), V_int.tolist(), W_int.tolist(), tol=tol):
        return U_int.tolist(), V_int.tolist(), W_int.tolist()
        
    return None

def _build_rank44_decomposition():
    """
    Build explicit rank-44 decomposition for 2x3 * 3x8.
    Uses 4 blocks of 2x3 * 3x2, each rank 11 (4 for outer product + 7 for Strassen).
    """
    n, m, p = 2, 3, 8
    rank_total = 44
    U = np.zeros((n*m, rank_total))  # 6 x 44
    V = np.zeros((m*p, rank_total))  # 24 x 44
    W = np.zeros((p*n, rank_total))  # 16 x 44
    
    col = 0
    for block in range(4):
        k0 = 2 * block
        k1 = 2 * block + 1
        
        # Part 1: Outer product A_col0 (2x1) * B_row0 (1x2) -> Rank 4
        # A_col0: indices 0 (a00), 3 (a10)
        # B_row0: indices k0 (b0k0), k1 (b0k1)
        # C: indices 2*k0 (c0k0), 2*k0+1 (c1k0), 2*k1 (c0k1), 2*k1+1 (c1k1)
        
        # t1 = a00 * b0k0 -> C0k0
        t = col
        U[0, t] = 1
        V[k0, t] = 1
        W[2*k0, t] = 1
        col += 1
        
        # t2 = a00 * b0k1 -> C0k1
        t = col
        U[0, t] = 1
        V[k1, t] = 1
        W[2*k1, t] = 1
        col += 1
        
        # t3 = a10 * b0k0 -> C1k0
        t = col
        U[3, t] = 1
        V[k0, t] = 1
        W[2*k0+1, t] = 1
        col += 1
        
        # t4 = a10 * b0k1 -> C1k1
        t = col
        U[3, t] = 1
        V[k1, t] = 1
        W[2*k1+1, t] = 1
        col += 1
        
        # Part 2: Strassen 2x2 * 2x2 for A_cols1,2 * B_rows1,2 -> Rank 7
        # A_cols1,2: 
        #   a01 -> U[1], a02 -> U[2]
        #   a11 -> U[4], a12 -> U[5]
        # B_rows1,2:
        #   b1k0 -> V[8+k0], b1k1 -> V[8+k1]
        #   b2k0 -> V[16+k0], b2k1 -> V[16+k1]
        # C: same as before
        
        # Strassen for 2x2:
        # M1 = (A11 + A22) * (B11 + B22)
        # A11=a01, A22=a12, B11=b1k0, B22=b2k1
        t = col
        U[1, t] = 1
        U[5, t] = 1
        V[8+k0, t] = 1
        V[16+k1, t] = 1
        W[2*k0, t] = 1    # C0k0
        W[2*k1+1, t] = 1  # C1k1
        col += 1
        
        # M2 = (A21 + A22) * B11
        # A21=a11, A22=a12, B11=b1k0
        t = col
        U[4, t] = 1
        U[5, t] = 1
        V[8+k0, t] = 1
        W[2*k0+1, t] = 1  # C1k0
        W[2*k1+1, t] = -1 # -C1k1
        col += 1
        
        # M3 = A11 * (B12 - B22)
        # A11=a01, B12=b1k1, B22=b2k1
        t = col
        U[1, t] = 1
        V[8+k1, t] = 1
        V[16+k1, t] = -1
        W[2*k1, t] = 1    # C0k1
        W[2*k1+1, t] = 1  # C1k1
        col += 1
        
        # M4 = A22 * (B21 - B11)
        # A22=a12, B21=b2k0, B11=b1k0
        t = col
        U[5, t] = 1
        V[16+k0, t] = 1
        V[8+k0, t] = -1
        W[2*k0, t] = 1    # C0k0
        W[2*k0+1, t] = 1  # C1k0
        col += 1
        
        # M5 = (A11 + A12) * B22
        # A11=a01, A12=a02, B22=b2k1
        t = col
        U[1, t] = 1
        U[2, t] = 1
        V[16+k1, t] = 1
        W[2*k0, t] = -1   # -C0k0
        W[2*k1, t] = 1    # C0k1
        col += 1
        
        # M6 = (A21 - A11) * (B11 + B12)
        # A21=a11, A11=a01, B11=b1k0, B12=b1k1
        t = col
        U[4, t] = 1
        U[1, t] = -1
        V[8+k0, t] = 1
        V[8+k1, t] = 1
        W[2*k1+1, t] = 1  # C1k1
        col += 1
        
        # M7 = (A12 - A22) * (B21 + B22)
        # A12=a02, A22=a12, B21=b2k0, B22=b2k1
        t = col
        U[2, t] = 1
        U[5, t] = -1
        V[16+k0, t] = 1
        V[16+k1, t] = 1
        W[2*k0, t] = 1    # C0k0
        col += 1
        
    assert col == 44, f"Expected 44 columns, got {col}"
    
    return U.tolist(), V.tolist(), W.tolist()

def _build_schoolbook():
    """Standard schoolbook decomposition of rank n*m*p = 48."""
    n, m, p = DIMENSIONS
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
    return U, V, W

def _als_decompose(rank, max_iters=50, deadline=None, seed=42, init_U=None, init_V=None, init_W=None):
    """
    Alternating Least Squares for tensor decomposition.
    """
    n, m, p = DIMENSIONS
    rng = np.random.RandomState(seed)
    
    if init_U is not None:
        U = np.array(init_U, dtype=float).copy()
        V = np.array(init_V, dtype=float).copy()
        W = np.array(init_W, dtype=float).copy()
    else:
        U = rng.randn(n*m, rank) * 0.1
        V = rng.randn(m*p, rank) * 0.1
        W = rng.randn(p*n, rank) * 0.1
        for r in range(rank):
            i = r % n
            j = (r // n) % m
            k = (r // (n*m)) % p
            U[i*m+j, r] += 1.0
            V[j*p+k, r] += 1.0
            W[k*n+i, r] += 1.0

    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
            
        # Update U
        for i in range(n):
            for j in range(m):
                a = i * m + j
                M = np.zeros((p, rank))
                for k in range(p):
                    b = j * p + k
                    c = k * n + i
                    M[k, :] = V[b, :] * W[c, :]
                try:
                    x, _, _, _ = np.linalg.lstsq(M, np.ones(p), rcond=None)
                    U[a, :] = x
                except:
                    pass
                    
        # Update V
        for j in range(m):
            for k in range(p):
                b = j * p + k
                M = np.zeros((n*p, rank))
                idx = 0
                for i in range(n):
                    for kk in range(p):
                        a = i * m + j
                        c = kk * n + i
                        M[idx, :] = U[a, :] * W[c, :]
                        idx += 1
                try:
                    x, _, _, _ = np.linalg.lstsq(M, np.ones(n*p), rcond=None)
                    V[b, :] = x
                except:
                    pass
                    
        # Update W
        for k in range(p):
            for i in range(n):
                c = k * n + i
                M = np.zeros((n*m, rank))
                idx = 0
                for ii in range(n):
                    for j in range(m):
                        a = ii * m + j
                        b = j * p + k
                        M[idx, :] = U[a, :] * V[b, :]
                        idx += 1
                try:
                    x, _, _, _ = np.linalg.lstsq(M, np.ones(n*m), rcond=None)
                    W[c, :] = x
                except:
                    pass
                    
        if it % 5 == 4:
            T = np.zeros((n*m, m*p, p*n))
            for r in range(rank):
                T += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
            
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        err = abs(T[a,b,c] - 1.0)
                        if err > max_err:
                            max_err = err
            target = np.zeros((n*m, m*p, p*n), dtype=bool)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        target[i*m+j, j*p+k, k*n+i] = True
            non_target_err = np.max(np.abs(T[~target])) if (~target).any() else 0.0
            
            if max_err < 1e-9 and non_target_err < 1e-9:
                return U.tolist(), V.tolist(), W.tolist()
            
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 48
    
    best_U, best_V, best_W = _build_schoolbook()
    best_rank = schoolbook_rank
    
    start_time = time.time()
    deadline = start_time + 130
    
    # 1. Try deterministic rank-44
    try:
        U44, V44, W44 = _build_rank44_decomposition()
        if _verify(U44, V44, W44, tol=1e-6):
            rounded = _round_and_verify(U44, V44, W44)
            if rounded:
                best_U, best_V, best_W = rounded
                best_rank = 44
            else:
                best_U, best_V, best_W = U44, V44, W44
                best_rank = 44
    except Exception:
        pass
    
    # 2. Try to find rank-40 (4 blocks of rank-10) using ALS initialized with rank-44
    # We project the rank-44 solution to rank-40 by keeping the first 40 columns?
    # Or we can just run ALS from scratch with rank 40.
    if time.time() < deadline - 20:
        for rank in range(40, 44):
            if time.time() > deadline - 5:
                break
            found = False
            for trial in range(10):
                if time.time() > deadline - 5:
                    break
                result = _als_decompose(rank, max_iters=100, deadline=deadline - 5, seed=trial*100+rank)
                if result:
                    U, V, W = result
                    if _verify(U, V, W, tol=1e-6):
                        rounded = _round_and_verify(U, V, W)
                        if rounded:
                            best_U, best_V, best_W = rounded
                            best_rank = rank
                        else:
                            best_U, best_V, best_W = U, V, W
                            best_rank = rank
                        found = True
                        break
            if found:
                break
                
    return {"U": best_U, "V": best_V, "W": best_W}
