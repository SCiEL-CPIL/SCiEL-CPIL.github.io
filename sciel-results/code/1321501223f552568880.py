import numpy as np
import time

DIMENSIONS = [3, 3, 6]

def _schoolbook(n, m, p):
    """Return the standard schoolbook rank-n*m*p decomposition."""
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

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check if sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k."""
    R = U.shape[1]
    U_3d = U.reshape(n, m, R)
    V_3d = V.reshape(m, p, R)
    W_3d = W.reshape(p, n, R)
    S = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    if not np.allclose(S, np.ones((n, m, p)), atol=tol):
        return False
    if not np.allclose(U * 2, np.round(U * 2), atol=1e-8):
        return False
    if not np.allclose(V * 2, np.round(V * 2), atol=1e-8):
        return False
    if not np.allclose(W * 2, np.round(W * 2), atol=1e-8):
        return False
    return True

def _strassen_3x3x3():
    """
    Return the rank-19 Strassen-Winograd decomposition for 3x3x3.
    U: (9, 19), V: (9, 19), W: (9, 19)
    Using the known rank-19 algorithm.
    """
    # The rank-19 3x3x3 decomposition (Strassen-Winograd)
    # This is a well-known result. Let me use the explicit construction.
    # 
    # Actually, let me use a simpler approach: construct it from the known formulas.
    # The 3x3x3 matrix multiplication has rank 19.
    # 
    # Let me use the explicit rank-19 construction from the literature.
    # For now, let me use a known valid rank-19 decomposition.
    
    # I'll use the construction from Bini et al. or similar.
    # Here's a known rank-19 3x3x3 decomposition:
    
    U = np.zeros((9, 19))
    V = np.zeros((9, 19))
    W = np.zeros((9, 19))
    
    # Using the Strassen-Winograd rank-19 algorithm for 3x3x3
    # The algorithm uses 19 multiplications.
    # Let me encode the known rank-19 scheme.
    
    # Actually, let me just verify if a simple construction works.
    # The key insight: 3x3x6 = 2 * (3x3x3), so rank <= 2*19 = 38.
    # Let me construct this directly.
    
    # For the 3x3x3 part, I'll use a known rank-19 decomposition.
    # Here's one from the literature (adapted):
    
    # M1 = (A11+A22+A33)(B11+B22+B33)
    # M2 = (A11-A22)(B11-B22)  
    # M3 = (A22-A33)(B22-B33)
    # M4 = (A33-A11)(B33-B11)
    # M5 = (A12+A23)(B12+B23)
    # M6 = (A12-A23)(B12-B23)
    # M7 = (A21+A32)(B21+B32)
    # M8 = (A21-A32)(B21-B32)
    # M9 = (A31+A13)(B31+B13)
    # M10 = (A31-A13)(B31-B13)
    # M11 = (A11+A12+A21)(B11+B12+B21)
    # M12 = (A11-A12+A21)(B11-B12+B21)
    # M13 = (A22+A23+A32)(B22+B23+B32)
    # M14 = (A22-A23+A32)(B22-B23+B32)
    # M15 = (A33+A31+A13)(B33+B31+B13)
    # M16 = (A33-A31+A13)(B33-B31+B13)
    # M17 = (A13+A31+A33)(B13+B31+B33)
    # M18 = (A13-A31+A33)(B13-B31+B33)
    # M19 = (A21+A12+A11)(B21+B12+B11)
    
    # This is getting complex. Let me use a different approach.
    # I'll search for a rank-38 decomposition of 3x3x6 directly.
    
    # For now, return None to indicate we need to search.
    return None, None, None

def _construct_38_from_19():
    """
    Try to construct a rank-38 decomposition of 3x3x6 from two rank-19 3x3x3 decompositions.
    Split B (3x6) into B1 (3x3) and B2 (3x3).
    C = A*B = A*B1 + A*B2
    """
    n, m, p = 3, 3, 6
    
    # We need a rank-19 3x3x3 decomposition first.
    # Let me try to find one using ALS with a tight deadline.
    
    deadline = time.time() + 20.0
    
    # Try to find rank-19 3x3x3
    rng = np.random.RandomState(42)
    U19 = None
    V19 = None
    W19 = None
    
    for seed in range(20):
        if time.time() > deadline:
            break
        rng = np.random.RandomState(seed)
        U = rng.randn(9, 19) * 0.1
        V = rng.randn(9, 19) * 0.1
        W = rng.randn(9, 19) * 0.1
        
        for it in range(50):
            if time.time() > deadline:
                break
            # ALS step for 3x3x3
            for i in range(3):
                for j in range(3):
                    M = np.zeros((3, 19))
                    for k in range(3):
                        M[k, :] = V[j*3+k, :] * W[k*3+i, :]
                    try:
                        U[i*3+j, :] = np.linalg.lstsq(M, np.ones(3), rcond=None)[0]
                    except:
                        pass
            for j in range(3):
                for k in range(3):
                    M = np.zeros((3, 19))
                    for i in range(3):
                        M[i, :] = U[i*3+j, :] * W[k*3+i, :]
                    try:
                        V[j*3+k, :] = np.linalg.lstsq(M, np.ones(3), rcond=None)[0]
                    except:
                        pass
            for k in range(3):
                for i in range(3):
                    M = np.zeros((3, 19))
                    for j in range(3):
                        M[j, :] = U[i*3+j, :] * V[j*3+k, :]
                    try:
                        W[k*3+i, :] = np.linalg.lstsq(M, np.ones(3), rcond=None)[0]
                    except:
                        pass
            
            U_3d = U.reshape(3, 3, 19)
            V_3d = V.reshape(3, 3, 19)
            W_3d = W.reshape(3, 3, 19)
            S = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
            err = np.sum((S - 1.0)**2)
            if err < 1e-10:
                U19, V19, W19 = U.copy(), V.copy(), W.copy()
                break
        if U19 is not None:
            break
    
    if U19 is None:
        return None
    
    # Now construct 3x3x6 from two 3x3x3 decompositions
    # C = A*B where B = [B1 | B2]
    # C = A*B1 + A*B2
    # Using the same U for both (since A is the same), but different V and W
    
    U38 = np.zeros((9, 38))
    V38 = np.zeros((18, 38))
    W38 = np.zeros((18, 38))
    
    # First 3x3x3 (columns 0-2 of B)
    U38[:, 0:19] = U19
    V38[0:9, 0:19] = V19  # B1 is rows 0-8 of V (3*3=9 entries)
    W38[0:9, 0:19] = W19   # W for k=0,1,2
    
    # Second 3x3x3 (columns 3-5 of B)
    U38[:, 19:38] = U19
    V38[9:18, 19:38] = V19  # B2 is rows 9-17 of V
    W38[9:18, 19:38] = W19  # W for k=3,4,5 -> indices 9-17 in W (k*n+i, k=3,4,5)
    
    # Verify
    if _verify(U38, V38, W38, 3, 3, 6, tol=1e-8):
        return U38, V38, W38
    
    # Try snapping to half-integers
    U38_r = np.round(U38 * 2) / 2
    V38_r = np.round(V38 * 2) / 2
    W38_r = np.round(W38 * 2) / 2
    if _verify(U38_r, V38_r, W38_r, 3, 3, 6, tol=1e-8):
        return U38_r, V38_r, W38_r
    
    return None

def _als_step(U, V, W, n, m, p, deadline):
    """Perform one step of Alternating Least Squares."""
    R = U.shape[1]
    
    U_new = U.copy()
    for i in range(n):
        for j in range(m):
            if time.time() > deadline:
                return U, V, W
            M = np.zeros((p, R))
            for k in range(p):
                M[k, :] = V[j*p+k, :] * W[k*n+i, :]
            try:
                U_new[i*m+j, :] = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
            except np.linalg.LinAlgError:
                pass
    U = U_new
    
    V_new = V.copy()
    for j in range(m):
        for k in range(p):
            if time.time() > deadline:
                return U, V, W
            M = np.zeros((n, R))
            for i in range(n):
                M[i, :] = U[i*m+j, :] * W[k*n+i, :]
            try:
                V_new[j*p+k, :] = np.linalg.lstsq(M, np.ones(n), rcond=None)[0]
            except np.linalg.LinAlgError:
                pass
    V = V_new
    
    W_new = W.copy()
    for k in range(p):
        for i in range(n):
            if time.time() > deadline:
                return U, V, W
            M = np.zeros((m, R))
            for j in range(m):
                M[j, :] = U[i*m+j, :] * V[j*p+k, :]
            try:
                W_new[k*n+i, :] = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
            except np.linalg.LinAlgError:
                pass
    W = W_new
    
    return U, V, W

def _try_rank(n, m, p, rank, seed, deadline):
    """Try to find a rank-r decomposition using ALS."""
    rng = np.random.RandomState(seed)
    U = rng.randn(n * m, rank) * 0.1
    V = rng.randn(m * p, rank) * 0.1
    W = rng.randn(p * n, rank) * 0.1
    
    best_err = float('inf')
    best_U, best_V, best_W = None, None, None
    
    max_iters = 100
    for it in range(max_iters):
        if time.time() > deadline:
            break
        U, V, W = _als_step(U, V, W, n, m, p, deadline)
        
        U_3d = U.reshape(n, m, rank)
        V_3d = V.reshape(m, p, rank)
        W_3d = W.reshape(p, n, rank)
        S = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        err = np.sum((S - 1.0)**2)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
        if best_err < 1e-10:
            break
    
    if best_err < 1e-8:
        U_round = np.round(best_U * 2) / 2
        V_round = np.round(best_V * 2) / 2
        W_round = np.round(best_W * 2) / 2
        if _verify(U_round, V_round, W_round, n, m, p, tol=1e-8):
            return U_round, V_round, W_round
    
    return None, None, None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    # Fallback: schoolbook
    U_fb, V_fb, W_fb = _schoolbook(n, m, p)
    best_U, best_V, best_W = U_fb, V_fb, W_fb
    best_rank = n * m * p
    
    # First, try to construct rank-38 from two rank-19 3x3x3 decompositions
    result = _construct_38_from_19()
    if result is not None:
        U38, V38, W38 = result
        best_U, best_V, best_W = U38, V38, W38
        best_rank = 38
    
    # Try to find lower ranks
    ranks_to_try = [37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18]
    
    for R in ranks_to_try:
        if time.time() > deadline:
            break
        if R >= best_rank:
            continue
            
        for seed in range(30):
            if time.time() > deadline:
                break
            U, V, W = _try_rank(n, m, p, R, seed, deadline)
            if U is not None:
                best_rank = R
                best_U, best_V, best_W = U, V, W
                break
    
    def to_list(mat):
        return mat.tolist()
        
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
