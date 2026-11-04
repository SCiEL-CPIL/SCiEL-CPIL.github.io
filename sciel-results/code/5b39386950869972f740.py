import time
import numpy as np

DIMENSIONS = [4, 4, 8]
START = time.time()
DEADLINE = START + 130.0   # safety margin under 150s kill

def _build_strassen_2x2x2():
    """
    Returns U, V, W for 2x2x2 Strassen decomposition (rank 7).
    Indices:
    U[i*2+j, r]
    V[j*2+k, r]
    W[k*2+i, r]
    """
    U = np.zeros((4, 7), dtype=np.int8)
    V = np.zeros((4, 7), dtype=np.int8)
    W = np.zeros((4, 7), dtype=np.int8)

    # r=0: M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1

    # r=1: M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1

    # r=2: M3 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1

    # r=3: M4 = A11(B10-B00)
    U[3, 3] = 1
    V[0, 3] = -1; V[2, 3] = 1
    W[0, 3] = 1; W[1, 3] = 1

    # r=4: M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1

    # r=5: M6 = (A10-A00)(B00+B01)
    U[0, 5] = -1; U[2, 5] = 1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1

    # r=6: M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1

    return U, V, W

def _build_rank98():
    """
    Builds a rank-98 decomposition for 4x4 * 4x8 matrix multiplication.
    Uses the tensor product of Strassen's 2x2x2 decomposition to create a
    rank-49 decomposition for 4x4x4, then duplicates it for the two 4x4 blocks
    of the 4x8 output.
    """
    U2, V2, W2 = _build_strassen_2x2x2()
    
    rank_444 = 49
    U4 = np.zeros((16, rank_444), dtype=np.int16)
    V4 = np.zeros((16, rank_444), dtype=np.int16)
    W4 = np.zeros((16, rank_444), dtype=np.int16)
    
    for c0 in range(7):
        for c1 in range(7):
            c = c0 * 7 + c1
            for i in range(4):
                i0, i1 = i // 2, i % 2
                for j in range(4):
                    j0, j1 = j // 2, j % 2
                    U4[i*4+j, c] = U2[i0*2+j0, c0] * U2[i1*2+j1, c1]
            
            for j in range(4):
                j0, j1 = j // 2, j % 2
                for k in range(4):
                    k0, k1 = k // 2, k % 2
                    V4[j*4+k, c] = V2[j0*2+k0, c0] * V2[j1*2+k1, c1]
            
            for k in range(4):
                k0, k1 = k // 2, k % 2
                for i in range(4):
                    i0, i1 = i // 2, i % 2
                    W4[k*4+i, c] = W2[k0*2+i0, c0] * W2[k1*2+i1, c1]
    
    total_rank = 98
    U = np.zeros((16, total_rank), dtype=np.int16)
    V = np.zeros((32, total_rank), dtype=np.int16)
    W = np.zeros((32, total_rank), dtype=np.int16)
    
    for block in range(2):
        col_start = block * 49
        col_end = (block + 1) * 49
        U[:, col_start:col_end] = U4
        
        for j in range(4):
            for k_local in range(4):
                k_global = block * 4 + k_local
                V[j*8 + k_global, col_start:col_end] = V4[j*4 + k_local, :]
        
        for k_local in range(4):
            k_global = block * 4 + k_local
            for i in range(4):
                W[k_global*4 + i, col_start:col_end] = W4[k_local*4 + i, :]
                
    # Vectorized verification
    n, m, p = 4, 4, 8
    result = np.zeros((n, m, p), dtype=np.float64)
    for r in range(total_rank):
        u_col = U[:, r].reshape(n, m)
        v_col = V[:, r].reshape(m, p)
        w_col = W[:, r].reshape(p, n)
        result += np.einsum('ij,jk,ki->ijk', u_col, v_col, w_col)
    
    if not np.allclose(result, 1.0, atol=1e-9):
        return None
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

def _schoolbook(n=4, m=4, p=8):
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

def _verify_decomposition(U, V, W, n=4, m=4, p=8, atol=1e-6):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if isinstance(U, list) else U.shape[1]
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    result = np.zeros((n, m, p), dtype=np.float64)
    for r in range(rank):
        u_col = U[:, r].reshape(n, m)
        v_col = V[:, r].reshape(m, p)
        w_col = W[:, r].reshape(p, n)
        result += np.einsum('ij,jk,ki->ijk', u_col, v_col, w_col)
    
    return np.allclose(result, 1.0, atol=atol)

def _try_rank_reduction(U_init, V_init, W_init, target_rank, max_iter=300, tol=1e-8):
    """
    Attempts to reduce the rank of a decomposition using iterative coordinate descent.
    U_init, V_init, W_init are the initial factor matrices (as numpy arrays).
    target_rank is the desired output rank.
    Returns (U, V, W) as numpy arrays if successful, else None.
    """
    n, m, p = 4, 4, 8
    current_rank = U_init.shape[1]
    
    # Start with the initial decomposition, truncate to target_rank
    U = U_init[:, :target_rank].copy().astype(np.float64)
    V = V_init[:, :target_rank].copy().astype(np.float64)
    W = W_init[:, :target_rank].copy().astype(np.float64)
    
    best_err = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        if time.time() > DEADLINE - 15:
            break
        
        # Coordinate descent: update each factor matrix one at a time
        # For U: fix V, W, solve for each row of U
        for i in range(n):
            for j in range(m):
                # Want: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k
                # Let A[k, r] = V[j*p+k, r] * W[k*n+i, r]
                # Then: A @ U[i*m+j, :] = ones(p)
                A = np.zeros((p, target_rank))
                for k in range(p):
                    A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                b = np.ones(p)
                try:
                    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    U[i*m+j, :] = x
                except:
                    pass
        
        # For V: fix U, W, solve for each row of V
        for j in range(m):
            for k in range(p):
                # Want: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i
                # Let A[i, r] = U[i*m+j, r] * W[k*n+i, r]
                # Then: A @ V[j*p+k, :] = ones(n)
                A = np.zeros((n, target_rank))
                for i in range(n):
                    A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                b = np.ones(n)
                try:
                    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    V[j*p+k, :] = x
                except:
                    pass
        
        # For W: fix U, V, solve for each row of W
        for k in range(p):
            for i in range(n):
                # Want: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all j
                # Let A[j, r] = U[i*m+j, r] * V[j*p+k, r]
                # Then: A @ W[k*n+i, :] = ones(m)
                A = np.zeros((m, target_rank))
                for j in range(m):
                    A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                b = np.ones(m)
                try:
                    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    W[k*n+i, :] = x
                except:
                    pass
        
        # Compute error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    err += (val - 1.0) ** 2
        err = np.sqrt(err / (n*m*p))
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < tol:
            break
    
    # Try to round to half-integers
    U_rounded = np.round(best_U * 2) / 2
    V_rounded = np.round(best_V * 2) / 2
    W_rounded = np.round(best_W * 2) / 2
    
    err_rounded = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_rounded[i*m+j, :] * V_rounded[j*p+k, :] * W_rounded[k*n+i, :])
                err_rounded += (val - 1.0) ** 2
    err_rounded = np.sqrt(err_rounded / (n*m*p))
    
    if err_rounded < 1e-5:
        return U_rounded, V_rounded, W_rounded
    
    # Try rounding to integers
    U_rounded = np.round(best_U)
    V_rounded = np.round(best_V)
    W_rounded = np.round(best_W)
    
    err_rounded = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_rounded[i*m+j, :] * V_rounded[j*p+k, :] * W_rounded[k*n+i, :])
                err_rounded += (val - 1.0) ** 2
    err_rounded = np.sqrt(err_rounded / (n*m*p))
    
    if err_rounded < 1e-5:
        return U_rounded, V_rounded, W_rounded
    
    return None

def _try_winograd_style(n=4, m=4, p=8):
    """
    Attempts a Winograd-style decomposition exploiting the block structure.
    The 4x4x8 tensor can be seen as two 4x4x4 blocks.
    We try to find a shared structure between the two blocks that reduces total rank.
    
    Strategy: Use a single set of "intermediate" computations that serve both blocks,
    with different projection matrices for each block's output.
    """
    # This is a heuristic search for a structured low-rank decomposition
    # We'll try random structured initializations and refine
    
    best_result = None
    best_rank = 98  # Start with known rank
    
    # Try various target ranks
    for target_rank in [40, 48, 56, 64, 72, 80, 88]:
        if time.time() > DEADLINE - 20:
            break
        
        # Multiple random restarts
        for restart in range(3):
            if time.time() > DEADLINE - 15:
                break
            
            np.random.seed(42 + target_rank * 100 + restart)
            
            # Initialize with structured random tensors
            # Use the Strassen-based rank-98 as a starting point and truncate
            U98, V98, W98 = _build_rank98()
            U98 = np.array(U98['U'], dtype=np.float64)
            V98 = np.array(V98['V'], dtype=np.float64)
            W98 = np.array(W98['W'], dtype=np.float64)
            
            # Truncate to target_rank
            U = U98[:, :target_rank].copy()
            V = V98[:, :target_rank].copy()
            W = W98[:, :target_rank].copy()
            
            # Add some random perturbation to escape local minima
            if restart > 0:
                U += np.random.randn(n*m, target_rank) * 0.1
                V += np.random.randn(m*p, target_rank) * 0.1
                W += np.random.randn(p*n, target_rank) * 0.1
            
            result = _try_rank_reduction(U, V, W, target_rank, max_iter=150)
            
            if result is not None:
                U_res, V_res, W_res = result
                if target_rank < best_rank:
                    best_rank = target_rank
                    best_result = {
                        "U": U_res.tolist(),
                        "V": V_res.tolist(),
                        "W": W_res.tolist()
                    }
    
    return best_result

def construct():
    # Try the rank-98 block-Strassen decomposition first (reliable)
    try:
        if time.time() < DEADLINE:
            res = _build_rank98()
            if res is not None:
                return res
    except Exception:
        pass
    
    # Try to find a lower-rank decomposition using Winograd-style search
    try:
        if time.time() < DEADLINE - 20:
            res = _try_winograd_style()
            if res is not None:
                # Verify the result
                if _verify_decomposition(res['U'], res['V'], res['W']):
                    return res
    except Exception:
        pass
    
    # Fallback to schoolbook
    return _schoolbook()

if __name__ == "__main__":
    # Quick self-test
    res = construct()
    print(f"Rank: {len(res['U'][0])}")
    # Verify manually if needed
    U, V, W = res['U'], res['V'], res['W']
    n, m, p = 4, 4, 8
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(len(U[0])):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > 1e-6:
                    valid = False
                    break
            if not valid: break
        if not valid: break
    print(f"Valid: {valid}")
