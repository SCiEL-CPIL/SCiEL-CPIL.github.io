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
    # Step 1: Get 2x2x2 Strassen tensors
    U2, V2, W2 = _build_strassen_2x2x2()
    
    # Step 2: Construct 4x4x4 tensors via tensor product (Kronecker product)
    # Rank is 7 * 7 = 49
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
    
    # Step 3: Construct 4x4x8 tensors
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
                
    # Step 4: Vectorized verification
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

def _try_cw_decomposition(n=4, m=4, p=8, target_rank=72, max_iter=200, seed=42):
    """
    Attempts to find a lower-rank decomposition using iterative refinement,
    inspired by Coppersmith-Winograd structured approaches.
    This is a heuristic search that may or may not succeed.
    """
    np.random.seed(seed)
    
    # Initialize with random tensors
    U = np.random.randn(n*m, target_rank)
    V = np.random.randn(m*p, target_rank)
    W = np.random.randn(p*n, target_rank)
    
    # Normalize initial values
    U /= np.linalg.norm(U)
    V /= np.linalg.norm(V)
    W /= np.linalg.norm(W)
    
    best_err = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        if time.time() > DEADLINE - 10:
            break
        
        # Compute error: T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        # We want this to equal 1 for all valid (i,j,k)
        
        # Update U: for each (i,j), solve least squares
        # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all k
        # Let A[i,j,k,r] = V[j*p+k,r] * W[k*n+i,r]
        # Then for fixed (i,j): sum_r U[i*m+j,r] * A[i,j,k,r] = 1 for all k
        # This is: A_sub @ U[i*m+j,:] = ones(p)
        
        for i in range(n):
            for j in range(m):
                # Build A matrix: (p, target_rank)
                A = np.zeros((p, target_rank))
                for k in range(p):
                    A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                b = np.ones(p)
                
                # Solve least squares: A @ x = b
                try:
                    x, residuals, rank_a, sv = np.linalg.lstsq(A, b, rcond=None)
                    U[i*m+j, :] = x
                except:
                    pass
        
        # Update V: for each (j,k), solve least squares
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, target_rank))
                for i in range(n):
                    A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                b = np.ones(n)
                
                try:
                    x, residuals, rank_a, sv = np.linalg.lstsq(A, b, rcond=None)
                    V[j*p+k, :] = x
                except:
                    pass
        
        # Update W: for each (k,i), solve least squares
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, target_rank))
                for j in range(m):
                    A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                b = np.ones(m)
                
                try:
                    x, residuals, rank_a, sv = np.linalg.lstsq(A, b, rcond=None)
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
        
        # Early termination if error is very small
        if err < 1e-8:
            break
        
        # If error is increasing significantly, try a different initialization
        if it > 50 and err > best_err * 2:
            U = np.random.randn(n*m, target_rank)
            V = np.random.randn(m*p, target_rank)
            W = np.random.randn(p*n, target_rank)
            U /= np.linalg.norm(U)
            V /= np.linalg.norm(V)
            W /= np.linalg.norm(W)
    
    # Verify the best solution
    if best_err < 1e-6:
        # Check if we can round to integers/half-integers
        # Try rounding and verify
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
        
        if err_rounded < 1e-6:
            # Convert to lists
            return {
                "U": U_rounded.tolist(),
                "V": V_rounded.tolist(),
                "W": W_rounded.tolist()
            }
    
    return None

def construct():
    # Try the rank-98 block-Strassen decomposition first (reliable)
    try:
        if time.time() < DEADLINE:
            res = _build_rank98()
            if res is not None:
                return res
    except Exception:
        pass
    
    # Try to find a lower-rank decomposition using Coppersmith-Winograd inspired search
    try:
        if time.time() < DEADLINE - 20:
            res = _try_cw_decomposition()
            if res is not None:
                # Verify the result
                U = np.array(res['U'])
                V = np.array(res['V'])
                W = np.array(res['W'])
                n, m, p = 4, 4, 8
                result = np.zeros((n, m, p), dtype=np.float64)
                for r in range(len(U[0])):
                    u_col = U[:, r].reshape(n, m)
                    v_col = V[:, r].reshape(m, p)
                    w_col = W[:, r].reshape(p, n)
                    result += np.einsum('ij,jk,ki->ijk', u_col, v_col, w_col)
                if np.allclose(result, 1.0, atol=1e-6):
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
