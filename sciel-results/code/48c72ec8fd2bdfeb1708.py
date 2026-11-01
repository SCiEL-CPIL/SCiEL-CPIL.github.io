import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def _verify_fast(U, V, W, n=2, m=4, p=8, trials=20, tol=1e-8):
    """Fast verification using vectorized numpy operations."""
    if not U or not V or not W:
        return False
    rank = len(U[0])
    if rank == 0:
        return False
    
    U_mat = np.array(U, dtype=float)
    V_mat = np.array(V, dtype=float)
    W_mat = np.array(W, dtype=float)
    
    if U_mat.shape != (n*m, rank) or V_mat.shape != (m*p, rank) or W_mat.shape != (p*n, rank):
        return False

    for _ in range(trials):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B

        U_reshaped = U_mat.reshape(n, m, rank)
        V_reshaped = V_mat.reshape(m, p, rank)
        W_reshaped = W_mat.reshape(p, n, rank)

        factor_A = np.einsum('ij,ijc->ic', A, U_reshaped)
        factor_B = np.einsum('jk,jkc->kc', B, V_reshaped)
        C_computed = np.einsum('ic,kc,kic->ik', factor_A, factor_B, W_reshaped)

        if np.max(np.abs(C_computed - C_expected)) > tol:
            return False
    return True

def _build_strassen_2x2x2():
    """Standard Strassen decomposition for 2x2 * 2x2 -> 2x2, rank 7."""
    U = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, -1],
        [0, 1, 0, 1]
    ], dtype=float)
    V = np.array([
        [1, 0, 0, 1],
        [0, 0, 0, 1],
        [1, 0, -1, 0],
        [0, -1, 0, 1],
        [0, 0, 1, 0],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ], dtype=float)
    W = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=float)
    return U.tolist(), V.tolist(), W.tolist()

def _build_strassen_56():
    """Build the standard block-Strassen rank-56 decomposition for 2x4x8."""
    n, m, p = 2, 4, 8
    rank = 56
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    U_s, V_s, W_s = _build_strassen_2x2x2()
    U_s = np.array(U_s)
    V_s = np.array(V_s)
    W_s = np.array(W_s)
    
    col_offset = 0
    for block in range(4):
        for f in range(7):
            col = col_offset + f
            for r in range(4):
                U[r, col] = U_s[f, r]
            for r in range(4):
                j = r // 2
                k = 2 * block + r % 2
                V[j * 8 + k, col] = V_s[f, r]
            for r in range(4):
                i = r // 2
                k = 2 * block + r % 2
                W[k * 2 + i, col] = W_s[r, f]
            col_offset += 1
    
    for block in range(4):
        for f in range(7):
            col = col_offset + f
            for r in range(4):
                U[2 + r, col] = U_s[f, r]
            for r in range(4):
                j = 2 + r // 2
                k = 2 * block + r % 2
                V[j * 8 + k, col] = V_s[f, r]
            for r in range(4):
                i = r // 2
                k = 2 * block + r % 2
                W[k * 2 + i, col] = W_s[r, f]
            col_offset += 1
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Build the standard Strassen decomposition (Rank 56)
    U_56, V_56, W_56 = _build_strassen_56()
    
    # Verify Rank 56
    if not _verify_fast(U_56.tolist(), V_56.tolist(), W_56.tolist(), n, m, p, trials=5, tol=1e-9):
        # Fallback to schoolbook
        schoolbook_rank = n * m * p
        U_sb = np.zeros((n*m, schoolbook_rank))
        V_sb = np.zeros((m*p, schoolbook_rank))
        W_sb = np.zeros((p*n, schoolbook_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m+j, col] = 1
                    V_sb[j*p+k, col] = 1
                    W_sb[k*n+i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    return {"U": U_56.tolist(), "V": V_56.tolist(), "W": W_56.tolist()}
