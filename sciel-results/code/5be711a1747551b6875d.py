"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def _build_strassen_2x2x2():
    """
    Build Strassen's rank-7 decomposition for 2x2 x 2x2 matrix multiplication.
    Returns U (4x7), V (4x7), W (4x7) following the trace convention W[k*n+i, r].
    """
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)
    
    # Term 1: M1 = (A00 + A11) * (B00 + B11)
    U[0, 0] = 1.0; U[3, 0] = 1.0
    V[0, 0] = 1.0; V[3, 0] = 1.0
    W[0, 0] = 1.0; W[3, 0] = 1.0
    
    # Term 2: M2 = (A10 + A11) * B00
    U[2, 1] = 1.0; U[3, 1] = 1.0
    V[0, 1] = 1.0
    W[2, 1] = 1.0; W[3, 1] = -1.0
    
    # Term 3: M3 = A00 * (B01 - B11)
    U[0, 2] = 1.0
    V[1, 2] = 1.0; V[3, 2] = -1.0
    W[1, 2] = 1.0; W[3, 2] = 1.0
    
    # Term 4: M4 = A11 * (B10 - B00)
    U[3, 3] = 1.0
    V[2, 3] = 1.0; V[0, 3] = -1.0
    W[0, 3] = 1.0; W[2, 3] = 1.0
    
    # Term 5: M5 = (A00 + A01) * B11
    U[0, 4] = 1.0; U[1, 4] = 1.0
    V[3, 4] = 1.0
    W[0, 4] = -1.0; W[1, 4] = 1.0
    
    # Term 6: M6 = (A10 - A00) * (B00 + B01)
    U[2, 5] = 1.0; U[0, 5] = -1.0
    V[0, 5] = 1.0; V[1, 5] = 1.0
    W[2, 5] = 1.0
    
    # Term 7: M7 = (A01 - A11) * (B10 + B11)
    U[1, 6] = 1.0; U[3, 6] = -1.0
    V[2, 6] = 1.0; V[3, 6] = 1.0
    W[0, 6] = 1.0
    
    return U, V, W

def _build_schoolbook_2x2x1():
    """Build schoolbook rank-4 decomposition for 2x2 x 2x1."""
    U = np.zeros((4, 4), dtype=np.float64)
    V = np.zeros((2, 4), dtype=np.float64)
    W = np.zeros((2, 4), dtype=np.float64)
    
    U[0, 0] = 1.0
    V[0, 0] = 1.0
    W[0, 0] = 1.0
    
    U[1, 1] = 1.0
    V[0, 1] = 1.0
    W[1, 1] = 1.0
    
    U[2, 2] = 1.0
    V[1, 2] = 1.0
    W[0, 2] = 1.0
    
    U[3, 3] = 1.0
    V[1, 3] = 1.0
    W[1, 3] = 1.0
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly decomposes the matrix multiplication tensor."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(U.shape[1]):
                    val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _als_2x2x3(rank, n_iter=500, tol=1e-10, deadline=None):
    """
    Try to find a rank-rank decomposition for 2x2x3 matrix multiplication using ALS.
    Returns (U, V, W) if found, None otherwise.
    """
    n, m, p = 2, 2, 3
    
    for restart in range(100):
        if deadline is not None and time.time() > deadline:
            return None
            
        rng = np.random.RandomState(restart * 12345 + 67890)
        U = rng.randn(4, rank) * 0.5
        V = rng.randn(6, rank) * 0.5
        W = rng.randn(6, rank) * 0.5
        
        for it in range(n_iter):
            if deadline is not None and time.time() > deadline:
                return None
            
            # Update U: for each (i,j), solve least squares
            for i in range(n):
                for j in range(m):
                    # For fixed (i,j), we need: sum_k sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
                    # This is: sum_r U[i,j,r] * (sum_k V[j,k,r] * W[k,i,r]) = 1
                    # Let M[r] = sum_k V[j,k,r] * W[k,i,r]
                    # Then: sum_r U[i,j,r] * M[r] = 1
                    # But we also need to satisfy all k simultaneously...
                    # Actually, for fixed (i,j), the constraint is:
                    # For each k: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
                    # This is a linear system in U[i,j,:]
                    # A[i,j] @ U[i,j,:] = 1 where A[i,j][k,r] = V[j,k,r] * W[k,i,r]
                    A = np.zeros((p, rank))
                    for k in range(p):
                        A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    try:
                        U[i * m + j, :] = np.linalg.lstsq(A, np.ones(p), rcond=None)[0]
                    except:
                        break
            
            # Update V: for each (j,k), solve least squares
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, rank))
                    for i in range(n):
                        A[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    try:
                        V[j * p + k, :] = np.linalg.lstsq(A, np.ones(n), rcond=None)[0]
                    except:
                        break
            
            # Update W: for each (k,i), solve least squares
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, rank))
                    for j in range(m):
                        A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    try:
                        W[k * n + i, :] = np.linalg.lstsq(A, np.ones(m), rcond=None)[0]
                    except:
                        break
            
            # Check convergence
            if it % 50 == 0 or it == n_iter - 1:
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for r in range(rank):
                                val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                            err += (val - 1.0) ** 2
                if err < tol:
                    return U, V, W
        
        # Check final error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    err += (val - 1.0) ** 2
        if err < 1e-8:
            return U, V, W
    
    return None

def _construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin
    
    # Build the known rank-53 solution first as fallback
    U_strassen, V_strassen, W_strassen = _build_strassen_2x2x2()
    U_school, V_school, W_school = _build_schoolbook_2x2x1()
    
    # Try to find a rank-10 decomposition for 2x2x3
    U_223, V_223, W_223 = None, None, None
    if time.time() < deadline - 30:
        U_223, V_223, W_223 = _als_2x2x3(10, n_iter=300, deadline=deadline - 5)
    
    if U_223 is not None:
        # Use 6 blocks of Strassen (2x2x2) + 1 block of rank-10 (2x2x3)
        # Total rank = 6*7 + 10 = 52
        total_rank = 6 * 7 + 10
        U = np.zeros((4, total_rank), dtype=np.float64)
        V = np.zeros((2 * 15, total_rank), dtype=np.float64)
        W = np.zeros((15 * 2, total_rank), dtype=np.float64)
        
        col_offset = 0
        # 6 blocks of Strassen at k = 0,2,4,6,8,10
        for block in range(6):
            k0 = 2 * block
            for r in range(7):
                c = col_offset + r
                for i in range(2):
                    for j in range(2):
                        U[i * 2 + j, c] = U_strassen[i * 2 + j, r]
                for j in range(2):
                    V[j * 15 + k0, c] = V_strassen[j * 2 + 0, r]
                    V[j * 15 + k0 + 1, c] = V_strassen[j * 2 + 1, r]
                for k in range(2):
                    kk = k0 + k
                    for i in range(2):
                        W[kk * 2 + i, c] = W_strassen[k * 2 + i, r]
            col_offset += 7
        
        # Last block: rank-10 for 2x2x3 at k = 12,13,14
        k_start = 12
        for r in range(10):
            c = col_offset + r
            for i in range(2):
                for j in range(2):
                    U[i * 2 + j, c] = U_223[i * 2 + j, r]
            for j in range(2):
                for k in range(3):
                    V[j * 15 + k_start + k, c] = V_223[j * 3 + k, r]
            for k in range(3):
                for i in range(2):
                    W[(k_start + k) * 2 + i, c] = W_223[k * 2 + i, r]
            col_offset += 1
        
        # Verify
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: rank-53 solution (7 blocks of Strassen + 1 block of schoolbook)
    total_rank = 7 * 7 + 4
    U = np.zeros((4, total_rank), dtype=np.float64)
    V = np.zeros((2 * 15, total_rank), dtype=np.float64)
    W = np.zeros((15 * 2, total_rank), dtype=np.float64)
    
    col_offset = 0
    for block in range(7):
        k0 = 2 * block
        for r in range(7):
            c = col_offset + r
            for i in range(2):
                for j in range(2):
                    U[i * 2 + j, c] = U_strassen[i * 2 + j, r]
            for j in range(2):
                V[j * 15 + k0, c] = V_strassen[j * 2 + 0, r]
                V[j * 15 + k0 + 1, c] = V_strassen[j * 2 + 1, r]
            for k in range(2):
                kk = k0 + k
                for i in range(2):
                    W[kk * 2 + i, c] = W_strassen[k * 2 + i, r]
        col_offset += 7
    
    # Last block: schoolbook for 2x2x1 at k = 14
    k_last = 14
    for r in range(4):
        c = col_offset + r
        for i in range(2):
            for j in range(2):
                U[i * 2 + j, c] = U_school[i * 2 + j, r]
        for j in range(2):
            V[j * 15 + k_last, c] = V_school[j, r]
        for i in range(2):
            W[k_last * 2 + i, c] = W_school[i, r]
        col_offset += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct():
    return _construct()
# EVOLVE-BLOCK-END
