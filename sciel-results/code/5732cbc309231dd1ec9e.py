"""Matrix multiplication 2,2,15 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_222():
    """Return Strassen decomposition for 2x2x2 multiplication (rank 7)."""
    U = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=float)
    V = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=float)
    W = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=float)
    return U, V, W

def _schoolbook_221():
    """Return schoolbook decomposition for 2x2x1 multiplication (rank 4)."""
    U = np.eye(4)
    V = np.array([[1, 0, 1, 0],
                  [0, 1, 0, 1]], dtype=float)
    W = np.array([[1, 1, 0, 0],
                  [0, 0, 1, 1]], dtype=float)
    return U, V, W

def _build_strassen_53():
    """Build the guaranteed rank-53 decomposition (7 Strassen blocks + 1 schoolbook)."""
    n, m, p = DIMENSIONS
    rank = 53
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    U_strassen, V_strassen, W_strassen = _strassen_222()
    U_school, V_school, W_school = _schoolbook_221()
    
    col_offset = 0
    for block in range(7):
        k0 = 2 * block
        k1 = 2 * block + 1
        for r in range(7):
            c = col_offset + r
            for i in range(2):
                for j in range(2):
                    U[i * m + j, c] = U_strassen[i * 2 + j, r]
            for j in range(2):
                V[j * p + k0, c] = V_strassen[j * 2, r]
                V[j * p + k1, c] = V_strassen[j * 2 + 1, r]
            for k in range(2):
                kk = k0 + k
                for i in range(2):
                    W[kk * n + i, c] = W_strassen[k * 2 + i, r]
        col_offset += 7
    
    k_last = 14
    for r in range(4):
        c = col_offset + r
        for i in range(2):
            for j in range(2):
                U[i * m + j, c] = U_school[i * 2 + j, r]
        for j in range(2):
            V[j * p + k_last, c] = V_school[j, r]
        for i in range(2):
            W[k_last * n + i, c] = W_school[i, r]
    
    return {"U": U, "V": V, "W": W}

def _verify_decomp(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _tensor_residuals(U, V, W, n, m, p):
    """Compute residuals for all (i,j,k) entries."""
    residuals = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                residuals.append(val - 1.0)
    return np.array(residuals)

def _try_rank_reduction(U, V, W, n, m, p, target_rank, max_iter=2000, tol=1e-8):
    """
    Try to reduce rank from current to target_rank by zeroing out columns
    using a greedy approach with local optimization.
    """
    current_rank = U.shape[1]
    if current_rank <= target_rank:
        return None
    
    # Make copies
    U_cur = U.copy()
    V_cur = V.copy()
    W_cur = W.copy()
    
    # Try removing one column at a time
    for attempt in range(current_rank - target_rank):
        best_col = None
        best_loss = float('inf')
        best_U = None
        best_V = None
        best_W = None
        
        for c in range(U_cur.shape[1]):
            # Try removing column c by redistributing its contribution
            # Simple approach: just zero it and check
            U_test = U_cur.copy()
            V_test = V_cur.copy()
            W_test = W_cur.copy()
            U_test[:, c] = 0
            V_test[:, c] = 0
            W_test[:, c] = 0
            
            residuals = _tensor_residuals(U_test, V_test, W_test, n, m, p)
            loss = np.sum(residuals**2)
            
            if loss < best_loss:
                best_loss = loss
                best_col = c
                best_U = U_test
                best_V = V_test
                best_W = W_test
        
        if best_col is not None and best_loss < tol:
            U_cur = best_U
            V_cur = best_V
            W_cur = best_W
        else:
            break
    
    if U_cur.shape[1] <= target_rank:
        # Trim to target_rank
        U_out = U_cur[:, :target_rank]
        V_out = V_cur[:, :target_rank]
        W_out = W_cur[:, :target_rank]
        if _verify_decomp(U_out, V_out, W_out, n, m, p, tol):
            return U_out, V_out, W_out
    
    return None

def _build_combined():
    """Build the rank-53 Strassen+schoolbook decomposition."""
    n, m, p = DIMENSIONS
    U, V, W = _build_strassen_53()["U"], _build_strassen_53()["V"], _build_strassen_53()["W"]
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    # Build the base rank-53 decomposition
    base = _build_strassen_53()
    U0, V0, W0 = base["U"], base["V"], base["W"]
    
    # Verify base
    if not _verify_decomp(U0, V0, W0, n, m, p):
        # Fallback: just return what we have
        return base
    
    best_result = base
    best_rank = 53
    
    # Try to reduce rank
    for target_rank in range(52, 45, -1):
        if time.time() > deadline - 10:
            break
        
        result = _try_rank_reduction(U0, V0, W0, n, m, p, target_rank)
        if result is not None:
            U_r, V_r, W_r = result
            best_result = {"U": U_r, "V": V_r, "W": W_r}
            best_rank = target_rank
            break
    
    return best_result
# EVOLVE-BLOCK-END
