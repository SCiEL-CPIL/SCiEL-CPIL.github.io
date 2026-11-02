"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = 150.0
_SAFETY_MARGIN = 5.0

def _remaining_time():
    return _DEADLINE - _SAFETY_MARGIN - (time.time() - _START_TIME)

def _schoolbook():
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
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, tol=1e-6):
    """Verify that the decomposition is correct."""
    n, m, p = DIMENSIONS
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    # Check all valid entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.sum(U_arr[a, :] * V_arr[b, :] * W_arr[c, :])
                if abs(val - 1.0) > tol:
                    return False
    
    # Check all invalid entries
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                # Decode indices
                i1, j1 = divmod(a, m)
                j2, k = divmod(b, p)
                k2, i2 = divmod(c, n)
                if i1 == i2 and j1 == j2 and k == k2:
                    continue  # This is a valid entry
                val = np.sum(U_arr[a, :] * V_arr[b, :] * W_arr[c, :])
                if abs(val) > tol:
                    return False
    return True

def _strassen_2x2x2(offset_a, offset_b, offset_c, n, m, p):
    """
    Return (U, V, W) for a 2x2x2 block at given offsets using Strassen.
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = 1
    
    # M3 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[2, 3] = 1; W[3, 3] = 1
    
    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[1, 4] = 1; W[3, 4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[2, 5] = 1; W[3, 5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1; W[3, 6] = 1
    
    U_full = np.zeros((n*m, 7))
    V_full = np.zeros((m*p, 7))
    W_full = np.zeros((p*n, 7))
    
    a_map = [offset_a, offset_a + 1, offset_a + m, offset_a + m + 1]
    b_map = [offset_b, offset_b + 1, offset_b + p, offset_b + p + 1]
    c_map = [offset_c, offset_c + 1, offset_c + n, offset_c + n + 1]
    
    for r in range(7):
        for local_idx in range(4):
            if U[local_idx, r] != 0:
                global_idx = a_map[local_idx]
                if 0 <= global_idx < n*m:
                    U_full[global_idx, r] = U[local_idx, r]
            if V[local_idx, r] != 0:
                global_idx = b_map[local_idx]
                if 0 <= global_idx < m*p:
                    V_full[global_idx, r] = V[local_idx, r]
            if W[local_idx, r] != 0:
                global_idx = c_map[local_idx]
                if 0 <= global_idx < p*n:
                    W_full[global_idx, r] = W[local_idx, r]
    
    return U_full, V_full, W_full

def _dot_1x2x2(offset_a, offset_b, offset_c, n, m, p):
    """
    Return (U, V, W) for a 1x2x2 block at given offsets.
    """
    U = np.zeros((2, 2))
    V = np.zeros((4, 2))
    W = np.zeros((2, 2))
    
    U[0, 0] = 1
    V[0, 0] = 1; V[1, 0] = 1
    W[0, 0] = 1; W[1, 0] = 1
    
    U[1, 1] = 1
    V[2, 1] = 1; V[3, 1] = 1
    W[0, 1] = 1; W[1, 1] = 1
    
    U_full = np.zeros((n*m, 2))
    V_full = np.zeros((m*p, 2))
    W_full = np.zeros((p*n, 2))
    
    a_map = [offset_a, offset_a + 1]
    b_map = [offset_b, offset_b + 1, offset_b + p, offset_b + p + 1]
    c_map = [offset_c, offset_c + 1]
    
    for r in range(2):
        for local_idx in range(2):
            if U[local_idx, r] != 0:
                global_idx = a_map[local_idx]
                if 0 <= global_idx < n*m:
                    U_full[global_idx, r] = U[local_idx, r]
        for local_idx in range(4):
            if V[local_idx, r] != 0:
                global_idx = b_map[local_idx]
                if 0 <= global_idx < m*p:
                    V_full[global_idx, r] = V[local_idx, r]
        for local_idx in range(2):
            if W[local_idx, r] != 0:
                global_idx = c_map[local_idx]
                if 0 <= global_idx < p*n:
                    W_full[global_idx, r] = W[local_idx, r]
    
    return U_full, V_full, W_full

def _try_strassen_blocks():
    """
    Construct a block-based Strassen decomposition for 3x4x8.
    Total rank: 2*14 + 2*4 = 28 + 8 = 36
    """
    n, m, p = DIMENSIONS  # 3, 4, 8
    
    U_total = np.zeros((n*m, 0))
    V_total = np.zeros((m*p, 0))
    W_total = np.zeros((p*n, 0))
    
    def _add_block(U_block, V_block, W_block):
        nonlocal U_total, V_total, W_total
        U_total = np.hstack([U_total, U_block])
        V_total = np.hstack([V_total, V_block])
        W_total = np.hstack([W_total, W_block])
    
    # C[0:2, 0:4] = A[0:2, :] * B[:, 0:4]
    U_b, V_b, W_b = _strassen_2x2x2(0, 0, 0, n, m, p)
    _add_block(U_b, V_b, W_b)
    
    U_b, V_b, W_b = _strassen_2x2x2(2, 16, 0, n, m, p)
    _add_block(U_b, V_b, W_b)
    
    # C[0:2, 4:8] = A[0:2, :] * B[:, 4:8]
    U_b, V_b, W_b = _strassen_2x2x2(0, 4, 12, n, m, p)
    _add_block(U_b, V_b, W_b)
    
    U_b, V_b, W_b = _strassen_2x2x2(2, 20, 12, n, m, p)
    _add_block(U_b, V_b, W_b)
    
    # C[2:3, 0:4] = A[2:3, :] * B[:, 0:4]
    U_b, V_b, W_b = _dot_1x2x2(8, 0, 2, n, m, p)
    _add_block(U_b, V_b, W_b)
    
    U_b, V_b, W_b = _dot_1x2x2(10, 16, 2, n, m, p)
    _add_block(U_b, V_b, W_b)
    
    # C[2:3, 4:8] = A[2:3, :] * B[:, 4:8]
    U_b, V_b, W_b = _dot_1x2x2(8, 4, 14, n, m, p)
    _add_block(U_b, V_b, W_b)
    
    U_b, V_b, W_b = _dot_1x2x2(10, 20, 14, n, m, p)
    _add_block(U_b, V_b, W_b)
    
    U_list = U_total.tolist()
    V_list = V_total.tolist()
    W_list = W_total.tolist()
    
    result = {"U": U_list, "V": V_list, "W": W_list}
    
    if _verify(U_list, V_list, W_list):
        return result, U_total.shape[1]
    else:
        return _schoolbook(), n * m * p

def _try_optimized_blocks():
    """
    Try to reduce rank by sharing Strassen factors across blocks.
    """
    n, m, p = DIMENSIONS  # 3, 4, 8
    
    U_total = np.zeros((n*m, 0))
    V_total = np.zeros((m*p, 0))
    W_total = np.zeros((p*n, 0))
    
    def _add_block(U_block, V_block, W_block):
        nonlocal U_total, V_total, W_total
        U_total = np.hstack([U_total, U_block])
        V_total = np.hstack([V_total, V_block])
        W_total = np.hstack([W_total, W_block])
    
    # C[0:2, 0:4] = A[0:2, :] * B[:, 0:4]
    U_b, V_b, W_b = _strassen_2x2x2(0, 0, 0, n, m, p)
    _add_block(U_b, V_b, W_b)
    
    U_b, V_b, W_b = _strassen_2x2x2(2, 16, 0, n, m, p)
    _add_block(U_b, V_b, W_b)
    
    # C[0:2, 4:8] = A[0:2, :] * B[:, 4:8]
    U_b, V_b, W_b = _strassen_2x2x2(0, 4, 12, n, m, p)
    _add_block(U_b, V_b, W_b)
    
    U_b, V_b, W_b = _strassen_2x2x2(2, 20, 12, n, m, p)
    _add_block(U_b, V_b, W_b)
    
    # C[2:3, 0:4] = A[2:3, :] * B[:, 0:4]
    U_b, V_b, W_b = _dot_1x2x2(8, 0, 2, n, m, p)
    _add_block(U_b, V_b, W_b)
    
    U_b, V_b, W_b = _dot_1x2x2(10, 16, 2, n, m, p)
    _add_block(U_b, V_b, W_b)
    
    # C[2:3, 4:8] = A[2:3, :] * B[:, 4:8]
    U_b, V_b, W_b = _dot_1x2x2(8, 4, 14, n, m, p)
    _add_block(U_b, V_b, W_b)
    
    U_b, V_b, W_b = _dot_1x2x2(10, 20, 14, n, m, p)
    _add_block(U_b, V_b, W_b)
    
    U_list = U_total.tolist()
    V_list = V_total.tolist()
    W_list = W_total.tolist()
    
    result = {"U": U_list, "V": V_list, "W": W_list}
    
    if _verify(U_list, V_list, W_list):
        return result, U_total.shape[1]
    else:
        return _schoolbook(), n * m * p

def _try_sdp_relaxation():
    """
    Try an SDP-based approach to find a lower rank decomposition.
    This is a heuristic that uses the SDP relaxation to guide the search.
    """
    n, m, p = DIMENSIONS
    
    # Build the tensor T
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Try various rank values
    for rank in range(20, 40):
        if _remaining_time() < 10:
            break
        
        # Initialize with random factors
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
        
        # Normalize
        U = U / np.linalg.norm(U, axis=1, keepdims=True)
        V = V / np.linalg.norm(V, axis=1, keepdims=True)
        W = W / np.linalg.norm(W, axis=1, keepdims=True)
        
        # Iteratively refine
        for iteration in range(100):
            if _remaining_time() < 5:
                break
            
            # Update U
            for a in range(n*m):
                for b in range(m*p):
                    for c in range(p*n):
                        if T[a, b, c] != 0:
                            U[a, :] += T[a, b, c] * (V[b, :] * W[c, :])
            
            # Update V
            for b in range(m*p):
                for a in range(n*m):
                    for c in range(p*n):
                        if T[a, b, c] != 0:
                            V[b, :] += T[a, b, c] * (U[a, :] * W[c, :])
            
            # Update W
            for c in range(p*n):
                for a in range(n*m):
                    for b in range(m*p):
                        if T[a, b, c] != 0:
                            W[c, :] += T[a, b, c] * (U[a, :] * V[b, :])
            
            # Check convergence
            if iteration % 10 == 0:
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                if _verify(U_list, V_list, W_list):
                    return {"U": U_list, "V": V_list, "W": W_list}, rank
    
    return None, None

def construct():
    n, m, p = DIMENSIONS
    
    # Try the block Strassen decomposition (rank 36)
    try:
        result, rank = _try_strassen_blocks()
        if rank < n * m * p:
            return result
    except Exception:
        pass
    
    # Try the optimized blocks
    try:
        result, rank = _try_optimized_blocks()
        if rank < n * m * p:
            return result
    except Exception:
        pass
    
    # Try SDP relaxation
    try:
        if _remaining_time() > 30:
            result, rank = _try_sdp_relaxation()
            if result is not None and rank < n * m * p:
                return result
    except Exception:
        pass
    
    # Fallback to schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
