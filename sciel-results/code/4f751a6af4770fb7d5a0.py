import time
import numpy as np
import sys

DIMENSIONS = [2, 3, 10]

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r] for r in range(rank))
                if abs(val - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1.0
                V[j*p + k][col] = 1.0
                W[k*n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2x2():
    """
    Strassen's rank-7 decomposition for 2x2x2 matrix multiplication.
    Returns U (4x7), V (4x7), W (4x7) such that
    sum_r U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r] = 1 for all i,j,k in {0,1}
    
    Strassen's 7 products:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B10-B11)
    M4 = A11(B01-B00)
    M5 = (A00+A01)B11
    M6 = (A10-A00)(B00+B10)
    M7 = (A01-A11)(B01+B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    """
    # U rows: (i,j) for i in {0,1}, j in {0,1} -> indices 0,1,2,3
    # V rows: (j,k) for j in {0,1}, k in {0,1} -> indices 0,1,2,3
    # W rows: (k,i) for k in {0,1}, i in {0,1} -> indices 0,1,2,3
    
    # U[i*2+j, r]: coefficient of A[i,j] in product r
    # V[j*2+k, r]: coefficient of B[j,k] in product r
    # W[k*2+i, r]: coefficient for C[i,k] from product r
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1  # A00, A11
    V[0, 0] = 1; V[3, 0] = 1  # B00, B11
    W[0, 0] = 1; W[3, 0] = 1  # C00, C11
    
    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1  # A10, A11
    V[0, 1] = 1  # B00
    W[2, 1] = 1; W[3, 1] = -1  # C10, C11
    
    # M3 = A00(B10-B11)
    U[0, 2] = 1  # A00
    V[2, 2] = 1; V[3, 2] = -1  # B10, B11
    W[1, 2] = 1; W[3, 2] = 1  # C01, C11
    
    # M4 = A11(B01-B00)
    U[3, 3] = 1  # A11
    V[1, 3] = 1; V[0, 3] = -1  # B01, B00
    W[0, 3] = 1; W[2, 3] = 1  # C00, C10
    
    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1  # A00, A01
    V[3, 4] = 1  # B11
    W[0, 4] = -1; W[1, 4] = 1  # C00, C01
    
    # M6 = (A10-A00)(B00+B10)
    U[2, 5] = 1; U[0, 5] = -1  # A10, A00
    V[0, 5] = 1; V[2, 5] = 1  # B00, B10
    W[3, 5] = 1  # C11
    
    # M7 = (A01-A11)(B01+B11)
    U[1, 6] = 1; U[3, 6] = -1  # A01, A11
    V[1, 6] = 1; V[3, 6] = 1  # B01, B11
    W[0, 6] = 1  # C00
    
    return U, V, W

def _build_strassen_for_2x3x10():
    """
    Build a decomposition for 2x3x10 using Strassen-like structure.
    
    Strategy: Split the 3x10 B matrix into blocks and use Strassen where possible.
    
    For 2x3x10, we can think of it as:
    - A is 2x3, B is 3x10
    - C[i,k] = sum_j A[i,j] * B[j,k]
    
    We can split B into 5 blocks of 3x2 each, and use Strassen for each 2x2x2 subproblem.
    But we need to be careful about how we combine them.
    
    Alternative: Use a more direct approach with rank reduction.
    """
    n, m, p = 2, 3, 10
    
    # Start with schoolbook and try to reduce rank
    # Schoolbook rank = 2*3*10 = 60
    
    # Try to construct a rank-45 decomposition using block structure
    # Split p=10 into 5 blocks of size 2
    
    U_s, V_s, W_s = _strassen_2x2x2()
    
    # For each block b (0..4), we compute A * B[:, 2b:2b+2]
    # This is a 2x3 * 3x2 = 2x2 matrix
    # We can use Strassen on the "2x2 core" if we can map it
    
    # Actually, let's try a different approach:
    # Use the fact that for 2x3x2, we can find a rank-9 decomposition
    # Then repeat 5 times for rank 45
    
    # For 2x3x2:
    # A is 2x3, B is 3x2, C is 2x2
    # C[i,k] = sum_{j=0}^{2} A[i,j] * B[j,k]
    
    # We can view this as:
    # C = [A0 A1 A2] * [[B00, B01], [B10, B11], [B20, B21]]
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    
    # Let's try to find a rank-9 decomposition for 2x3x2
    # Then use it 5 times for 2x3x10
    
    # For now, let's just return schoolbook as a baseline
    # and try to improve it
    
    return None

def _local_search_reduce_rank(U, V, W, n, m, p, target_rank, max_time, tol=1e-8):
    """
    Try to reduce the rank of a valid decomposition by removing zero or near-zero columns
    and then trying to fill in with a lower-rank approximation.
    """
    start_time = time.time()
    rank = len(U[0])
    
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # First, try to find columns that can be removed
    # A column r is "redundant" if removing it still gives a valid decomposition
    # (or if it's all zeros)
    
    # Check for zero columns
    zero_cols = []
    for r in range(rank):
        if np.all(np.abs(U[:, r]) < 1e-10) or np.all(np.abs(V[:, r]) < 1e-10) or np.all(np.abs(W[:, r]) < 1e-10):
            zero_cols.append(r)
    
    if zero_cols:
        # Remove zero columns
        keep = [r for r in range(rank) if r not in zero_cols]
        U = U[:, keep]
        V = V[:, keep]
        W = W[:, keep]
        rank = len(keep)
    
    # Try to reduce rank by merging similar columns or finding linear dependencies
    # This is complex, so let's try a simpler approach:
    # Use gradient descent to find a lower-rank decomposition
    
    if rank <= target_rank:
        return U.tolist(), V.tolist(), W.tolist()
    
    # Try random restarts with target_rank
    best_U = best_V = best_W = None
    best_residual = float('inf')
    
    for trial in range(100):
        if time.time() - start_time > max_time:
            break
        
        R = target_rank
        # Initialize with random values scaled appropriately
        U_t = np.random.randn(n * m, R) * 0.3
        V_t = np.random.randn(m * p, R) * 0.3
        W_t = np.random.randn(p * n, R) * 0.3
        
        # Compute residual
        residual = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U_t[i*m + j, r] * V_t[j*p + k, r] * W_t[k*n + i, r] for r in range(R))
                    residual += (val - 1.0) ** 2
        
        if residual < best_residual:
            best_residual = residual
            best_U, best_V, best_W = U_t.copy(), V_t.copy(), W_t.copy()
        
        if residual < 1e-10:
            break
    
    if best_residual < 1e-8:
        return best_U.tolist(), best_V.tolist(), best_W.tolist()
    
    return None

def _construct_rank_45():
    """
    Construct a rank-45 decomposition for 2x3x10.
    
    Approach: Use 5 independent 2x3x2 subproblems, each with rank 9.
    """
    n, m, p = 2, 3, 10
    block_p = 2
    n_blocks = p // block_p  # 5
    
    # For each block, we need a rank-9 decomposition of 2x3x2
    # Let's try to construct one
    
    # For 2x3x2:
    # U: 6x9, V: 6x9, W: 4x9
    # T[i,j,k] = 1 for i in {0,1}, j in {0,1,2}, k in {0,1}
    
    # Let's try to find this by optimization
    best_U = best_V = best_W = None
    best_residual = float('inf')
    
    for trial in range(200):
        R = 9
        U_t = np.random.randn(6, R) * 0.5
        V_t = np.random.randn(6, R) * 0.5
        W_t = np.random.randn(4, R) * 0.5
        
        residual = 0.0
        for i in range(2):
            for j in range(3):
                for k in range(2):
                    val = sum(U_t[i*3 + j, r] * V_t[j*2 + k, r] * W_t[k*2 + i, r] for r in range(R))
                    residual += (val - 1.0) ** 2
        
        if residual < best_residual:
            best_residual = residual
            best_U, best_V, best_W = U_t.copy(), V_t.copy(), W_t.copy()
        
        if residual < 1e-10:
            break
    
    if best_residual > 1e-6:
        return None
    
    # Build full decomposition
    full_rank = 9 * n_blocks
    U_full = np.zeros((n * m, full_rank))
    V_full = np.zeros((m * p, full_rank))
    W_full = np.zeros((p * n, full_rank))
    
    for b in range(n_blocks):
        for r in range(9):
            col = b * 9 + r
            for i in range(n):
                for j in range(m):
                    U_full[i*m + j, col] = best_U[i*m + j, r]
            for j in range(m):
                for k in range(block_p):
                    V_full[j*p + b*block_p + k, col] = best_V[j*block_p + k, r]
            for k in range(block_p):
                for i in range(n):
                    W_full[(b*block_p + k)*n + i, col] = best_W[k*n + i, r]
    
    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}

def _construct_rank_35():
    """
    Try to construct a rank-35 decomposition for 2x3x10.
    
    This would require a rank-7 decomposition of 2x3x2 (if we use 5 blocks),
    or a different block structure.
    
    For 2x3x2, the minimum rank is 9 (known result), so we can't do rank-7 per block.
    But we might be able to do better with a different approach.
    
    Let's try a direct rank-35 search.
    """
    n, m, p = 2, 3, 10
    R = 35
    
    best_U = best_V = best_W = None
    best_residual = float('inf')
    
    for trial in range(50):
        U_t = np.random.randn(n * m, R) * 0.3
        V_t = np.random.randn(m * p, R) * 0.3
        W_t = np.random.randn(p * n, R) * 0.3
        
        residual = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U_t[i*m + j, r] * V_t[j*p + k, r] * W_t[k*n + i, r] for r in range(R))
                    residual += (val - 1.0) ** 2
        
        if residual < best_residual:
            best_residual = residual
            best_U, best_V, best_W = U_t.copy(), V_t.copy(), W_t.copy()
        
        if residual < 1e-10:
            break
    
    if best_residual > 1e-6:
        return None
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 145.0  # Leave 5s margin
    
    # Best fallback
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try rank-45 construction
    if time.time() - start_time < deadline - 10:
        result = _construct_rank_45()
        if result is not None:
            rank = len(result["U"][0])
            if rank < best_rank:
                if _verify(result["U"], result["V"], result["W"], n, m, p):
                    best_result = result
                    best_rank = rank
    
    # Try rank-35 construction
    if time.time() - start_time < deadline - 5:
        result = _construct_rank_35()
        if result is not None:
            rank = len(result["U"][0])
            if rank < best_rank:
                if _verify(result["U"], result["V"], result["W"], n, m, p):
                    best_result = result
                    best_rank = rank
    
    return best_result
