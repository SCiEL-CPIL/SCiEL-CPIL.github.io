import time
import numpy as np

DIMENSIONS = [2, 2, 15]
TIME_LIMIT = 140  # Safety margin before 150s kill

def _build_schoolbook(n, m, p):
    """Build the schoolbook decomposition of rank n*m*p."""
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

def _build_strassen_2x2x2():
    """
    Build Strassen's rank-7 decomposition for 2x2 x 2x2 matrix multiplication.
    Returns U (4x7), V (4x7), W (4x7) following the trace convention W[k*n+i, r].
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Term 1: M1 = (A00 + A11) * (B00 + B11)
    U[0*2+0, 0] = 1  # A00
    U[1*2+1, 0] = 1  # A11
    V[0*2+0, 0] = 1  # B00
    V[1*2+1, 0] = 1  # B11
    W[0*2+0, 0] = 1  # C00
    W[1*2+1, 0] = 1  # C11
    
    # Term 2: M2 = (A10 + A11) * B00
    U[1*2+0, 1] = 1  # A10
    U[1*2+1, 1] = 1  # A11
    V[0*2+0, 1] = 1  # B00
    W[0*2+1, 1] = 1  # C10
    W[1*2+1, 1] = -1 # C11
    
    # Term 3: M3 = A00 * (B01 - B11)
    U[0*2+0, 2] = 1  # A00
    V[0*2+1, 2] = 1  # B01
    V[1*2+1, 2] = -1 # B11
    W[1*2+0, 2] = 1  # C01
    W[1*2+1, 2] = 1  # C11
    
    # Term 4: M4 = A11 * (B10 - B00)
    U[1*2+1, 3] = 1  # A11
    V[1*2+0, 3] = 1  # B10
    V[0*2+0, 3] = -1 # B00
    W[0*2+0, 3] = 1  # C00
    W[0*2+1, 3] = 1  # C10
    
    # Term 5: M5 = (A00 + A01) * B11
    U[0*2+0, 4] = 1  # A00
    U[0*2+1, 4] = 1  # A01
    V[1*2+1, 4] = 1  # B11
    W[0*2+0, 4] = -1 # C00
    W[1*2+0, 4] = 1  # C01
    
    # Term 6: M6 = (A10 - A00) * (B00 + B01)
    U[1*2+0, 5] = 1  # A10
    U[0*2+0, 5] = -1 # A00
    V[0*2+0, 5] = 1  # B00
    V[0*2+1, 5] = 1  # B01
    W[1*2+1, 5] = 1  # C11
    
    # Term 7: M7 = (A01 - A11) * (B10 + B11)
    U[0*2+1, 6] = 1  # A01
    U[1*2+1, 6] = -1 # A11
    V[1*2+0, 6] = 1  # B10
    V[1*2+1, 6] = 1  # B11
    W[0*2+0, 6] = 1  # C00
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that the decomposition is correct."""
    rank = U.shape[1]
    if U.shape != (n*m, rank):
        return False
    if V.shape != (m*p, rank):
        return False
    if W.shape != (p*n, rank):
        return False
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _find_rank10_223():
    """
    Attempt to find a rank-10 decomposition for 2x2x3 matrix multiplication.
    Returns (U, V, W) if found, else None.
    """
    n, m, p = 2, 2, 3
    rank = 10
    
    # Target tensor T[i, j, k] = 1 for all i,j,k
    # Flatten indices:
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    # We want to solve U V^T W^T = T (in tensor form)
    # Using ALS:
    # Fix V, W. For each row u of U (corresponding to i,j), we have:
    # sum_r U[i*m+j, r] * (V[j*p+0, r] * W[0*n+i, r] + V[j*p+1, r] * W[1*n+i, r] + V[j*p+2, r] * W[2*n+i, r]) = 1
    # This is a linear system for U[i*m+j, :].
    
    # Initialize U, V, W with random small values
    np.random.seed(42)
    U = np.random.randn(n * m, rank) * 0.1
    V = np.random.randn(m * p, rank) * 0.1
    W = np.random.randn(p * n, rank) * 0.1
    
    # Precompute structure for faster ALS
    # For U update:
    # For each (i, j), we need to solve for U[i*m+j, :]
    # Let X = V[j*p+0, :] * W[0*n+i, :] + V[j*p+1, :] * W[1*n+i, :] + V[j*p+2, :] * W[2*n+i, :]
    # We want U[i*m+j, :] @ X = 1.
    # This is underdetermined (rank 10, 1 equation). We can add a regularization term ||U||^2 to get a unique solution.
    # Minimize (U @ X - 1)^2 + lambda ||U||^2
    # Solution: U = (X @ X^T + lambda * I)^-1 X
    
    lambda_reg = 1e-6
    
    start_time = time.time()
    max_time = 20  # Limit search time to 20 seconds
    
    for it in range(1000):
        if time.time() - start_time > max_time:
            break
            
        # Update U
        for i in range(n):
            for j in range(m):
                idx_u = i * m + j
                # Construct X
                x = np.zeros(rank)
                for k in range(p):
                    idx_v = j * p + k
                    idx_w = k * n + i
                    x += V[idx_v, :] * W[idx_w, :]
                # Solve (X X^T + lambda I) U = X
                # Note: X is a row vector (1 x rank) effectively, but we treat it as a column for the system?
                # U is a row vector. We want U @ x = 1.
                # Let u be the column vector of U[idx_u, :].
                # x^T u = 1.
                # Minimize (x^T u - 1)^2 + lambda ||u||^2
                # Gradient: 2 x (x^T u - 1) + 2 lambda u = 0
                # (x x^T + lambda I) u = x
                A = np.outer(x, x) + lambda_reg * np.eye(rank)
                u = np.linalg.solve(A, x)
                U[idx_u, :] = u
                
        # Update V
        for j in range(m):
            for k in range(p):
                idx_v = j * p + k
                x = np.zeros(rank)
                for i in range(n):
                    idx_u = i * m + j
                    idx_w = k * n + i
                    x += U[idx_u, :] * W[idx_w, :]
                A = np.outer(x, x) + lambda_reg * np.eye(rank)
                v = np.linalg.solve(A, x)
                V[idx_v, :] = v
                
        # Update W
        for k in range(p):
            for i in range(n):
                idx_w = k * n + i
                x = np.zeros(rank)
                for j in range(m):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    x += U[idx_u, :] * V[idx_v, :]
                A = np.outer(x, x) + lambda_reg * np.eye(rank)
                w = np.linalg.solve(A, x)
                W[idx_w, :] = w
        
        # Check error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    err += (val - 1.0)**2
        if err < 1e-8:
            # Try to snap to half-integers
            U_snap = np.round(U * 2) / 2
            V_snap = np.round(V * 2) / 2
            W_snap = np.round(W * 2) / 2
            
            if _verify_decomposition(U_snap, V_snap, W_snap, n, m, p, tol=1e-9):
                return U_snap, V_snap, W_snap
            # If snap fails, try integers
            U_snap = np.round(U)
            V_snap = np.round(V)
            W_snap = np.round(W)
            if _verify_decomposition(U_snap, V_snap, W_snap, n, m, p, tol=1e-9):
                return U_snap, V_snap, W_snap
            
            # If snap fails but error is low, maybe return the float version? 
            # The prompt requires integer/half-integer. So we must snap.
            # If we can't snap, we fail this attempt.
            pass
            
    return None

def _build_rank53():
    """Build the rank-53 decomposition using 7 Strassen blocks + 1 schoolbook."""
    n, m, p = DIMENSIONS
    block_sizes = [2, 2, 2, 2, 2, 2, 2, 1]
    num_blocks = len(block_sizes)
    
    U_str, V_str, W_str = _build_strassen_2x2x2()
    U_sb, V_sb, W_sb = _build_schoolbook(2, 2, 1)
    
    total_rank = 7 * 7 + 4  # 53
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col_offset = 0
    for b in range(num_blocks):
        block_size = block_sizes[b]
        start_k = sum(block_sizes[:b])
        
        if block_size == 2:
            U[:, col_offset:col_offset+7] = U_str
            for j in range(m):
                for k_local in range(2):
                    k_global = start_k + k_local
                    V[j * p + k_global, col_offset:col_offset+7] = V_str[j * 2 + k_local, :]
            for k_local in range(2):
                k_global = start_k + k_local
                for i in range(n):
                    W[k_global * n + i, col_offset:col_offset+7] = W_str[k_local * 2 + i, :]
            col_offset += 7
        else:
            U[:, col_offset:col_offset+4] = U_sb
            for j in range(m):
                V[j * p + start_k, col_offset:col_offset+4] = V_sb[j * 1 + 0, :]
            for i in range(n):
                W[start_k * n + i, col_offset:col_offset+4] = W_sb[0 * 2 + i, :]
            col_offset += 4
    
    return U, V, W

def construct():
    """
    Construct a rank-minimal exact tensor decomposition
    for multiplying a 2x2 matrix by a 2x15 matrix.
    """
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Baseline: rank 53
    U, V, W = _build_rank53()
    best_rank = 53
    best_U, best_V, best_W = U, V, W
    
    # Verify baseline
    if not _verify_decomposition(U, V, W, n, m, p):
        U, V, W = _build_schoolbook(n, m, p)
        best_rank = 60
        best_U, best_V, best_W = U, V, W
    
    # Attempt to find rank-52 solution using rank-10 2x2x3 block
    if time.time() - start_time < TIME_LIMIT - 30:
        print("Attempting rank-10 2x2x3 search...")
        U_223, V_223, W_223 = _find_rank10_223()
        if U_223 is not None:
            # Construct rank-52 decomposition
            # 6 blocks of 2x2x2 (Strassen) + 1 block of 2x2x3 (Rank 10)
            block_sizes = [2, 2, 2, 2, 2, 2, 3]
            num_blocks = len(block_sizes)
            
            U_str, V_str, W_str = _build_strassen_2x2x2()
            
            total_rank = 6 * 7 + 10  # 52
            
            U_new = np.zeros((n * m, total_rank))
            V_new = np.zeros((m * p, total_rank))
            W_new = np.zeros((p * n, total_rank))
            
            col_offset = 0
            for b in range(num_blocks):
                block_size = block_sizes[b]
                start_k = sum(block_sizes[:b])
                
                if block_size == 2:
                    U_new[:, col_offset:col_offset+7] = U_str
                    for j in range(m):
                        for k_local in range(2):
                            k_global = start_k + k_local
                            V_new[j * p + k_global, col_offset:col_offset+7] = V_str[j * 2 + k_local, :]
                    for k_local in range(2):
                        k_global = start_k + k_local
                        for i in range(n):
                            W_new[k_global * n + i, col_offset:col_offset+7] = W_str[k_local * 2 + i, :]
                    col_offset += 7
                else: # block_size == 3
                    U_new[:, col_offset:col_offset+10] = U_223
                    for j in range(m):
                        for k_local in range(3):
                            k_global = start_k + k_local
                            V_new[j * p + k_global, col_offset:col_offset+10] = V_223[j * 3 + k_local, :]
                    for k_local in range(3):
                        k_global = start_k + k_local
                        for i in range(n):
                            W_new[k_global * n + i, col_offset:col_offset+10] = W_223[k_local * 2 + i, :]
                    col_offset += 10
            
            if _verify_decomposition(U_new, V_new, W_new, n, m, p):
                best_rank = 52
                best_U, best_V, best_W = U_new, V_new, W_new
                print("Found rank 52 decomposition!")
            else:
                print("Rank 52 verification failed.")
        else:
            print("Rank 10 2x2x3 search failed.")

    # Final time check
    elapsed = time.time() - start_time
    if elapsed > TIME_LIMIT - 5:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
