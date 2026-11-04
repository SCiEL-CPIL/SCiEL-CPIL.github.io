"""Matrix multiplication 2,4,6 rank minimization via Strassen block decomposition + ALS rank reduction."""
import numpy as np
import time

DIMENSIONS = [2, 4, 6]
START_TIME = time.time()
TIME_LIMIT = 140  # safety margin before 150s kill

def _elapsed():
    return time.time() - START_TIME

def _deadline_reached():
    return _elapsed() > TIME_LIMIT

def _build_target_tensor(n, m, p):
    """Build the target multiplication tensor T[a,b,d] where a=i*m+j, b=j*p+k, d=k*n+i."""
    target = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i * m + j, j * p + k, k * n + i] = 1.0
    return target

def _verify_decomposition(n, m, p, U, V, W, tol=1e-8):
    """Verify that the decomposition exactly reconstructs the target tensor."""
    target = _build_target_tensor(n, m, p)
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    rank = U_arr.shape[1]
    
    # Reconstruct: T[a,b,d] = sum_c U[a,c] * V[b,c] * W[d,c]
    recon = np.einsum('ac,bc,dc->abd', U_arr, V_arr, W_arr)
    error = np.linalg.norm(target - recon)
    return error < tol, error

def _strassen_decomposition(n, m, p):
    """
    Build a rank-42 decomposition using 6 Strassen 2x2x2 blocks.
    
    For 2x4x6:
    - A is 2x4, B is 4x6
    - Split A into columns: A = [A1, A2] where A1, A2 are 2x2
    - Split B into rows: B = [B1; B2] where B1, B2 are 2x6
    - C = A1*B1 + A2*B2
    
    Each 2x2x2 multiplication can be done with rank 7 (Strassen).
    We have two such multiplications: A1*B1 and A2*B2.
    But B1 and B2 are 2x6, not 2x2. So we need to split further.
    
    Actually, let's think differently. The tensor is:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    For n=2, m=4, p=6:
    C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k]
    
    Group terms:
    C[i,k] = (A[i,0]*B[0,k] + A[i,1]*B[1,k]) + (A[i,2]*B[2,k] + A[i,3]*B[3,k])
    
    First group: 2x2 matrix (A[:,0:2]) times 2x6 matrix (B[0:2,:])
    Second group: 2x2 matrix (A[:,2:4]) times 2x6 matrix (B[2:4,:])
    
    For a 2x2 times 2x6 multiplication:
    C[i,k] = sum_{j=0}^{1} A[i,j] * B[j,k]
    This is a 2x2x6 tensor. We can split the 6 columns into three 2-column blocks:
    C[:, 0:2] = A * B[:, 0:2]  (2x2 x 2x2, rank 7 via Strassen)
    C[:, 2:4] = A * B[:, 2:4]  (2x2 x 2x2, rank 7 via Strassen)
    C[:, 4:6] = A * B[:, 4:6]  (2x2 x 2x2, rank 7 via Strassen)
    
    So for each 2x2x6 block, we need 3 * 7 = 21 rank.
    Two such blocks: 2 * 21 = 42 rank total.
    """
    # Strassen 2x2x2 decomposition
    # For 2x2 matrices: C = A*B where A,B,C are 2x2
    # Strassen uses 7 products:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # For 2x2 x 2x2, the tensor indices are:
    # a = i*2 + j (i in 0,1; j in 0,1) -> A[i,j]
    # b = j*2 + k (j in 0,1; k in 0,1) -> B[j,k]
    # d = k*2 + i (k in 0,1; i in 0,1) -> C[i,k]
    
    # Strassen factors for 2x2x2 (rank 7):
    # U[a, c], V[b, c], W[d, c] for c in 0..6
    strassen_U = np.array([
        [1, 0, 1, 0, 1, -1, 0],   # A00
        [0, 1, 0, 0, 0, 1, 0],    # A10
        [1, 0, 0, 0, 1, 0, 1],    # A01
        [1, 1, 0, 1, 0, 0, -1],   # A11
    ], dtype=np.float64)
    
    strassen_V = np.array([
        [1, 1, 1, -1, 0, 1, 1],   # B00
        [1, 0, -1, -1, 1, 1, 1],  # B10
        [0, 0, 1, 0, 1, 1, 0],    # B01
        [1, 0, 0, 0, 1, 0, 1],    # B11
    ], dtype=np.float64)
    
    strassen_W = np.array([
        [1, 0, 0, 1, -1, 0, 1],   # C00
        [0, 0, 1, 0, 1, 0, 0],    # C01
        [0, 1, 0, 1, 0, 0, 0],    # C10
        [1, -1, 1, 0, 0, 1, 0],   # C11
    ], dtype=np.float64)
    
    # Verify Strassen decomposition
    # T[a,b,d] = sum_c U[a,c] V[b,c] W[d,c]
    recon = np.einsum('ac,bc,dc->abd', strassen_U, strassen_V, strassen_W)
    # Target: T[i*2+j, j*2+k, k*2+i] = 1
    target_222 = np.zeros((4, 4, 4), dtype=np.float64)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                target_222[i*2+j, j*2+k, k*2+i] = 1.0
    err = np.linalg.norm(target_222 - recon)
    if err > 1e-10:
        # Fallback to schoolbook for 2x2x2
        strassen_U = np.zeros((4, 7), dtype=np.float64)
        strassen_V = np.zeros((4, 7), dtype=np.float64)
        strassen_W = np.zeros((4, 7), dtype=np.float64)
        col = 0
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    strassen_U[i*2+j, col] = 1
                    strassen_V[j*2+k, col] = 1
                    strassen_W[k*2+i, col] = 1
                    col += 1
        # Pad to 7
        strassen_U[:, 4:] = 0
        strassen_V[:, 4:] = 0
        strassen_W[:, 4:] = 0
    
    # Now build the full 2x4x6 decomposition
    # We have 6 blocks: 2 (for the two A-halves) x 3 (for the three B-column-pairs)
    # Total rank: 6 * 7 = 42
    
    rank = 42
    U = np.zeros((n * m, rank), dtype=np.float64)  # (8, 42)
    V = np.zeros((m * p, rank), dtype=np.float64)  # (24, 42)
    W = np.zeros((p * n, rank), dtype=np.float64)  # (12, 42)
    
    block_idx = 0
    for a_half in range(2):  # A[:, 0:2] and A[:, 2:4]
        for b_col_pair in range(3):  # B[0:2, 0:2], B[0:2, 2:4], B[0:2, 4:6] and B[2:4, ...]
            # Wait, I need to reconsider the blocking.
            # 
            # A is 2x4. Split into A[:, 0:2] and A[:, 2:4], each 2x2.
            # B is 4x6. Split into B[0:2, :] and B[2:4, :], each 2x6.
            # 
            # C = A[:, 0:2] * B[0:2, :] + A[:, 2:4] * B[2:4, :]
            #
            # For A[:, 0:2] * B[0:2, :]:
            #   This is 2x2 * 2x6. Split B[0:2, :] into three 2x2 blocks.
            #   Block 0: A[:, 0:2] * B[0:2, 0:2]
            #   Block 1: A[:, 0:2] * B[0:2, 2:4]
            #   Block 2: A[:, 0:2] * B[0:2, 4:6]
            #
            # For A[:, 2:4] * B[2:4, :]:
            #   Similarly three blocks.
            #
            # So 6 blocks total, each 2x2x2.
            
            # For block (a_half, b_col_pair):
            # A indices: i in 0,1; j in 2*a_half, 2*a_half+1
            # B indices: j in 2*a_half, 2*a_half+1; k in 2*b_col_pair, 2*b_col_pair+1
            # C indices: i in 0,1; k in 2*b_col_pair, 2*b_col_pair+1
            
            col_offset = block_idx * 7
            
            for i in range(2):
                for j_local in range(2):
                    j = 2 * a_half + j_local
                    a_idx = i * m + j
                    for c in range(7):
                        U[a_idx, col_offset + c] = strassen_U[i * 2 + j_local, c]
            
            for j_local in range(2):
                j = 2 * a_half + j_local
                for k_local in range(2):
                    k = 2 * b_col_pair + k_local
                    b_idx = j * p + k
                    for c in range(7):
                        V[b_idx, col_offset + c] = strassen_V[j_local * 2 + k_local, c]
            
            for i in range(2):
                for k_local in range(2):
                    k = 2 * b_col_pair + k_local
                    d_idx = k * n + i
                    for c in range(7):
                        W[d_idx, col_offset + c] = strassen_W[k_local * 2 + i, c]
            
            block_idx += 1
    
    return U.tolist(), V.tolist(), W.tolist()

def _snap_to_half_integers(arr, tol=1e-6):
    """Snap values close to half-integers to exact half-integers."""
    result = np.array(arr, dtype=np.float64)
    doubled = result * 2
    rounded = np.round(doubled)
    close = np.abs(doubled - rounded) < tol
    result[close] = rounded[close] / 2
    return result

def _try_reduce_rank(n, m, p, U, V, W, target, max_iter=30):
    """
    Try to reduce the rank by setting some columns to zero and re-solving.
    This is a heuristic: for each rank level below current, try to see if
    a valid decomposition exists.
    """
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    current_rank = U_arr.shape[1]
    
    # Try reducing rank by 1, 2, 3...
    for new_rank in range(current_rank - 1, max(current_rank - 6, 0), -1):
        if _deadline_reached():
            break
        
        # Try to find a valid decomposition of rank new_rank
        # Use random restarts with ALS refinement
        best_valid = False
        best_U, best_V, best_W = None, None, None
        
        for restart in range(3):
            if _deadline_reached():
                break
            
            # Initialize with random matrices
            rng = np.random.RandomState(42 + restart)
            U_try = rng.randn(n * m, new_rank) * 0.1
            V_try = rng.randn(m * p, new_rank) * 0.1
            W_try = rng.randn(p * n, new_rank) * 0.1
            
            # ALS refinement
            for it in range(max_iter):
                if _deadline_reached():
                    break
                
                # Update U
                M = V_try[:, None, :] * W_try[None, :, :]  # (m*p, p*n, rank)
                P_flat = M.reshape(new_rank, -1)  # (rank, m*p*p*n)
                P_T = P_flat.T  # (m*p*p*n, rank)
                P_pinv = np.linalg.pinv(P_T)
                
                new_U = np.zeros((n * m, new_rank))
                for a in range(n * m):
                    y = target[a, :, :].reshape(-1)
                    new_U[a, :] = y @ P_pinv
                U_try = new_U
                
                # Update V
                Q_flat = (U_try[:, None, :] * W_try[None, :, :]).reshape(new_rank, -1)
                Q_T = Q_flat.T
                Q_pinv = np.linalg.pinv(Q_T)
                
                new_V = np.zeros((m * p, new_rank))
                for b in range(m * p):
                    y = target[:, b, :].reshape(-1)
                    new_V[b, :] = y @ Q_pinv
                V_try = new_V
                
                # Update W
                R_flat = (U_try[:, None, :] * V_try[None, :, :]).reshape(new_rank, -1)
                R_T = R_flat.T
                R_pinv = np.linalg.pinv(R_T)
                
                new_W = np.zeros((p * n, new_rank))
                for d in range(p * n):
                    y = target[:, :, d].reshape(-1)
                    new_W[d, :] = y @ R_pinv
                W_try = new_W
            
            # Check validity
            recon = np.einsum('ac,bc,dc->abd', U_try, V_try, W_try)
            err = np.linalg.norm(target - recon)
            
            if err < 1e-8:
                best_valid = True
                best_U, best_V, best_W = U_try, V_try, W_try
                break
        
        if best_valid:
            # Snap to half-integers
            U_snapped = _snap_to_half_integers(best_U)
            V_snapped = _snap_to_half_integers(best_V)
            W_snapped = _snap_to_half_integers(best_W)
            
            recon = np.einsum('ac,bc,dc->abd', U_snapped, V_snapped, W_snapped)
            err = np.linalg.norm(target - recon)
            
            if err < 1e-8:
                return U_snapped.tolist(), V_snapped.tolist(), W_snapped.tolist()
            else:
                return best_U.tolist(), best_V.tolist(), best_W.tolist()
    
    # No reduction found, return original
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    
    # Start with Strassen-based rank-42 decomposition
    U, V, W = _strassen_decomposition(n, m, p)
    
    # Verify
    target = _build_target_tensor(n, m, p)
    valid, err = _verify_decomposition(n, m, p, U, V, W)
    
    if not valid:
        # Fallback to schoolbook
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
    
    # Try to reduce rank
    if not _deadline_reached():
        U, V, W = _try_reduce_rank(n, m, p, U, V, W, target)
    
    return {"U": U, "V": V, "W": W}
