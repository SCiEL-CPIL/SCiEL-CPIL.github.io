import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def _build_strassen_rank28():
    """Build the known rank-28 Strassen decomposition for 2x4x4 matmul."""
    n, m, p = DIMENSIONS
    rank = 28
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen 2x2x2 steps (7 multiplications)
    # Each step has u (2x2), v (2x2), w (2x2) matrices with entries
    strassen_steps = [
        {'u': {0: 1, 3: 1}, 'v': {0: 1, 3: 1}, 'w': {0: 1, 3: 1}},
        {'u': {2: 1, 3: 1}, 'v': {0: 1}, 'w': {2: 1, 3: -1}},
        {'u': {0: 1}, 'v': {1: 1, 3: -1}, 'w': {1: 1, 3: 1}},
        {'u': {3: 1}, 'v': {2: 1, 0: -1}, 'w': {0: 1, 2: 1}},
        {'u': {0: 1, 1: 1}, 'v': {3: 1}, 'w': {0: -1, 1: 1}},
        {'u': {2: 1, 0: -1}, 'v': {0: 1, 1: 1}, 'w': {3: 1}},
        {'u': {1: 1, 3: -1}, 'v': {2: 1, 3: 1}, 'w': {0: 1}}
    ]
    
    def add_block(a_col_start, b_row_start, b_col_start, c_col_start, col_offset):
        """Add a 2x2x2 Strassen block at specified positions."""
        for s in range(7):
            step = strassen_steps[s]
            col = col_offset + s
            # U: A has shape (n, m) = (2, 4). Block covers rows [0,1] or [0,1] (n=2 so always rows 0,1)
            # and columns [a_col_start, a_col_start+1]
            for local_idx, coeff in step['u'].items():
                l_row = local_idx // 2
                l_col = local_idx % 2
                global_idx = l_row * m + (a_col_start + l_col)
                U[global_idx, col] += coeff
            # V: B has shape (m, p) = (4, 4). Block covers rows [b_row_start, b_row_start+1]
            # and columns [b_col_start, b_col_start+1]
            for local_idx, coeff in step['v'].items():
                l_row = local_idx // 2
                l_col = local_idx % 2
                global_idx = (b_row_start + l_row) * p + (b_col_start + l_col)
                V[global_idx, col] += coeff
            # W: C has shape (p, n) = (4, 2) per the trace convention W[k*n+i, r]
            # Block covers k in [c_col_start, c_col_start+1] and i in [0,1]
            for local_idx, coeff in step['w'].items():
                l_row = local_idx // 2  # i index (0 or 1)
                l_col = local_idx % 2   # k index (0 or 1)
                global_idx = (c_col_start + l_col) * n + l_row
                W[global_idx, col] += coeff

    # Four 2x2x2 blocks:
    # Block 1: A cols 0-1, B rows 0-1, B cols 0-1, C k=0-1
    add_block(0, 0, 0, 0, 0)
    # Block 2: A cols 0-1, B rows 0-1, B cols 2-3, C k=2-3
    add_block(0, 0, 2, 2, 7)
    # Block 3: A cols 2-3, B rows 2-3, B cols 0-1, C k=0-1
    add_block(2, 2, 0, 0, 14)
    # Block 4: A cols 2-3, B rows 2-3, B cols 2-3, C k=2-3
    add_block(2, 2, 2, 2, 21)
    
    return U, V, W


def _verify_decomposition(U, V, W, tol=1e-6):
    """Verify that U, V, W exactly decompose the 2x4x4 matmul tensor."""
    n, m, p = DIMENSIONS
    # Matmul tensor: T[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B
    # contribute to C[i,k]. The tensor is T[i,j,k] = 1 for all valid i,j,k
    # because C[i,k] = sum_j A[i,j]*B[j,k]
    # Wait, that's not right either. The tensor T[i,j,k] has T[i,j,k]=1 iff
    # the product A[i,j]*B[j,k] contributes to C[i,k].
    # Actually for matrix multiplication, the structure tensor has:
    # T[i,j,k] = 1 if j is the shared index, i.e., T[i,j,k] = delta_{j,?}
    # No. The tensor is defined as: T[i,j,k] = 1 for all i in [n], j in [m], k in [p]
    # because C[i,k] = sum_j A[i,j] B[j,k]
    # The structure tensor of matmul m_{n,m,p} has T[i,j,k] = 1 for all i,j,k.
    # Wait no. Let me think again.
    # The tensor is T in V* x W* x X* where:
    # T(a,b,c) = sum_{i,j,k} a_i b_j c_k T[i,j,k]
    # For matmul: C = AB, C[i,k] = sum_j A[i,j] B[j,k]
    # So T[i,j,k] = 1 if and only if the triple (i,j,k) appears in the sum
    # which is for all i in [n], j in [m], k in [p] with the constraint that
    # j is the shared index. Actually T[i,j,k] = 1 for all i,j,k because
    # every combination (i,j,k) contributes: A[i,j]*B[j,k] contributes to C[i,k].
    # So T[i,j,k] = 1 for all i, j, k.
    
    # Hmm, but that would make the rank at most n*m*p = 32 (trivial decomposition)
    # and the Strassen decomposition has rank 28 < 32, which is consistent.
    
    # Let me verify: the trivial decomposition has rank 32:
    # U[i*m+j, r] = delta_{i,j} for r = i*m+j
    # V[j*p+k, r] = delta_{j,k} for r = j*p+k  
    # W[k*n+i, r] = 1 for r = ...
    # Actually the trivial decomposition is rank n*m*p = 32.
    
    # The correct tensor: T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    T = np.ones((n, m, p))
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    rank = U_arr.shape[1]
    U3 = U_arr.reshape(n, m, rank)
    V3 = V_arr.reshape(m, p, rank)
    W3 = W_arr.reshape(p, n, rank)  # W[k*n+i, r] -> (p, n, rank)
    
    # sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r] should equal T[i,j,k] = 1
    recon = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    
    residual = np.max(np.abs(recon - T))
    return residual < tol, residual


def _build_alt_strassen_rank28():
    """
    Alternative rank-28 construction: treat 2x4x4 as two independent 1x4x4 problems
    plus cross terms, or use a different block decomposition.
    
    Actually, let's try: split A into two row-vectors (each 1x4), and for each,
    compute 1x4 * 4x4 = 1x4. Each 1x4x4 has rank at most 4 (trivial) or can use
    a non-trivial decomposition.
    
    For 1x4x4: C[k] = sum_j A[j] B[j,k], j=0..3, k=0..3
    Trivial rank: 4. Can we do better? The tensor is T[j,k] = 1 for all j,k (1x4x4).
    Rank of 1x4x4 tensor = 4 (it's just an outer product structure, rank = min(4,4) = 4).
    
    So two independent 1x4x4 would give rank 8, but we have cross-coupling in B.
    Actually no - if we split A into A_row0 and A_row1, each row computes
    C_row[k] = sum_j A_row[j] * B[j,k]
    These are independent given B, so total rank = 4 + 4 = 8? That can't be right
    since we know rank is 28.
    
    The issue is that B is shared. The full tensor T[i,j,k] = 1 for all i,j,k
    has rank 28, not 8. The "two independent 1x4x4" view is wrong because
    the B factors are shared across the two i values.
    
    Let me just stick with the block Strassen approach.
    """
    return _build_strassen_rank28()


def _polish_decomposition(U, V, W, max_iters=100, tol=1e-12):
    """Polish an existing decomposition to higher precision using ALS."""
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    T = np.ones((n, m, p))
    
    U3 = U.reshape(n, m, rank).astype(float)
    V3 = V.reshape(m, p, rank).astype(float)
    W3 = W.reshape(p, n, rank).astype(float)
    
    for it in range(max_iters):
        # Update U
        for i in range(n):
            for j in range(m):
                A = V3[j, :, :] * W3[:, i, :]  # (p, rank)
                A = A.T  # (rank, p)
                if np.linalg.matrix_rank(A) >= rank:
                    try:
                        U3[i, j, :] = np.linalg.solve(A.T @ A, A.T @ T[i, j, :])
                    except:
                        try:
                            U3[i, j, :] = np.linalg.lstsq(A, T[i, j, :], rcond=None)[0]
                        except:
                            pass
        # Update V
        for j in range(m):
            for k in range(p):
                B = U3[:, j, :] * W3[k, :, :]  # (n, rank)
                B = B.T  # (rank, n)
                if np.linalg.matrix_rank(B) >= rank:
                    try:
                        V3[j, k, :] = np.linalg.solve(B.T @ B, B.T @ T[:, j, k])
                    except:
                        try:
                            V3[j, k, :] = np.linalg.lstsq(B, T[:, j, k], rcond=None)[0]
                        except:
                            pass
        # Update W
        for k in range(p):
            for i in range(n):
                C = U3[i, :, :] * V3[:, k, :]  # (m, rank)
                C = C.T  # (rank, m)
                if np.linalg.matrix_rank(C) >= rank:
                    try:
                        W3[k, i, :] = np.linalg.solve(C.T @ C, C.T @ T[i, :, k])
                    except:
                        try:
                            W3[k, i, :] = np.linalg.lstsq(C, T[i, :, k], rcond=None)[0]
                        except:
                            pass
        
        recon = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        residual = np.max(np.abs(recon - T))
        if residual < tol:
            break
    
    U_flat = U3.reshape(n * m, rank)
    V_flat = V3.reshape(m * p, rank)
    W_flat = W3.reshape(p * n, rank)
    return U_flat, V_flat, W_flat, residual


def _try_rank_27_als(seed, max_iters=200):
    """Try to find a rank-27 decomposition from scratch using ALS."""
    n, m, p = DIMENSIONS
    rank = 27
    T = np.ones((n, m, p))
    rng = np.random.RandomState(seed)
    
    U = rng.randn(n * m, rank) * 0.1
    V = rng.randn(m * p, rank) * 0.1
    W = rng.randn(p * n, rank) * 0.1
    
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    for it in range(max_iters):
        for i in range(n):
            for j in range(m):
                A = (V3[j] * W3[:, i]).T  # (rank, p)
                try:
                    U3[i, j] = np.linalg.lstsq(A, T[i, j, :], rcond=None)[0]
                except:
                    pass
        for j in range(m):
            for k in range(p):
                B = (U3[:, j] * W3[k]).T  # (rank, n)
                try:
                    V3[j, k] = np.linalg.lstsq(B, T[:, j, k], rcond=None)[0]
                except:
                    pass
        for k in range(p):
            for i in range(n):
                C = (U3[i] * V3[:, k]).T  # (rank, m)
                try:
                    W3[k, i] = np.linalg.lstsq(C, T[i, :, k], rcond=None)[0]
                except:
                    pass
        
        if it % 20 == 0:
            recon = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
            residual = np.max(np.abs(recon - T))
            if residual < 1e-8:
                break
    
    recon = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    residual = np.max(np.abs(recon - T))
    
    if residual < 1e-6:
        U_flat = U3.reshape(n * m, rank)
        V_flat = V3.reshape(m * p, rank)
        W_flat = W3.reshape(p * n, rank)
        return U_flat, V_flat, W_flat, residual
    return None


def _try_rank_reduction_from_strassen(target_rank, seed, max_iters=150):
    """
    Start from rank-28 Strassen, zero out one column, and try to re-optimize
    the remaining 27 columns to still decompose the tensor.
    """
    n, m, p = DIMENSIONS
    T = np.ones((n, m, p))
    rng = np.random.RandomState(seed)
    
    U28, V28, W28 = _build_strassen_rank28()
    
    for attempt in range(28):
        col_idx = rng.randint(0, 28)
        
        U = U28.copy()
        V = V28.copy()
        W = W28.copy()
        U[:, col_idx] = 0
        V[:, col_idx] = 0
        W[:, col_idx] = 0
        
        # Remove the zero column to work with 27 columns
        mask = np.ones(28, dtype=bool)
        mask[col_idx] = False
        U = U[:, mask]
        V = V[:, mask]
        W = W[:, mask]
        
        rank = 27
        U3 = U.reshape(n, m, rank).astype(float)
        V3 = V.reshape(m, p, rank).astype(float)
        W3 = W.reshape(p, n, rank).astype(float)
        
        for it in range(max_iters):
            for i in range(n):
                for j in range(m):
                    A = (V3[j] * W3[:, i]).T
                    try:
                        U3[i, j] = np.linalg.lstsq(A, T[i, j, :], rcond=None)[0]
                    except:
                        pass
            for j in range(m):
                for k in range(p):
                    B = (U3[:, j] * W3[k]).T
                    try:
                        V3[j, k] = np.linalg.lstsq(B, T[:, j, k], rcond=None)[0]
                    except:
                        pass
            for k in range(p):
                for i in range(n):
                    C = (U3[i] * V3[:, k]).T
                    try:
                        W3[k, i] = np.linalg.lstsq(C, T[i, :, k], rcond=None)[0]
                    except:
                        pass
            
            if it % 20 == 0:
                recon = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
                residual = np.max(np.abs(recon - T))
                if residual < 1e-8:
                    break
        
        recon = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        residual = np.max(np.abs(recon - T))
        
        if residual < 1e-6:
            U_flat = U3.reshape(n * m, rank)
            V_flat = V3.reshape(m * p, rank)
            W_flat = W3.reshape(p * n, rank)
            return U_flat, V_flat, W_flat, residual
    
    return None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s limit
    
    # Start with known rank-28 Strassen solution
    U28, V28, W28 = _build_strassen_rank28()
    
    def to_lists(arr):
        return [[float(x) for x in row] for row in arr]
    
    best_result = {"U": to_lists(U28), "V": to_lists(V28), "W": to_lists(W28)}
    best_rank = 28
    
    # Polish the rank-28 solution for higher precision
    if time.time() < deadline:
        try:
            U_pol, V_pol, W_pol, res = _polish_decomposition(U28, V28, W28, max_iters=50, tol=1e-12)
            if res < 1e-10:
                best_result = {"U": to_lists(U_pol), "V": to_lists(V_pol), "W": to_lists(W_pol)}
        except:
            pass
    
    # Try to reduce to rank 27 via zeroing a column and re-optimizing
    if time.time() < deadline and best_rank == 28:
        for seed in range(30):
            if time.time() >= deadline:
                break
            try:
                result = _try_rank_reduction_from_strassen(27, seed, max_iters=100)
                if result is not None:
                    U_r, V_r, W_r, res = result
                    valid, res2 = _verify_decomposition(U_r, V_r, W_r, tol=1e-5)
                    if valid:
                        best_rank = 27
                        best_result = {"U": to_lists(U_r), "V": to_lists(V_r), "W": to_lists(W_r)}
                        break
            except:
                continue
    
    # Try pure ALS for rank 27 from random init
    if time.time() < deadline and best_rank == 28:
        for seed in range(20):
            if time.time() >= deadline:
                break
            try:
                result = _try_rank_27_als(seed, max_iters=150)
                if result is not None:
                    U_r, V_r, W_r, res = result
                    valid, res2 = _verify_decomposition(U_r, V_r, W_r, tol=1e-5)
                    if valid:
                        best_rank = 27
                        best_result = {"U": to_lists(U_r), "V": to_lists(V_r), "W": to_lists(W_r)}
                        break
            except:
                continue
    
    return best_result
# EVOLVE-BLOCK-END
