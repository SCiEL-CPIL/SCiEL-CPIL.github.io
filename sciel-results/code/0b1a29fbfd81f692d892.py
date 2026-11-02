import numpy as np
import time

def _build_target_tensor(n, m, p):
    """Build the matrix multiplication tensor T[a,b,c] = 1 if c = k*n+i where a=i*m+j, b=j*p+k"""
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T[a_idx, b_idx, c_idx] = 1.0
    return T

def _build_block_tensor(n, m, p_block):
    """Build the 2x3x2 block tensor (one block of B columns)."""
    T = np.zeros((n*m, m*p_block, p_block*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p_block):
                a_idx = i * m + j
                b_idx = j * p_block + k
                c_idx = k * n + i
                T[a_idx, b_idx, c_idx] = 1.0
    return T

def _als_decompose(T, rank, max_iter=200, n_restarts=10, seed=42):
    """
    Alternating Least Squares to find a rank decomposition of tensor T.
    T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    """
    n_a, n_b, n_c = T.shape
    rng = np.random.RandomState(seed)
    best_error = np.inf
    best_result = None
    
    for restart in range(n_restarts):
        U = rng.randn(n_a, rank) * 0.1
        V = rng.randn(n_b, rank) * 0.1
        W = rng.randn(n_c, rank) * 0.1
        
        for it in range(max_iter):
            # Solve for U
            T_reshaped = T.reshape(n_a, n_b * n_c)
            Z = np.zeros((n_b * n_c, rank))
            for b in range(n_b):
                for c in range(n_c):
                    Z[b * n_c + c, :] = V[b, :] * W[c, :]
            try:
                U_new = T_reshaped @ np.linalg.pinv(Z.T)
            except:
                break
            
            # Solve for V
            T_reshaped_v = np.zeros((n_b, n_a * n_c))
            for b in range(n_b):
                for a in range(n_a):
                    for c in range(n_c):
                        T_reshaped_v[b, a * n_c + c] = T[a, b, c]
            Z2 = np.zeros((n_a * n_c, rank))
            for a in range(n_a):
                for c in range(n_c):
                    Z2[a * n_c + c, :] = U_new[a, :] * W[c, :]
            try:
                V_new = T_reshaped_v @ np.linalg.pinv(Z2.T)
            except:
                break
            
            # Solve for W
            T_reshaped_w = np.zeros((n_c, n_a * n_b))
            for c in range(n_c):
                for a in range(n_a):
                    for b in range(n_b):
                        T_reshaped_w[c, a * n_b + b] = T[a, b, c]
            Z3 = np.zeros((n_a * n_b, rank))
            for a in range(n_a):
                for b in range(n_b):
                    Z3[a * n_b + b, :] = U_new[a, :] * V_new[b, :]
            try:
                W_new = T_reshaped_w @ np.linalg.pinv(Z3.T)
            except:
                break
            
            U, V, W = U_new, V_new, W_new
        
        # Check error
        T_check = np.einsum('ar,br,cr->abc', U, V, W)
        error = np.max(np.abs(T_check - T))
        
        if error < best_error:
            best_error = error
            best_result = (U.copy(), V.copy(), W.copy())
        
        if error < 1e-10:
            break
    
    return best_result, best_error

def _try_round(U, V, W, T, scales=[1, 2, 3, 4, 5, 6, 8, 10, 12]):
    """Try rounding to integers or half-integers at various scales and verify."""
    for s in scales:
        U_r = np.round(U * s) / s
        V_r = np.round(V * s) / s
        W_r = np.round(W * s) / s
        T_check = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
        if np.max(np.abs(T_check - T)) < 1e-8:
            return (U_r, V_r, W_r)
    return None

def _known_rank11_block():
    """Return the known rank-11 decomposition for a 2x3x2 block (Strassen + 4 rank-1)."""
    n, m, p_block = 2, 3, 2
    rank = 11
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p_block, rank))
    W = np.zeros((p_block * n, rank))
    
    k0, k1 = 0, 1
    
    # Strassen part
    # m1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[2, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # m2 = (A10+A11)B00
    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # m3 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # m4 = A11(B10-B00)
    U[4, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # m5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # m6 = (A10-A00)(B00+B01)
    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # m7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[4, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    # Extra: A[:,2] * B[2,:]
    U[2, 7] = 1; V[4, 7] = 1; W[0, 7] = 1
    U[2, 8] = 1; V[5, 8] = 1; W[2, 8] = 1
    U[5, 9] = 1; V[4, 9] = 1; W[1, 9] = 1
    U[5, 10] = 1; V[5, 10] = 1; W[3, 10] = 1
    
    return U, V, W

def construct():
    """
    Rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x8 matrix.
    Structure: 4 independent 2x3x2 blocks.
    Tries to find rank-10 per block (total 40), falls back to rank-11 (total 44).
    """
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s limit
    
    n, m, p = 2, 3, 8
    num_blocks = p // 2  # 4
    
    # Build the block tensor
    T_block = _build_block_tensor(n, m, 2)
    
    # First, verify our known rank-11 solution works
    U11, V11, W11 = _known_rank11_block()
    T_check = np.einsum('ar,br,cr->abc', U11, V11, W11)
    known_error = np.max(np.abs(T_check - T_block))
    
    # Try to find a rank-10 decomposition
    rank10_found = False
    U10, V10, W10 = None, None, None
    
    if time.time() < deadline:
        # Try ALS with rank 10
        result, error = _als_decompose(T_block, 10, max_iter=300, n_restarts=30, seed=42)
        if result is not None and error < 1e-8:
            rounded = _try_round(result[0], result[1], result[2], T_block)
            if rounded is not None:
                U10, V10, W10 = rounded
                rank10_found = True
    
    if not rank10_found and time.time() < deadline:
        # Try more ALS restarts with different seeds
        for seed in [123, 456, 789, 101, 202, 303]:
            if time.time() >= deadline:
                break
            result, error = _als_decompose(T_block, 10, max_iter=200, n_restarts=5, seed=seed)
            if result is not None and error < 1e-8:
                rounded = _try_round(result[0], result[1], result[2], T_block)
                if rounded is not None:
                    U10, V10, W10 = rounded
                    rank10_found = True
                    break
    
    # Decide which rank to use
    if rank10_found:
        rank_per_block = 10
        U_block, V_block, W_block = U10, V10, W10
    else:
        rank_per_block = 11
        U_block, V_block, W_block = U11, V11, W11
    
    total_rank = rank_per_block * num_blocks
    
    # Assemble full decomposition
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    for blk in range(num_blocks):
        base = blk * rank_per_block
        k0 = 2 * blk
        k1 = 2 * blk + 1
        
        # Copy U block (same for all blocks since A is shared)
        U[:, base:base+rank_per_block] = U_block
        
        # Copy V block with column offset
        for j in range(m):
            for kk in range(2):
                src_col = kk  # column index within block
                dst_col = k0 + kk  # actual column index
                V[j * p + dst_col, base:base+rank_per_block] = V_block[j * 2 + src_col, :]
        
        # Copy W block with row offset
        for kk in range(2):
            src_row = kk
            dst_row = k0 + kk
            for i in range(n):
                W[dst_row * n + i, base:base+rank_per_block] = W_block[src_row * n + i, :]
    
    # Final verification
    T_full = _build_target_tensor(n, m, p)
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    final_error = np.max(np.abs(T_recon - T_full))
    
    if final_error > 1e-8:
        # Fallback to known rank-11 if something went wrong
        U, V, W = np.zeros((n*m, 44)), np.zeros((m*p, 44)), np.zeros((p*n, 44))
        for blk in range(num_blocks):
            base = blk * 11
            k0 = 2 * blk
            k1 = 2 * blk + 1
            U[:, base:base+11] = U11
            for j in range(m):
                for kk in range(2):
                    V[j * p + k0 + kk, base:base+11] = V11[j * 2 + kk, :]
            for kk in range(2):
                for i in range(n):
                    W[(k0+kk) * n + i, base:base+11] = W11[kk * n + i, :]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
