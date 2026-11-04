import time
import numpy as np

DIMENSIONS = [4, 4, 8]
START = time.time()
DEADLINE = START + 140.0  # Safety margin under 150s kill

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

def _verify_decomposition(U, V, W, n=4, m=4, p=8, atol=1e-9):
    """Vectorized verification of a tensor decomposition."""
    rank = U.shape[1]
    result = np.zeros((n, m, p), dtype=np.float64)
    for r in range(rank):
        u_col = U[:, r].reshape(n, m)
        v_col = V[:, r].reshape(m, p)
        w_col = W[:, r].reshape(p, n)
        result += np.einsum('ij,jk,ki->ijk', u_col, v_col, w_col)
    return np.allclose(result, 1.0, atol=atol)

def _try_shared_left_factor_search(time_budget):
    """
    Attempts to find a lower-rank decomposition by exploiting the direct-sum
    structure: 4x4x8 = 4x4x4 ⊕ 4x4x4 where both blocks share the same A factor.
    
    Strategy: Use L-BFGS-B to minimize the residual norm while iteratively
    zeroing out the smallest-magnitude entries in the factor matrices to
    encourage sparsity and potentially reduce the effective rank.
    """
    if time.time() > time_budget:
        return None
    
    from scipy.optimize import minimize
    
    U2, V2, W2 = _build_strassen_2x2x2()
    
    # Build the rank-49 4x4x4 decomposition
    rank_444 = 49
    U4 = np.zeros((16, rank_444), dtype=np.float64)
    V4 = np.zeros((16, rank_444), dtype=np.float64)
    W4 = np.zeros((16, rank_444), dtype=np.float64)
    
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
    
    # Construct the full 4x4x8 with direct sum
    total_rank = 98
    U = np.zeros((16, total_rank), dtype=np.float64)
    V = np.zeros((32, total_rank), dtype=np.float64)
    W = np.zeros((32, total_rank), dtype=np.float64)
    
    for block in range(2):
        cs = block * 49
        ce = (block + 1) * 49
        U[:, cs:ce] = U4
        for j in range(4):
            for kl in range(4):
                kg = block * 4 + kl
                V[j*8 + kg, cs:ce] = V4[j*4 + kl, :]
        for kl in range(4):
            kg = block * 4 + kl
            for i in range(4):
                W[kg*4 + i, cs:ce] = W4[kl*4 + i, :]
    
    # Verify the rank-98 solution is correct
    if not _verify_decomposition(U, V, W, atol=1e-9):
        return None
    
    # Attempt rank reduction via continuous optimization
    # We'll try to minimize the residual while encouraging sparsity
    # by adding an L1 penalty, then check if we can round to integers
    
    n, m, p = 4, 4, 8
    target = np.ones((n, m, p), dtype=np.float64)
    
    def _residual(params, U_shape, V_shape, W_shape):
        """Compute residual norm from flattened parameters."""
        U_flat, V_flat, W_flat = params
        U_mat = U_flat.reshape(U_shape)
        V_mat = V_flat.reshape(V_shape)
        W_mat = W_flat.reshape(W_shape)
        
        result = np.zeros((n, m, p), dtype=np.float64)
        for r in range(total_rank):
            u_col = U_mat[:, r].reshape(n, m)
            v_col = V_mat[:, r].reshape(m, p)
            w_col = W_mat[:, r].reshape(p, n)
            result += np.einsum('ij,jk,ki->ijk', u_col, v_col, w_col)
        
        return np.linalg.norm(result - target) ** 2
    
    def _objective(params):
        """Objective: residual + L1 sparsity penalty."""
        U_flat, V_flat, W_flat = params
        residual = _residual(params, (16, total_rank), (32, total_rank), (32, total_rank))
        # L1 penalty to encourage sparsity
        l1 = (np.abs(U_flat).sum() + np.abs(V_flat).sum() + np.abs(W_flat).sum()) * 0.01
        return residual + l1
    
    # Initialize parameters
    U_init = U.flatten()
    V_init = V.flatten()
    W_init = W.flatten()
    params_init = [U_init, V_init, W_init]
    
    # Try optimization with time limit
    opt_start = time.time()
    opt_timeout = time_budget - opt_start - 5.0  # 5s safety margin
    
    if opt_timeout > 2.0:
        try:
            # Use L-BFGS-B with time limit
            best_params = params_init
            best_obj = _objective(params_init)
            
            # Split into smaller optimization runs to respect time limit
            max_iter = 100
            while time.time() < opt_start + opt_timeout and max_iter > 0:
                batch_size = min(max_iter, 20)
                
                def _callback(xk):
                    nonlocal best_params, best_obj
                    obj_val = _objective([xk[0:16*total_rank], xk[16*total_rank:16*total_rank+32*total_rank], xk[16*total_rank+32*total_rank:]])
                    if obj_val < best_obj:
                        best_obj = obj_val
                        best_params = [xk[0:16*total_rank], xk[16*total_rank:16*total_rank+32*total_rank], xk[16*total_rank+32*total_rank:]]
                    return time.time() > opt_start + opt_timeout
                
                # Flatten all parameters into a single vector for scipy
                U_flat_opt = best_params[0].copy()
                V_flat_opt = best_params[1].copy()
                W_flat_opt = best_params[2].copy()
                x0 = np.concatenate([U_flat_opt, V_flat_opt, W_flat_opt])
                
                def _obj_flat(x):
                    U_f = x[0:16*total_rank]
                    V_f = x[16*total_rank:16*total_rank+32*total_rank]
                    W_f = x[16*total_rank+32*total_rank:]
                    return _objective([U_f, V_f, W_f])
                
                res = minimize(_obj_flat, x0, method='L-BFGS-B', 
                              options={'maxiter': batch_size, 'maxfun': batch_size * 10,
                                      'callback': _callback})
                
                if res.success or res.fun < best_obj:
                    x_opt = res.x
                    U_opt = x_opt[0:16*total_rank].reshape(16, total_rank)
                    V_opt = x_opt[16*total_rank:16*total_rank+32*total_rank].reshape(32, total_rank)
                    W_opt = x_opt[16*total_rank+32*total_rank:].reshape(32, total_rank)
                    
                    # Check if this gives a valid decomposition
                    if _verify_decomposition(U_opt, V_opt, W_opt, atol=1e-6):
                        # Try to reduce rank by zeroing out small entries
                        U_rounded = np.where(np.abs(U_opt) < 1e-3, 0, U_opt)
                        V_rounded = np.where(np.abs(V_opt) < 1e-3, 0, V_opt)
                        W_rounded = np.where(np.abs(W_opt) < 1e-3, 0, W_opt)
                        
                        # Check if rounded version still works
                        if _verify_decomposition(U_rounded, V_rounded, W_rounded, atol=1e-6):
                            # Count non-zero columns to estimate effective rank
                            U_nz = np.any(U_rounded != 0, axis=0).sum()
                            V_nz = np.any(V_rounded != 0, axis=0).sum()
                            W_nz = np.any(W_rounded != 0, axis=0).sum()
                            effective_rank = max(U_nz, V_nz, W_nz)
                            
                            if effective_rank < 98:
                                # Try to round to integers
                                if (np.allclose(U_rounded, np.round(U_rounded), atol=1e-4) and 
                                    np.allclose(V_rounded, np.round(V_rounded), atol=1e-4) and 
                                    np.allclose(W_rounded, np.round(W_rounded), atol=1e-4)):
                                    U_int = np.round(U_rounded).astype(np.int16)
                                    V_int = np.round(V_rounded).astype(np.int16)
                                    W_int = np.round(W_rounded).astype(np.int16)
                                    return {
                                        "U": U_int.tolist(),
                                        "V": V_int.tolist(),
                                        "W": W_int.tolist()
                                    }
                        best_params = [U_opt.flatten(), V_opt.flatten(), W_opt.flatten()]
                
                max_iter -= batch_size
                if time.time() > opt_start + opt_timeout:
                    break
        except Exception:
            pass
    
    # If optimization didn't find a better solution, return the original rank-98
    if np.allclose(U, np.round(U), atol=1e-9) and np.allclose(V, np.round(V), atol=1e-9) and np.allclose(W, np.round(W), atol=1e-9):
        U_int = np.round(U).astype(np.int16)
        V_int = np.round(V).astype(np.int16)
        W_int = np.round(W).astype(np.int16)
    else:
        U_int = U.astype(np.float64)
        V_int = V.astype(np.float64)
        W_int = W.astype(np.float64)
    
    return {
        "U": U_int.tolist(),
        "V": V_int.tolist(),
        "W": W_int.tolist()
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

def construct():
    try:
        if time.time() < DEADLINE:
            res = _build_rank98()
            if res is not None:
                # Try to find a better solution with remaining time
                remaining = DEADLINE - time.time() - 10.0  # 10s safety margin
                if remaining > 5.0:
                    better = _try_shared_left_factor_search(START + remaining + time.time())
                    if better is not None:
                        return better
                return res
    except Exception:
        pass
    return _schoolbook()

if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['U'][0])}")
