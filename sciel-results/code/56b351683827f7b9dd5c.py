import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 4, 4]

def _build_tensor(n, m, p):
    """
    Build the multiplication tensor T of shape (n*m, m*p, p*n).
    """
    T = np.zeros((n * m, m * p, p * n))
    for r in range(n):
        for c in range(m):
            for k in range(p):
                i = r * m + c
                j = c * p + k
                kk = k * n + r
                T[i, j, kk] = 1.0
    return T

def _build_tensor_242():
    """
    Build tensor for 2x4 * 4x2 -> 2x2.
    Dimensions: n=2, m=4, p=2.
    """
    n, m, p = 2, 4, 2
    T = np.zeros((n * m, m * p, p * n))
    for r in range(n):
        for c in range(m):
            for k in range(p):
                i = r * m + c
                j = c * p + k
                kk = k * n + r
                T[i, j, kk] = 1.0
    return T

def _strassen_decomposition(n, m, p):
    """
    Construct a rank-28 decomposition for 2x4x4 using heterogeneous block structure.
    """
    rank = 28
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    patterns = [
        {'a': {0: 1, 3: 1}, 'b': {0: 1, 3: 1}, 'c': {0: 1, 3: 1}},
        {'a': {2: 1, 3: 1}, 'b': {0: 1}, 'c': {2: 1, 3: -1}},
        {'a': {0: 1}, 'b': {1: 1, 3: -1}, 'c': {1: 1, 3: 1}},
        {'a': {3: 1}, 'b': {2: 1, 0: -1}, 'c': {0: 1, 2: 1}},
        {'a': {0: 1, 1: 1}, 'b': {3: 1}, 'c': {0: -1, 1: 1}},
        {'a': {2: 1, 0: -1}, 'b': {0: 1, 1: 1}, 'c': {3: 1}},
        {'a': {1: 1, 3: -1}, 'b': {2: 1, 3: 1}, 'c': {0: 1}},
    ]
    
    blocks = [
        (0, 0, 0, 0),
        (2, 2, 0, 0),
        (0, 0, 2, 2),
        (2, 2, 2, 2),
    ]
    
    col_idx = 0
    for a_off, b_r_off, b_c_off, c_off in blocks:
        for pat in patterns:
            for local_i, coeff in pat['a'].items():
                r = local_i // 2
                c = local_i % 2
                global_i = r * m + (a_off + c)
                U[global_i, col_idx] = coeff
            
            for local_j, coeff in pat['b'].items():
                r = local_j // 2
                c = local_j % 2
                global_j = (b_r_off + r) * p + (b_c_off + c)
                V[global_j, col_idx] = coeff
            
            for local_k, coeff in pat['c'].items():
                r = local_k // 2
                c = local_k % 2
                global_k = c_off + c
                global_i = r
                W[global_k * n + global_i, col_idx] = coeff
            
            col_idx += 1
            
    return U, V, W

def _verify_decomposition(U, V, W, T, atol=1e-9):
    T_check = np.einsum('it,jt,kt->ijk', U, V, W)
    return np.allclose(T_check, T, atol=atol), np.linalg.norm(T_check - T)

def _try_optimize_rank(T, n, m, p, target_rank, seed=0, time_limit=10.0):
    """
    ALS optimization for rank reduction.
    """
    start_time = time.time()
    dim_u, dim_v, dim_w = n * m, m * p, p * n
    T_flat = T.flatten()
    
    def compute_residual_norm(U, V, W):
        T_calc = np.einsum('it,jt,kt->ijk', U, V, W)
        return np.linalg.norm(T_calc - T)
    
    max_iters = 200
    tol = 1e-12
    
    for attempt in range(10):
        if time.time() - start_time > time_limit:
            break
            
        np.random.seed(seed + attempt)
        
        # SVD Initialization
        # Flatten T to (dim_u, dim_v * dim_w)
        T_reshaped = T.reshape(dim_u, dim_v * dim_w)
        U_svd, S, Vh_svd = np.linalg.svd(T_reshaped, full_matrices=False)
        
        # Truncate to target_rank
        U = U_svd[:, :target_rank]
        # V and W need to be reconstructed
        # This is tricky. Random init for V and W is often better for ALS.
        
        U = np.random.randn(dim_u, target_rank)
        V = np.random.randn(dim_v, target_rank)
        W = np.random.randn(dim_w, target_rank)
        
        prev_norm = np.inf
        for it in range(max_iters):
            if time.time() - start_time > time_limit:
                break
                
            # Update U
            B = np.einsum('jt,kt->jkt', V, W)
            B_flat = B.reshape(dim_v * dim_w, target_rank)
            T_reshaped_u = T.reshape(dim_u, dim_v * dim_w)
            try:
                U_new = np.linalg.lstsq(B_flat.T, T_reshaped_u.T, rcond=None)[0].T
            except:
                break
                
            # Update V
            A = np.einsum('it,kt->ikt', U_new, W)
            A_flat = A.reshape(dim_u * dim_w, target_rank)
            T_reshaped_v = T.transpose(1, 0, 2).reshape(dim_v, dim_u * dim_w)
            try:
                V_new = np.linalg.lstsq(A_flat.T, T_reshaped_v.T, rcond=None)[0].T
            except:
                break
                
            # Update W
            C = np.einsum('it,jt->ijt', U_new, V_new)
            C_flat = C.reshape(dim_u * dim_v, target_rank)
            T_reshaped_w = T.transpose(2, 0, 1).reshape(dim_w, dim_u * dim_v)
            try:
                W_new = np.linalg.lstsq(C_flat.T, T_reshaped_w.T, rcond=None)[0].T
            except:
                break
                
            U, V, W = U_new, V_new, W_new
            current_norm = compute_residual_norm(U, V, W)
            
            if np.abs(prev_norm - current_norm) < tol:
                if current_norm < 1e-9:
                    return U, V, W
                break
            prev_norm = current_norm
            
            if current_norm < 1e-9:
                return U, V, W
                
    return None

def _find_rank_13_242(time_limit=15.0):
    """
    Try to find a rank-13 decomposition for 2x4x2 multiplication.
    """
    n, m, p = 2, 4, 2
    T = _build_tensor_242()
    
    # Try rank 13
    res = _try_optimize_rank(T, n, m, p, 13, seed=42, time_limit=time_limit)
    if res is not None:
        U, V, W = res
        # Snap to half-integers
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        
        T_check = np.einsum('it,jt,kt->ijk', U_snap, V_snap, W_snap)
        if np.allclose(T_check, T, atol=1e-9):
            return U_snap, V_snap, W_snap
            
    # Try rank 14 (just in case 13 is impossible but 14 is easy)
    res = _try_optimize_rank(T, n, m, p, 14, seed=42, time_limit=time_limit/2)
    if res is not None:
        U, V, W = res
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        T_check = np.einsum('it,jt,kt->ijk', U_snap, V_snap, W_snap)
        if np.allclose(T_check, T, atol=1e-9):
            return U_snap, V_snap, W_snap
            
    return None

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    deadline = start_time + 115.0
    
    # 1. Get the known rank-28 solution
    U28, V28, W28 = _strassen_decomposition(n, m, p)
    T = _build_tensor(n, m, p)
    
    valid28, err28 = _verify_decomposition(U28, V28, W28, T)
    
    best_U, best_V, best_W = U28, V28, W28
    best_rank = 28
    
    if not valid28:
        # Fallback to schoolbook
        rank32 = n * m * p
        U32 = np.zeros((n * m, rank32))
        V32 = np.zeros((m * p, rank32))
        W32 = np.zeros((p * n, rank32))
        idx = 0
        for r in range(n):
            for c in range(m):
                for k in range(p):
                    i = r * m + c
                    j = c * p + k
                    kk = k * n + r
                    U32[i, idx] = 1.0
                    V32[j, idx] = 1.0
                    W32[kk, idx] = 1.0
                    idx += 1
        best_U, best_V, best_W = U32, V32, W32
        best_rank = 32
        
    # 2. Try to find a better solution using the slicing strategy
    # Look for rank 13 in 2x4x2, which implies rank 26 in 2x4x4
    if time.time() < deadline - 30:
        sol_242 = _find_rank_13_242(time_limit=20.0)
        if sol_242 is not None:
            U_242, V_242, W_242 = sol_242
            r_242 = U_242.shape[1]
            
            # Construct 2x4x4 solution
            # C_left = A * B_left (4x2 part of B)
            # C_right = A * B_right (4x2 part of B)
            # We need to map the 2x4x2 solution to the full problem.
            # The 2x4x2 tensor T_242 corresponds to:
            # A (2x4), B (4x2), C (2x2)
            # In the full 2x4x4 problem:
            # A is the same.
            # B is 4x4. B_left is cols 0,1. B_right is cols 2,3.
            # C is 2x4. C_left is cols 0,1. C_right is cols 2,3.
            
            U_full = np.zeros((8, 2 * r_242))
            V_full = np.zeros((16, 2 * r_242))
            W_full = np.zeros((8, 2 * r_242))
            
            # Copy for Left half (C cols 0,1)
            # A is same -> U_full[:, 0:r_242] = U_242
            U_full[:, 0:r_242] = U_242
            
            # B_left is rows 0-3, cols 0-1 of B_full.
            # In V_242, indices are j = c*p + k. p=2.
            # j = c*2 + k. c in 0..3, k in 0..1.
            # This matches the first 8 rows of V_full (which corresponds to B cols 0,1).
            # V_full indices: j = c*p + k. p=4.
            # B_left corresponds to k in 0..1.
            # So V_full[0:8, 0:r_242] should be mapped from V_242.
            # V_242 has 8 rows (4*2). V_full has 16 rows (4*4).
            # The mapping is: V_242[j] corresponds to B[c, k] where j=c*2+k.
            # In V_full, B[c, k] is at index c*4+k.
            # So for each j in V_242:
            #   c = j // 2
            #   k = j % 2
            #   idx_full = c * 4 + k
            for j in range(8):
                c = j // 2
                k = j % 2
                idx_full = c * 4 + k
                V_full[idx_full, 0:r_242] = V_242[j, :]
            
            # C_left corresponds to C[r, k] where k in 0..1.
            # In W_242, indices are k = k''*n + r. n=2.
            # k = k''*2 + r. k'' in 0..1, r in 0..1.
            # In W_full, C[r, k] is at index k*2 + r.
            # The mapping is: W_242[j] corresponds to C[r, k] where j=k*2+r.
            # In W_full, C[r, k] is at index k*2+r.
            # So the indices are the same!
            W_full[0:4, 0:r_242] = W_242[0:4, :] # C cols 0,1
            # Wait, W_242 has 4 rows (2*2). W_full has 8 rows (4*2).
            # W_242 covers C[r, k] for k in 0..1.
            # In W_full, these are indices 0..3 (k=0,1; r=0,1).
            # So W_full[0:4, 0:r_242] = W_242.
            
            # Copy for Right half (C cols 2,3)
            U_full[:, r_242:2*r_242] = U_242
            
            # B_right is rows 0-3, cols 2-3 of B_full.
            # V_242[j] corresponds to B[c, k] where k in 0..1.
            # We want to place this at B[c, k+2].
            # In V_full, index is c*4 + (k+2).
            for j in range(8):
                c = j // 2
                k = j % 2
                idx_full = c * 4 + (k + 2)
                V_full[idx_full, r_242:2*r_242] = V_242[j, :]
            
            # C_right corresponds to C[r, k] where k in 2..3.
            # In W_242, index is k''*2 + r.
            # In W_full, index is k*2 + r where k in 2..3.
            # So W_full[4:8, r_242:2*r_242] = W_242.
            W_full[4:8, r_242:2*r_242] = W_242
            
            # Verify
            T_check = np.einsum('it,jt,kt->ijk', U_full, V_full, W_full)
            if np.allclose(T_check, T, atol=1e-9):
                best_U, best_V, best_W = U_full, V_full, W_full
                best_rank = 2 * r_242
                print(f"Found rank {best_rank} solution via slicing!")
    
    # 3. If no better solution found, try general rank reduction on 2x4x4
    if best_rank > 26 and time.time() < deadline - 20:
        # Try rank 27 on full tensor
        res = _try_optimize_rank(T, n, m, p, 27, seed=1, time_limit=10.0)
        if res is not None:
            U, V, W = res
            U_snap = np.round(U * 2) / 2
            V_snap = np.round(V * 2) / 2
            W_snap = np.round(W * 2) / 2
            T_check = np.einsum('it,jt,kt->ijk', U_snap, V_snap, W_snap)
            if np.allclose(T_check, T, atol=1e-9):
                best_U, best_V, best_W = U_snap, V_snap, W_snap
                best_rank = 27

    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}

if __name__ == "__main__":
    res = construct()
    print("Rank used:", len(res['U'][0]))
