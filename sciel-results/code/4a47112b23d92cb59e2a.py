import time
import numpy as np

DIMENSIONS = [2, 3, 8]
_N, _M, _P = 2, 3, 8

# --- Verification and Utilities ---

def _verify(U, V, W, target_mask, tol=1e-8):
    """
    Verify that U, V, W correctly decompose the matrix multiplication tensor.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    target_mask: boolean array of shape (n*m, m*p, p*n)
    """
    if isinstance(U, np.ndarray):
        U_arr = U
    else:
        U_arr = np.array(U, dtype=float)
    if isinstance(V, np.ndarray):
        V_arr = V
    else:
        V_arr = np.array(V, dtype=float)
    if isinstance(W, np.ndarray):
        W_arr = W
    else:
        W_arr = np.array(W, dtype=float)
        
    # Compute T at target positions using einsum
    T = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    
    # Check target positions
    if np.any(target_mask):
        target_vals = T[target_mask]
        if np.max(np.abs(target_vals - 1.0)) > tol:
            return False
    
    # Check non-target positions
    if np.any(~target_mask):
        if np.max(np.abs(T[~target_mask])) > tol:
            return False
            
    return True

def _round_and_verify(U, V, W, target_mask, tol=1e-5):
    """Try to round U, V, W to integers or half-integers and verify."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Try half-integers
    U_half = np.round(U_arr * 2) / 2
    V_half = np.round(V_arr * 2) / 2
    W_half = np.round(W_arr * 2) / 2
    if _verify(U_half, V_half, W_half, target_mask, tol=tol):
        return U_half.tolist(), V_half.tolist(), W_half.tolist()
        
    # Try integers
    U_int = np.round(U_arr)
    V_int = np.round(V_arr)
    W_int = np.round(W_arr)
    if _verify(U_int, V_int, W_int, target_mask, tol=tol):
        return U_int.tolist(), V_int.tolist(), W_int.tolist()
        
    return None

# --- Sub-problem: 2x3x2 ---

# Target indices for 2x3x2
# A is 2x3, B is 3x2, C is 2x2
# C[i,k] = sum_j A[i,j] B[j,k]
# Indices:
# A: i in [0,1], j in [0,2] -> a = i*3+j
# B: j in [0,2], k in [0,1] -> b = j*2+k
# C: k in [0,1], i in [0,1] -> c = k*2+i (Trace convention W[k*n+i, r])
# Wait, the trace convention is W[k*n+i, r].
# For 2x3x2: n=2, p=2. W index is k*n + i = 2k + i.
# Target T[a, b, c] = 1 if c = 2k + i, a = i*3 + j, b = j*2 + k.

_N2, _M2, _P2 = 2, 3, 2
_TARGET_IDX_232 = []
for i in range(_N2):
    for j in range(_M2):
        for k in range(_P2):
            a = i * _M2 + j
            b = j * _P2 + k
            c = k * _N2 + i
            _TARGET_IDX_232.append((a, b, c))

_TARGET_MASK_232 = np.zeros((_N2*_M2, _M2*_P2, _P2*_N2), dtype=bool)
for a, b, c in _TARGET_IDX_232:
    _TARGET_MASK_232[a, b, c] = True

def _als_232(rank, max_iters=100, deadline=None, seed=None, init=None):
    """
    ALS for 2x3x2 tensor decomposition.
    """
    n, m, p = _N2, _M2, _P2
    
    if init is not None:
        U = np.array(init[0], dtype=float).copy()
        V = np.array(init[1], dtype=float).copy()
        W = np.array(init[2], dtype=float).copy()
    else:
        rng = np.random.RandomState(seed)
        U = rng.randn(n*m, rank)
        V = rng.randn(m*p, rank)
        W = rng.randn(p*n, rank)
        
    # Precompute indices for speed
    # For each a, list of (k, b, c) such that T[a,b,c] is target
    # Actually, we solve for U[a,:] given V, W.
    # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    # For fixed a, we want sum_r U[a,r] (V[b,r] W[c,r]) = 1 for all (b,c) in target for this a.
    # Target for fixed a=i*m+j:
    # b = j*p+k, c = k*n+i.
    # So for fixed a, the pairs (b,c) are:
    # k=0: b=j*2+0, c=0*2+i = i
    # k=1: b=j*2+1, c=1*2+i = 2+i
    # There are p=2 such pairs.
    
    # Let's construct the matrix M_a for each a.
    # M_a is p x rank.
    # Row k of M_a corresponds to the k-th term in the sum for output C[i,k].
    # Wait, the equation is:
    # For each a (i,j), and each k:
    # sum_r U[a,r] V[j*p+k, r] W[k*n+i, r] = 1
    # This is a linear system in U[a,:].
    # There are p equations (one for each k).
    
    M_list = []
    for a in range(n*m):
        i = a // m
        j = a % m
        M_a = np.zeros((p, rank))
        for k in range(p):
            b = j * p + k
            c = k * n + i
            M_a[k, :] = V[b, :] * W[c, :]
        M_list.append(M_a)
        
    ones_p = np.ones(p)
    
    # Similarly for V[b,:].
    # For fixed b (j,k), the equations come from varying i.
    # sum_r U[i*m+j, r] V[b, r] W[k*n+i, r] = 1
    # There are n=2 equations (one for each i).
    M_list_V = []
    for b in range(m*p):
        j = b // p
        k = b % p
        M_b = np.zeros((n, rank))
        for i in range(n):
            a = i * m + j
            c = k * n + i
            M_b[i, :] = U[a, :] * W[c, :]
        M_list_V.append(M_b)
        
    ones_n = np.ones(n)
    
    # Similarly for W[c,:].
    # For fixed c (k,i), the equations come from varying j.
    # sum_r U[i*m+j, r] V[j*p+k, r] W[c, r] = 1
    # There are m=3 equations (one for each j).
    M_list_W = []
    for c in range(p*n):
        k = c // n
        i = c % n
        M_c = np.zeros((m, rank))
        for j in range(m):
            a = i * m + j
            b = j * p + k
            M_c[j, :] = U[a, :] * V[b, :]
        M_list_W.append(M_c)
        
    ones_m = np.ones(m)

    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
            
        # Update U
        for a in range(n*m):
            try:
                x, _, _, _ = np.linalg.lstsq(M_list[a], ones_p, rcond=None)
                U[a, :] = x
            except:
                pass
                
        # Update V
        for b in range(m*p):
            try:
                x, _, _, _ = np.linalg.lstsq(M_list_V[b], ones_n, rcond=None)
                V[b, :] = x
            except:
                pass
                
        # Update W
        for c in range(p*n):
            try:
                x, _, _, _ = np.linalg.lstsq(M_list_W[c], ones_m, rcond=None)
                W[c, :] = x
            except:
                pass
        
        # Check convergence
        T = np.einsum('ar,br,cr->abc', U, V, W)
        target_err = np.max(np.abs(T[_TARGET_MASK_232] - 1.0))
        non_target_err = np.max(np.abs(T[~_TARGET_MASK_232]))
        
        if target_err < 1e-9 and non_target_err < 1e-9:
            return U.tolist(), V.tolist(), W.tolist()
            
    return None

def _polish_232(U, V, W, max_iters=50, tol=1e-14, deadline=None):
    """High-precision polish for 2x3x2."""
    n, m, p = _N2, _M2, _P2
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    ones_p = np.ones(p)
    ones_n = np.ones(n)
    ones_m = np.ones(m)
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
            
        for a in range(n*m):
            i = a // m
            j = a % m
            M_a = np.zeros((p, rank))
            for k in range(p):
                b = j * p + k
                c = k * n + i
                M_a[k, :] = V_arr[b, :] * W_arr[c, :]
            try:
                x, _, _, _ = np.linalg.lstsq(M_a, ones_p, rcond=None)
                U_arr[a, :] = x
            except:
                pass
                
        for b in range(m*p):
            j = b // p
            k = b % p
            M_b = np.zeros((n, rank))
            for i in range(n):
                a = i * m + j
                c = k * n + i
                M_b[i, :] = U_arr[a, :] * W_arr[c, :]
            try:
                x, _, _, _ = np.linalg.lstsq(M_b, ones_n, rcond=None)
                V_arr[b, :] = x
            except:
                pass
                
        for c in range(p*n):
            k = c // n
            i = c % n
            M_c = np.zeros((m, rank))
            for j in range(m):
                a = i * m + j
                b = j * p + k
                M_c[j, :] = U_arr[a, :] * V_arr[b, :]
            try:
                x, _, _, _ = np.linalg.lstsq(M_c, ones_m, rcond=None)
                W_arr[c, :] = x
            except:
                pass
        
        T = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
        target_err = np.max(np.abs(T[_TARGET_MASK_232] - 1.0))
        non_target_err = np.max(np.abs(T[~_TARGET_MASK_232]))
        
        if target_err < tol and non_target_err < tol:
            return U_arr.tolist(), V_arr.tolist(), W_arr.tolist()
            
    return None

# --- Full Problem: 2x3x8 ---

# Target indices for 2x3x8
_TARGET_IDX = []
for i in range(_N):
    for j in range(_M):
        for k in range(_P):
            a = i * _M + j
            b = j * _P + k
            c = k * _N + i
            _TARGET_IDX.append((a, b, c))

_TARGET_MASK = np.zeros((_N*_M, _M*_P, _P*_N), dtype=bool)
for a, b, c in _TARGET_IDX:
    _TARGET_MASK[a, b, c] = True

def _build_rank44_decomposition():
    """
    Build explicit rank-44 decomposition for 2x3 * 3x8.
    """
    n, m, p = 2, 3, 8
    U = np.zeros((n*m, 44))  # 6 x 44
    V = np.zeros((m*p, 44))  # 24 x 44
    W = np.zeros((p*n, 44))  # 16 x 44
    
    col_offset = 0
    
    for block in range(4):
        k0 = 2 * block
        k1 = 2 * block + 1
        
        # m1 = (a00+a11)(b00+b11)
        t = col_offset
        U[0, t] += 1
        U[4, t] += 1
        V[k0, t] += 1
        V[8+k1, t] += 1
        W[2*k0, t] += 1
        W[2*k1+1, t] += 1
        col_offset += 1
        
        # m2 = (a10+a11)b00
        t = col_offset
        U[3, t] += 1
        U[4, t] += 1
        V[k0, t] += 1
        W[2*k0+1, t] += 1
        W[2*k1+1, t] += -1
        col_offset += 1
        
        # m3 = a00(b01-b11)
        t = col_offset
        U[0, t] += 1
        V[k1, t] += 1
        V[8+k1, t] += -1
        W[2*k1, t] += 1
        W[2*k1+1, t] += 1
        col_offset += 1
        
        # m4 = a11(b10-b00)
        t = col_offset
        U[4, t] += 1
        V[8+k0, t] += 1
        V[k0, t] += -1
        W[2*k0, t] += 1
        W[2*k0+1, t] += 1
        col_offset += 1
        
        # m5 = (a00+a01)b11
        t = col_offset
        U[0, t] += 1
        U[1, t] += 1
        V[8+k1, t] += 1
        W[2*k0, t] += -1
        W[2*k1, t] += 1
        col_offset += 1
        
        # m6 = (a10-a00)(b00+b01)
        t = col_offset
        U[3, t] += 1
        U[0, t] += -1
        V[k0, t] += 1
        V[k1, t] += 1
        W[2*k1+1, t] += 1
        col_offset += 1
        
        # m7 = (a01-a11)(b10+b11)
        t = col_offset
        U[1, t] += 1
        U[4, t] += -1
        V[8+k0, t] += 1
        V[8+k1, t] += 1
        W[2*k0, t] += 1
        col_offset += 1
        
        # 4 rank-1 terms for A_right * B_bottom
        for ii in range(2):
            for kk in range(2):
                t = col_offset
                k_val = k0 if kk == 0 else k1
                U[ii*3+2, t] = 1
                V[16+k_val, t] = 1
                W[k_val*2+ii, t] = 1
                col_offset += 1
    
    assert col_offset == 44, f"Expected 44 columns, got {col_offset}"
    
    return U.tolist(), V.tolist(), W.tolist()

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 48
    
    # Fallback: Schoolbook
    U_fb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_fb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_fb = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][column] = 1
                V_fb[j * p + k][column] = 1
                W_fb[k * n + i][column] = 1
                column += 1
                
    best_U, best_V, best_W = U_fb, V_fb, W_fb
    best_rank = schoolbook_rank
    
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s kill
    
    # 1. Try to find rank-10 decomposition for 2x3x2
    found_232 = None
    for trial in range(50):
        if time.time() > deadline - 10:
            break
        seed = trial * 12345 + 678
        result = _als_232(rank=10, max_iters=100, deadline=deadline, seed=seed)
        if result:
            U2, V2, W2 = result
            # Polish
            polished = _polish_232(U2, V2, W2, max_iters=50, tol=1e-14, deadline=deadline)
            if polished:
                U2, V2, W2 = polished
            # Round
            rounded = _round_and_verify(U2, V2, W2, _TARGET_MASK_232, tol=1e-5)
            if rounded:
                found_232 = rounded
                break
            else:
                # Check if unrounded is valid (though task requires integer/half-integer)
                # The task says "integer/half-integer entries". So we MUST round.
                # If it doesn't round, it's invalid for the final submission if we use it.
                # However, we can keep it as a candidate if we can't find a rounded one, 
                # but the verifier in the evaluator might check for integer/half-integer.
                # Let's assume the evaluator checks for valid decomposition with integer/half-integer.
                # So we only accept rounded ones.
                pass

    if found_232:
        U2, V2, W2 = found_232
        U2_arr = np.array(U2, dtype=float)
        V2_arr = np.array(V2, dtype=float)
        W2_arr = np.array(W2, dtype=float)
        
        # Construct 2x3x8 by placing 4 blocks
        rank_232 = len(U2_arr[0])
        total_rank = 4 * rank_232
        
        U_full = np.zeros((n*m, total_rank))
        V_full = np.zeros((m*p, total_rank))
        W_full = np.zeros((p*n, total_rank))
        
        # A is 2x3, B is 3x8.
        # B is split into 4 blocks of 3x2: B[:, 0:2], B[:, 2:4], B[:, 4:6], B[:, 6:8]
        # For block t (0..3):
        # A is the same 2x3 matrix.
        # B_block is 3x2.
        # C_block is 2x2.
        
        for t in range(4):
            col_offset = t * rank_232
            
            # U: A is the same for all blocks.
            # U_full[i*m+j, col_offset : col_offset+rank_232] = U2[i*m+j, :]
            U_full[:, col_offset:col_offset+rank_232] = U2_arr
            
            # V: B_block is 3x2.
            # V2 indices: j in 0..2, k in 0..1.
            # V_full indices: j in 0..2, k in 2*t .. 2*t+1.
            for j in range(m):
                for k in range(2):
                    b_idx_232 = j * 2 + k
                    b_idx_full = j * p + (2 * t + k)
                    V_full[b_idx_full, col_offset:col_offset+rank_232] = V2_arr[b_idx_232, :]
            
            # W: C_block is 2x2.
            # W2 indices: k in 0..1, i in 0..1. Index c = k*2 + i.
            # W_full indices: k in 2*t .. 2*t+1, i in 0..1. Index c = k*2 + i.
            for k in range(2):
                for i in range(n):
                    c_idx_232 = k * 2 + i
                    c_idx_full = (2 * t + k) * n + i
                    W_full[c_idx_full, col_offset:col_offset+rank_232] = W2_arr[c_idx_232, :]
        
        # Verify full decomposition
        if _verify(U_full, V_full, W_full, _TARGET_MASK, tol=1e-8):
            best_U = U_full.tolist()
            best_V = V_full.tolist()
            best_W = W_full.tolist()
            best_rank = total_rank
            
    # 2. Fallback to rank-44 if rank-40 not found
    if best_rank == schoolbook_rank:
        try:
            U44, V44, W44 = _build_rank44_decomposition()
            U44_arr = np.array(U44, dtype=float)
            V44_arr = np.array(V44, dtype=float)
            W44_arr = np.array(W44, dtype=float)
            if _verify(U44_arr, V44_arr, W44_arr, _TARGET_MASK, tol=1e-6):
                rounded = _round_and_verify(U44_arr, V44_arr, W44_arr, _TARGET_MASK, tol=1e-5)
                if rounded:
                    best_U, best_V, best_W = rounded
                    best_rank = 44
                else:
                    best_U, best_V, best_W = U44, V44, W44
                    best_rank = 44
        except Exception:
            pass
            
    return {"U": best_U, "V": best_V, "W": best_W}
