import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 118
    
    n, m, p = 3, 4, 6
    
    # First, build the rank-66 baseline
    U66 = [[0] * 66 for _ in range(n * m)]
    V66 = [[0] * 66 for _ in range(m * p)]
    W66 = [[0] * 66 for _ in range(p * n)]
    
    col = 0
    def add_strassen(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        A00 = a_r * m + a_c
        A01 = a_r * m + a_c + 1
        A10 = (a_r + 1) * m + a_c
        A11 = (a_r + 1) * m + a_c + 1
        B00 = b_r * p + b_c
        B01 = b_r * p + b_c + 1
        B10 = (b_r + 1) * p + b_c
        B11 = (b_r + 1) * p + b_c + 1
        C00 = c_c * n + c_r
        C01 = (c_c + 1) * n + c_r
        C10 = c_c * n + (c_r + 1)
        C11 = (c_c + 1) * n + (c_r + 1)
        
        U66[A00][col] += 1; U66[A11][col] += 1
        V66[B00][col] += 1; V66[B11][col] += 1
        W66[C00][col] += 1; W66[C11][col] += 1
        col += 1
        U66[A10][col] += 1; U66[A11][col] += 1
        V66[B00][col] += 1
        W66[C10][col] += 1; W66[C11][col] -= 1
        col += 1
        U66[A00][col] += 1
        V66[B01][col] += 1; V66[B11][col] -= 1
        W66[C01][col] += 1; W66[C11][col] += 1
        col += 1
        U66[A11][col] += 1
        V66[B10][col] += 1; V66[B00][col] -= 1
        W66[C00][col] += 1; W66[C10][col] += 1
        col += 1
        U66[A00][col] += 1; U66[A01][col] += 1
        V66[B11][col] += 1
        W66[C00][col] -= 1; W66[C01][col] += 1
        col += 1
        U66[A10][col] += 1; U66[A00][col] -= 1
        V66[B00][col] += 1; V66[B01][col] += 1
        W66[C11][col] += 1
        col += 1
        U66[A01][col] += 1; U66[A11][col] -= 1
        V66[B10][col] += 1; V66[B11][col] += 1
        W66[C00][col] += 1
        col += 1
    
    def add_1x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        A0 = a_r * m + a_c
        A1 = a_r * m + a_c + 1
        B00 = b_r * p + b_c
        B01 = b_r * p + b_c + 1
        B10 = (b_r + 1) * p + b_c
        B11 = (b_r + 1) * p + b_c + 1
        C0 = c_c * n + c_r
        C1 = (c_c + 1) * n + c_r
        
        U66[A0][col] += 1; V66[B00][col] += 1; W66[C0][col] += 1
        col += 1
        U66[A1][col] += 1; V66[B10][col] += 1; W66[C0][col] += 1
        col += 1
        U66[A0][col] += 1; V66[B01][col] += 1; W66[C1][col] += 1
        col += 1
        U66[A1][col] += 1; V66[B11][col] += 1; W66[C1][col] += 1
    
    add_1x2_2x2(0, 0, 0, 0, 0, 0)
    add_1x2_2x2(0, 2, 2, 0, 0, 0)
    add_1x2_2x2(0, 0, 0, 2, 0, 2)
    add_1x2_2x2(0, 2, 2, 2, 0, 2)
    add_1x2_2x2(0, 0, 0, 4, 0, 4)
    add_1x2_2x2(0, 2, 2, 4, 0, 4)
    add_strassen(1, 0, 0, 0, 1, 0)
    add_strassen(1, 2, 2, 0, 1, 0)
    add_strassen(1, 0, 0, 2, 1, 2)
    add_strassen(1, 2, 2, 2, 1, 2)
    add_strassen(1, 0, 0, 4, 1, 4)
    add_strassen(1, 2, 2, 4, 1, 4)
    
    U_np = np.array(U66, dtype=float)
    V_np = np.array(V66, dtype=float)
    W_np = np.array(W66, dtype=float)
    
    # Build the full multiplication tensor T[i,j,k] = delta_{j,k}
    # We'll try rank-1 peeling: find a rank-1 tensor that is "forced" by the structure
    
    # The multiplication tensor has T[i,j,k] = 1 if j==k, else 0
    # For rank-1 peeling, we look for vectors u, v, w such that
    # u[i]*v[j]*w[k] is "uniquely determined" by the tensor constraints
    
    # Strategy: The tensor is sparse. If we fix certain entries, we can determine
    # what rank-1 components must exist.
    
    # Try a different approach: use the null space method more aggressively
    # Build tensor matrix for all 66 columns
    tensor_matrix = np.zeros((n * m * p, 66))
    for l in range(66):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    tensor_matrix[i * m * p + j * p + k, l] = \
                        U_np[i * m + j, l] * V_np[j * p + k, l] * W_np[k * n + i, l]
    
    # Try multiple elimination strategies
    best_result = None
    best_rank = 66
    
    # Strategy 1: Direct null space elimination (as before but more aggressive)
    try:
        U_svd, s, Vt = np.linalg.svd(tensor_matrix, full_matrices=False)
        tol = s[0] * max(tensor_matrix.shape) * np.finfo(float).eps
        rank_tensor = int(np.sum(s > tol))
        
        if rank_tensor < 66:
            num_null = 66 - rank_tensor
            null_space = Vt[rank_tensor:, :]
            
            # Try all combinations of null vectors for iterative elimination
            U_cur = U_np.copy()
            V_cur = V_np.copy()
            W_cur = W_np.copy()
            active = list(range(66))
            
            max_iter = 10
            for iteration in range(max_iter):
                if time.time() > deadline - 20:
                    break
                if len(active) <= 1:
                    break
                    
                # Rebuild tensor matrix for active columns
                tensor_active = np.zeros((n * m * p, len(active)))
                for idx, l in enumerate(active):
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                tensor_active[i * m * p + j * p + k, idx] = \
                                    U_cur[i * m + j, l] * V_cur[j * p + k, l] * W_cur[k * n + i, l]
                
                # SVD to find null space
                try:
                    Ua, sa, Vta = np.linalg.svd(tensor_active, full_matrices=False)
                    tol_a = sa[0] * max(tensor_active.shape) * np.finfo(float).eps
                    rank_a = int(np.sum(sa > tol_a))
                except:
                    break
                
                if rank_a >= len(active):
                    break  # No more null vectors
                
                nulls_a = Vta[rank_a:, :]
                eliminated = False
                
                for nidx in range(nulls_a.shape[0]):
                    if time.time() > deadline - 10:
                        break
                    v = nulls_a[nidx, :]
                    abs_v = np.abs(v)
                    candidates_a = np.where(abs_v > 1e-10)[0]
                    
                    # Try each candidate for elimination
                    best_elim_idx = None
                    best_err = 1e9
                    
                    for elim_idx in candidates_a:
                        if time.time() > deadline - 5:
                            break
                        elim_col = active[elim_idx]
                        
                        # Build new active set
                        new_active = [l for idx, l in enumerate(active) if idx != elim_idx]
                        new_rank = len(new_active)
                        
                        U_new = np.zeros((n * m, new_rank))
                        V_new = np.zeros((m * p, new_rank))
                        W_new = np.zeros((p * n, new_rank))
                        
                        new_col = 0
                        valid = True
                        for idx, l in enumerate(active):
                            if idx == elim_idx:
                                continue
                            scale = 1.0 - v[idx] / v[elim_idx]
                            if abs(scale) < 1e-15:
                                valid = False
                                break
                            U_new[:, new_col] = U_cur[:, l]
                            V_new[:, new_col] = V_cur[:, l]
                            W_new[:, new_col] = W_cur[:, l] * scale
                            new_col += 1
                        
                        if not valid or new_col != new_rank:
                            continue
                        
                        # Verify
                        err = 0.0
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    s_val = 0.0
                                    for l in range(new_rank):
                                        s_val += U_new[i * m + j, l] * V_new[j * p + k, l] * W_new[k * n + i, l]
                                    expected = 1.0 if j == k else 0.0
                                    err = max(err, abs(s_val - expected))
                        
                        if err < 1e-9 and err < best_err:
                            best_err = err
                            best_elim_idx = elim_idx
                            U_new_best = U_new.copy()
                            V_new_best = V_new.copy()
                            W_new_best = W_new.copy()
                            new_active_best = new_active.copy()
                    
                    if best_elim_idx is not None:
                        U_cur = U_new_best
                        V_cur = V_new_best
                        W_cur = W_new_best
                        active = new_active_best
                        eliminated = True
                        if len(active) < best_rank:
                            best_rank = len(active)
                            best_result = (U_cur.copy(), V_cur.copy(), W_cur.copy())
                        break
                if not eliminated:
                    break
            
            if best_result is not None:
                U_out = best_result[0].tolist()
                V_out = best_result[1].tolist()
                W_out = best_result[2].tolist()
                return {"U": U_out, "V": V_out, "W": W_out}
    except Exception:
        pass
    
    # Strategy 2: Try to find a rank-1 peeling directly from the tensor structure
    # The tensor T[i,j,k] = delta_{j,k}
    # If we can find u, v, w such that u[i]*v[j]*w[k] matches a subset of entries,
    # we can peel it off
    
    # For the 3x4x6 case, let's try to identify forced rank-1 terms
    # by looking at the support pattern
    
    # The tensor has support only when j==k
    # So T[i,j,j] = 1 for all i,j
    
    # A rank-1 tensor u⊗v⊗w has T[i,j,k] = u[i]*v[j]*w[k]
    # For this to be a "peelable" component, we need it to be uniquely determined
    
    # Let's try: suppose w is supported only on some indices
    # If w[k] != 0 only for k in S, then the rank-1 term only affects entries with k in S
    
    # Try peeling with w supported on single index
    for k0 in range(p):
        if time.time() > deadline - 15:
            break
        # w[k0] = 1, w[other] = 0
        # Then T[i,j,k0] = u[i]*v[j]*1 for j=k0, else 0
        # So we need u[i]*v[k0] = 1 for all i (from T[i,k0,k0]=1)
        # This forces u[i] = 1/v[k0] for all i, and v[k0] = 1
        # But then for j != k0, T[i,j,k0] = u[i]*v[j]*1 = (1/v[k0])*v[j] = v[j]
        # But T[i,j,k0] should be 0 for j != k0, so v[j] = 0 for j != k0
        # This gives a rank-1 term: u[i]=1, v[k0]=1, w[k0]=1, others 0
        # This is just the "diagonal" term for (k0,k0)
        
        # This is trivial and doesn't help reduce rank below 66
        
        pass
    
    # Strategy 3: Use a different block decomposition that might have lower rank
    # Try: split A as 1+2, B as 2+2, C as 2+2+2 (different from before)
    # Or try other partitions
    
    # For now, return the best we found
    if best_result is not None:
        U_out = best_result[0].tolist()
        V_out = best_result[1].tolist()
        W_out = best_result[2].tolist()
        return {"U": U_out, "V": V_out, "W": W_out}
    
    # Return the rank-66 decomposition
    return {"U": U66, "V": V66, "W": W66}
