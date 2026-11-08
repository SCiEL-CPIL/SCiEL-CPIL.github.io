import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    def verify_full(U, V, W, tol=1e-10):
        """Verify full 2x3 x 3x6 decomposition."""
        n, m, p = DIMENSIONS
        # Check all non-zero entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1) > tol:
                        return False
        # Check zero entries - sample
        np.random.seed(42)
        for _ in range(500):
            a = np.random.randint(0, n*m)
            b = np.random.randint(0, m*p)
            c = np.random.randint(0, p*n)
            i, j = divmod(a, m)
            j2, k = divmod(b, p)
            k2, i2 = divmod(c, n)
            if not (j == j2 and i == i2 and k == k2):
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(val) > tol:
                    return False
        return True
    
    def verify_sub(U_sub, V_sub, W_sub, tol=1e-10):
        """Verify 2x3 x 3x2 sub-decomposition.
        U_sub: (6, r), V_sub: (6, r), W_sub: (4, r)
        W_sub[k*2+i, r] for k in {0,1}, i in {0,1}
        """
        n_sub, m_sub, p_sub = 2, 3, 2
        # Check all non-zero entries
        for i in range(n_sub):
            for j in range(m_sub):
                for k in range(p_sub):
                    val = np.sum(U_sub[i*m_sub+j, :] * V_sub[j*p_sub+k, :] * W_sub[k*n_sub+i, :])
                    if abs(val - 1) > tol:
                        return False
        # Check zero entries
        np.random.seed(42)
        for _ in range(200):
            a = np.random.randint(0, n_sub*m_sub)
            b = np.random.randint(0, m_sub*p_sub)
            c = np.random.randint(0, p_sub*n_sub)
            i, j = divmod(a, m_sub)
            j2, k = divmod(b, p_sub)
            k2, i2 = divmod(c, n_sub)
            if not (j == j2 and i == i2 and k == k2):
                val = np.sum(U_sub[a, :] * V_sub[b, :] * W_sub[c, :])
                if abs(val) > tol:
                    return False
        return True
    
    # Schoolbook rank 36
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n*m, rank_schoolbook))
    V_sb = np.zeros((m*p, rank_schoolbook))
    W_sb = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()
    best_rank = rank_schoolbook
    
    # Try to find rank-11 decomposition for 2x3 x 3x2
    # Then assemble 3-block decomposition for rank 33
    
    n_sub, m_sub, p_sub = 2, 3, 2
    target_sub_rank = 11
    
    U_sub_best = None
    V_sub_best = None
    W_sub_best = None
    
    # Try random search for sub-decomposition
    for attempt in range(200):
        if time.time() > deadline - 30:
            break
        
        np.random.seed(attempt * 7919 + 42)
        
        r = target_sub_rank
        U_sub = np.random.randn(n_sub*m_sub, r) * 0.5
        V_sub = np.random.randn(m_sub*p_sub, r) * 0.5
        
        # Solve for W_sub: for each (a,b,c) with T=1:
        # sum_r U_sub[a,r]*V_sub[b,r]*W_sub[c,r] = 1
        # For each (a,b,c) with T=0:
        # sum_r U_sub[a,r]*V_sub[b,r]*W_sub[c,r] = 0
        
        # Build linear system for W_sub
        num_nonzero = n_sub * m_sub * p_sub  # 12
        # Use all 12 nonzero + sample of zeros
        num_zero_sample = 100
        
        A_sys = np.zeros((num_nonzero + num_zero_sample, p_sub*n_sub*r))
        b_sys = np.zeros(num_nonzero + num_zero_sample)
        
        idx = 0
        for i in range(n_sub):
            for j in range(m_sub):
                for k in range(p_sub):
                    a = i*m_sub+j
                    b = j*p_sub+k
                    c = k*n_sub+i
                    for r_idx in range(r):
                        A_sys[idx, c*r+r_idx] = U_sub[a, r_idx] * V_sub[b, r_idx]
                    b_sys[idx] = 1
                    idx += 1
        
        # Add zero constraints
        zero_indices = []
        for a in range(n_sub*m_sub):
            for b in range(m_sub*p_sub):
                for c in range(p_sub*n_sub):
                    i, j = divmod(a, m_sub)
                    j2, k = divmod(b, p_sub)
                    k2, i2 = divmod(c, n_sub)
                    if not (j == j2 and i == i2 and k == k2):
                        zero_indices.append((a, b, c))
        
        np.random.seed(attempt * 13 + 7)
        if len(zero_indices) > num_zero_sample:
            sample_idx = np.random.choice(len(zero_indices), num_zero_sample, replace=False)
            zero_sample = [zero_indices[i] for i in sample_idx]
        else:
            zero_sample = zero_indices
        
        for (a, b, c) in zero_sample:
            for r_idx in range(r):
                A_sys[idx, c*r+r_idx] = U_sub[a, r_idx] * V_sub[b, r_idx]
            b_sys[idx] = 0
            idx += 1
        
        try:
            W_flat, residuals, rank_sys, sv = np.linalg.lstsq(A_sys, b_sys, rcond=None)
            W_sub = W_flat.reshape(p_sub*n_sub, r)
            
            if verify_sub(U_sub, V_sub, W_sub):
                U_sub_best = U_sub.copy()
                V_sub_best = V_sub.copy()
                W_sub_best = W_sub.copy()
                print(f"Found rank-{r} sub-decomposition at attempt {attempt}!")
                break
        except:
            pass
    
    if U_sub_best is not None:
        # Assemble 3-block decomposition
        # U: (6, 33) = [U_sub, U_sub, U_sub]
        # V: (18, 33) where for block b, V_sub[j, k_sub] maps to V_full[j*p + b*p_sub + k_sub]
        # W: (12, 33) where for block b, W_sub[k_sub, i] maps to W_full[(b*p_sub + k_sub)*n + i]
        
        rank_total = 3 * target_sub_rank
        U_full = np.zeros((n*m, rank_total))
        V_full = np.zeros((m*p, rank_total))
        W_full = np.zeros((p*n, rank_total))
        
        for block in range(3):
            col_offset = block * target_sub_rank
            
            # U is same for all blocks
            U_full[:, col_offset:col_offset+target_sub_rank] = U_sub_best
            
            # V: block b uses B columns b*p_sub and b*p_sub+1
            for j in range(m_sub):
                for k_sub in range(p_sub):
                    v_row = j * p + (block * p_sub + k_sub)
                    v_sub_idx = j * p_sub + k_sub
                    V_full[v_row, col_offset:col_offset+target_sub_rank] = V_sub_best[v_sub_idx, :]
            
            # W: block b uses C columns b*p_sub and b*p_sub+1
            for k_sub in range(p_sub):
                for i in range(n_sub):
                    w_row = (block * p_sub + k_sub) * n + i
                    w_sub_idx = k_sub * n_sub + i
                    W_full[w_row, col_offset:col_offset+target_sub_rank] = W_sub_best[w_sub_idx, :]
        
        if verify_full(U_full, V_full, W_full):
            best_U, best_V, best_W = U_full.copy(), V_full.copy(), W_full.copy()
            best_rank = rank_total
            print(f"Assembled rank-{rank_total} full decomposition!")
        else:
            print("Assembled decomposition verification failed!")
    
    # Try to find even lower rank with random search
    for r in range(25, best_rank):
        if time.time() > deadline - 10:
            break
        
        for attempt in range(30):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(attempt * 31 + r * 1000)
            
            U_r = np.random.randn(n*m, r) * 0.5
            V_r = np.random.randn(m*p, r) * 0.5
            
            # Solve for W
            num_nonzero = n * m * p  # 36
            num_zero_sample = 150
            
            A_sys = np.zeros((num_nonzero + num_zero_sample, p*n*r))
            b_sys = np.zeros(num_nonzero + num_zero_sample)
            
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i*m+j
                        b = j*p+k
                        c = k*n+i
                        for r_idx in range(r):
                            A_sys[idx, c*r+r_idx] = U_r[a, r_idx] * V_r[b, r_idx]
                        b_sys[idx] = 1
                        idx += 1
            
            zero_indices = []
            for a in range(n*m):
                for b in range(m*p):
                    for c in range(p*n):
                        i, j = divmod(a, m)
                        j2, k = divmod(b, p)
                        k2, i2 = divmod(c, n)
                        if not (j == j2 and i == i2 and k == k2):
                            zero_indices.append((a, b, c))
            
            np.random.seed(attempt * 17 + r * 500)
            if len(zero_indices) > num_zero_sample:
                sample_idx = np.random.choice(len(zero_indices), num_zero_sample, replace=False)
                zero_sample = [zero_indices[i] for i in sample_idx]
            else:
                zero_sample = zero_indices
            
            for (a, b, c) in zero_sample:
                for r_idx in range(r):
                    A_sys[idx, c*r+r_idx] = U_r[a, r_idx] * V_r[b, r_idx]
                b_sys[idx] = 0
                idx += 1
            
            try:
                W_flat, residuals, rank_sys, sv = np.linalg.lstsq(A_sys, b_sys, rcond=None)
                W_r = W_flat.reshape(p*n, r)
                
                if verify_full(U_r, V_r, W_r):
                    best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
                    best_rank = r
                    print(f"Found rank-{r} full decomposition!")
                    break
            except:
                pass
    
    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
