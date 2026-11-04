import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    nm, mp, pn = n * m, m * p, p * n

    # Build target tensor indices
    nonzero_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                nonzero_entries.append((a, b, c))
    
    nz_set = set(nonzero_entries)
    
    # Schoolbook fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1
                    V[b, col] = 1
                    W[c, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    fallback = schoolbook()

    # Fast verification using vectorized operations
    def verify_fast(U, V, W, tol=1e-5):
        # U: (nm, r), V: (mp, r), W: (pn, r)
        # M_ab_r = U[a, r] * V[b, r]
        M = U[:, np.newaxis, :] * V[np.newaxis, :, :]
        # R_ab_c = sum_r M_ab_r * W[c, r]
        R = np.einsum('abr,cr->abc', M, W)
        
        a_idx, b_idx, c_idx = zip(*nonzero_entries)
        err_nz = np.max(np.abs(R[a_idx, b_idx, c_idx] - 1.0))
        
        # Create mask for zero entries
        T_mask = np.zeros((nm, mp, pn), dtype=bool)
        T_mask[a_idx, b_idx, c_idx] = True
        err_zero = np.max(np.abs(R[~T_mask]))
        
        max_err = max(err_nz, err_zero)
        return max_err <= tol, max_err

    def try_rank(r, max_restarts=3, max_iters=30):
        if time.time() > deadline:
            return None
            
        # Pre-compute SVD initializations
        # Unfoldings
        # T1: (nm, mp*pn)
        # T2: (mp, nm*pn)
        # T3: (pn, nm*mp)
        
        # Construct sparse T unfoldings efficiently
        # T1[a, b*pn + c] = 1 if (a,b,c) in nz
        T1 = np.zeros((nm, mp * pn))
        T2 = np.zeros((mp, nm * pn))
        T3 = np.zeros((pn, nm * mp))
        
        for (a, b, c) in nonzero_entries:
            T1[a, b * pn + c] = 1.0
            T2[b, a * pn + c] = 1.0
            T3[c, a * mp + b] = 1.0
            
        U_svd, _, _ = np.linalg.svd(T1, full_matrices=False)
        V_svd, _, _ = np.linalg.svd(T2, full_matrices=False)
        W_svd, _, _ = np.linalg.svd(T3, full_matrices=False)
        
        for attempt in range(max_restarts):
            if time.time() > deadline:
                return None
                
            # Initialize
            U = U_svd[:, :r].copy()
            V = V_svd[:, :r].copy()
            W = W_svd[:, :r].copy()
            
            # Add small noise to break symmetry
            U += np.random.randn(nm, r) * 0.01
            V += np.random.randn(mp, r) * 0.01
            W += np.random.randn(pn, r) * 0.01
            
            # ALS Iterations
            for it in range(max_iters):
                if time.time() > deadline:
                    return None
                    
                # Update U
                for a in range(nm):
                    # Equations: sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c]
                    # A * U[a,:] = b_vec
                    # Rows of A correspond to (b,c) pairs
                    # A[eq, r] = V[b, r] * W[c, r]
                    # b_vec[eq] = T[a, b, c]
                    
                    # Build A and b_vec
                    # Number of equations: mp * pn = 30 * 24 = 720
                    # But we only care about non-zeros? No, zeros are constraints too.
                    # However, for speed, we can use all.
                    
                    # Vectorized construction of A
                    # V is (mp, r), W is (pn, r)
                    # We need outer product V[b,:] * W[c,:] for all b,c
                    # This is (mp, pn, r)
                    # Reshape to (mp*pn, r)
                    A = (V[:, np.newaxis, :] * W[np.newaxis, :, :]).reshape(mp * pn, r)
                    # b_vec is T1[a, :] reshaped
                    b_vec = T1[a, :]
                    
                    # Solve least squares
                    U[a, :] = np.linalg.lstsq(A, b_vec, rcond=None)[0]
                    
                # Update V
                for b in range(mp):
                    A = (U[:, np.newaxis, :] * W[np.newaxis, :, :]).reshape(nm * pn, r)
                    b_vec = T2[b, :]
                    V[b, :] = np.linalg.lstsq(A, b_vec, rcond=None)[0]
                    
                # Update W
                for c in range(pn):
                    A = (U[:, np.newaxis, :] * V[np.newaxis, :, :]).reshape(nm * mp, r)
                    b_vec = T3[c, :]
                    W[c, :] = np.linalg.lstsq(A, b_vec, rcond=None)[0]
                    
            valid, err = verify_fast(U, V, W)
            if valid:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
        return None

    best_result = None
    
    # Try ranks from 119 down to 100
    # 120 is schoolbook.
    for r in range(119, 99, -1):
        if time.time() > deadline - 10:
            break
        result = try_rank(r)
        if result is not None:
            best_result = result
            break
            
    if best_result is None:
        best_result = fallback
        
    return best_result
