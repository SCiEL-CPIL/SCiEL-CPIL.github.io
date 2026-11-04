import numpy as np
import time

def construct():
    n, m, p = 2, 3, 9
    deadline = time.time() + 115  # safety margin before 150s kill

    # Build the full target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    # Schoolbook fallback: rank = n*m*p = 54
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1

    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb

    def to_list(mat):
        return [[float(x) for x in row] for row in mat]

    def calculate_error(T, U, V, W):
        T_rec = np.einsum('ar,br,cr->abc', U, V, W)
        return np.sum((T - T_rec)**2)

    def snap_and_verify(T, U, V, W, atol=1e-9):
        """
        Snaps U, V, W to nearest half-integers and verifies exactness.
        Returns (success, U_snapped, V_snapped, W_snapped)
        """
        U_s = np.round(U * 2) / 2
        V_s = np.round(V * 2) / 2
        W_s = np.round(W * 2) / 2
        
        T_rec = np.einsum('ar,br,cr->abc', U_s, V_s, W_s)
        if np.allclose(T, T_rec, atol=atol, rtol=1e-9):
            return True, U_s, V_s, W_s
        return False, None, None, None

    # ============================================================
    # Strategy: Block decomposition - 2x3x9 as 3 independent 2x3x3 blocks
    # ============================================================
    n2, m2, p2 = 2, 3, 3
    T2 = np.zeros((n2*m2, m2*p2, p2*n2))
    for i in range(n2):
        for j in range(m2):
            for k in range(p2):
                T2[i*m2+j, j*p2+k, k*n2+i] = 1.0

    def als_step_233(T2, U, V, W, r):
        """
        One ALS step for 2x3x3 tensor.
        """
        # Update U: for each row a, solve min ||T[a,:,:] - sum_r U[a,r] V[:,r] W[:,r]||
        T_flat = T2.reshape(n2*m2, m2*p2*p2*n2)
        M_VW = np.einsum('br,cr->bcr', V, W).reshape(m2*p2*p2*n2, r)
        
        try:
            U_new, _, _, _ = np.linalg.lstsq(M_VW, T_flat.T, rcond=None)
            U = U_new.T
        except:
            pass

        # Update V
        T_flat_V = T2.transpose(1, 0, 2).reshape(m2*p2, n2*m2*p2*n2)
        M_UW = np.einsum('ar,cr->acr', U, W).reshape(n2*m2*p2*n2, r)
        
        try:
            V_new, _, _, _ = np.linalg.lstsq(M_UW, T_flat_V.T, rcond=None)
            V = V_new.T
        except:
            pass

        # Update W
        T_flat_W = T2.transpose(2, 0, 1).reshape(p2*n2, n2*m2*m2*p2)
        M_UV = np.einsum('ar,br->abr', U, V).reshape(n2*m2*m2*p2, r)
        
        try:
            W_new, _, _, _ = np.linalg.lstsq(M_UV, T_flat_W.T, rcond=None)
            W = W_new.T
        except:
            pass

        return U, V, W

    def als_step_full(T, U, V, W, r):
        """
        One ALS step for full 2x3x9 tensor.
        """
        # Update U
        T_flat = T.reshape(n*m, m*p*p*n)
        M_VW = np.einsum('br,cr->bcr', V, W).reshape(m*p*p*n, r)
        
        try:
            U_new, _, _, _ = np.linalg.lstsq(M_VW, T_flat.T, rcond=None)
            U = U_new.T
        except:
            pass

        # Update V
        T_flat_V = T.transpose(1, 0, 2).reshape(m*p, n*m*p*n)
        M_UW = np.einsum('ar,cr->acr', U, W).reshape(n*m*p*n, r)
        
        try:
            V_new, _, _, _ = np.linalg.lstsq(M_UW, T_flat_V.T, rcond=None)
            V = V_new.T
        except:
            pass

        # Update W
        T_flat_W = T.transpose(2, 0, 1).reshape(p*n, n*m*m*p)
        M_UV = np.einsum('ar,br->abr', U, V).reshape(n*m*m*p, r)
        
        try:
            W_new, _, _, _ = np.linalg.lstsq(M_UV, T_flat_W.T, rcond=None)
            W = W_new.T
        except:
            pass

        return U, V, W

    def build_full_from_block(U2, V2, W2, r_block):
        """
        Build full 2x3x9 decomposition from a 2x3x3 block decomposition.
        """
        r_full = r_block * 3
        U_full = np.zeros((n*m, r_full))
        V_full = np.zeros((m*p, r_full))
        W_full = np.zeros((p*n, r_full))

        # U is same for all 3 blocks (depends only on i,j)
        U_full[:, 0:r_block] = U2
        U_full[:, r_block:2*r_block] = U2
        U_full[:, 2*r_block:3*r_block] = U2

        # V depends on (j,k): row index = j*9+k
        for j in range(m):
            for k in range(p):
                v_row = j * p + k
                block = k // p2
                k_local = k % p2
                v_233_row = j * p2 + k_local
                col_start = block * r_block
                V_full[v_row, col_start:col_start+r_block] = V2[v_233_row, :]

        # W depends on (k,i): row index = k*2+i
        for k in range(p):
            for i in range(n):
                w_row = k * n + i
                block = k // p2
                k_local = k % p2
                w_233_row = k_local * n + i
                col_start = block * r_block
                W_full[w_row, col_start:col_start+r_block] = W2[w_233_row, :]

        return U_full, V_full, W_full, r_full

    def polish_als(T, U, V, W, r, n_iters=50, atol=1e-10):
        """
        Final polish pass with tighter tolerance.
        """
        for _ in range(n_iters):
            if time.time() > deadline - 15:
                break
            U, V, W = als_step_full(T, U, V, W, r)
            error = calculate_error(T, U, V, W)
            if error < 1e-10:
                snap_success, U_s, V_s, W_s = snap_and_verify(T, U, V, W, atol=1e-10)
                if snap_success:
                    return True, U_s, V_s, W_s
        return False, U, V, W

    # Try ranks for 2x3x3: schoolbook is 18, known lower bound ~7-8
    # The optimal rank for 2x3x3 is believed to be around 11-14
    for r_block in range(11, 19):
        if time.time() > deadline - 25:
            break
        r_full = r_block * 3
        if r_full >= best_rank:
            break

        # Try multiple random initializations for 2x3x3
        for trial in range(300):
            if time.time() > deadline - 25:
                break

            np.random.seed(42 + r_block * 1000 + trial)
            U2 = np.random.randn(n2*m2, r_block)
            V2 = np.random.randn(m2*p2, r_block)
            W2 = np.random.randn(p2*n2, r_block)

            # ALS iterations for 2x3x3
            success = False
            for iteration in range(200):
                if time.time() > deadline - 25:
                    break

                U2, V2, W2 = als_step_233(T2, U2, V2, W2, r_block)
                
                error = calculate_error(T2, U2, V2, W2)
                
                if error < 1e-6:
                    snap_success, U2_s, V2_s, W2_s = snap_and_verify(T2, U2, V2, W2)
                    if snap_success:
                        U2, V2, W2 = U2_s, V2_s, W2_s
                        success = True
                        break
                elif error > 1e4:
                    break

            if success:
                # Build full 2x3x9 decomposition from 3 blocks
                U_full, V_full, W_full, r_full = build_full_from_block(U2, V2, W2, r_block)

                # Verify the full decomposition
                full_error = calculate_error(T, U_full, V_full, W_full)

                if full_error < 1e-8:
                    best_rank = r_full
                    best_U, best_V, best_W = U_full, V_full, W_full
                    break

        if best_rank == r_full:
            break

    # If we found a good rank, try to improve with direct ALS on full tensor
    if best_rank < rank_sb and time.time() < deadline - 35:
        r_try = best_rank - 1
        if r_try >= 30:  # Reasonable lower bound
            for trial in range(80):
                if time.time() > deadline - 15:
                    break
                
                np.random.seed(1000 + trial)
                U = np.random.randn(n*m, r_try)
                V = np.random.randn(m*p, r_try)
                W = np.random.randn(p*n, r_try)
                
                success = False
                for iteration in range(150):
                    if time.time() > deadline - 15:
                        break
                    
                    U, V, W = als_step_full(T, U, V, W, r_try)
                    error = calculate_error(T, U, V, W)
                    
                    if error < 1e-6:
                        snap_success, U_s, V_s, W_s = snap_and_verify(T, U, V, W)
                        if snap_success:
                            best_rank = r_try
                            best_U, best_V, best_W = U_s, V_s, W_s
                            success = True
                            break
                    elif error > 1e4:
                        break
                
                if success:
                    break

    # Also try a few more ranks between the found rank and schoolbook
    if best_rank < rank_sb and time.time() < deadline - 20:
        for r_try in range(best_rank - 1, max(best_rank - 10, 30), -1):
            if r_try >= best_rank:
                continue
            if time.time() > deadline - 15:
                break
            
            for trial in range(50):
                if time.time() > deadline - 15:
                    break
                
                np.random.seed(2000 + r_try * 100 + trial)
                U = np.random.randn(n*m, r_try)
                V = np.random.randn(m*p, r_try)
                W = np.random.randn(p*n, r_try)
                
                success = False
                for iteration in range(100):
                    if time.time() > deadline - 15:
                        break
                    
                    U, V, W = als_step_full(T, U, V, W, r_try)
                    error = calculate_error(T, U, V, W)
                    
                    if error < 1e-6:
                        snap_success, U_s, V_s, W_s = snap_and_verify(T, U, V, W)
                        if snap_success:
                            best_rank = r_try
                            best_U, best_V, best_W = U_s, V_s, W_s
                            success = True
                            break
                    elif error > 1e4:
                        break
                
                if success:
                    break
            if best_rank == r_try:
                break

    # Final polish pass on the best solution found
    if best_rank < rank_sb and time.time() < deadline - 10:
        polish_success, U_p, V_p, W_p = polish_als(T, best_U, best_V, best_W, best_rank)
        if polish_success:
            best_U, best_V, best_W = U_p, V_p, W_p

    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
