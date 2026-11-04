import numpy as np
import time

def construct():
    n, m, p = 2, 3, 9
    deadline = time.time() + 110

    def to_list(mat):
        return [[float(x) for x in row] for row in mat]

    # Schoolbook fallback
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

    # Build target tensor for verification
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    def verify(U, V, W):
        """Check if U, V, W give exact decomposition of T"""
        # T_rec[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        T_rec = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_rec, T, atol=1e-10)

    def verify_error(U, V, W):
        T_rec = np.einsum('ar,br,cr->abc', U, V, W)
        return np.sum((T_rec - T)**2)

    # Strategy: Use known rank-8 decomposition of 2x3x3
    # The 2x3x3 multiplication tensor has rank 8
    # For 2x3x9 = 3 blocks of 2x3x3, naive gives rank 24
    # But we can potentially do better
    
    # Let's try to construct an explicit rank-8 decomposition for 2x3x3
    # Then replicate for 2x3x9 (giving rank 24)
    # Then try to find a direct lower-rank decomposition of 2x3x9
    
    # First, let's try ALS with better initialization and more aggressive search
    def als_step(T, U, V, W, n, m, p, r):
        nm, mp, pn = n*m, m*p, p*n
        # Update U
        M_VW = np.einsum('br,cr->bcr', V, W).reshape(mp*pn, r)
        T_flat = T.reshape(nm, mp*pn)
        try:
            U_new, _, _, _ = np.linalg.lstsq(M_VW, T_flat.T, rcond=None)
            U = U_new.T
        except:
            pass

        # Update V
        M_UW = np.einsum('ar,cr->acr', U, W).reshape(nm*pn, r)
        T_flat_V = T.transpose(1, 0, 2).reshape(mp, nm*pn)
        try:
            V_new, _, _, _ = np.linalg.lstsq(M_UW, T_flat_V.T, rcond=None)
            V = V_new.T
        except:
            pass

        # Update W
        M_UV = np.einsum('ar,br->abr', U, V).reshape(nm*mp, r)
        T_flat_W = T.transpose(2, 0, 1).reshape(pn, nm*mp)
        try:
            W_new, _, _, _ = np.linalg.lstsq(M_UV, T_flat_W.T, rcond=None)
            W = W_new.T
        except:
            pass

        return U, V, W

    # Try direct search on full 2x3x9 with various ranks
    # Start from rank 24 (3*8) and go down
    for r in range(24, rank_sb):
        if time.time() > deadline - 30:
            break
        if r >= best_rank:
            continue

        found = False
        # Try multiple random initializations
        for trial in range(200):
            if time.time() > deadline - 30:
                break

            np.random.seed(42 + r * 1000 + trial)
            U = np.random.randn(n*m, r) * 0.1
            V = np.random.randn(m*p, r) * 0.1
            W = np.random.randn(p*n, r) * 0.1

            # Run ALS with convergence check
            prev_error = np.inf
            for iteration in range(1000):
                if time.time() > deadline - 30:
                    break

                U, V, W = als_step(T, U, V, W, n, m, p, r)
                error = verify_error(U, V, W)

                if error < 1e-12:
                    found = True
                    break
                elif error > prev_error * 1.5 and iteration > 10:
                    # Diverging, break early
                    break
                prev_error = error

            if found:
                best_rank = r
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                break

        if found:
            break

    # If we found rank 24, try to see if we can do better with structured initialization
    # Use the block structure: 2x3x9 = 3 copies of 2x3x3
    if best_rank >= 24:
        # Try structured initialization based on 2x3x3 blocks
        # Initialize with block structure and refine
        r_struct = 24
        U_struct = np.zeros((n*m, r_struct))
        V_struct = np.zeros((m*p, r_struct))
        W_struct = np.zeros((p*n, r_struct))

        # For each block b (0,1,2), use a random rank-8-ish init for 2x3x3
        for b in range(3):
            col_start = b * 8
            # Random init for this block
            U_block = np.random.randn(n*m, 8) * 0.1
            V_block = np.random.randn(m*3, 8) * 0.1
            W_block = np.random.randn(3*n, 8) * 0.1

            # Place into full matrices
            U_struct[:, col_start:col_start+8] = U_block

            # V: row j*p+k, block b covers k in [3b, 3b+2]
            for j in range(m):
                for k_local in range(3):
                    k = 3*b + k_local
                    v_row = j*p + k
                    v2_row = j*3 + k_local
                    V_struct[v_row, col_start:col_start+8] = V_block[v2_row, :]

            # W: row k*n+i, block b covers k in [3b, 3b+2]
            for k_local in range(3):
                k = 3*b + k_local
                for i in range(n):
                    w_row = k*n + i
                    w2_row = k_local*n + i
                    W_struct[w_row, col_start:col_start+8] = W_block[w2_row, :]

        # Refine with ALS
        U_ref, V_ref, W_ref = U_struct, V_struct, W_struct
        prev_error = np.inf
        for iteration in range(500):
            if time.time() > deadline - 10:
                break
            U_ref, V_ref, W_ref = als_step(T, U_ref, V_ref, W_ref, n, m, p, r_struct)
            error = verify_error(U_ref, V_ref, W_ref)
            if error < 1e-12:
                if r_struct < best_rank:
                    best_rank = r_struct
                    best_U, best_V, best_W = U_ref.copy(), V_ref.copy(), W_ref.copy()
                break
            elif error > prev_error * 1.5 and iteration > 10:
                break
            prev_error = error

    # Final: try a few more direct searches at lower ranks if time permits
    if best_rank > 20 and time.time() < deadline - 20:
        for r in range(20, min(best_rank, 24)):
            if time.time() > deadline - 10:
                break
            if r >= best_rank:
                continue

            found = False
            for trial in range(100):
                if time.time() > deadline - 10:
                    break

                np.random.seed(999 + r * 1000 + trial)
                U = np.random.randn(n*m, r) * 0.1
                V = np.random.randn(m*p, r) * 0.1
                W = np.random.randn(p*n, r) * 0.1

                prev_error = np.inf
                for iteration in range(800):
                    if time.time() > deadline - 10:
                        break
                    U, V, W = als_step(T, U, V, W, n, m, p, r)
                    error = verify_error(U, V, W)
                    if error < 1e-12:
                        found = True
                        break
                    elif error > prev_error * 1.5 and iteration > 10:
                        break
                    prev_error = error

                if found:
                    best_rank = r
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    break

            if found:
                break

    if time.time() > deadline - 5:
        return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}

    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
