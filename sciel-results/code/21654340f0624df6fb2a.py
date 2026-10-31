import time
import numpy as np

DIMENSIONS = [4, 4, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS

    # Strassen coefficients for 2x2 matrix multiplication
    A_coeff = np.zeros((7, 2, 2), dtype=int)
    B_coeff = np.zeros((7, 2, 2), dtype=int)
    C_coeff = np.zeros((7, 2, 2), dtype=int)

    # s=0: M0 = (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
    A_coeff[0, 0, 0] = 1; A_coeff[0, 1, 1] = 1
    B_coeff[0, 0, 0] = 1; B_coeff[0, 1, 1] = 1
    C_coeff[0, 0, 0] = 1; C_coeff[0, 1, 1] = 1

    # s=1: M1 = (A10+A11)B00 -> C10 += 1, C11 -= 1
    A_coeff[1, 1, 0] = 1; A_coeff[1, 1, 1] = 1
    B_coeff[1, 0, 0] = 1
    C_coeff[1, 1, 0] = 1; C_coeff[1, 1, 1] = -1

    # s=2: M2 = A00(B01-B11) -> C01 += 1, C11 += 1
    A_coeff[2, 0, 0] = 1
    B_coeff[2, 0, 1] = 1; B_coeff[2, 1, 1] = -1
    C_coeff[2, 0, 1] = 1; C_coeff[2, 1, 1] = 1

    # s=3: M3 = A11(B10-B00) -> C00 += 1, C10 += 1
    A_coeff[3, 1, 1] = 1
    B_coeff[3, 1, 0] = 1; B_coeff[3, 0, 0] = -1
    C_coeff[3, 0, 0] = 1; C_coeff[3, 1, 0] = 1

    # s=4: M4 = (A00+A01)B11 -> C00 -= 1, C01 += 1
    A_coeff[4, 0, 0] = 1; A_coeff[4, 0, 1] = 1
    B_coeff[4, 1, 1] = 1
    C_coeff[4, 0, 0] = -1; C_coeff[4, 0, 1] = 1

    # s=5: M5 = (A10-A00)(B00+B01) -> C11 += 1
    A_coeff[5, 1, 0] = 1; A_coeff[5, 0, 0] = -1
    B_coeff[5, 0, 0] = 1; B_coeff[5, 0, 1] = 1
    C_coeff[5, 1, 1] = 1

    # s=6: M6 = (A01-A11)(B10+B11) -> C00 += 1
    A_coeff[6, 0, 1] = 1; A_coeff[6, 1, 1] = -1
    B_coeff[6, 1, 0] = 1; B_coeff[6, 1, 1] = 1
    C_coeff[6, 0, 0] = 1

    # Build all 49 Strassen⊗Strassen terms
    U_full = np.zeros((16, 49), dtype=float)
    V_full = np.zeros((16, 49), dtype=float)
    W_full = np.zeros((16, 49), dtype=float)

    for s in range(7):
        for t in range(7):
            r = 7 * s + t

            for I in range(2):
                for J in range(2):
                    ca_out = A_coeff[s, I, J]
                    if ca_out == 0: continue
                    for i2 in range(2):
                        for j2 in range(2):
                            ca_in = A_coeff[t, i2, j2]
                            if ca_in == 0: continue
                            i = 2 * I + i2
                            j = 2 * J + j2
                            U_full[i * 4 + j, r] = ca_out * ca_in

            for J in range(2):
                for K in range(2):
                    cb_out = B_coeff[s, J, K]
                    if cb_out == 0: continue
                    for j2 in range(2):
                        for k2 in range(2):
                            cb_in = B_coeff[t, j2, k2]
                            if cb_in == 0: continue
                            j = 2 * J + j2
                            k = 2 * K + k2
                            V_full[j * 4 + k, r] = cb_out * cb_in

            for I in range(2):
                for K in range(2):
                    cc_out = C_coeff[s, I, K]
                    if cc_out == 0: continue
                    for i2 in range(2):
                        for k2 in range(2):
                            cc_in = C_coeff[t, i2, k2]
                            if cc_in == 0: continue
                            i = 2 * I + i2
                            k = 2 * K + k2
                            W_full[k * 4 + i, r] = cc_out * cc_in

    # Verify 49-term works
    np.random.seed(123)
    A_test = np.random.randn(4, 4)
    B_test = np.random.randn(4, 4)
    C_test = A_test @ B_test

    C_49 = np.zeros((4, 4))
    for i in range(4):
        for k in range(4):
            total = 0.0
            for j in range(4):
                for r in range(49):
                    total += U_full[i*4+j, r] * V_full[j*4+k, r] * W_full[k*4+i, r]
            C_49[i, k] = total

    if not np.allclose(C_49, C_test, atol=1e-9):
        return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}

    # Check for linear dependencies among the 49 terms
    # Flatten each term r into a vector of length 4096
    # T_r[i, j, k] = U_full[i*4+j, r] * V_full[j*4+k, r] * W_full[k*4+i, r]
    
    terms = np.zeros((49, 4096), dtype=float)
    
    for r in range(49):
        u_r = U_full[:, r]
        v_r = V_full[:, r]
        w_r = W_full[:, r]
        
        # Reshape to 4x4 for easier indexing
        # u_r index: i*4+j
        # v_r index: j*4+k
        # w_r index: k*4+i
        
        # We can compute the outer product structure
        # But 4096 is small, so direct loop is fine
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    terms[r, i*16 + j*4 + k] = u_r[i*4+j] * v_r[j*4+k] * w_r[k*4+i]

    # Compute rank of the 49x4096 matrix
    # Use SVD to find rank and null space
    U_svd, s_svd, Vh_svd = np.linalg.svd(terms, full_matrices=False)
    
    # Determine numerical rank
    tol = 1e-10
    rank = np.sum(s_svd > tol)
    
    if rank < 49:
        # There are linear dependencies
        # The null space is spanned by the last (49-rank) columns of Vh_svd (transposed)
        # Actually, Vh_svd is (4096, 49). The null space vectors are rows of Vh_svd corresponding to zero singular values.
        # Wait, SVD: A = U S Vh. A is (49, 4096).
        # U is (49, 49). S is (49,). Vh is (49, 4096).
        # The null space of A (vectors x such that A x = 0) is spanned by the last columns of V?
        # No, A is (49, 4096). We want x in R^49 such that sum x_r T_r = 0.
        # This is A x = 0? No, A is the matrix of terms.
        # If we treat T_r as columns, M is (4096, 49).
        # We want x such that M x = 0.
        # SVD of M: M = U_m S_m V_mh.
        # M is (4096, 49).
        # U_m is (4096, 49). S_m is (49,). V_mh is (49, 49).
        # The null space of M is spanned by the last columns of V_m (which is V_mh.T).
        
        # Let's re-run SVD on the transposed matrix for clarity
        M = terms.T # (4096, 49)
        U_m, s_m, Vh_m = np.linalg.svd(M, full_matrices=False)
        
        tol = 1e-10
        rank_m = np.sum(s_m > tol)
        
        if rank_m < 49:
            # Null space vectors are in V_mh.T
            V_m = Vh_m.T
            # The last (49 - rank_m) columns of V_m span the null space
            null_vecs = V_m[:, rank_m:]
            
            # We want to reduce the rank.
            # Pick a null vector x.
            # Identify an index k where x[k] is significant.
            # We can eliminate term k.
            # New terms for r != k: u_r' = u_r + x[r]/x[k] * u_k ?
            # No, T_k = - sum_{r!=k} (x[r]/x[k]) T_r.
            # T = sum_{r!=k} T_r + T_k = sum_{r!=k} T_r - sum_{r!=k} (x[r]/x[k]) T_r = sum_{r!=k} (1 - x[r]/x[k]) T_r.
            # This is a sum of 48 terms.
            
            x = null_vecs[:, 0]
            # Find k with max |x[k]|
            k = np.argmax(np.abs(x))
            if abs(x[k]) < tol:
                return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
            
            U_new = np.zeros((16, 48), dtype=float)
            V_new = np.zeros((16, 48), dtype=float)
            W_new = np.zeros((16, 48), dtype=float)
            
            idx = 0
            for r in range(49):
                if r == k:
                    continue
                coeff = 1.0 - x[r] / x[k]
                U_new[:, idx] = coeff * U_full[:, r]
                V_new[:, idx] = coeff * V_full[:, r]
                W_new[:, idx] = coeff * W_full[:, r]
                idx += 1
            
            # Verify new decomposition
            C_new = np.zeros((4, 4))
            for i in range(4):
                for kk in range(4):
                    total = 0.0
                    for j in range(4):
                        for r in range(48):
                            total += U_new[i*4+j, r] * V_new[j*4+kk, r] * W_new[kk*4+i, r]
                    C_new[i, kk] = total
            
            if np.allclose(C_new, C_test, atol=1e-9):
                return {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}
            else:
                # Fallback to 49 if verification fails
                return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}

    # No linear dependencies found (rank 49)
    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
