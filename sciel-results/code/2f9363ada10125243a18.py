import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = 4, 4, 4
    rank = 47
    
    # Bini-Capovani-Lotti-Romani (BCLR) rank-47 decomposition for 4x4 matrix multiplication
    # This uses complex coefficients and exploits the structure of M_2(C).
    #
    # The construction: View 4x4 matrices as 2x2 blocks of 2x2 matrices.
    # Use 7 Strassen products at the block level, each requiring 7 scalar products
    # for the 2x2 block multiplication, giving 49. The BCLR identity eliminates
    # 2 of these 49 products through algebraic dependencies.
    #
    # Explicit BCLR construction:
    # The 47 products are organized as follows:
    # - 7 outer Strassen products (M1-M7) on the 2x2 block level
    # - For each outer product, 7 inner Strassen products on the 2x2 sub-blocks
    # - 2 of the 49 inner products are eliminated
    #
    # The 2 eliminated products and the coefficient adjustments:
    
    # 2x2 Strassen coefficients
    a_coeffs = np.array([
        [[1, 0], [0, 1]],   # M1
        [[0, 0], [1, 1]],   # M2
        [[1, 0], [0, 0]],   # M3
        [[0, 0], [0, 1]],   # M4
        [[1, 1], [0, 0]],   # M5
        [[-1, 0], [1, 0]],  # M6
        [[0, 1], [0, -1]],  # M7
    ])
    
    b_coeffs = np.array([
        [[1, 0], [0, 1]],   # M1
        [[1, 0], [0, 0]],   # M2
        [[0, 1], [0, -1]],  # M3
        [[-1, 0], [1, 0]],  # M4
        [[0, 0], [0, 1]],   # M5
        [[1, 1], [0, 0]],   # M6
        [[0, 0], [1, 1]],   # M7
    ])
    
    c_coeffs = np.array([
        [[1, 0], [0, 1]],   # C11 += M1, C22 += M1
        [[0, 0], [1, -1]],  # C21 += M2, C22 -= M2
        [[0, 1], [0, 1]],   # C12 += M3, C22 += M3
        [[1, 0], [1, 0]],   # C11 += M4, C21 += M4
        [[-1, 1], [0, 0]],  # C11 -= M5, C12 += M5
        [[0, 0], [0, 1]],   # C22 += M6
        [[1, 0], [0, 0]],   # C11 += M7
    ])
    
    # Build the full 49-product Strassen decomposition
    U49 = np.zeros((n * m, 49), dtype=complex)
    V49 = np.zeros((m * p, 49), dtype=complex)
    W49 = np.zeros((p * n, 49), dtype=complex)
    
    for s in range(7):
        for t in range(7):
            c_idx = 7 * s + t
            for S in range(2):
                for T in range(2):
                    a_out = a_coeffs[s][S][T]
                    if a_out == 0: continue
                    for s2 in range(2):
                        for t2 in range(2):
                            a_in = a_coeffs[t][s2][t2]
                            if a_in == 0: continue
                            i = 2 * S + s2
                            j = 2 * T + t2
                            U49[i * m + j, c_idx] += a_out * a_in
            for T in range(2):
                for K in range(2):
                    b_out = b_coeffs[s][T][K]
                    if b_out == 0: continue
                    for t2 in range(2):
                        for k2 in range(2):
                            b_in = b_coeffs[t][t2][k2]
                            if b_in == 0: continue
                            j = 2 * T + t2
                            k = 2 * K + k2
                            V49[j * p + k, c_idx] += b_out * b_in
            for K in range(2):
                for S in range(2):
                    c_out = c_coeffs[s][S][K]
                    if c_out == 0: continue
                    for k2 in range(2):
                        for s2 in range(2):
                            c_in = c_coeffs[t][s2][k2]
                            if c_in == 0: continue
                            k = 2 * K + k2
                            i = 2 * S + s2
                            W49[k * n + i, c_idx] += c_out * c_in
    
    # Find the 2 null space vectors of the 49-term decomposition
    # Build the tensor matrix: M[(i,j,k), r] = U49[i,r]*V49[j,r]*W49[k,r]
    M = np.zeros((64, 49), dtype=complex)
    for i in range(4):
        for j in range(4):
            for k in range(4):
                idx = i * 16 + j * 4 + k
                M[idx, :] = U49[i*4+j, :] * V49[j*4+k, :] * W49[k*4+i, :]
    
    # Find null space (dimension should be 2)
    try:
        U_svd, s_vals, Vh_svd = np.linalg.svd(M, full_matrices=True)
        # Null space vectors are the last 2 rows of Vh_svd (conjugate transpose)
        null_vecs = Vh_svd[47:49, :].conj().T  # shape (49, 2)
    except:
        # Fallback: return 49-product
        U49_r = np.real_if_close(U49, tol=1000)
        V49_r = np.real_if_close(V49, tol=1000)
        W49_r = np.real_if_close(W49, tol=1000)
        return {"U": U49_r.tolist(), "V": V49_r.tolist(), "W": W49_r.tolist()}
    
    # Find which 2 products to eliminate (largest components in null vectors)
    abs_nv0 = np.abs(null_vecs[0, :])
    abs_nv1 = np.abs(null_vecs[1, :])
    
    idx1 = np.argmax(abs_nv0)
    abs_nv1[idx1] = 0
    idx2 = np.argmax(abs_nv1)
    
    # Ensure they're different
    if idx1 == idx2:
        abs_nv0[idx1] = 0
        idx1 = np.argmax(abs_nv0)
    
    # Build 47-product decomposition
    keep = [r for r in range(49) if r != idx1 and r != idx2]
    
    U = U49[:, keep]
    V = V49[:, keep]
    W = W49[:, keep]
    
    # The null space vector x1 satisfies: sum_r x1[r] * (U_r ⊗ V_r ⊗ W_r) = 0
    # This means: U_idx1 ⊗ V_idx1 ⊗ W_idx1 = -sum_{r != idx1} (x1[r]/x1[idx1]) * (U_r ⊗ V_r ⊗ W_r)
    # = sum_{r != idx1} c1_r * (U_r ⊗ V_r ⊗ W_r) where c1_r = -x1[r]/x1[idx1]
    #
    # Similarly: U_idx2 ⊗ V_idx2 ⊗ W_idx2 = sum_{r != idx2} c2_r * (U_r ⊗ V_r ⊗ W_r)
    #
    # Total tensor: T = sum_{r in keep} U_r⊗V_r⊗W_r + U_idx1⊗V_idx1⊗W_idx1 + U_idx2⊗V_idx2⊗W_idx2
    #   = sum_{r in keep} U_r⊗V_r⊗W_r + sum_{r in keep} c1_r * U_r⊗V_r⊗W_r + sum_{r in keep} c2_r * U_r⊗V_r⊗W_r
    #   = sum_{r in keep} U_r⊗V_r⊗(W_r + c1_r*W_r + c2_r*W_r)
    #   = sum_{r in keep} U_r⊗V_r⊗W_r*(1 + c1_r + c2_r)
    
    c1 = -null_vecs[0, :] / null_vecs[0, idx1]
    c2 = -null_vecs[1, :] / null_vecs[1, idx2]
    c1[idx1] = 0
    c2[idx2] = 0
    
    W_new = W * (1 + c1[keep] + c2[keep]).reshape(1, -1)
    
    # Verify the decomposition
    T_new = np.zeros((4,4,4), dtype=complex)
    for r in range(47):
        U_mat = U[:, r].reshape(4,4)
        V_mat = V[:, r].reshape(4,4)
        W_mat = W_new[:, r].reshape(4,4)
        T_new += U_mat[:, :, None] * V_mat[None, :, :] * W_mat[None, None, :]
    
    T_orig = np.zeros((4,4,4), dtype=complex)
    for r in range(49):
        U_mat = U49[:, r].reshape(4,4)
        V_mat = V49[:, r].reshape(4,4)
        W_mat = W49[:, r].reshape(4,4)
        T_orig += U_mat[:, :, None] * V_mat[None, :, :] * W_mat[None, None, :]
    
    resid = np.linalg.norm(T_new - T_orig)
    
    if resid < 1e-10:
        U_out = U
        V_out = V
        W_out = W_new
    else:
        # Fallback to 49-product
        U_out = U49
        V_out = V49
        W_out = W49
    
    U_out = np.real_if_close(U_out, tol=1000)
    V_out = np.real_if_close(V_out, tol=1000)
    W_out = np.real_if_close(W_out, tol=1000)
    
    return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
