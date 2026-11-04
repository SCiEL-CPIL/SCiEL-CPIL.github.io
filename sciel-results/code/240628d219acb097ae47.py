import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 110.0

    # Strassen rank-7 factors for 2x2
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=int)

    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=int)

    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=int)

    # Build rank-49 decomposition via Kronecker product with correct permutation
    U4_kron = np.kron(U2, U2)  # shape (16, 49)
    V4_kron = np.kron(V2, V2)  # shape (16, 49)
    W4_kron = np.kron(W2, W2)  # shape (16, 49)

    # Permutation: map standard row-major index S to Kronecker index L4
    # For U (A matrix): S = r*4 + c, r = r_out*2 + r_in, c = c_out*2 + c_in
    # L2_out = r_out*2 + c_out, L2_in = r_in*2 + c_in
    # L4 = L2_out * 4 + L2_in
    perm_U = np.zeros(16, dtype=int)
    for S in range(16):
        r = S // 4
        c = S % 4
        r_out, r_in = r // 2, r % 2
        c_out, c_in = c // 2, c % 2
        L2_out = r_out * 2 + c_out
        L2_in = r_in * 2 + c_in
        perm_U[S] = L2_out * 4 + L2_in

    # For V (B matrix): S = j*4 + k, j = j_out*2 + j_in, k = k_out*2 + k_in
    perm_V = np.zeros(16, dtype=int)
    for S in range(16):
        j = S // 4
        k = S % 4
        j_out, j_in = j // 2, j % 2
        k_out, k_in = k // 2, k % 2
        L2_out = j_out * 2 + k_out
        L2_in = j_in * 2 + k_in
        perm_V[S] = L2_out * 4 + L2_in

    # For W (C matrix): S = k*4 + i, k = k_out*2 + k_in, i = i_out*2 + i_in
    perm_W = np.zeros(16, dtype=int)
    for S in range(16):
        k = S // 4
        i = S % 4
        k_out, k_in = k // 2, k % 2
        i_out, i_in = i // 2, i % 2
        L2_out = k_out * 2 + i_out
        L2_in = k_in * 2 + i_in
        perm_W[S] = L2_out * 4 + L2_in

    U_mat = U4_kron[perm_U, :]
    V_mat = V4_kron[perm_V, :]
    W_mat = W4_kron[perm_W, :]

    # Verify correctness of rank-49
    valid = True
    for i in range(4):
        for j in range(4):
            for k in range(4):
                prod = np.sum(U_mat[i * 4 + j] * V_mat[j * 4 + k] * W_mat[k * 4 + i])
                if prod != 1:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break

    if not valid or time.time() - start_time > deadline:
        # Fallback: naive rank-64
        n, m, p = 4, 4, 4
        rank = n * m * p
        U_f = [[0] * rank for _ in range(n * m)]
        V_f = [[0] * rank for _ in range(m * p)]
        W_f = [[0] * rank for _ in range(p * n)]
        col = 0
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    U_f[ii * m + jj][col] = 1
                    V_f[jj * p + kk][col] = 1
                    W_f[kk * n + ii][col] = 1
                    col += 1
        return {"U": U_f, "V": V_f, "W": W_f}

    # Pre-compute term vectors for rank reduction search
    # term_s(i,j,k) = U_mat[i*4+j, s] * V_mat[j*4+k, s] * W_mat[k*4+i, s]
    # We want sum_s term_s(i,j,k) = 1 for all (i,j,k)
    
    # Build the 49 x 64 terms matrix
    terms = np.zeros((49, 64), dtype=float)
    for s in range(49):
        for idx in range(64):
            i = idx // 16
            j = (idx // 4) % 4
            k = idx % 4
            terms[s, idx] = U_mat[i*4+j, s] * V_mat[j*4+k, s] * W_mat[k*4+i, s]
    
    T = np.ones(64, dtype=float)
    
    # Try to find rank-48 by removing one term and solving
    norms = np.linalg.norm(terms, axis=1)
    order = np.argsort(norms)
    
    for c_to_remove in order:
        if time.time() - start_time > deadline:
            break
        
        remaining = [c for c in range(49) if c != c_to_remove]
        M = terms[remaining]  # Shape (48, 64)
        
        try:
            lambda_sol, residuals, rank_M, s_vals = np.linalg.lstsq(M.T, T, rcond=None)
            recon = M.T @ lambda_sol
            if np.max(np.abs(recon - T)) < 1e-9:
                # Check if lambda values are integers or half-integers
                valid_lambda = True
                for val in lambda_sol:
                    val_round = round(val)
                    val_half = round(val * 2) / 2.0
                    if abs(val - val_round) > 1e-6 and abs(val - val_half) > 1e-6:
                        valid_lambda = False
                        break
                
                if valid_lambda:
                    new_U = U_mat[:, remaining].copy().astype(float)
                    new_V = V_mat[:, remaining].copy().astype(float)
                    new_W = W_mat[:, remaining].copy().astype(float)
                    
                    for idx, lam in enumerate(lambda_sol):
                        new_U[:, idx] *= lam
                        new_V[:, idx] *= lam
                        new_W[:, idx] *= lam
                    
                    # Final verification
                    valid_final = True
                    for i in range(4):
                        for j in range(4):
                            for k in range(4):
                                prod = np.dot(new_U[i*4+j], new_V[j*4+k] * new_W[k*4+i])
                                if abs(prod - 1) > 1e-6:
                                    valid_final = False
                                    break
                            if not valid_final:
                                break
                        if not valid_final:
                            break
                    
                    if valid_final:
                        return {
                            "U": new_U.tolist(),
                            "V": new_V.tolist(),
                            "W": new_W.tolist()
                        }
        except Exception:
            pass
    
    # If no rank-48 found, return rank-49
    return {
        "U": U_mat.tolist(),
        "V": V_mat.tolist(),
        "W": W_mat.tolist()
    }
