import numpy as np
import time
from itertools import combinations

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

    # Build rank-49 decomposition via Kronecker product of Strassen factors
    i_vals = np.repeat(np.arange(4), 4)
    j_vals = np.tile(np.arange(4), 4)
    i_hi = (i_vals // 2).astype(int)
    i_lo = (i_vals % 2).astype(int)
    j_hi = (j_vals // 2).astype(int)
    j_lo = (j_vals % 2).astype(int)

    u_hi_vecs = U2[i_hi * 2 + j_hi]
    u_lo_vecs = U2[i_lo * 2 + j_lo]
    u_outer = u_hi_vecs[:, :, np.newaxis] * u_lo_vecs[:, np.newaxis, :]
    u_mat = u_outer.reshape(16, 49)

    j_vals_v = np.repeat(np.arange(4), 4)
    k_vals_v = np.tile(np.arange(4), 4)
    j_hi_v = (j_vals_v // 2).astype(int)
    j_lo_v = (j_vals_v % 2).astype(int)
    k_hi_v = (k_vals_v // 2).astype(int)
    k_lo_v = (k_vals_v % 2).astype(int)

    v_hi_vecs = V2[j_hi_v * 2 + k_hi_v]
    v_lo_vecs = V2[j_lo_v * 2 + k_lo_v]
    v_outer = v_hi_vecs[:, :, np.newaxis] * v_lo_vecs[:, np.newaxis, :]
    v_mat = v_outer.reshape(16, 49)

    k_vals_w = np.repeat(np.arange(4), 4)
    i_vals_w = np.tile(np.arange(4), 4)
    k_hi_w = (k_vals_w // 2).astype(int)
    k_lo_w = (k_vals_w % 2).astype(int)
    i_hi_w = (i_vals_w // 2).astype(int)
    i_lo_w = (i_vals_w % 2).astype(int)

    w_hi_vecs = W2[k_hi_w * 2 + i_hi_w]
    w_lo_vecs = W2[k_lo_w * 2 + i_lo_w]
    w_outer = w_hi_vecs[:, :, np.newaxis] * w_lo_vecs[:, np.newaxis, :]
    w_mat = w_outer.reshape(16, 49)

    # Verify correctness of rank-49
    valid = True
    for i in range(4):
        for j in range(4):
            for k in range(4):
                prod = np.sum(u_mat[i * 4 + j] * v_mat[j * 4 + k] * w_mat[k * 4 + i])
                if prod != 1:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break

    if not valid or time.time() - start_time > deadline:
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

    # Search for rank-48 decomposition
    # Target tensor T
    T = np.zeros(64, dtype=int)
    for i in range(4):
        for j in range(4):
            for k in range(4):
                T[k * 16 + i * 4 + j] = 1  # Indexing: T[k, i, j] corresponds to (i, j, k) in U, V, W?
                # Wait, the trace convention is W[k*n+i, r].
                # The term is U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
                # So for a given (i, j, k), the contribution is to T[i, j, k].
                # Let's flatten T in order (i, j, k).
                # Index = i * 16 + j * 4 + k.
                # Let's re-verify the index in the previous loop.
                # prod = sum(u_mat[i*4+j] * v_mat[j*4+k] * w_mat[k*4+i])
                # This sum is for the coefficient of (i, j, k).
                # So T[i, j, k] should be 1.
                pass

    T = np.zeros(64, dtype=int)
    for i in range(4):
        for j in range(4):
            for k in range(4):
                T[i * 16 + j * 4 + k] = 1

    # Pre-compute the 49 term vectors
    terms = np.zeros((49, 64), dtype=int)
    for c in range(49):
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    terms[c, i * 16 + j * 4 + k] = u_mat[i * 4 + j, c] * v_mat[j * 4 + k, c] * w_mat[k * 4 + i, c]

    # Check if any term is zero
    nonzero_terms = [c for c in range(49) if np.any(terms[c] != 0)]
    
    # Try to remove one term
    for c_to_remove in range(49):
        if time.time() - start_time > deadline:
            break
            
        remaining = [c for c in range(49) if c != c_to_remove]
        
        # Construct matrix M where columns are the remaining terms
        M = terms[remaining]  # Shape (48, 64)
        
        # We want to solve M.T @ lambda = T
        # M.T is 64x48. T is 64x1.
        # This is an overdetermined system.
        try:
            # Use least squares to find lambda
            # If the residual is zero, we have a solution.
            lambda_sol, residuals, rank_M, s = np.linalg.lstsq(M.T, T, rcond=None)
            
            # Check residual
            if residuals.size > 0 and residuals[0] < 1e-9:
                # Verify exactness
                recon = M.T @ lambda_sol
                if np.allclose(recon, T, atol=1e-9):
                    # Check if lambda_sol has rational/integer values?
                    # The problem allows integer/half-integer entries.
                    # If lambda_sol are not nice, we might still accept them if they are exact rationals.
                    # But the output format requires integer/half-integer.
                    # Let's check if they are close to integers or half-integers.
                    valid_lambda = True
                    for val in lambda_sol:
                        # Check if val is close to an integer or half-integer
                        if abs(val - round(val)) > 1e-6 and abs(val - (round(val*2)/2.0)) > 1e-6:
                            valid_lambda = False
                            break
                    if valid_lambda:
                        # Construct the new factors
                        new_U = u_mat[:, remaining]
                        new_V = v_mat[:, remaining]
                        new_W = w_mat[:, remaining]
                        
                        # Scale the columns by lambda_sol
                        for idx, lam in enumerate(lambda_sol):
                            new_U[:, idx] *= lam
                            new_V[:, idx] *= lam
                            new_W[:, idx] *= lam
                        
                        # Verify again with the scaled factors
                        valid_final = True
                        for i in range(4):
                            for j in range(4):
                                for k in range(4):
                                    prod = np.sum(new_U[i * 4 + j] * new_V[j * 4 + k] * new_W[k * 4 + i])
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
        "U": u_mat.tolist(),
        "V": v_mat.tolist(),
        "W": w_mat.tolist()
    }
