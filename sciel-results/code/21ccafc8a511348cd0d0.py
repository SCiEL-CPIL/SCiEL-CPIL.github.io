"""Tensor rank minimization for 2x5x6 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 105.0

    # Build rank-54 base: 6 Strassen 2x2x2 blocks (42) + 12 standard terms
    def build_base():
        R = 54
        U = np.zeros((n * m, R), dtype=float)
        V = np.zeros((m * p, R), dtype=float)
        W = np.zeros((p * n, R), dtype=float)
        c = 0

        def put(u_pairs, v_pairs, w_pairs):
            nonlocal c
            for idx, val in u_pairs:
                U[idx, c] += val
            for idx, val in v_pairs:
                V[idx, c] += val
            for idx, val in w_pairs:
                W[idx, c] += val
            c += 1

        def strassen(ja, jb, ka, kb):
            ua0 = 0 * m + ja
            ub0 = 0 * m + jb
            ua1 = 1 * m + ja
            ub1 = 1 * m + jb
            va_ka = ja * p + ka
            va_kb = ja * p + kb
            vb_ka = jb * p + ka
            vb_kb = jb * p + kb
            wk0_0 = ka * n + 0
            wk0_1 = ka * n + 1
            wk1_0 = kb * n + 0
            wk1_1 = kb * n + 1

            put([(ua0, 1), (ub1, 1)], [(va_ka, 1), (vb_kb, 1)], [(wk0_0, 1), (wk1_1, 1)])
            put([(ua1, 1), (ub1, 1)], [(va_ka, 1)], [(wk0_1, 1), (wk1_1, -1)])
            put([(ua0, 1)], [(va_kb, 1), (vb_kb, -1)], [(wk1_0, 1), (wk1_1, 1)])
            put([(ub1, 1)], [(vb_ka, 1), (va_ka, -1)], [(wk0_0, 1), (wk0_1, 1)])
            put([(ua0, 1), (ub0, 1)], [(vb_kb, 1)], [(wk0_0, -1), (wk1_0, 1)])
            put([(ua1, 1), (ua0, -1)], [(va_ka, 1), (va_kb, 1)], [(wk1_1, 1)])
            put([(ub0, 1), (ub1, -1)], [(vb_ka, 1), (vb_kb, 1)], [(wk0_0, 1)])

        for (ja, jb) in [(0, 1), (2, 3)]:
            for (ka, kb) in [(0, 1), (2, 3), (4, 5)]:
                strassen(ja, jb, ka, kb)

        for k in range(p):
            for i in range(n):
                U[i * m + 4, c] = 1.0
                V[4 * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1

        return U, V, W, c

    U, V, W, R = build_base()

    def verify(Um, Vm, Wm, Rm):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = float(np.dot(Um[i * m + j, :Rm] * Vm[j * p + k, :Rm], Wm[k * n + i, :Rm]))
                    if abs(s - 1.0) > 1e-10:
                        return False
        return True

    if not verify(U, V, W, R):
        Uf = np.zeros((n * m, n * m * p), dtype=float)
        Vf = np.zeros((m * p, n * m * p), dtype=float)
        Wf = np.zeros((p * n, n * m * p), dtype=float)
        cc = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    Uf[i * m + j, cc] = 1.0
                    Vf[j * p + k, cc] = 1.0
                    Wf[k * n + i, cc] = 1.0
                    cc += 1
        return {"U": Uf.tolist(), "V": Vf.tolist(), "W": Wf.tolist()}

    best_rank = R
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    # Try to reduce rank by finding a column that is in the "span" of others
    # in the tensor sense. For each column r, check if we can write
    # T_r = sum_{s != r} alpha_s T_s for some coefficients.
    # This is a system of n*m*p = 60 equations in R-1 unknowns.
    # The equation for (i,j,k) is:
    #   U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = sum_{s!=r} alpha_s * U[i*m+j,s]*V[j*p+k,s]*W[k*n+i,s]
    # This is linear in alpha_s!
    
    for col_to_remove in range(R):
        if time.time() > deadline - 10.0:
            break
        
        # Build the linear system for alpha_s (s != col_to_remove)
        # A[eq, s] = U[i*m+j,s] * V[j*p+k,s] * W[k*n+i,s] for s != col_to_remove
        # b[eq] = U[i*m+j,col] * V[j*p+k,col] * W[k*n+i,col]
        
        other_cols = [s for s in range(R) if s != col_to_remove]
        A = np.zeros((n * m * p, R - 1), dtype=float)
        b = np.zeros(n * m * p, dtype=float)
        
        eq = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    b[eq] = U[u_idx, col_to_remove] * V[v_idx, col_to_remove] * W[w_idx, col_to_remove]
                    for s_idx, s in enumerate(other_cols):
                        A[eq, s_idx] = U[u_idx, s] * V[v_idx, s] * W[w_idx, s]
                    eq += 1
        
        # Solve least squares
        try:
            alpha, residuals, rank_sys, sv = np.linalg.lstsq(A, b, rcond=None)
        except:
            continue
        
        # Check if the residual is small
        residual = np.linalg.norm(A @ alpha - b)
        if residual < 1e-8:
            # We can remove this column!
            new_R = R - 1
            new_U = np.zeros((n * m, new_R), dtype=float)
            new_V = np.zeros((m * p, new_R), dtype=float)
            new_W = np.zeros((p * n, new_R), dtype=float)
            
            new_col = 0
            for s in other_cols:
                new_U[:, new_col] = U[:, s]
                new_V[:, new_col] = V[:, s]
                new_W[:, new_col] = W[:, s]
                new_col += 1
            
            # Verify
            if verify(new_U, new_V, new_W, new_R):
                best_rank = new_R
                best_U, best_V, best_W = new_U, new_V, new_W
                R = new_R
                U, V, W = new_U, new_V, new_W
                break  # Restart the loop with the reduced set

    if best_rank < 54:
        return {"U": best_U[:, :best_rank].tolist(), "V": best_V[:, :best_rank].tolist(), "W": best_W[:, :best_rank].tolist()}
    
    return {"U": U[:, :R].tolist(), "V": V[:, :R].tolist(), "W": W[:, :R].tolist()}
# EVOLVE-BLOCK-END
