import numpy as np
import time

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 6
    start_time = time.time()
    deadline = start_time + 110

    def build_schoolbook():
        r = n * m * p  # 48
        U = np.zeros((n*m, r))
        V = np.zeros((m*p, r))
        W = np.zeros((p*n, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W

    def build_strassen_42():
        rank = 42
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        # Strassen coefficients for 2x2*2x2:
        # M1 = (P00+P11)*(Q00+Q11)
        # M2 = (P10+P11)*Q00
        # M3 = P00*(Q01-Q11)
        # M4 = P11*(Q10-Q00)
        # M5 = (P00+P01)*Q11
        # M6 = (P10-P00)*(Q00+Q01)
        # M7 = (P01-P11)*(Q10+Q11)
        #
        # Each term t has:
        #   U_factors: list of (p_row, p_col, coeff)
        #   V_factors: list of (q_row, q_col, coeff)
        #   W_factors: list of (r_row, r_col, coeff)
        
        strassen_terms = [
            # M1 = (P00+P11)*(Q00+Q11) -> contributes to R00+R11
            ([((0,0,1), (1,1,1))],
             [((0,0,1), (1,1,1))],
             [((0,0,1), (1,1,1))]),
            # M2 = (P10+P11)*Q00 -> contributes to R10-R11
            ([((1,0,1), (1,1,1))],
             [((0,0,1))],
             [((1,0,1), (1,1,-1))]),
            # M3 = P00*(Q01-Q11) -> contributes to R01+R11
            ([((0,0,1))],
             [((0,1,1), (1,1,-1))],
             [((0,1,1), (1,1,1))]),
            # M4 = P11*(Q10-Q00) -> contributes to R00+R10
            ([((1,1,1))],
             [((1,0,1), (0,0,-1))],
             [((0,0,1), (1,0,1))]),
            # M5 = (P00+P01)*Q11 -> contributes to -R00+R01
            ([((0,0,1), (0,1,1))],
             [((1,1,1))],
             [((0,0,-1), (0,1,1))]),
            # M6 = (P10-P00)*(Q00+Q01) -> contributes to R11
            ([((1,0,1), (0,0,-1))],
             [((0,0,1), (0,1,1))],
             [((1,1,1))]),
            # M7 = (P01-P11)*(Q10+Q11) -> contributes to R00
            ([((0,1,1), (1,1,-1))],
             [((1,0,1), (1,1,1))],
             [((0,0,1))]),
        ]
        
        col_idx = 0
        for source in range(2):  # 0=top (A cols 0,1; B rows 0,1), 1=bot (A cols 2,3; B rows 2,3)
            for c in range(3):  # column block 0,1,2 (B cols 2c, 2c+1; C cols 2c, 2c+1)
                for t in range(7):
                    u_factors, v_factors, w_factors = strassen_terms[t]
                    
                    # U: P[pi, pj] -> A[pi, pj + 2*source] -> U[pi*m + pj + 2*source, col_idx]
                    for (pi, pj, coeff) in u_factors:
                        u_row = pi * m + (pj + 2 * source)
                        U[u_row, col_idx] += coeff
                    
                    # V: Q[qi, qj] -> B[qi + 2*source, 2*c + qj] -> V[(qi+2*source)*p + 2*c + qj, col_idx]
                    for (qi, qj, coeff) in v_factors:
                        v_row = (qi + 2 * source) * p + (2 * c + qj)
                        V[v_row, col_idx] += coeff
                    
                    # W: R[ri, rj] -> C[ri, 2*c + rj] -> W[(2*c + rj)*n + ri, col_idx]
                    for (ri, rj, coeff) in w_factors:
                        w_row = (2 * c + rj) * n + ri
                        W[w_row, col_idx] += coeff
                    
                    col_idx += 1
        
        return U, V, W

    def tensor_error(U, V, W):
        U = np.asarray(U, dtype=np.float64)
        V = np.asarray(V, dtype=np.float64)
        W = np.asarray(W, dtype=np.float64)
        r = U.shape[1]
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err

    def check_valid(U, V, W, tol=1e-6):
        return tensor_error(U, V, W) < tol

    U42, V42, W42 = build_strassen_42()
    
    if not check_valid(U42, V42, W42):
        U, V, W = build_schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_rank = 42
    best_U, best_V, best_W = U42, V42, W42

    def solve_for_U(V, W, rank):
        U_sol = np.zeros((n*m, rank))
        for i in range(n):
            for j in range(m):
                A_uj = np.zeros((p, rank))
                for k in range(p):
                    A_uj[k, :] = V[j*p+k, :] * W[k*n+i, :]
                try:
                    x, _, _, _ = np.linalg.lstsq(A_uj, np.ones(p), rcond=None)
                    U_sol[i*m+j, :] = x
                except:
                    U_sol[i*m+j, :] = 0
        return U_sol

    def solve_for_V(U, W, rank):
        V_sol = np.zeros((m*p, rank))
        for j in range(m):
            for k in range(p):
                A_vk = np.zeros((n, rank))
                for i in range(n):
                    A_vk[i, :] = U[i*m+j, :] * W[k*n+i, :]
                try:
                    x, _, _, _ = np.linalg.lstsq(A_vk, np.ones(n), rcond=None)
                    V_sol[j*p+k, :] = x
                except:
                    V_sol[j*p+k, :] = 0
        return V_sol

    def solve_for_W(U, V, rank):
        W_sol = np.zeros((p*n, rank))
        for k in range(p):
            for i in range(n):
                A_wk = np.zeros((m, rank))
                for j in range(m):
                    A_wk[j, :] = U[i*m+j, :] * V[j*p+k, :]
                try:
                    x, _, _, _ = np.linalg.lstsq(A_wk, np.ones(m), rcond=None)
                    W_sol[k*n+i, :] = x
                except:
                    W_sol[k*n+i, :] = 0
        return W_sol

    # Try to reduce rank using ALS with structured initialization
    for target_rank in range(41, 33, -1):
        if time.time() > deadline - 15:
            break
        
        # Initialize from Strassen decomposition truncated
        U_cur = best_U[:, :target_rank].copy()
        V_cur = best_V[:, :target_rank].copy()
        W_cur = best_W[:, :target_rank].copy()
        
        # Run ALS
        for it in range(20):
            if time.time() > deadline - 5:
                break
            U_new = solve_for_U(V_cur, W_cur, target_rank)
            V_new = solve_for_V(U_new, W_cur, target_rank)
            W_new = solve_for_W(U_new, V_new, target_rank)
            
            U_cur, V_cur, W_cur = U_new, V_new, W_new
            
            err = tensor_error(U_cur, V_cur, W_cur)
            if err < 1e-6:
                best_rank = target_rank
                best_U, best_V, best_W = U_cur, V_cur, W_cur
                # Try to round to half-integers
                U_r = np.round(U_cur * 2) / 2
                V_r = np.round(V_cur * 2) / 2
                W_r = np.round(W_cur * 2) / 2
                if check_valid(U_r, V_r, W_r, tol=1e-4):
                    best_U, best_V, best_W = U_r, V_r, W_r
                break
        
        if best_rank == target_rank:
            break

    # Final check
    if not check_valid(best_U, best_V, best_W, tol=1e-5):
        U, V, W = build_schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
