import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # 10s safety margin before 150s kill

    n, m, p = DIMENSIONS  # n=4, m=4, p=6

    # Strassen coefficients for 2x2x2
    alpha = [
        {0: 1, 1: 0, 2: 0, 3: 1},
        {0: 0, 1: 0, 2: 1, 3: 1},
        {0: 1, 1: 0, 2: 0, 3: 0},
        {0: 0, 1: 0, 2: 0, 3: 1},
        {0: 1, 1: 1, 2: 0, 3: 0},
        {0: -1, 1: 0, 2: 1, 3: 0},
        {0: 0, 1: 1, 2: 0, 3: -1}
    ]
    beta = [
        {0: 1, 1: 0, 2: 0, 3: 1},
        {0: 1, 1: 0, 2: 0, 3: 0},
        {0: 0, 1: 1, 2: 0, 3: -1},
        {0: -1, 1: 0, 2: 1, 3: 0},
        {0: 0, 1: 0, 2: 0, 3: 1},
        {0: 1, 1: 1, 2: 0, 3: 0},
        {0: 0, 1: 0, 2: 1, 3: 1}
    ]
    recon = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, -1), (0, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)]
    ]

    # Build rank-77 decomposition
    R = 77
    U = np.zeros((16, R))
    V = np.zeros((24, R))
    W = np.zeros((24, R))

    col = 0
    for s in range(7):
        # Part 1: Inner Strassen for 2x2 x 2x2 (columns 0,1 of Y_s)
        for si in range(7):
            for a in range(4):
                if alpha[si][a] != 0:
                    x_row, x_col = a // 2, a % 2
                    for r in range(4):
                        if alpha[s][r] != 0:
                            ar, br = r // 2, r % 2
                            u_idx = (ar * 2 + x_row) * 4 + (br * 2 + x_col)
                            U[u_idx, col] += alpha[si][a] * alpha[s][r]

            for b in range(4):
                if beta[si][b] != 0:
                    y_row, y_inner_col = b // 2, b % 2
                    for c in range(4):
                        if beta[s][c] != 0:
                            ac, bc = c // 2, c % 2
                            v_idx = (ac * 2 + y_row) * 6 + (bc * 3 + y_inner_col)
                            V[v_idx, col] += beta[si][b] * beta[s][c]

            for a, b, gamma in recon[s]:
                for za, zb, zgamma in recon[si]:
                    w_idx = (b * 3 + zb) * 4 + (a * 2 + za)
                    W[w_idx, col] += gamma * zgamma

            col += 1

        # Part 2: Schoolbook for 2x2 x 2x1 (column 2 of Y_s)
        for u in range(2):
            for v in range(2):
                for r in range(4):
                    if alpha[s][r] != 0:
                        ar, br = r // 2, r % 2
                        u_idx = (ar * 2 + u) * 4 + (br * 2 + v)
                        U[u_idx, col] += alpha[s][r]

                for c in range(4):
                    if beta[s][c] != 0:
                        ac, bc = c // 2, c % 2
                        v_idx = (ac * 2 + v) * 6 + (bc * 3 + 2)
                        V[v_idx, col] += beta[s][c]

                for a, b, gamma in recon[s]:
                    w_idx = (b * 3 + 2) * 4 + (a * 2 + u)
                    W[w_idx, col] += gamma

                col += 1

    # Verify the decomposition is correct
    def _check_decomp(U_mat, V_mat, W_mat):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_true = A @ B
        
        C_comp = np.zeros((n, p))
        for r in range(U_mat.shape[1]):
            u_r = U_mat[:, r]
            v_r = V_mat[:, r]
            w_r = W_mat[:, r]
            
            for i in range(n):
                for j in range(p):
                    s1 = 0.0
                    for k in range(m):
                        s1 += A[i, k] * u_r[i * m + k]
                    s2 = 0.0
                    for k in range(m):
                        s2 += v_r[k * p + j] * B[k, j]
                    C_comp[i, j] += s1 * s2 * w_r[j * n + i]
        
        return np.max(np.abs(C_comp - C_true))

    err = _check_decomp(U, V, W)

    # Try to reduce rank by attempting to find a lower-rank decomposition
    # using a coordinate descent approach on the tensor equation
    def tensor_err(U_mat, V_mat, W_mat):
        # Compute ||T - sum_r U_r ⊗ V_r ⊗ W_r||_F^2
        # T[i*m+j, j*p+k, k*n+i] = 1
        # recon[i*m+j, j*p+k, k*n+i] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        recon = np.einsum('ar,br,cr->abc', U_mat, V_mat, W_mat)
        # Build target
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m+j, j*p+k, k*n+i] = 1.0
        return np.sum((T - recon) ** 2)

    # Attempt rank reduction: try removing one column at a time and see if we can adjust
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = R

    # Try to reduce from current rank
    for target_rank in range(R - 1, max(R - 10, 60), -1):
        if time.time() > DEADLINE - 20:
            break
        
        # Start from a subset of columns
        U_try = best_U[:, :target_rank].copy()
        V_try = best_V[:, :target_rank].copy()
        W_try = best_W[:, :target_rank].copy()
        
        # Coordinate descent to minimize tensor error
        err = tensor_err(U_try, V_try, W_try)
        
        for it in range(50):
            if time.time() > DEADLINE - 15:
                break
            if err < 1e-12:
                break
            
            improved = False
            for c in range(target_rank):
                if time.time() > DEADLINE - 15:
                    break
                
                Uo, Vo, Wo = U_try[:, c].copy(), V_try[:, c].copy(), W_try[:, c].copy()
                
                # Small random perturbation
                scale = 0.05
                U_try[:, c] += np.random.randn(n * m) * scale
                V_try[:, c] += np.random.randn(m * p) * scale
                W_try[:, c] += np.random.randn(p * n) * scale
                
                new_err = tensor_err(U_try, V_try, W_try)
                if new_err < err:
                    err = new_err
                    improved = True
                else:
                    U_try[:, c] = Uo
                    V_try[:, c] = Vo
                    W_try[:, c] = Wo
            
            if not improved:
                break
        
        if err < 1e-10:
            best_U, best_V, best_W = U_try, V_try, W_try
            best_rank = target_rank
    
    # Final verification
    final_err = _check_decomp(best_U, best_V, best_W)
    
    # If final check fails, fall back to original
    if final_err > 1e-6:
        best_U, best_V, best_W = U, V, W
        best_rank = R

    def to_list(mat):
        result = []
        for row in mat:
            result.append([int(round(x)) if abs(x - round(x)) < 1e-9 else float(x) for x in row])
        return result

    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
