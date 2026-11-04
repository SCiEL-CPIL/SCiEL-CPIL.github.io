import numpy as np
import time
import sys

def construct():
    n, m, p = 2, 3, 7
    N_U = n * m  # 6
    N_V = m * p  # 21
    N_W = p * n  # 14

    start_time = time.time()
    deadline = 140.0

    def try_rank_numerical(rank, max_iters=5000, tol=1e-8):
        """Try to find a rank decomposition numerically."""
        if time.time() > deadline - 10:
            return None
        np.random.seed(42)
        U = np.random.randn(N_U, rank)
        V = np.random.randn(N_V, rank)
        W = np.random.randn(N_W, rank)
        
        T_target = np.zeros((N_U, N_V, N_W), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_target[i*m+j, j*p+k, k*n+i] = 1.0
        
        T_flat_target = T_target.flatten()
        
        def objective(params):
            U = params[:N_U*rank].reshape(N_U, rank)
            V = params[N_U*rank:N_U*rank+N_V*rank].reshape(N_V, rank)
            W = params[N_U*rank+N_V*rank:].reshape(N_W, rank)
            T_calc = np.einsum('ar,br,cr->abc', U, V, W)
            return T_calc.flatten() - T_flat_target
        
        from scipy.optimize import least_squares
        params0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        try:
            result = least_squares(objective, params0, method='lm', max_nfev=max_iters, xtol=1e-10, ftol=1e-10)
            if result.success or np.linalg.norm(result.fun) < tol:
                U_final = result.x[:N_U*rank].reshape(N_U, rank)
                V_final = result.x[N_U*rank:N_U*rank+N_V*rank].reshape(N_V, rank)
                W_final = result.x[N_U*rank+N_V*rank:].reshape(N_W, rank)
                # Verify
                T_calc = np.einsum('ar,br,cr->abc', U_final, V_final, W_final)
                if np.allclose(T_calc, T_target, atol=1e-6):
                    return U_final, V_final, W_final
        except Exception:
            pass
        return None

    def get_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(N_U)]
        V = [[0] * rank for _ in range(N_V)]
        W = [[0] * rank for _ in range(N_W)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    def get_strassen_2x2x2():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        return U, V, W

    def get_2x1x2_rank4():
        U = np.zeros((2, 4))
        V = np.zeros((2, 4))
        W = np.zeros((4, 4))
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[0, 1] = 1; V[1, 1] = 1; W[1, 1] = 1
        U[1, 2] = 1; V[0, 2] = 1; W[2, 2] = 1
        U[1, 3] = 1; V[1, 3] = 1; W[3, 3] = 1
        return U, V, W

    def get_2x3x1_rank6():
        U = np.zeros((6, 6))
        V = np.zeros((3, 6))
        W = np.zeros((2, 6))
        col = 0
        for i in range(2):
            for j in range(3):
                U[i*3+j, col] = 1
                V[j, col] = 1
                W[i, col] = 1
                col += 1
        return U, V, W

    def get_2x3x2_rank11():
        U = np.zeros((6, 11))
        V = np.zeros((9, 11))
        W = np.zeros((4, 11))
        U_s, V_s, W_s = get_strassen_2x2x2()
        U_212, V_212, W_212 = get_2x1x2_rank4()
        U[0, 0:7] = U_s[0, :]
        U[1, 0:7] = U_s[1, :]
        U[3, 0:7] = U_s[2, :]
        U[4, 0:7] = U_s[3, :]
        V[0, 0:7] = V_s[0, :]
        V[1, 0:7] = V_s[1, :]
        V[3, 0:7] = V_s[2, :]
        V[4, 0:7] = V_s[3, :]
        W[0, 0:7] = W_s[0, :]
        W[1, 0:7] = W_s[1, :]
        W[2, 0:7] = W_s[2, :]
        W[3, 0:7] = W_s[3, :]
        U[2, 7:11] = U_212[0, :]
        U[5, 7:11] = U_212[1, :]
        V[6, 7:11] = V_212[0, :]
        V[7, 7:11] = V_212[1, :]
        W[0, 7:11] = W_212[0, :]
        W[1, 7:11] = W_212[1, :]
        W[2, 7:11] = W_212[2, :]
        W[3, 7:11] = W_212[3, :]
        return U, V, W

    def build_2x3x7():
        U11_a, V11_a, W11_a = get_2x3x2_rank11()
        U11_b, V11_b, W11_b = get_2x3x2_rank11()
        U11_c, V11_c, W11_c = get_2x3x2_rank11()
        U6, V6, W6 = get_2x3x1_rank6()
        R_total = 39
        U_full = np.zeros((N_U, R_total))
        V_full = np.zeros((N_V, R_total))
        W_full = np.zeros((N_W, R_total))
        
        U_full[:, 0:11] = U11_a
        for j in range(3):
            for k in range(2):
                V_full[j*7+k, 0:11] = V11_a[j*2+k, :]
        for k in range(2):
            for i in range(2):
                W_full[k*2+i, 0:11] = W11_a[k*2+i, :]
                
        U_full[:, 11:22] = U11_b
        for j in range(3):
            for k in range(2):
                V_full[j*7+(k+2), 11:22] = V11_b[j*2+k, :]
        for k in range(2):
            for i in range(2):
                W_full[(k+2)*2+i, 11:22] = W11_b[k*2+i, :]
                
        U_full[:, 22:33] = U11_c
        for j in range(3):
            for k in range(2):
                V_full[j*7+(k+4), 22:33] = V11_c[j*2+k, :]
        for k in range(2):
            for i in range(2):
                W_full[(k+4)*2+i, 22:33] = W11_c[k*2+i, :]
                
        U_full[:, 33:39] = U6
        for j in range(3):
            V_full[j*7+6, 33:39] = V6[j, :]
        for i in range(2):
            W_full[6*2+i, 33:39] = W6[i, :]
            
        return U_full, V_full, W_full

    def try_better_decomposition():
        """Try to find a decomposition with rank < 39."""
        if time.time() > deadline - 20:
            return None
        
        # Try ranks from 38 down to 30
        for rank in range(38, 29, -1):
            if time.time() > deadline - 15:
                break
            result = try_rank_numerical(rank, max_iters=3000)
            if result is not None:
                return result
        return None

    def verify(U, V, W):
        T_calc = np.einsum('ar,br,cr->abc', U, V, W)
        T_target = np.zeros((N_U, N_V, N_W), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_target[i*m+j, j*p+k, k*n+i] = 1.0
        return np.allclose(T_calc, T_target, atol=1e-9)

    U_full, V_full, W_full = build_2x3x7()

    if not verify(U_full, V_full, W_full):
        return get_schoolbook()

    best_result = (U_full, V_full, W_full)
    best_rank = 39

    # Try to find a better decomposition
    better = try_better_decomposition()
    if better is not None:
        U_b, V_b, W_b = better
        if verify(U_b, V_b, W_b):
            best_result = (U_b, V_b, W_b)
            best_rank = U_b.shape[1]

    return {
        "U": best_result[0].tolist(),
        "V": best_result[1].tolist(),
        "W": best_result[2].tolist()
    }
