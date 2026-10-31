import time
import numpy as np
import scipy.optimize

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = 4, 4, 4
    
    # Build target tensor
    target = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for k in range(m):
            for j in range(p):
                a_idx = i * m + k
                b_idx = k * p + j
                c_idx = j * n + i
                target[a_idx, b_idx, c_idx] = 1.0

    def verify(U, V, W):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        recon = np.einsum('ir,jr,kr->ijk', U_arr, V_arr, W_arr, optimize=True)
        return np.allclose(recon, target, atol=1e-12)

    def strassen_2x2():
        U2 = np.zeros((4, 7), dtype=float)
        V2 = np.zeros((4, 7), dtype=float)
        W2 = np.zeros((4, 7), dtype=float)

        U2[0, 0] = 1; U2[3, 0] = 1
        V2[0, 0] = 1; V2[3, 0] = 1
        W2[0, 0] = 1; W2[3, 0] = 1

        U2[2, 1] = 1; U2[3, 1] = 1
        V2[0, 1] = 1
        W2[1, 1] = 1; W2[3, 1] = -1

        U2[0, 2] = 1
        V2[1, 2] = 1; V2[3, 2] = -1
        W2[2, 2] = 1; W2[3, 2] = 1

        U2[3, 3] = 1
        V2[2, 3] = 1; V2[0, 3] = -1
        W2[0, 3] = 1; W2[1, 3] = 1

        U2[0, 4] = 1; U2[1, 4] = 1
        V2[3, 4] = 1
        W2[0, 4] = -1; W2[2, 4] = 1

        U2[2, 5] = 1; U2[0, 5] = -1
        V2[0, 5] = 1; V2[1, 5] = 1
        W2[3, 5] = 1

        U2[1, 6] = 1; U2[3, 6] = -1
        V2[2, 6] = 1; V2[3, 6] = 1
        W2[0, 6] = 1

        return U2, V2, W2

    def strassen_kronecker_4x4():
        U2, V2, W2 = strassen_2x2()
        rank = 49
        U49 = np.zeros((16, rank), dtype=float)
        V49 = np.zeros((16, rank), dtype=float)
        W49 = np.zeros((16, rank), dtype=float)

        for I in range(2):
            for J in range(2):
                for ii in range(2):
                    for jj in range(2):
                        a_row = (I * 2 + ii) * 4 + (J * 2 + jj)
                        for s in range(7):
                            for t in range(7):
                                coeff = U2[I * 2 + J, s] * U2[ii * 2 + jj, t]
                                if coeff != 0:
                                    U49[a_row, s * 7 + t] = coeff

        for K in range(2):
            for L in range(2):
                for kk in range(2):
                    for ll in range(2):
                        b_row = (K * 2 + kk) * 4 + (L * 2 + ll)
                        for s in range(7):
                            for t in range(7):
                                coeff = V2[K * 2 + L, s] * V2[kk * 2 + ll, t]
                                if coeff != 0:
                                    V49[b_row, s * 7 + t] = coeff

        for I in range(2):
            for J in range(2):
                for ii in range(2):
                    for jj in range(2):
                        w_row = (J * 2 + jj) * 4 + (I * 2 + ii)
                        for s in range(7):
                            for t in range(7):
                                coeff = W2[I * 2 + J, s] * W2[ii * 2 + jj, t]
                                if coeff != 0:
                                    W49[w_row, s * 7 + t] = coeff

        return U49, V49, W49

    def naive():
        rank = 64
        U = np.zeros((16, rank), dtype=float)
        V = np.zeros((16, rank), dtype=float)
        W = np.zeros((16, rank), dtype=float)
        for i in range(4):
            for k in range(4):
                for j in range(4):
                    r = i * 16 + k * 4 + j
                    U[i * 4 + k, r] = 1.0
                    V[k * 4 + j, r] = 1.0
                    W[j * 4 + i, r] = 1.0
        return U, V, W

    # 1. Get known rank 49 solution
    U49, V49, W49 = strassen_kronecker_4x4()
    if verify(U49.tolist(), V49.tolist(), W49.tolist()):
        best_U, best_V, best_W = U49, V49, W49
        best_rank = 49
    else:
        U_n, V_n, W_n = naive()
        best_U, best_V, best_W = U_n, V_n, W_n
        best_rank = 64

    # 2. Attempt to find rank 48 decomposition
    # Use remaining time for a local search
    elapsed = time.time() - start_time
    if elapsed < 90.0:
        try:
            rank_target = 48
            # Initialize with random noise added to a subset of the 49 factors?
            # Or just random initialization?
            # Random initialization is unlikely to work.
            # Let's try to optimize a rank 48 tensor initialized with the first 48 columns of the rank 49 solution.
            U_init = U49[:, :48].copy()
            V_init = V49[:, :48].copy()
            W_init = W49[:, :48].copy()
            
            # Add small noise to escape local minima
            np.random.seed(42)
            U_init += np.random.normal(0, 0.1, U_init.shape)
            V_init += np.random.normal(0, 0.1, V_init.shape)
            W_init += np.random.normal(0, 0.1, W_init.shape)

            def objective(params):
                if time.time() > deadline:
                    return np.inf
                U_r = params[:16*48].reshape(16, 48)
                V_r = params[16*48:32*48].reshape(16, 48)
                W_r = params[32*48:48*48].reshape(16, 48)
                recon = np.einsum('ir,jr,kr->ijk', U_r, V_r, W_r, optimize=True)
                diff = recon - target
                return np.sum(diff**2)

            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            # Use L-BFGS-B
            res = scipy.optimize.minimize(objective, params_init, method='L-BFGS-B', 
                                          options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12, 'disp': False})
            
            if res.success or res.fun < 1e-10:
                U_opt = res.x[:16*48].reshape(16, 48)
                V_opt = res.x[16*48:32*48].reshape(16, 48)
                W_opt = res.x[32*48:48*48].reshape(16, 48)
                
                if verify(U_opt.tolist(), V_opt.tolist(), W_opt.tolist()):
                    best_U, best_V, best_W = U_opt, V_opt, W_opt
                    best_rank = 48
        except Exception as e:
            # If search fails, keep rank 49
            pass

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
