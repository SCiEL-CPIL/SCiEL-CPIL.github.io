import time
import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = DIMENSIONS

    def make_rank_55():
        rank = 55
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))

        idx = 0
        # Part 1: Strassen for 2x2 x 2x10 (5 blocks of 2x2x2)
        for block in range(5):
            k0 = 2 * block
            k1 = 2 * block + 1

            # M1 = (A00 + A11) * (B0,k0 + B1,k1)
            U[0 * 3 + 0, idx] = 1.0
            U[1 * 3 + 1, idx] = 1.0
            V[0 * 10 + k0, idx] = 1.0
            V[1 * 10 + k1, idx] = 1.0
            W[k0 * 2 + 0, idx] = 1.0
            W[k1 * 2 + 1, idx] = 1.0
            idx += 1

            # M2 = (A10 + A11) * B0,k0
            U[1 * 3 + 0, idx] = 1.0
            U[1 * 3 + 1, idx] = 1.0
            V[0 * 10 + k0, idx] = 1.0
            W[k0 * 2 + 1, idx] = 1.0
            W[k1 * 2 + 1, idx] = -1.0
            idx += 1

            # M3 = A00 * (B0,k1 - B1,k1)
            U[0 * 3 + 0, idx] = 1.0
            V[0 * 10 + k1, idx] = 1.0
            V[1 * 10 + k1, idx] = -1.0
            W[k1 * 2 + 0, idx] = 1.0
            W[k1 * 2 + 1, idx] = 1.0
            idx += 1

            # M4 = A11 * (B1,k0 - B0,k0)
            U[1 * 3 + 1, idx] = 1.0
            V[1 * 10 + k0, idx] = 1.0
            V[0 * 10 + k0, idx] = -1.0
            W[k0 * 2 + 0, idx] = 1.0
            W[k0 * 2 + 1, idx] = 1.0
            idx += 1

            # M5 = (A00 + A01) * B1,k1
            U[0 * 3 + 0, idx] = 1.0
            U[0 * 3 + 1, idx] = 1.0
            V[1 * 10 + k1, idx] = 1.0
            W[k0 * 2 + 0, idx] = -1.0
            W[k1 * 2 + 0, idx] = 1.0
            idx += 1

            # M6 = (A10 - A00) * (B0,k0 + B0,k1)
            U[1 * 3 + 0, idx] = 1.0
            U[0 * 3 + 0, idx] = -1.0
            V[0 * 10 + k0, idx] = 1.0
            V[0 * 10 + k1, idx] = 1.0
            W[k1 * 2 + 1, idx] = 1.0
            idx += 1

            # M7 = (A01 - A11) * (B1,k0 + B1,k1)
            U[0 * 3 + 1, idx] = 1.0
            U[1 * 3 + 1, idx] = -1.0
            V[1 * 10 + k0, idx] = 1.0
            V[1 * 10 + k1, idx] = 1.0
            W[k0 * 2 + 0, idx] = 1.0
            idx += 1

        # Part 2: Schoolbook for 2x1 x 1x10
        for i in range(2):
            for k in range(10):
                U[i * 3 + 2, idx] = 1.0
                V[2 * 10 + k, idx] = 1.0
                W[k * 2 + i, idx] = 1.0
                idx += 1

        return {"U": U, "V": V, "W": W}

    def validate(U, V, W):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    lhs = np.dot(U_arr[i * m + j] * V_arr[j * p + k], W_arr[k * n + i])
                    if not np.isclose(lhs, 1.0, atol=1e-9):
                        return False
        return True

    def als_decompose(target_rank, max_iters=50, tol=1e-12):
        # Initialize with random or truncated 55-rank
        U_init = np.random.randn(n * m, target_rank)
        V_init = np.random.randn(m * p, target_rank)
        W_init = np.random.randn(p * n, target_rank)
        
        # Build target tensor T
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] = 1.0

        # Precompute indices for speed?
        # We will solve linear systems.
        
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            # Update U
            # T_ijk = sum_r U_im+j,r * V_jp+k,r * W_kn+i,r
            # For fixed r, U_im+j,r is unknown.
            # We can solve for all U at once?
            # Let's construct the linear system A_U vec(U) = vec(T)
            # vec(U) has size n*m*R
            # A_U is 60 x (6*R)
            A_U = np.zeros((n * m * p, n * m * target_rank))
            for r in range(target_rank):
                for i in range(n):
                    for j in range(m):
                        u_idx = (i * m + j) * target_rank + r
                        for k in range(p):
                            t_idx = i * m * p + j * p + k
                            coeff = V_init[j * p + k, r] * W_init[k * n + i, r]
                            A_U[t_idx, u_idx] = coeff
            
            # Solve A_U vec(U) = vec(T)
            # Use lstsq
            sol_U, _, _, _ = np.linalg.lstsq(A_U, T.flatten(), rcond=None)
            U_new = sol_U.reshape(n * m, target_rank)
            
            # Update V
            A_V = np.zeros((n * m * p, m * p * target_rank))
            for r in range(target_rank):
                for j in range(m):
                    for k in range(p):
                        v_idx = (j * p + k) * target_rank + r
                        for i in range(n):
                            t_idx = i * m * p + j * p + k
                            coeff = U_new[i * m + j, r] * W_init[k * n + i, r]
                            A_V[t_idx, v_idx] = coeff
                            
            sol_V, _, _, _ = np.linalg.lstsq(A_V, T.flatten(), rcond=None)
            V_new = sol_V.reshape(m * p, target_rank)
            
            # Update W
            A_W = np.zeros((n * m * p, p * n * target_rank))
            for r in range(target_rank):
                for k in range(p):
                    for i in range(n):
                        w_idx = (k * n + i) * target_rank + r
                        for j in range(m):
                            t_idx = i * m * p + j * p + k
                            coeff = U_new[i * m + j, r] * V_new[j * p + k, r]
                            A_W[t_idx, w_idx] = coeff
                            
            sol_W, _, _, _ = np.linalg.lstsq(A_W, T.flatten(), rcond=None)
            W_new = sol_W.reshape(p * n, target_rank)
            
            # Check convergence
            error = np.linalg.norm(T.flatten() - (U_new * V_new * W_new).flatten())
            if error < tol:
                # Round to half-integers
                U_round = np.round(U_new * 2) / 2
                V_round = np.round(V_new * 2) / 2
                W_round = np.round(W_new * 2) / 2
                
                if validate(U_round, V_round, W_round):
                    return {"U": U_round, "V": V_round, "W": W_round}
            
            U_init = U_new
            V_init = V_new
            W_init = W_new
            
        return None

    try:
        # Try to reduce rank
        best_result = None
        for r in range(54, 30, -1):
            if time.time() > deadline - 10:
                break
            res = als_decompose(r)
            if res:
                best_result = res
                break
                
        if best_result:
            return {"U": best_result['U'].tolist(), "V": best_result['V'].tolist(), "W": best_result['W'].tolist()}
            
        result = make_rank_55()
        if validate(result['U'], result['V'], result['W']):
            return {"U": result['U'].tolist(), "V": result['V'].tolist(), "W": result['W'].tolist()}
    except Exception:
        pass

    # Fallback: Schoolbook
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
