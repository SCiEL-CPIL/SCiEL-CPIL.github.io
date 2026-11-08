import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110

    # Build target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    # Schoolbook fallback
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1

    best_rank = rank_school
    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()

    # Try to find rank 28 decomposition
    target_rank = 28
    num_total = n*m * m*p * p*n  # 900

    def compute_residual(U, V, W):
        T_approx = np.einsum('ic,jc,kc->ijk', U, V, W)
        return np.linalg.norm(T_approx - T)

    def alternate_minimize(U, V, W, max_iters=100, tol=1e-14):
        for it in range(max_iters):
            if time.time() > deadline:
                break

            # Solve for W given U, V
            A_W = np.zeros((num_total, p*n*target_rank))
            b_W = np.zeros(num_total)
            eq_idx = 0
            for i in range(n*m):
                for j in range(m*p):
                    for k in range(p*n):
                        for c in range(target_rank):
                            A_W[eq_idx, k*target_rank + c] = U[i,c] * V[j,c]
                        b_W[eq_idx] = T[i,j,k]
                        eq_idx += 1
            W_flat, _, _, _ = np.linalg.lstsq(A_W, b_W, rcond=None)
            W_new = W_flat.reshape(p*n, target_rank)

            # Solve for V given U, W
            A_V = np.zeros((num_total, m*p*target_rank))
            b_V = np.zeros(num_total)
            eq_idx = 0
            for i in range(n*m):
                for j in range(m*p):
                    for k in range(p*n):
                        for c in range(target_rank):
                            A_V[eq_idx, j*target_rank + c] = U[i,c] * W[k,c]
                        b_V[eq_idx] = T[i,j,k]
                        eq_idx += 1
            V_flat, _, _, _ = np.linalg.lstsq(A_V, b_V, rcond=None)
            V_new = V_flat.reshape(m*p, target_rank)

            # Solve for U given V, W
            A_U = np.zeros((num_total, n*m*target_rank))
            b_U = np.zeros(num_total)
            eq_idx = 0
            for i in range(n*m):
                for j in range(m*p):
                    for k in range(p*n):
                        for c in range(target_rank):
                            A_U[eq_idx, i*target_rank + c] = V[j,c] * W[k,c]
                        b_U[eq_idx] = T[i,j,k]
                        eq_idx += 1
            U_flat, _, _, _ = np.linalg.lstsq(A_U, b_U, rcond=None)
            U_new = U_flat.reshape(n*m, target_rank)

            old_res = compute_residual(U, V, W)
            U, V, W = U_new, V_new, W_new
            new_res = compute_residual(U, V, W)

            if new_res < 1e-10:
                break
            if abs(old_res - new_res) < tol:
                break

        return U, V, W, new_res

    # Try multiple random initializations
    np.random.seed(42)
    for restart in range(20):
        if time.time() > deadline:
            break

        U = np.random.randn(n*m, target_rank)
        V = np.random.randn(m*p, target_rank)
        W = np.random.randn(p*n, target_rank)

        U, V, W, residual = alternate_minimize(U, V, W)

        if residual < 1e-10:
            best_rank = target_rank
            best_U = U
            best_V = V
            best_W = W
            break

    if time.time() > deadline:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
