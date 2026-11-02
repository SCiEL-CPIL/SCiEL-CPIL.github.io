import time
import numpy as np
from fractions import Fraction
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    n, m, p = 3, 3, 6
    nm = n * m  # 9
    mp = m * p  # 18
    pn = p * n  # 18

    # Build target tensor
    T_target = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0

    def check_decomposition(U, V, W, tol=1e-7):
        r = U.shape[1]
        T_recon = np.einsum('at,bt,ct->abc', U, V, W)
        return np.allclose(T_recon, T_target, atol=tol, rtol=0)

    def schoolbook_decomposition():
        rank = n * m * p  # 54
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W, rank

    U_sb, V_sb, W_sb, sb_rank = schoolbook_decomposition()
    best_U, best_V, best_W, best_rank = U_sb.copy(), V_sb.copy(), W_sb.copy(), sb_rank

    def try_rank_decomposition(target_rank, num_trials=5, max_iter=2000):
        if target_rank >= best_rank:
            return None
        if time.time() > deadline:
            return None

        from scipy.optimize import minimize

        for trial in range(num_trials):
            if time.time() > deadline:
                return None

            np.random.seed(trial * 1000 + 42)
            U_t = np.random.randn(nm, target_rank) * 0.5
            V_t = np.random.randn(mp, target_rank) * 0.5
            W_t = np.random.randn(pn, target_rank) * 0.5

            x0 = np.concatenate([U_t.flatten(), V_t.flatten(), W_t.flatten()])

            def objective(x):
                U_f = x[:nm * target_rank].reshape(nm, target_rank)
                V_f = x[nm * target_rank: nm * target_rank + mp * target_rank].reshape(mp, target_rank)
                W_f = x[nm * target_rank + mp * target_rank:].reshape(pn, target_rank)
                T_recon = np.einsum('at,bt,ct->abc', U_f, V_f, W_f)
                error = T_target - T_recon
                return np.sum(error ** 2)

            def gradient(x):
                U_f = x[:nm * target_rank].reshape(nm, target_rank)
                V_f = x[nm * target_rank: nm * target_rank + mp * target_rank].reshape(mp, target_rank)
                W_f = x[nm * target_rank + mp * target_rank:].reshape(pn, target_rank)

                T_recon = np.einsum('at,bt,ct->abc', U_f, V_f, W_f)
                error = T_target - T_recon

                dU = -2.0 * np.einsum('abc,bt,ct->at', error, V_f, W_f)
                dV = -2.0 * np.einsum('abc,at,ct->bt', error, U_f, W_f)
                dW = -2.0 * np.einsum('abc,at,bt->ct', error, U_f, V_f)

                return np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])

            try:
                result = minimize(objective, x0, jac=gradient, method='L-BFGS-B',
                                  options={'maxiter': max_iter, 'gtol': 1e-12, 'maxfun': 10000})
            except Exception:
                continue

            if time.time() > deadline:
                return None

            U_final = result.x[:nm * target_rank].reshape(nm, target_rank)
            V_final = result.x[nm * target_rank: nm * target_rank + mp * target_rank].reshape(mp, target_rank)
            W_final = result.x[nm * target_rank + mp * target_rank:].reshape(pn, target_rank)

            if check_decomposition(U_final, V_final, W_final, tol=1e-7):
                # Polish
                if time.time() < deadline - 10:
                    x_polish = np.concatenate([U_final.flatten(), V_final.flatten(), W_final.flatten()])
                    try:
                        result_polish = minimize(objective, x_polish, jac=gradient, method='L-BFGS-B',
                                                options={'maxiter': 500, 'gtol': 1e-14, 'maxfun': 5000})
                        U_polish = result_polish.x[:nm * target_rank].reshape(nm, target_rank)
                        V_polish = result_polish.x[nm * target_rank: nm * target_rank + mp * target_rank].reshape(mp, target_rank)
                        W_polish = result_polish.x[nm * target_rank + mp * target_rank:].reshape(pn, target_rank)
                        if check_decomposition(U_polish, V_polish, W_polish, tol=1e-8):
                            return U_polish, V_polish, W_polish, target_rank
                    except Exception:
                        pass
                return U_final, V_final, W_final, target_rank

        return None

    # Try ranks lower than 54
    # Based on 2 * rank(3x3x3) ~ 2 * 19 = 38, try around there
    ranks_to_try = [38, 40, 42, 45, 48, 50, 52, 53]
    
    for r in ranks_to_try:
        if time.time() > deadline:
            break
        res = try_rank_decomposition(r, num_trials=3, max_iter=1500)
        if res is not None:
            U_t, V_t, W_t, rank_found = res
            best_U, best_V, best_W, best_rank = U_t, V_t, W_t, rank_found
            # If we found a good one, try to go lower immediately
            if rank_found <= 40:
                for lower_r in range(rank_found - 1, 30, -1):
                    if time.time() > deadline:
                        break
                    res2 = try_rank_decomposition(lower_r, num_trials=2, max_iter=1000)
                    if res2 is not None:
                        U_t2, V_t2, W_t2, r2 = res2
                        best_U, best_V, best_W, best_rank = U_t2, V_t2, W_t2, r2
                        break
                break

    # Final check
    if best_rank < sb_rank:
        if not check_decomposition(best_U, best_V, best_W, tol=1e-6):
            best_U, best_V, best_W, best_rank = U_sb.copy(), V_sb.copy(), W_sb.copy(), sb_rank

    def to_list(mat):
        result = []
        for row in mat:
            row_list = []
            for val in row:
                frac = Fraction(val).limit_denominator(10000)
                if abs(float(frac) - val) < 1e-9:
                    if frac.denominator == 1:
                        row_list.append(frac.numerator)
                    elif frac.denominator == 2:
                        row_list.append(frac.numerator / 2.0)
                    else:
                        row_list.append(float(frac))
                else:
                    row_list.append(float(val))
            result.append(row_list)
        return result

    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
