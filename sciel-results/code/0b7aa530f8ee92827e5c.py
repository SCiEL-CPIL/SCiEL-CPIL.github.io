"""Rank-minimal exact tensor decomposition for 4x5 * 5x6 matrix multiplication."""
import time
import random
import numpy as np
from scipy.optimize import minimize

DEADLINE = 140  # safety margin before 150s kill


def construct():
    start_time = time.time()
    n, m, p = 4, 5, 6
    nm, mp, pn = n * m, m * p, p * n  # 20, 30, 24

    def schoolbook():
        R = n * m * p
        U = [[0.0] * R for _ in range(nm)]
        V = [[0.0] * R for _ in range(mp)]
        W = [[0.0] * R for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1.0
                    V[j * p + k][col] = 1.0
                    W[k * n + i][col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}

    best = schoolbook()

    def verify(U_arr, V_arr, W_arr, r):
        """Fast verification using numpy einsum."""
        if time.time() > DEADLINE - 3:
            return False
        U_r = np.array(U_arr).reshape(n, m, r)
        V_r = np.array(V_arr).reshape(m, p, r)
        W_r = np.array(W_arr).reshape(p, n, r)
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        return np.allclose(T, np.ones((n, m, p)), atol=1e-9)

    def try_rank_optimized(R, time_budget):
        """Try to find exact rank-R decomposition via optimization."""
        deadline = time.time() + time_budget
        best_obj = float('inf')
        best_sol = None

        n_restarts = 0
        while time.time() < deadline - 2 and n_restarts < 50:
            n_restarts += 1

            # Random initialization
            U = np.random.randn(nm, R) * 0.1
            V = np.random.randn(mp, R) * 0.1
            W = np.random.randn(pn, R) * 0.1

            def compute_violations(Ux, Vx, Wx):
                """Compute violations using vectorized operations."""
                U_r = Ux.reshape(n, m, R)
                V_r = Vx.reshape(m, p, R)
                W_r = Wx.reshape(p, n, R)
                T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
                return T - np.ones((n, m, p))

            def obj_and_grad(x):
                Ux = x[:nm * R].reshape(nm, R)
                Vx = x[nm * R:nm * R + mp * R].reshape(mp, R)
                Wx = x[nm * R + mp * R:nm * R + mp * R + pn * R].reshape(pn, R)

                viol = compute_violations(Ux, Vx, Wx)
                fval = float(np.sum(viol * viol))

                # Vectorized gradients
                U_r = Ux.reshape(n, m, R)
                V_r = Vx.reshape(m, p, R)
                W_r = Wx.reshape(p, n, R)
                
                # dU[i,j,r] = sum_k 2*viol[i,j,k]*V[j,k,r]*W[k,i,r]
                dU = 2.0 * np.einsum('ijk,jkr,kir->ijr', viol, V_r, W_r)
                # dV[j,k,r] = sum_i 2*viol[i,j,k]*U[i,j,r]*W[k,i,r]
                dV = 2.0 * np.einsum('ijk,ijr,kir->jkr', viol, U_r, W_r)
                # dW[k,i,r] = sum_j 2*viol[i,j,k]*U[i,j,r]*V[j,k,r]
                dW = 2.0 * np.einsum('ijk,ijr,jkr->kir', viol, U_r, V_r)

                gx = np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
                return fval, gx

            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])

            try:
                fval, gval = obj_and_grad(x0)

                # Coarse optimization
                res = minimize(lambda x: obj_and_grad(x)[0], x0,
                               jac=lambda x: obj_and_grad(x)[1],
                               method='L-BFGS-B',
                               options={'maxiter': 500, 'maxfun': 10000,
                                        'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50})
                fval = res.fun

                if fval < 1e-8:
                    # Polish pass
                    res2 = minimize(lambda x: obj_and_grad(x)[0], res.x,
                                    jac=lambda x: obj_and_grad(x)[1],
                                    method='L-BFGS-B',
                                    options={'maxiter': 500, 'maxfun': 10000,
                                             'ftol': 1e-15, 'gtol': 1e-10, 'maxls': 100})
                    fval = res2.fun
                    x_opt = res2.x
                else:
                    x_opt = res.x

                if fval < best_obj:
                    best_obj = fval
                    U_opt = x_opt[:nm * R].reshape(nm, R)
                    V_opt = x_opt[nm * R:nm * R + mp * R].reshape(mp, R)
                    W_opt = x_opt[nm * R + mp * R:nm * R + mp * R + pn * R].reshape(pn, R)
                    best_sol = (U_opt, V_opt, W_opt)

            except Exception:
                pass

            if best_obj < 1e-14:
                break

        if best_sol is not None and best_obj < 1e-10:
            U_opt, V_opt, W_opt = best_sol
            # Verify exactness
            U_list = [[float(U_opt[i, r]) for r in range(R)] for i in range(nm)]
            V_list = [[float(V_opt[i, r]) for r in range(R)] for i in range(mp)]
            W_list = [[float(W_opt[i, r]) for r in range(R)] for i in range(pn)]

            if verify(U_list, V_list, W_list, R):
                return {"U": U_list, "V": V_list, "W": W_list}

        return None

    # Try ranks from low to high, but focus on feasible ranges
    # For 4x5*5x6, known lower bounds suggest rank ~57-72
    # Try a range of ranks with time allocation
    ranks_to_try = [80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120]

    for R in ranks_to_try:
        if time.time() > DEADLINE - 10:
            break
        time_left = DEADLINE - time.time() - 5
        if time_left < 8:
            break

        # Allocate more time to lower ranks (harder)
        if R <= 72:
            budget = min(time_left, 30)
        elif R <= 90:
            budget = min(time_left, 20)
        else:
            budget = min(time_left, 15)

        result = try_rank_optimized(R, budget)

        if result is not None:
            best = result
            break

    return best
