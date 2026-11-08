import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin before 120s kill

    def make_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify_exact(U_list, V_list, W_list, R):
        """Check if decomposition is exact: T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(R):
                        s += U_list[i*m+j][r] * V_list[j*p+k][r] * W_list[k*n+i][r]
                    if s != 1:
                        return False
        return True

    def verify_float(U_np, V_np, W_np, R, tol=1e-9):
        """Quick float check"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(R):
                        s += U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def try_rank(R, max_iters=300, n_restarts=5, seed_base=0):
        """Try to find exact decomposition of rank R using ALS-like refinement"""
        for restart in range(n_restarts):
            if time.time() > deadline:
                return None
            rng = np.random.RandomState(seed_base + restart)
            U = rng.randn(n*m, R)
            V = rng.randn(m*p, R)
            W = rng.randn(p*n, R)

            for it in range(max_iters):
                if time.time() > deadline:
                    return None

                # Update U: for each (i,j), solve M^T @ U[i*m+j,:] = 1
                for i in range(n):
                    for j in range(m):
                        M = np.zeros((p, R))
                        for k in range(p):
                            M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                        try:
                            U[i*m+j, :] = np.linalg.lstsq(M.T, np.ones(p), rcond=None)[0]
                        except:
                            pass

                # Update V: for each (j,k), solve N^T @ V[j*p+k,:] = 1
                for j in range(m):
                    for k in range(p):
                        N = np.zeros((n, R))
                        for i in range(n):
                            N[i, :] = U[i*m+j, :] * W[k*n+i, :]
                        try:
                            V[j*p+k, :] = np.linalg.lstsq(N.T, np.ones(n), rcond=None)[0]
                        except:
                            pass

                # Update W: for each (k,i), solve P^T @ W[k*n+i,:] = 1
                for k in range(p):
                    for i in range(n):
                        P = np.zeros((m, R))
                        for j in range(m):
                            P[j, :] = U[i*m+j, :] * V[j*p+k, :]
                        try:
                            W[k*n+i, :] = np.linalg.lstsq(P.T, np.ones(m), rcond=None)[0]
                        except:
                            pass

                # Check convergence
                if verify_float(U, V, W, R, tol=1e-8):
                    # Convert to exact rationals/ints
                    U_list = [[0]*R for _ in range(n*m)]
                    V_list = [[0]*R for _ in range(m*p)]
                    W_list = [[0]*R for _ in range(p*n)]
                    for idx in range(n*m):
                        for r in range(R):
                            val = U[idx, r]
                            # Try to find a simple rational
                            found = False
                            for den in [1, 2, 3, 4, 5, 6, 8, 10, 12]:
                                num = round(val * den)
                                if abs(val - num/den) < 1e-6:
                                    if num == 0:
                                        U_list[idx][r] = 0
                                    elif den == 1:
                                        U_list[idx][r] = num
                                    else:
                                        U_list[idx][r] = num / den
                                    found = True
                                    break
                            if not found:
                                U_list[idx][r] = val
                    for idx in range(m*p):
                        for r in range(R):
                            val = V[idx, r]
                            found = False
                            for den in [1, 2, 3, 4, 5, 6, 8, 10, 12]:
                                num = round(val * den)
                                if abs(val - num/den) < 1e-6:
                                    if num == 0:
                                        V_list[idx][r] = 0
                                    elif den == 1:
                                        V_list[idx][r] = num
                                    else:
                                        V_list[idx][r] = num / den
                                    found = True
                                    break
                            if not found:
                                V_list[idx][r] = val
                    for idx in range(p*n):
                        for r in range(R):
                            val = W[idx, r]
                            found = False
                            for den in [1, 2, 3, 4, 5, 6, 8, 10, 12]:
                                num = round(val * den)
                                if abs(val - num/den) < 1e-6:
                                    if num == 0:
                                        W_list[idx][r] = 0
                                    elif den == 1:
                                        W_list[idx][r] = num
                                    else:
                                        W_list[idx][r] = num / den
                                    found = True
                                    break
                            if not found:
                                W_list[idx][r] = val

                    if verify_exact(U_list, V_list, W_list, R):
                        return {"U": U_list, "V": V_list, "W": W_list}

            # Try with different scaling
            U = U / (np.linalg.norm(U) + 1e-10)
            V = V / (np.linalg.norm(V) + 1e-10)
            W = W / (np.linalg.norm(W) + 1e-10)

        return None

    # Try ranks from low to high
    # Lower bound: rank >= max(n, m, p) = 5, but realistically higher
    for R in range(10, 35):
        if time.time() > deadline:
            break
        result = try_rank(R, max_iters=200, n_restarts=3, seed_base=R*100)
        if result is not None:
            return result

    # Fallback to Schoolbook
    return make_schoolbook()
# EVOLVE-BLOCK-END
