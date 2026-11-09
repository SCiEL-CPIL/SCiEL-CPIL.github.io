import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin before 150s kill

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

    def to_rational(val, max_den=30, tol=1e-7):
        """Convert float to simple rational"""
        if abs(val) < tol:
            return 0
        for den in range(1, max_den + 1):
            num = round(val * den)
            if abs(val - num/den) < tol:
                if num == 0:
                    return 0
                elif den == 1:
                    return num
                else:
                    return num / den
        return val

    def to_rational_matrix(mat, max_den=30, tol=1e-7):
        """Convert matrix to rationals"""
        return [[to_rational(val, max_den, tol) for val in row] for row in mat]

    def try_rank_als(R, max_iters=500, n_restarts=10, seed_base=0, init_scale=1.0):
        """Try to find exact decomposition of rank R using ALS-like refinement"""
        best_result = None
        for restart in range(n_restarts):
            if time.time() > deadline:
                return best_result
            rng = np.random.RandomState(seed_base + restart * 7 + 42)
            
            # Initialize with some structure
            U = rng.randn(n*m, R) * init_scale
            V = rng.randn(m*p, R) * init_scale
            W = rng.randn(p*n, R) * init_scale

            # Normalize to avoid numerical issues
            U = U / (np.linalg.norm(U) + 1e-10)
            V = V / (np.linalg.norm(V) + 1e-10)
            W = W / (np.linalg.norm(W) + 1e-10)

            converged = False
            for it in range(max_iters):
                if time.time() > deadline:
                    return best_result

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

                # Check convergence with tight tolerance
                if verify_float(U, V, W, R, tol=1e-8):
                    # Try to convert to exact rationals
                    U_list = to_rational_matrix(U.tolist(), max_den=40, tol=1e-6)
                    V_list = to_rational_matrix(V.tolist(), max_den=40, tol=1e-6)
                    W_list = to_rational_matrix(W.tolist(), max_den=40, tol=1e-6)

                    if verify_exact(U_list, V_list, W_list, R):
                        return {"U": U_list, "V": V_list, "W": W_list}
                    else:
                        # Try with slightly different tolerance
                        U_list2 = to_rational_matrix(U.tolist(), max_den=60, tol=1e-5)
                        V_list2 = to_rational_matrix(V.tolist(), max_den=60, tol=1e-5)
                        W_list2 = to_rational_matrix(W.tolist(), max_den=60, tol=1e-5)
                        if verify_exact(U_list2, V_list2, W_list2, R):
                            return {"U": U_list2, "V": V_list2, "W": W_list2}

            # If not converged, track best attempt
            if best_result is None:
                # Compute error
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for r in range(R):
                                s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                            max_err = max(max_err, abs(s - 1.0))
                if max_err < 0.1:
                    U_list = to_rational_matrix(U.tolist(), max_den=20, tol=1e-4)
                    V_list = to_rational_matrix(V.tolist(), max_den=20, tol=1e-4)
                    W_list = to_rational_matrix(W.tolist(), max_den=20, tol=1e-4)
                    if verify_exact(U_list, V_list, W_list, R):
                        return {"U": U_list, "V": V_list, "W": W_list}

        return best_result

    def try_rank_jacobi(R, max_iters=200, n_restarts=5, seed_base=0):
        """Try Jacobi-like approach"""
        for restart in range(n_restarts):
            if time.time() > deadline:
                return None
            rng = np.random.RandomState(seed_base + restart * 13 + 7)
            
            U = rng.randn(n*m, R)
            V = rng.randn(m*p, R)
            W = rng.randn(p*n, R)

            for it in range(max_iters):
                if time.time() > deadline:
                    return None

                # Gradient descent step
                lr = 0.01 / (1 + it * 0.01)

                # Compute gradients
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)

                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            # T[i,j,k] should be 1
                            # dT/dU[i,j,r] = V[j,k,r] * W[k,i,r]
                            # dT/dV[j,k,r] = U[i,j,r] * W[k,i,r]
                            # dT/dW[k,i,r] = U[i,j,r] * V[j,k,r]
                            for r in range(R):
                                val = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                                err = 1.0 - val
                                dU[i*m+j, r] += err * V[j*p+k, r] * W[k*n+i, r]
                                dV[j*p+k, r] += err * U[i*m+j, r] * W[k*n+i, r]
                                dW[k*n+i, r] += err * U[i*m+j, r] * V[j*p+k, r]

                U -= lr * dU
                V -= lr * dV
                W -= lr * dW

                if verify_float(U, V, W, R, tol=1e-8):
                    U_list = to_rational_matrix(U.tolist(), max_den=30, tol=1e-6)
                    V_list = to_rational_matrix(V.tolist(), max_den=30, tol=1e-6)
                    W_list = to_rational_matrix(W.tolist(), max_den=30, tol=1e-6)
                    if verify_exact(U_list, V_list, W_list, R):
                        return {"U": U_list, "V": V_list, "W": W_list}

        return None

    # Try ranks from low to high
    # Start with reasonable lower bound
    for R in range(12, 30):
        if time.time() > deadline:
            break
        
        # Try ALS first (usually more effective)
        result = try_rank_als(R, max_iters=300, n_restarts=8, seed_base=R*100, init_scale=1.0)
        if result is not None:
            return result

        # Try with different initialization scale
        result = try_rank_als(R, max_iters=200, n_restarts=5, seed_base=R*200+50, init_scale=0.5)
        if result is not None:
            return result

        # Try Jacobi if time permits
        if time.time() < deadline - 20:
            result = try_rank_jacobi(R, max_iters=100, n_restarts=3, seed_base=R*300)
            if result is not None:
                return result

    # Fallback to Schoolbook
    return make_schoolbook()
# EVOLVE-BLOCK-END

