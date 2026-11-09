import numpy as np
import time
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin before 150s kill

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
                        u_val = U_list[i*m+j][r]
                        v_val = V_list[j*p+k][r]
                        w_val = W_list[k*n+i][r]
                        # Handle both int and float
                        s += u_val * v_val * w_val
                    if abs(s - 1) > 1e-10:
                        return False
        return True

    def verify_exact_rational(U_list, V_list, W_list, R):
        """Exact verification using Fraction"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for r in range(R):
                        u_val = U_list[i*m+j][r]
                        v_val = V_list[j*p+k][r]
                        w_val = W_list[k*n+i][r]
                        # Convert to Fraction
                        if isinstance(u_val, float):
                            u_val = Fraction(u_val).limit_denominator(1000000)
                        if isinstance(v_val, float):
                            v_val = Fraction(v_val).limit_denominator(1000000)
                        if isinstance(w_val, float):
                            w_val = Fraction(w_val).limit_denominator(1000000)
                        s += Fraction(u_val) * Fraction(v_val) * Fraction(w_val)
                    if s != 1:
                        return False
        return True

    def try_to_rationalize(val, max_den=12):
        """Try to find a simple rational approximation"""
        if val == 0:
            return 0
        for den in range(1, max_den + 1):
            num = round(val * den)
            if abs(val - num/den) < 1e-9:
                if num == 0:
                    return 0
                if den == 1:
                    return num
                return num / den
        return None

    def rationalize_matrix(M, max_den=12):
        """Try to rationalize all entries of a matrix"""
        R, C = M.shape
        result = [[0]*C for _ in range(R)]
        for i in range(R):
            for j in range(C):
                val = M[i, j]
                if abs(val) < 1e-10:
                    result[i][j] = 0
                    continue
                rat = try_to_rationalize(val, max_den)
                if rat is not None:
                    result[i][j] = rat
                else:
                    return None
        return result

    def try_rank_als(R, max_iters=500, n_restarts=10, seed_base=0):
        """Try to find exact decomposition of rank R using ALS with multiple strategies"""
        best_result = None
        for restart in range(n_restarts):
            if time.time() > deadline:
                return best_result
            
            rng = np.random.RandomState(seed_base + restart * 7919)
            
            # Different initialization strategies
            if restart % 3 == 0:
                U = rng.randn(n*m, R)
                V = rng.randn(m*p, R)
                W = rng.randn(p*n, R)
            elif restart % 3 == 1:
                # Sparse initialization
                U = np.zeros((n*m, R))
                V = np.zeros((m*p, R))
                W = np.zeros((p*n, R))
                for r in range(R):
                    # Each rank-1 component touches a few entries
                    for idx in rng.choice(n*m, size=min(3, n*m), replace=False):
                        U[idx, r] = rng.randn()
                    for idx in rng.choice(m*p, size=min(3, m*p), replace=False):
                        V[idx, r] = rng.randn()
                    for idx in rng.choice(p*n, size=min(3, p*n), replace=False):
                        W[idx, r] = rng.randn()
            else:
                # Symmetric-ish initialization
                U = rng.randn(n*m, R)
                V = rng.randn(m*p, R)
                W = rng.randn(p*n, R)
                # Add some structure
                for r in range(R):
                    U[:, r] *= rng.choice([-1, 1], size=n*m)
                    V[:, r] *= rng.choice([-1, 1], size=m*p)
                    W[:, r] *= rng.choice([-1, 1], size=p*n)

            for it in range(max_iters):
                if time.time() > deadline:
                    return best_result

                # Update U
                for i in range(n):
                    for j in range(m):
                        M = np.zeros((p, R))
                        for k in range(p):
                            M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                        try:
                            U[i*m+j, :] = np.linalg.lstsq(M.T, np.ones(p), rcond=None)[0]
                        except:
                            pass

                # Update V
                for j in range(m):
                    for k in range(p):
                        N = np.zeros((n, R))
                        for i in range(n):
                            N[i, :] = U[i*m+j, :] * W[k*n+i, :]
                        try:
                            V[j*p+k, :] = np.linalg.lstsq(N.T, np.ones(n), rcond=None)[0]
                        except:
                            pass

                # Update W
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
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for r in range(R):
                                s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                            max_err = max(max_err, abs(s - 1.0))
                
                if max_err < 1e-8:
                    # Try to rationalize
                    for max_den in [2, 3, 4, 5, 6, 8, 10, 12]:
                        U_list = rationalize_matrix(U, max_den)
                        V_list = rationalize_matrix(V, max_den)
                        W_list = rationalize_matrix(W, max_den)
                        if U_list is not None and V_list is not None and W_list is not None:
                            if verify_exact(U_list, V_list, W_list, R):
                                return {"U": U_list, "V": V_list, "W": W_list}
                            if verify_exact_rational(U_list, V_list, W_list, R):
                                return {"U": U_list, "V": V_list, "W": W_list}
                    # If rationalization fails, try with larger denominators
                    for max_den in [15, 20, 24]:
                        U_list = rationalize_matrix(U, max_den)
                        V_list = rationalize_matrix(V, max_den)
                        W_list = rationalize_matrix(W, max_den)
                        if U_list is not None and V_list is not None and W_list is not None:
                            if verify_exact(U_list, V_list, W_list, R):
                                return {"U": U_list, "V": V_list, "W": W_list}

        return best_result

    def try_rank_direct(R, n_trials=100):
        """Try to find decomposition by directly searching for small integer solutions"""
        # This is a heuristic search for small integer entries
        if R > 20:  # Too expensive for direct search
            return None
        
        for trial in range(n_trials):
            if time.time() > deadline:
                return None
            
            rng = np.random.RandomState(trial * 104729 + R)
            
            # Generate random small integer/half-integer matrices
            def gen_small_int_matrix(rows, cols, max_val=3):
                M = np.zeros((rows, cols))
                for i in range(rows):
                    for j in range(cols):
                        if rng.rand() < 0.3:  # Sparse
                            M[i, j] = rng.randint(-max_val, max_val + 1)
                        else:
                            M[i, j] = 0
                return M
            
            U = gen_small_int_matrix(n*m, R)
            V = gen_small_int_matrix(m*p, R)
            W = gen_small_int_matrix(p*n, R)
            
            # Check if this works
            if verify_exact(U.tolist(), V.tolist(), W.tolist(), R):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
            # Also try with half-integers
            U = gen_small_int_matrix(n*m, R) * 0.5
            V = gen_small_int_matrix(m*p, R) * 0.5
            W = gen_small_int_matrix(p*n, R) * 0.5
            # Scale to make it work - this is just a guess
            # Actually, let's just check
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            if verify_exact(U_list, V_list, W_list, R):
                return {"U": U_list, "V": V_list, "W": W_list}

        return None

    # Try ranks from low to high
    # Theoretical lower bound is higher than max(n,m,p) for non-trivial cases
    # For 2x4x5, let's try from 12 upward
    for R in range(12, 41):
        if time.time() > deadline:
            break
        
        # Try ALS first
        result = try_rank_als(R, max_iters=300, n_restarts=8, seed_base=R*1000)
        if result is not None:
            return result
        
        # Try direct search for small R
        if R <= 20:
            result = try_rank_direct(R, n_trials=50)
            if result is not None:
                return result

    # Fallback to Schoolbook
    return make_schoolbook()
