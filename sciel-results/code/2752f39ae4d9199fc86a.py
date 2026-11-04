"""
Matrix multiplication 2x3 * 3x9 tensor decomposition.
We search for a rank-minimal decomposition.
"""
import numpy as np
import time
import itertools
from fractions import Fraction
import scipy.optimize as optimize

DIMENSIONS = [2, 3, 9]
n, m, p = DIMENSIONS

def construct():
    start_time = time.time()
    deadline = 115.0  # seconds
    
    # 1. Default schoolbook decomposition
    rank_schoolbook = n * m * p
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U, V, W

    def check_validity(rank, U_list, V_list, W_list):
        """
        Checks if the decomposition is valid using exact arithmetic.
        U: (n*m) x R
        V: (m*p) x R
        W: (p*n) x R
        Condition: For all (i, j, k), sum_r U[i,j,r] V[j,k,r] W[k,i,r] == 1
        """
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    u_row = U_list[i*m + j]
                    v_row = V_list[j*p + k]
                    w_row = W_list[k*n + i]
                    for r in range(rank):
                        s += u_row[r] * v_row[r] * w_row[r]
                    if s != 1:
                        return False
        return True

    def check_validity_fast(rank, U_list, V_list, W_list):
        """
        Faster check using numpy with float64.
        """
        U_np = np.array(U_list, dtype=np.float64)
        V_np = np.array(V_list, dtype=np.float64)
        W_np = np.array(W_list, dtype=np.float64)
        
        U_r = U_np.reshape(n, m, rank)
        V_r = V_np.reshape(m, p, rank)
        W_r = W_np.reshape(p, n, rank)
        
        for k in range(p):
            for i in range(n):
                w_row = W_r[k, i, :]
                for j in range(m):
                    u_row = U_r[i, j, :]
                    v_row = V_r[j, k, :]
                    val = np.dot(np.dot(u_row, v_row), w_row)
                    if not np.isclose(val, 1.0, rtol=1e-10, atol=1e-12):
                        return False
        return True

    def try_rank_numerical(rank):
        nonlocal best_rank, best_U, best_V, best_W
        if rank >= best_rank:
            return False
        
        # Variables: U (6*R), V (27*R), W (18*R)
        num_U = n * m
        num_V = m * p
        num_W = p * n
        total_vars = (num_U + num_V + num_W) * rank
        
        # Initial guess: random small floats
        np.random.seed(42)
        x0 = np.random.randn(total_vars) * 0.1
        
        def residuals(x):
            U_flat = x[:num_U * rank]
            V_flat = x[num_U * rank : num_U * rank + num_V * rank]
            W_flat = x[num_U * rank + num_V * rank :]
            
            U_r = U_flat.reshape(n, m, rank)
            V_r = V_flat.reshape(m, p, rank)
            W_r = W_flat.reshape(p, n, rank)
            
            res = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(np.dot(U_r[i, j, :], V_r[j, k, :]), W_r[k, i, :])
                        res.append(val - 1.0)
            return np.array(res)
        
        try:
            result = optimize.least_squares(residuals, x0, method='lm', max_nfev=1000)
            if result.success or np.allclose(result.fun, 0, atol=1e-8):
                x_sol = result.x
                U_flat = x_sol[:num_U * rank]
                V_flat = x_sol[num_U * rank : num_U * rank + num_V * rank]
                W_flat = x_sol[num_U * rank + num_V * rank :]
                
                # Round to nearest integer or half-integer?
                # Try rounding to nearest integer first
                U_int = np.round(U_flat.reshape(num_U, rank)).astype(int)
                V_int = np.round(V_flat.reshape(num_V, rank)).astype(int)
                W_int = np.round(W_flat.reshape(num_W, rank)).astype(int)
                
                U_list = U_int.tolist()
                V_list = V_int.tolist()
                W_list = W_int.tolist()
                
                if check_validity(rank, U_list, V_list, W_list):
                    best_rank = rank
                    best_U, best_V, best_W = U_list, V_list, W_list
                    return True
                
                # Try rounding to half-integers (multiply by 2, round, divide)
                U_half = np.round(U_flat.reshape(num_U, rank) * 2) / 2
                V_half = np.round(V_flat.reshape(num_V, rank) * 2) / 2
                W_half = np.round(W_flat.reshape(num_W, rank) * 2) / 2
                
                # Convert to Fraction for exact check
                U_frac = [[Fraction(x).limit_denominator(2) for x in row] for row in U_half]
                V_frac = [[Fraction(x).limit_denominator(2) for x in row] for row in V_half]
                W_frac = [[Fraction(x).limit_denominator(2) for x in row] for row in W_half]
                
                if check_validity(rank, U_frac, V_frac, W_frac):
                    best_rank = rank
                    best_U, best_V, best_W = U_frac, V_frac, W_frac
                    return True
        except Exception as e:
            pass
            
        return False

    # Search for lower ranks
    lb = max(n*m, m*p, p*n)
    
    for r in range(lb, best_rank):
        if time.time() > deadline:
            break
        if try_rank_numerical(r):
            break
            
    return {"U": best_U, "V": best_V, "W": best_W}
