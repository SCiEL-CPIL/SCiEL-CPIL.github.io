import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    start_time = time.time()
    time_limit = 130  # seconds, safety margin before 150s kill

    def is_valid(U, V, W):
        R = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = Fraction(0)
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    for r in range(R):
                        val += U[u_idx][r] * V[v_idx][r] * W[w_idx][r]
                    if val != 1:
                        return False
        return True

    def to_list(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, list):
            return [to_list(x) for x in obj]
        else:
            return obj

    def try_solve_for_W(U_np, V_np, R):
        """Given U and V as numpy arrays, solve for W. Returns dict or None."""
        num_vars = p * n * R
        A_mat = np.zeros((n * m * p, num_vars))
        b_vec = np.ones(n * m * p)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_base = k * n + i
                    for r in range(R):
                        var_idx = w_base * R + r
                        A_mat[eq_idx, var_idx] = U_np[u_idx, r] * V_np[v_idx, r]
                    eq_idx += 1
        
        try:
            W_flat, residuals, rank, s = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            
            # Try multiple denominators: 1, 2, 4
            for denom in [1, 2, 4]:
                W_scaled = np.round(W_flat * denom) / denom
                if np.allclose(A_mat @ W_scaled, b_vec, atol=1e-8):
                    W_sol = W_scaled.reshape((p * n, R))
                    U_list = [[Fraction(int(x)) for x in row] for row in U_np.tolist()]
                    V_list = [[Fraction(int(x)) for x in row] for row in V_np.tolist()]
                    W_list = [[Fraction(int(x * denom)) / denom for x in row] for row in W_sol.tolist()]
                    if is_valid(U_list, V_list, W_list):
                        return {"U": U_list, "V": V_list, "W": W_list}
        except:
            pass
        return None

    # Search for R from rank_schoolbook - 1 down to 20
    for R in range(rank_schoolbook - 1, 20, -1):
        if time.time() - start_time > time_limit:
            break
        
        attempts = 0
        max_attempts = 500
        
        while attempts < max_attempts and (time.time() - start_time) < time_limit:
            attempts += 1
            
            U_np = np.random.randint(-1, 2, size=(n * m, R))
            V_np = np.random.randint(-1, 2, size=(m * p, R))
            
            result = try_solve_for_W(U_np, V_np, R)
            if result is not None:
                return {
                    "U": to_list(result["U"]),
                    "V": to_list(result["V"]),
                    "W": to_list(result["W"])
                }

    # Fallback to Schoolbook
    rank = rank_schoolbook
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
