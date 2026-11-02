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
        # U: (n*m, R), V: (m*p, R), W: (p*n, R)
        # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
        # U, V, W are lists of lists of Fractions or ints
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

    # Try to find a decomposition with rank < rank_schoolbook
    # Strategy: Random search for integer/half-integer solutions
    
    # We will search for R from rank_schoolbook - 1 down to some lower bound
    # For each R, we try random U, V and solve for W
    
    for R in range(rank_schoolbook - 1, 20, -1):
        if time.time() - start_time > time_limit:
            break
        
        # Number of attempts depends on remaining time
        # Let's try a few hundred attempts per rank
        attempts = 0
        max_attempts = 1000
        
        while attempts < max_attempts and (time.time() - start_time) < time_limit:
            attempts += 1
            
            # Generate random U, V with entries in {-1, 0, 1}
            # Use numpy for speed, then convert to list of Fractions for exact arithmetic
            U_np = np.random.randint(-1, 2, size=(n * m, R))
            V_np = np.random.randint(-1, 2, size=(m * p, R))
            
            # Construct linear system for W
            # Variables: W[k*n+i, r] for k in 0..p-1, i in 0..n-1, r in 0..R-1
            # Total variables: p * n * R
            # Equations: n * m * p
            # Eq (i, j, k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            
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
            
            # Solve A_mat @ W_flat = b_vec
            # We want a solution where W entries are integers or half-integers.
            # Let W = 2 * X, where X are half-integers? Or just solve and check.
            # Let's solve for W directly.
            # If we find a rational solution, check if denominator is 1 or 2.
            
            try:
                # Use lstsq to find a solution
                W_flat, residuals, rank, s = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                
                # Check if W_flat is close to integers
                W_int = np.round(W_flat)
                if np.allclose(A_mat @ W_int, b_vec, atol=1e-9):
                    # Verify strictly
                    W_sol = W_int.reshape((p * n, R))
                    # Convert to list of Fractions for is_valid
                    U_list = [[Fraction(x) for x in row] for row in U_np.tolist()]
                    V_list = [[Fraction(x) for x in row] for row in V_np.tolist()]
                    W_list = [[Fraction(x) for x in row] for row in W_sol.tolist()]
                    
                    if is_valid(U_list, V_list, W_list):
                        return {
                            "U": to_list(U_list),
                            "V": to_list(V_list),
                            "W": to_list(W_list)
                        }
                
                # Check if W_flat is close to half-integers (multiply by 2 and round)
                W_half = 2 * np.round(W_flat * 2) / 2
                if np.allclose(A_mat @ W_half, b_vec, atol=1e-9):
                    W_sol = W_half.reshape((p * n, R))
                    U_list = [[Fraction(x) for x in row] for row in U_np.tolist()]
                    V_list = [[Fraction(x) for x in row] for row in V_np.tolist()]
                    W_list = [[Fraction(x) for x in row] for row in W_sol.tolist()]
                    
                    if is_valid(U_list, V_list, W_list):
                        return {
                            "U": to_list(U_list),
                            "V": to_list(V_list),
                            "W": to_list(W_list)
                        }
                        
            except:
                pass

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
