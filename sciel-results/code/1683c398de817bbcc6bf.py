import numpy as np
import time
import itertools
from scipy.optimize import least_squares
from scipy.linalg import lstsq

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    dim_u = n * m  # 9
    dim_v = m * p  # 15
    dim_w = p * n  # 15
    
    # Equations: (u_idx, v_idx, w_idx)
    eqs = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                eqs.append((u_idx, v_idx, w_idx))
                
    num_eqs = len(eqs)
    
    def solve_for_W(U, V, R):
        """
        Solves A x = b for W, given U and V.
        x is the flattened W matrix of size (dim_w * R).
        """
        # Construct A
        # A has shape (num_eqs, dim_w * R)
        A = np.zeros((num_eqs, dim_w * R))
        b = np.ones(num_eqs)
        
        for e_idx, (u_idx, v_idx, w_idx) in enumerate(eqs):
            for r in range(R):
                col_idx = w_idx * R + r
                A[e_idx, col_idx] = U[u_idx, r] * V[v_idx, r]
        
        # Solve A x = b
        # Use lstsq to handle underdetermined systems (finds min norm solution)
        x, residuals, rank, s = lstsq(A, b)
        
        # Reshape x to W
        W = x.reshape((dim_w, R))
        return W, residuals

    def check_solution(U, V, W, tol=1e-6):
        """
        Verifies if U, V, W form a valid decomposition.
        Also checks if entries are close to integers/half-integers.
        """
        max_res = 0
        for u_idx, v_idx, w_idx in eqs:
            val = np.sum(U[:, u_idx] * V[:, v_idx] * W[:, w_idx])
            max_res = max(max_res, abs(val - 1.0))
        
        if max_res > tol:
            return False, max_res, 0.0
            
        # Check integrality
        def dist_to_half_int(arr):
            # Distance to nearest multiple of 0.5
            rounded = np.round(arr * 2) / 2.0
            return np.sum(np.abs(arr - rounded))
            
        dist_u = dist_to_half_int(U)
        dist_v = dist_to_half_int(V)
        dist_w = dist_to_half_int(W)
        total_dist = dist_u + dist_v + dist_w
        
        return True, max_res, total_dist

    def round_to_half_int(arr):
        return np.round(arr * 2) / 2.0

    best_rank = n * m * p
    best_solution = None
    
    # Try ranks from 44 down to 30
    # We prioritize finding a valid integer/half-integer solution.
    
    # First, let's see if we can find a solution for R=44 with random integer U, V
    # If U, V are integers, and we solve for W, W might not be integer.
    
    for R in range(44, 29, -1):
        if time.time() > deadline - 5:
            break
            
        found = False
        attempts = 200 if R > 40 else 100
        
        for attempt in range(attempts):
            if time.time() > deadline - 2:
                break
                
            # Randomly initialize U and V with small integers or half-integers
            # Values from {-1, -0.5, 0, 0.5, 1}
            grid_vals = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
            
            U = np.random.choice(grid_vals, size=(R, dim_u))
            V = np.random.choice(grid_vals, size=(R, dim_v))
            
            # Ensure U and V are not all zero in a way that makes A singular
            # Actually, if A is rank deficient, lstsq still returns a solution,
            # but it might not satisfy the equations if b is not in the column space.
            # However, for random U, V, A will likely have full row rank (num_eqs=45)
            # if R is large enough.
            
            W, residuals = solve_for_W(U, V, R)
            
            # Check if the solution is valid and close to integer/half-integer
            is_valid, max_res, total_dist = check_solution(U, V, W)
            
            if is_valid and total_dist < 1e-4:
                # Found a good solution
                U_rounded = round_to_half_int(U)
                V_rounded = round_to_half_int(V)
                W_rounded = round_to_half_int(W)
                
                # Verify rounded solution
                is_valid_r, max_res_r, _ = check_solution(U_rounded, V_rounded, W_rounded)
                if is_valid_r:
                    best_rank = R
                    best_solution = {
                        "U": U_rounded.tolist(),
                        "V": V_rounded.tolist(),
                        "W": W_rounded.tolist()
                    }
                    found = True
                    break
        
        if found:
            break

    if best_solution:
        return best_solution
    else:
        # Fallback to schoolbook
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
