import numpy as np
import time
import random

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # Try to find a solution with rank r
    for r in range(45, 55):
        if time.time() > deadline:
            break
        sol = find_solution_fast(n, m, p, r, start_time, deadline)
        if sol:
            return sol
            
    return schoolbook(n, m, p)

def schoolbook(n, m, p):
    rank = n * m * p
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

def find_solution_fast(n, m, p, r, start_time, deadline):
    # Precompute indices for fast error calculation
    # We want to minimize sum_{i,j,k} (sum_l U_ij_l V_jk_l W_ik_l - 1)^2
    
    # Use a random restart with local search
    for restart in range(20):
        if time.time() > deadline:
            return None
            
        U = np.random.randint(-1, 2, (n * m, r))
        V = np.random.randint(-1, 2, (m * p, r))
        W = np.random.randint(-1, 2, (p * n, r))
        
        def get_error(U, V, W):
            # Compute the product tensor
            # T[i,j,k] = sum_l U[i*m+j, l] * V[j*p+k, l] * W[k*n+i, l]
            # We can do this with loops or einsum?
            # U is (nm, r), V is (mp, r), W is (pn, r)
            # This is not a standard matmul.
            
            err = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for l in range(r):
                            val += U[i * m + j, l] * V[j * p + k, l] * W[k * n + i, l]
                        err += (val - 1) ** 2
            return err

        current_err = get_error(U, V, W)
        
        # Local search
        steps = 500
        for step in range(steps):
            if time.time() > deadline:
                return None
            if current_err == 0:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
            # Pick random element
            mat_idx = random.choice([0, 1, 2])
            if mat_idx == 0:
                row = random.randint(0, n * m - 1)
                col = random.randint(0, r - 1)
                old_val = U[row, col]
                delta = random.choice([-1, 1])
                U[row, col] = old_val + delta
                new_err = get_error(U, V, W)
                if new_err < current_err:
                    current_err = new_err
                else:
                    U[row, col] = old_val
            elif mat_idx == 1:
                row = random.randint(0, m * p - 1)
                col = random.randint(0, r - 1)
                old_val = V[row, col]
                delta = random.choice([-1, 1])
                V[row, col] = old_val + delta
                new_err = get_error(U, V, W)
                if new_err < current_err:
                    current_err = new_err
                else:
                    V[row, col] = old_val
            else:
                row = random.randint(0, p * n - 1)
                col = random.randint(0, r - 1)
                old_val = W[row, col]
                delta = random.choice([-1, 1])
                W[row, col] = old_val + delta
                new_err = get_error(U, V, W)
                if new_err < current_err:
                    current_err = new_err
                else:
                    W[row, col] = old_val
    return None
