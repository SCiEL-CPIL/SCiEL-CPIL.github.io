import time
import random
import numpy as np

def construct():
    n, m, p = 3, 3, 4
    deadline = time.time() + 110

    # Schoolbook rank-36 decomposition
    U_school = [[0] * (n*m*p) for _ in range(n*m)]
    V_school = [[0] * (n*m*p) for _ in range(m*p)]
    W_school = [[0] * (n*m*p) for _ in range(p*n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][c] = 1
                V_school[j*p+k][c] = 1
                W_school[k*n+i][c] = 1
                c += 1

    best_rank = n * m * p
    best_factors = (U_school, V_school, W_school)

    def _verify(U, V, W):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def _round_and_verify(U, V, W):
        U_r = [[round(x * 2) / 2.0 for x in row] for row in U]
        V_r = [[round(x * 2) / 2.0 for x in row] for row in V]
        W_r = [[round(x * 2) / 2.0 for x in row] for row in W]
        if _verify(U_r, V_r, W_r):
            return U_r, V_r, W_r
        return None

    def _error(U, V, W):
        rank = len(U[0])
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    err += (s - 1.0) ** 2
        return err

    def _local_search(U, V, W, max_iters=500, time_limit=None):
        """
        Coordinate Descent with vectorized error.
        """
        U = [row[:] for row in U]
        V = [row[:] for row in V]
        W = [row[:] for row in W]
        rank = len(U[0])
        
        cur_err = _error(U, V, W)
        if cur_err < 1e-9:
            return U, V, W, cur_err

        # Precompute indices for speed
        # We update one entry at a time
        # To speed up, we can compute the gradient or just try small steps
        
        for it in range(max_iters):
            if time_limit and time.time() > time_limit:
                break
            improved = False
            # Try updating a random entry in U, V, or W
            mat = random.choice([0, 1, 2])
            if mat == 0:
                row = random.randint(0, n*m-1)
                col = random.randint(0, rank-1)
                old = U[row][col]
                # Try a few steps
                best_val = old
                best_err = cur_err
                for step in [-0.5, -0.1, 0.1, 0.5, 1.0]:
                    U[row][col] = old + step
                    new_err = _error(U, V, W)
                    if new_err < best_err:
                        best_err = new_err
                        best_val = old + step
                U[row][col] = best_val
                if best_err < cur_err:
                    cur_err = best_err
                    improved = True
            elif mat == 1:
                row = random.randint(0, m*p-1)
                col = random.randint(0, rank-1)
                old = V[row][col]
                best_val = old
                best_err = cur_err
                for step in [-0.5, -0.1, 0.1, 0.5, 1.0]:
                    V[row][col] = old + step
                    new_err = _error(U, V, W)
                    if new_err < best_err:
                        best_err = new_err
                        best_val = old + step
                V[row][col] = best_val
                if best_err < cur_err:
                    cur_err = best_err
                    improved = True
            else:
                row = random.randint(0, p*n-1)
                col = random.randint(0, rank-1)
                old = W[row][col]
                best_val = old
                best_err = cur_err
                for step in [-0.5, -0.1, 0.1, 0.5, 1.0]:
                    W[row][col] = old + step
                    new_err = _error(U, V, W)
                    if new_err < best_err:
                        best_err = new_err
                        best_val = old + step
                W[row][col] = best_val
                if best_err < cur_err:
                    cur_err = best_err
                    improved = True
            
            if not improved:
                # Try larger steps if stuck
                pass # Or break

        return U, V, W, cur_err

    # Try ranks from 30 upward
    for target_rank in [30, 31, 32, 33, 34, 35]:
        if time.time() > deadline:
            break
        found = False
        trials = 0
        while time.time() < deadline and trials < 500:
            trials += 1
            U = [[random.uniform(-2, 2) for _ in range(target_rank)] for _ in range(n*m)]
            V = [[random.uniform(-2, 2) for _ in range(target_rank)] for _ in range(m*p)]
            W = [[random.uniform(-2, 2) for _ in range(target_rank)] for _ in range(p*n)]
            
            if _verify(U, V, W):
                best_rank = target_rank
                best_factors = (U, V, W)
                found = True
                break
            
            U_ls, V_ls, W_ls, err = _local_search(U, V, W, max_iters=200, time_limit=deadline)
            if err < 1e-9:
                rounded = _round_and_verify(U_ls, V_ls, W_ls)
                if rounded:
                    best_rank = target_rank
                    best_factors = rounded
                    found = True
                    break
        
        if found:
            break

    return {"U": best_factors[0], "V": best_factors[1], "W": best_factors[2]}
