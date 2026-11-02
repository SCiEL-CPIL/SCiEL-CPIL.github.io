import time
import random
import numpy as np

def construct():
    n, m, p = 3, 3, 4
    deadline = time.time() + 115

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

    # Schoolbook rank-36 as fallback
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

    # Try to construct Rank 32 solution:
    # 23 terms for 3x3x3 + 9 terms for 3x3x1 (4th column)
    # I will use a known Rank 23 algorithm for 3x3x3.
    # If I cannot verify it, I skip.
    
    # Placeholder for Rank 23 coefficients.
    # Since I cannot guarantee the coefficients from memory, I will attempt a search for Rank 35.
    
    # Improved Search: Start from Rank 35, 34, 33.
    # Use a faster error function and local search.
    
    def _error_fast(U, V, W):
        U_arr = np.array(U)
        V_arr = np.array(V)
        W_arr = np.array(W)
        # Tensor contraction
        # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        # We want T[i,j,k] = 1 for all i,j,k.
        # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
        U_reshaped = U_arr.reshape(n, m, -1)
        V_reshaped = V_arr.reshape(m, p, -1)
        W_reshaped = W_arr.reshape(p, n, -1)
        
        # Compute T[i,j,k]
        # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # This is a batched dot product.
        # (n, m, rank) * (m, p, rank) -> need to align j.
        # Let's use einsum: 'ijr,jkr,kir->ijk'
        T = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
        
        err = np.sum((T - 1.0) ** 2)
        return err

    def _local_search(U, V, W, lr=0.01, max_iters=200, time_limit=None):
        U = [row[:] for row in U]
        V = [row[:] for row in V]
        W = [row[:] for row in W]
        rank = len(U[0])
        
        for it in range(max_iters):
            if time_limit and time.time() > time_limit:
                break
            
            # Compute gradients numerically for a subset
            U_grad = np.zeros_like(U)
            V_grad = np.zeros_like(V)
            W_grad = np.zeros_like(W)
            
            eps = 1e-4
            base_err = _error_fast(U, V, W)
            
            # Sample 10 random coordinates
            for _ in range(10):
                mat = random.choice([0, 1, 2])
                if mat == 0:
                    row = random.randint(0, n*m-1)
                    col = random.randint(0, rank-1)
                    old = U[row][col]
                    U[row][col] = old + eps
                    err_plus = _error_fast(U, V, W)
                    U[row][col] = old - eps
                    err_minus = _error_fast(U, V, W)
                    U[row][col] = old
                    U_grad[row][col] = (err_plus - err_minus) / (2*eps)
                elif mat == 1:
                    row = random.randint(0, m*p-1)
                    col = random.randint(0, rank-1)
                    old = V[row][col]
                    V[row][col] = old + eps
                    err_plus = _error_fast(U, V, W)
                    V[row][col] = old - eps
                    err_minus = _error_fast(U, V, W)
                    V[row][col] = old
                    V_grad[row][col] = (err_plus - err_minus) / (2*eps)
                else:
                    row = random.randint(0, p*n-1)
                    col = random.randint(0, rank-1)
                    old = W[row][col]
                    W[row][col] = old + eps
                    err_plus = _error_fast(U, V, W)
                    W[row][col] = old - eps
                    err_minus = _error_fast(U, V, W)
                    W[row][col] = old
                    W_grad[row][col] = (err_plus - err_minus) / (2*eps)
            
            # Update
            for i in range(n*m):
                for r in range(rank):
                    U[i][r] -= lr * U_grad[i][r]
            for i in range(m*p):
                for r in range(rank):
                    V[i][r] -= lr * V_grad[i][r]
            for i in range(p*n):
                for r in range(rank):
                    W[i][r] -= lr * W_grad[i][r]
        
        return U, V, W, _error_fast(U, V, W)

    def _round_and_verify(U, V, W):
        U_r = [[round(x * 2) / 2.0 for x in row] for row in U]
        V_r = [[round(x * 2) / 2.0 for x in row] for row in V]
        W_r = [[round(x * 2) / 2.0 for x in row] for row in W]
        if _verify(U_r, V_r, W_r):
            return U_r, V_r, W_r
        return None

    # Try ranks from 35 down to 32
    for target_rank in [35, 34, 33, 32]:
        if time.time() > deadline - 10:
            break
        found = False
        trials = 0
        max_trials = 20
        while time.time() < deadline - 10 and trials < max_trials:
            trials += 1
            U = [[random.uniform(-1, 1) for _ in range(target_rank)] for _ in range(n*m)]
            V = [[random.uniform(-1, 1) for _ in range(target_rank)] for _ in range(m*p)]
            W = [[random.uniform(-1, 1) for _ in range(target_rank)] for _ in range(p*n)]
            
            U_ls, V_ls, W_ls, err = _local_search(U, V, W, lr=0.01, max_iters=100, time_limit=deadline - 10)
            if err < 1e-7:
                rounded = _round_and_verify(U_ls, V_ls, W_ls)
                if rounded:
                    best_rank = target_rank
                    best_factors = rounded
                    found = True
                    break
        
        if found:
            break

    return {"U": best_factors[0], "V": best_factors[1], "W": best_factors[2]}
