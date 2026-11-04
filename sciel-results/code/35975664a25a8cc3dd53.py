import numpy as np
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # safety margin

    # Schoolbook rank
    schoolbook_rank = n * m * p
    
    # Try to find a lower rank
    # Based on small matrix multiplication ranks, 60 is likely minimal, 
    # but we check 59, 58, etc.
    # We prioritize higher ranks first as they are easier to solve?
    # Or lower ranks first to get the best score?
    # Score is 1/rank. Lower rank is better.
    # So we should try lower ranks first.
    
    # However, if rank 30 is impossible, we waste time.
    # Let's try a range.
    
    # Optimization: If we have a known good solution, use it.
    # For now, we search.
    
    # We will try ranks from 59 down to 55? 
    # Or 30 to 60?
    # The previous code tried 30 to 60.
    # Let's try 59, 58, 57, 56, 55 with limited restarts.
    # If none found, try 60 (immediate).
    
    # Actually, if rank 60 is the answer, we should return it fast.
    # Let's try a few restarts for 59.
    
    ranks_to_try = [59, 58, 57, 56, 55, 54, 53, 52, 51, 50]
    
    for rank in ranks_to_try:
        if time.time() > deadline:
            break
        result = _try_rank_fast(n, m, p, rank, deadline)
        if result is not None:
            return result
            
    # Fallback to schoolbook
    return _schoolbook(n, m, p)

def _try_rank_fast(n, m, p, rank, deadline):
    max_restarts = 10
    max_iter = 100
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        rng = np.random.RandomState(12345 + restart * 7919)
        U = rng.randn(n * m, rank)
        V = rng.randn(m * p, rank)
        W = rng.randn(p * n, rank)
        
        # Scale to avoid overflow/underflow?
        # Normalize rows?
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
                
            # --- Update U ---
            # Vectorize if possible, but loops are small.
            for i in range(n):
                for j in range(m):
                    # M[k, r] = V[j*p + k, r] * W[k*n + i, r]
                    # We can compute this product efficiently
                    # V_j = V[j*p:(j+1)*p, :] shape (p, rank)
                    # W_i = W.reshape(p, n, rank)[:, i, :] shape (p, rank)
                    # M = V_j * W_i (element-wise)
                    V_j = V[j * p:(j + 1) * p, :]
                    W_i = W[:, i, :] # W is (p, n, rank) if reshaped, but stored flat
                    # W is stored as (p*n, rank). Index k*n+i.
                    # So W_i is rows k*n+i for k in 0..p-1.
                    # This is not contiguous.
                    # We must construct M.
                    M = np.empty((p, rank))
                    for k in range(p):
                        M[k, :] = V_j[k, :] * W[k * n + i, :]
                    
                    try:
                        # Solve M.T @ u = ones(p)
                        # M.T is (rank, p). u is (rank,).
                        # This is an underdetermined system (rank > p usually? No, rank is 50+, p is 6).
                        # So rank >> p.
                        # We want the minimum norm solution?
                        # lstsq gives min norm.
                        U[i * m + j, :] = np.linalg.lstsq(M.T, np.ones(p), rcond=None)[0]
                    except np.linalg.LinAlgError:
                        pass

            # --- Update V ---
            for j in range(m):
                for k in range(p):
                    # N[i, r] = U[i*m + j, r] * W[k*n + i, r]
                    N_mat = np.empty((n, rank))
                    for i in range(n):
                        N_mat[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    try:
                        V[j * p + k, :] = np.linalg.lstsq(N_mat.T, np.ones(n), rcond=None)[0]
                    except np.linalg.LinAlgError:
                        pass

            # --- Update W ---
            for k in range(p):
                for i in range(n):
                # P[j, r] = U[i*m + j, r] * V[j*p + k, r]
                    P_mat = np.empty((m, rank))
                    for j in range(m):
                        P_mat[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    try:
                        W[k * n + i, :] = np.linalg.lstsq(P_mat.T, np.ones(m), rcond=None)[0]
                    except np.linalg.LinAlgError:
                        pass

            # Check convergence
            if it % 5 == 4:
                max_err = 0.0
                # Vectorized check
                # U: (nm, R), V: (mp, R), W: (pn, R)
                # We need sum_r U_ij,r V_jk,r W_ki,r
                # This is tricky to vectorize directly without loops.
                # But n,m,p are small.
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                            err = abs(val - 1.0)
                            if err > max_err:
                                max_err = err
                if max_err < 1e-8:
                    result = _rationalize(U, V, W, n, m, p, rank)
                    if result is not None:
                        return result

    return None

def _rationalize(U, V, W, n, m, p, rank):
    U_r = np.round(U * 2) / 2
    V_r = np.round(V * 2) / 2
    W_r = np.round(W * 2) / 2

    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_r[i * m + j, :], V_r[j * p + k, :] * W_r[k * n + i, :])
                if abs(val - 1.0) > 1e-9:
                    return None

    U_list = [[float(U_r[i, j]) for j in range(rank)] for i in range(n * m)]
    V_list = [[float(V_r[i, j]) for j in range(rank)] for i in range(m * p)]
    W_list = [[float(W_r[i, j]) for j in range(rank)] for i in range(p * n)]
    return {"U": U_list, "V": V_list, "W": W_list}

def _schoolbook(n, m, p):
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
# EVOLVE-BLOCK-END
