import time
import numpy as np

DIMENSIONS = [4, 5, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Schoolbook decomposition
    def get_schoolbook(rank):
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if col < rank:
                        U[i * m + j, col] = 1
                        V[j * p + k, col] = 1
                        W[k * n + i, col] = 1
                        col += 1
        return U, V, W

    U_school, V_school, W_school = get_schoolbook(rank_schoolbook)
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_school, V_school, W_school

    # Try to find a lower rank decomposition
    # We attempt rank 119
    target_rank = rank_schoolbook - 1
    
    if time.time() < deadline - 5:
        # Initialize with random values
        np.random.seed(42)
        U = np.random.uniform(-1, 1, size=(n * m, target_rank))
        V = np.random.uniform(-1, 1, size=(m * p, target_rank))
        W = np.random.uniform(-1, 1, size=(p * n, target_rank))
        
        # Reshape for easier indexing in ALS
        # U: (n, m, R)
        # V: (m, p, R)
        # W: (p, n, R)
        
        U_r = U.reshape(n, m, target_rank)
        V_r = V.reshape(m, p, target_rank)
        W_r = W.reshape(p, n, target_rank)
        
        # ALS Iterations
        max_iter = 200
        for it in range(max_iter):
            if time.time() > deadline - 10:
                break
                
            # Update U
            for i in range(n):
                for j in range(m):
                    # M[k, r] = V[j, k, r] * W[k, i, r]
                    # V_r[j] is (p, R)
                    # W_r[:, i] is (p, R)
                    M = V_r[j] * W_r[:, i]
                    # Solve M @ u = ones(p)
                    # M is (p, R). We want u (R,) such that M u = 1
                    # lstsq expects (M, N) where we solve A x = b
                    # Here A is M.T (R, p), b is ones(p)
                    # Or use lstsq on M directly if we treat it as finding x in M x = b
                    # np.linalg.lstsq(M, b) works if M is (p, R) and b is (p,)
                    try:
                        u_sol, residuals, rank_m, sv = np.linalg.lstsq(M, np.ones(p), rcond=None)
                        U_r[i, j] = u_sol
                    except np.linalg.LinAlgError:
                        pass
                        
            # Update V
            for j in range(m):
                for k in range(p):
                    # M[i, r] = U[i, j, r] * W[k, i, r]
                    # U_r[:, j] is (n, R)
                    # W_r[k] is (n, R)
                    M = U_r[:, j] * W_r[k]
                    try:
                        v_sol, residuals, rank_m, sv = np.linalg.lstsq(M, np.ones(n), rcond=None)
                        V_r[j, k] = v_sol
                    except np.linalg.LinAlgError:
                        pass
                        
            # Update W
            for k in range(p):
                for i in range(n):
                    # M[j, r] = U[i, j, r] * V[j, k, r]
                    # U_r[i] is (m, R)
                    # V_r[:, k] is (m, R)
                    M = U_r[i] * V_r[:, k]
                    try:
                        w_sol, residuals, rank_m, sv = np.linalg.lstsq(M, np.ones(m), rcond=None)
                        W_r[k, i] = w_sol
                    except np.linalg.LinAlgError:
                        pass

        # After ALS, check if we can round to half-integers
        # Round to nearest 0.5
        U_rounded = np.round(U_r * 2) / 2
        V_rounded = np.round(V_r * 2) / 2
        W_rounded = np.round(W_r * 2) / 2
        
        # Verify exactness
        valid = True
        # Check a subset or all? All is 120 checks. Fast enough.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_rounded[i, j] * V_rounded[j, k] * W_rounded[k, i])
                    if abs(val - 1.0) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            best_rank = target_rank
            best_U = U_rounded.flatten().tolist()
            best_V = V_rounded.flatten().tolist()
            best_W = W_rounded.flatten().tolist()
            return {"U": best_U, "V": best_V, "W": best_W}

    # If no better rank found, return schoolbook
    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
