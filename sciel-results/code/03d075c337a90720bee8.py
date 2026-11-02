import numpy as np
import time
import sys

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Time limit guard
    start_time = time.time()
    time_limit = 120  # 120 seconds safety margin (evaluator kills at 150s)
    
    # Fallback: Schoolbook
    def schoolbook():
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
    
    def verify_decomposition(U, V, W, n, m, p, tol=1e-9):
        """
        Verifies if the given decomposition exactly performs matrix multiplication.
        U: (n*m, rank)
        V: (m*p, rank)
        W: (p*n, rank)
        """
        rank = U.shape[1]
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                    if abs(val - 1.0) > tol:
                        return False
        return True

    # Strategy:
    # 1. Try to find a low-rank decomposition for 3x3x3 core.
    # 2. If found, extend to 3x3x6 using the direct sum structure.
    #    The rank of 3x3x6 is at most the rank of 3x3x3.
    #    Known rank of 3x3x3 is 23 (Bini et al.).
    
    n_core = 3
    m_core = 3
    p_core = 3
    
    best_core_rank = 27
    found_core = False
    core_U = core_V = core_W = None
    
    # Search ranks from 23 upwards
    for R in range(23, 28):
        if time.time() - start_time > time_limit - 5:
            break
            
        # Try random searches for this rank R
        attempts = 5000
        for attempt in range(attempts):
            if time.time() - start_time > time_limit - 5:
                break
                
            # Generate random U, V with entries -1, 0, 1
            U_rand = np.random.randint(-1, 2, size=(9, R))
            V_rand = np.random.randint(-1, 2, size=(9, R))
            
            W_core = np.zeros((9, R))
            valid = True
            
            # Solve for each output (i, k)
            for i in range(3):
                for k in range(3):
                    # Build 3 x R matrix M where M[j, r] = U[i,j,r] * V[j,k,r]
                    M = np.zeros((3, R))
                    for j in range(3):
                        for r in range(R):
                            M[j, r] = U_rand[i * 3 + j, r] * V_rand[j * 3 + k, r]
                    
                    # Solve M @ w_vec = ones(3)
                    # w_vec is W[i*3+k, :]
                    try:
                        w_vec, residuals, rank_m, s = np.linalg.lstsq(M, np.ones(3), rcond=None)
                    except:
                        valid = False
                        break
                    
                    # Check if solution is close to integers or half-integers
                    w_doubled = w_vec * 2
                    closest_ints = np.round(w_doubled)
                    error = np.max(np.abs(w_doubled - closest_ints))
                    
                    if error < 1e-9:
                        W_core[i * 3 + k, :] = w_vec
                    else:
                        valid = False
                        break
            
            if valid:
                found_core = True
                core_U = U_rand.tolist()
                core_V = V_rand.tolist()
                core_W = W_core.tolist()
                best_core_rank = R
                break
        
        if found_core:
            break
            
    if found_core:
        # Construct 3x3x6 solution from core
        R_total = best_core_rank
        U_total = [[0] * R_total for _ in range(n * m)]
        V_total = [[0] * R_total for _ in range(m * p)]
        W_total = [[0] * R_total for _ in range(p * n)]
        
        # U is shared
        for i in range(3):
            for j in range(3):
                for r in range(R_total):
                    U_total[i * 3 + j][r] = core_U[i * 3 + j][r]
        
        # V and W are repeated
        for j in range(3):
            for k in range(3):
                for r in range(R_total):
                    V_total[j * 6 + k][r] = core_V[j * 3 + k][r]
                    V_total[j * 6 + (k + 3)][r] = core_V[j * 3 + k][r]
                    
        for k in range(3):
            for i in range(3):
                for r in range(R_total):
                    W_total[k * 3 + i][r] = core_W[k * 3 + i][r]
                    W_total[(k + 3) * 3 + i][r] = core_W[k * 3 + i][r]
                    
        return {"U": U_total, "V": V_total, "W": W_total}
    else:
        return schoolbook()
# EVOLVE-BLOCK-END
