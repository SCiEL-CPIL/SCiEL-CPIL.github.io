import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin before 150s kill
    
    # Schoolbook solution
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_solution = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_sb
    
    def verify_solution(U, V, W, R):
        """Verify that U, V, W with rank R correctly decompose the matrix multiplication tensor."""
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for l in range(m):
                        val = 0.0
                        for r in range(R):
                            val += W[k * n + i, r] * U[i * m + j, r] * V[l * p + k, r]
                        expected = 1.0 if j == l else 0.0
                        if abs(val - expected) > 1e-9:
                            return False
        return True
    
    def try_rank(R, max_trials=200):
        """Try to find a valid decomposition of rank R."""
        for trial in range(max_trials):
            if time.time() > deadline - 10:
                return None
            
            # Random U, V with entries in {-1, 0, 1}
            U_flat = np.random.randint(-1, 2, (n * m, R))
            V_flat = np.random.randint(-1, 2, (m * p, R))
            
            # Solve for W
            W_flat = np.zeros((p * n, R))
            valid = True
            
            for i in range(n):
                for k in range(p):
                    # M_ik[j*m+l, r] = U[i*m+j, r] * V[l*p+k, r]
                    M_ik = np.zeros((m * m, R))
                    for j in range(m):
                        for l in range(m):
                            M_ik[j * m + l, :] = U_flat[i * m + j, :] * V_flat[l * p + k, :]
                    
                    b_ik = np.zeros(m * m)
                    for j in range(m):
                        b_ik[j * m + j] = 1
                    
                    try:
                        w, residuals, rank, s = np.linalg.lstsq(M_ik, b_ik, rcond=None)
                    except:
                        valid = False
                        break
                    
                    if len(residuals) == 0 or residuals[0] > 1e-9:
                        valid = False
                        break
                    
                    W_flat[k * n + i, :] = w
            
            if not valid:
                continue
            
            # Snap to half integers
            W_snapped = np.round(W_flat * 2) / 2
            
            # Verify
            if verify_solution(U_flat.tolist(), V_flat.tolist(), W_snapped.tolist(), R):
                return {
                    "U": [[float(v) for v in row] for row in U_flat],
                    "V": [[float(v) for v in row] for row in V_flat],
                    "W": [[float(v) for v in row] for row in W_snapped]
                }
        
        return None
    
    # Try ranks from a reasonable minimum up to schoolbook
    # Known: 4x4x4 has rank 21, so 4x4x6 should be at most 21+8=29
    # But let's search more broadly
    for R in range(25, best_rank + 1):
        if time.time() > deadline - 10:
            break
        
        result = try_rank(R, max_trials=100)
        if result is not None:
            best_solution = result
            best_rank = R
            break
    
    return best_solution
