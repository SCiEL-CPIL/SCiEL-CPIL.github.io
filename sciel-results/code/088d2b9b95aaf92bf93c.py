import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
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
    
    def try_rank_structured(R, max_trials=200):
        """Try to find a valid decomposition of rank R using structured block patterns."""
        for trial in range(max_trials):
            if time.time() > deadline - 5:
                return None
            
            # Use block-structured initialization based on 2x2 blocks
            U_flat = np.zeros((n * m, R))
            V_flat = np.zeros((m * p, R))
            
            # Strategy: assign rank factors to 2x2 blocks with Strassen-like patterns
            for bi in range(2):  # Block row for A (i dimension)
                for bj in range(2):  # Block col for A (j dimension)
                    # Assign R/4 factors to this 2x2 block
                    r_block_size = R // 4
                    r_start = (bi * 2 + bj) * r_block_size
                    r_end = min(r_start + r_block_size, R)
                    
                    for r in range(r_start, r_end):
                        # Use a randomized Strassen-like pattern
                        signs = np.random.choice([-1, 1], size=4)
                        
                        # U entries for this 2x2 block
                        for di in range(2):
                            for dj in range(2):
                                i_idx = bi * 2 + di
                                j_idx = bj * 2 + dj
                                if np.random.random() < 0.85:
                                    U_flat[i_idx * m + j_idx, r] = signs[di * 2 + dj]
                        
                        # V entries: for each j in this block, assign to k blocks
                        for dj in range(2):
                            j_idx = bj * 2 + dj
                            for bk in range(3):  # 3 blocks of 2 in k dimension
                                k_idx = bk * 2 + np.random.randint(2)
                                if np.random.random() < 0.85:
                                    V_flat[j_idx * p + k_idx, r] = signs[bk % 2]
            
            # Solve for W
            W_flat = np.zeros((p * n, R))
            valid = True
            
            for i in range(n):
                for k in range(p):
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
    
    # Try ranks from low to high with structured search
    for R in range(16, best_rank + 1):
        if time.time() > deadline - 5:
            break
        
        result = try_rank_structured(R, max_trials=150)
        if result is not None:
            best_solution = result
            best_rank = R
            break
    
    return best_solution
