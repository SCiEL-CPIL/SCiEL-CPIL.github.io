import numpy as np
import time

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # Schoolbook solution
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school
    
    values = [0, 1, -1, 0.5, -0.5]
    
    def evaluate(U, V, W):
        # U: (n*m, R), V: (m*p, R), W: (p*n, R)
        # Reshape to (n, m, R), (m, p, R), (p, n, R)
        U_t = U.reshape(n, m, -1)
        V_t = V.reshape(m, p, -1)
        W_t = W.reshape(p, n, -1)
        
        # E[i,j,k] = sum_r U_t[i,j,r] * V_t[j,k,r] * W_t[k,i,r]
        # einsum: 'ijr,jkr,kir->ijk'
        E = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
        return np.sum((E - 1.0)**2)

    def random_init(rank, density=0.3):
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        for r in range(rank):
            mask_u = np.random.rand(n * m) < density
            mask_v = np.random.rand(m * p) < density
            mask_w = np.random.rand(p * n) < density
            
            U[mask_u, r] = np.random.choice([1, -1, 0.5, -0.5], size=np.sum(mask_u))
            V[mask_v, r] = np.random.choice([1, -1, 0.5, -0.5], size=np.sum(mask_v))
            W[mask_w, r] = np.random.choice([1, -1, 0.5, -0.5], size=np.sum(mask_w))
            
        return U, V, W

    # Try ranks from 55 down to 20
    # We prioritize lower ranks, but start from a feasible one to ensure we have a baseline?
    # No, schoolbook is baseline. We just want to beat 60.
    # So we can try 50, 40, 30, 20.
    
    ranks_to_try = [50, 45, 40, 35, 30, 25, 20]
    
    for rank in ranks_to_try:
        if time.time() > deadline - 10:
            break
            
        # If we already found a solution with rank < current rank, we can skip?
        # No, we want the MINIMAL rank.
        # So we should try lower ranks first?
        # If we try 20 first and fail, then 25, etc.
        # But 20 might be impossible.
        # It's better to try in increasing order of difficulty (decreasing rank).
        # So 20, 25, 30...
        # But if 20 is impossible, we waste time.
        # Let's stick to descending order from a high guess?
        # No, if we find a solution at 50, we should check 40.
        # So we iterate 50, 40, 30...
        
        # Actually, let's just iterate from 50 down to 20.
        # If we find a solution at 50, we update best.
        # Then we try 40. If we find 40, we update best.
        # Then 30...
        
        max_restarts = 100
        found = False
        
        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                break
                
            U, V, W = random_init(rank, density=0.4)
            error = evaluate(U, V, W)
            
            # Local search
            for step in range(500):
                if time.time() > deadline - 10:
                    break
                    
                if error < 1e-6:
                    found = True
                    best_result = {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist()
                    }
                    best_rank = rank
                    break
                    
                # Pick random change
                mat_idx = np.random.randint(3)
                if mat_idx == 0:
                    mat = U
                    rows = n * m
                elif mat_idx == 1:
                    mat = V
                    rows = m * p
                else:
                    mat = W
                    rows = p * n
                    
                r_idx = np.random.randint(rank)
                row_idx = np.random.randint(rows)
                
                old_val = mat[row_idx, r_idx]
                new_val = np.random.choice(values)
                if new_val == old_val:
                    continue
                    
                mat[row_idx, r_idx] = new_val
                
                if mat_idx == 0:
                    new_error = evaluate(mat, V, W)
                elif mat_idx == 1:
                    new_error = evaluate(U, mat, W)
                else:
                    new_error = evaluate(U, V, mat)
                    
                if new_error < error:
                    error = new_error
                else:
                    mat[row_idx, r_idx] = old_val
                    
            if found:
                break # Stop restarting for this rank, move to lower rank

    return best_result
