import numpy as np
import time

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook fallback
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
    
    # Try to find rank 24
    target_rank = 24
    start_time = time.time()
    timeout = 100 # 100 seconds for search
    
    # Pre-compute consistent and inconsistent triplets
    consistent = []
    inconsistent = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                x = i * m + j
                y = j * p + k
                z = k * n + i
                consistent.append((x, y, z))
                
    all_triplets = [(x, y, z) for x in range(n*m) for y in range(m*p) for z in range(p*n)]
    consistent_set = set(consistent)
    for t in all_triplets:
        if t not in consistent_set:
            inconsistent.append(t)
            
    # Build the linear system matrix M and vector b
    # Variables: W[z, r] for z in 0..pn-1, r in 0..rank-1
    # Total vars: pn * rank
    num_vars = p * n * target_rank
    num_eqs = len(all_triplets)
    
    # We will iterate over random U, V
    for trial in range(500): # Limit trials
        if time.time() - start_time > timeout:
            break
            
        # Random U, V with -1, 0, 1
        U = np.random.randint(-1, 2, (n * m, target_rank))
        V = np.random.randint(-1, 2, (m * p, target_rank))
        
        # Build M and b
        M = np.zeros((num_eqs, num_vars))
        b = np.zeros(num_eqs)
        
        for idx, (x, y, z) in enumerate(all_triplets):
            # Equation: sum_r U[x, r] V[y, r] W[z, r] = target
            target = 1 if (x, y, z) in consistent_set else 0
            b[idx] = target
            for r in range(target_rank):
                var_idx = z * target_rank + r
                M[idx, var_idx] = U[x, r] * V[y, r]
                
        # Solve M W_flat = b
        try:
            W_flat, residuals, rank_M, s = np.linalg.lstsq(M, b, rcond=None)
            # Check residual
            if len(residuals) > 0 and residuals[0] < 1e-9:
                W = W_flat.reshape(p * n, target_rank)
                # Convert to list of lists
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                return {"U": U_list, "V": V_list, "W": W_list}
        except:
            pass
            
    # Fallback to schoolbook
    return {"U": U_school, "V": V_school, "W": W_school}
