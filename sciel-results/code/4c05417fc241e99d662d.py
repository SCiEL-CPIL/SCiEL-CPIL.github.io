import numpy as np
import time

DIMENSIONS = [2, 3, 4]

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
                
    best_rank = rank_school
    best_result = None
    deadline = time.time() + 110  # 110s limit to be safe
    
    # Try ranks from 19 up to 23
    # Prioritize 19, but if it fails quickly, try others?
    # Or try 19 with many attempts, then 20, etc.
    # Let's try 19 first.
    
    candidate_ranks = [19, 20, 21, 22, 23]
    
    # Precompute equation indices
    # eq_idx maps to (i, j, jp, k)
    eq_indices = []
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    eq_indices.append((i, j, jp, k))
    num_eqs = len(eq_indices)
    
    for R in candidate_ranks:
        if time.time() > deadline - 10: # Keep 10s buffer
            break
            
        # Number of attempts per rank
        # If R=19, try more?
        max_attempts = 5000
        start_time_rank = time.time()
        attempts_per_sec = 1000 # estimate
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
                
            # Random U, V
            # Use small integers to keep numbers small
            U_rand = np.random.randint(-1, 2, size=(n * m, R))
            V_rand = np.random.randint(-1, 2, size=(m * p, R))
            
            valid = True
            W_sol = np.zeros((p * n, R))
            
            # Solve for each (kp, ip) block
            for kp in range(p):
                for ip in range(n):
                    # Build A_block (num_eqs x R)
                    # A_block[eq_idx, r] = U[i*m+j, r] * V[jp*p+k, r]
                    # where (i,j,jp,k) is eq_indices[eq_idx]
                    
                    # Vectorized construction of A_block
                    # We need to map eq_indices to rows.
                    # Let's build A_block efficiently.
                    # For each r, A_block[:, r] is the element-wise product of 
                    # U[:, r] (mapped to eq indices) and V[:, r] (mapped to eq indices).
                    
                    # U_row_indices for eq_indices:
                    # i, j -> i*m + j
                    # V_row_indices for eq_indices:
                    # jp, k -> jp*p + k
                    
                    u_rows = np.array([eq[0] * m + eq[1] for eq in eq_indices])
                    v_rows = np.array([eq[2] * p + eq[3] for eq in eq_indices])
                    
                    # A_block[eq_idx, r] = U_rand[u_rows[eq_idx], r] * V_rand[v_rows[eq_idx], r]
                    # This is (num_eqs x R)
                    # U_sel = U_rand[u_rows, :] -> (num_eqs x R)
                    # V_sel = V_rand[v_rows, :] -> (num_eqs x R)
                    # A_block = U_sel * V_sel
                    
                    U_sel = U_rand[u_rows, :]
                    V_sel = V_rand[v_rows, :]
                    A_block = U_sel * V_sel
                    
                    # Build b_block
                    # b_block[eq_idx] = 1 if i==ip and j==jp and k==kp else 0
                    b_block = np.zeros(num_eqs)
                    for eq_idx, (i, j, jp, k) in enumerate(eq_indices):
                        if i == ip and j == jp and k == kp:
                            b_block[eq_idx] = 1.0
                    
                    # Solve A_block x = b_block
                    try:
                        x, residuals, rank, sv = np.linalg.lstsq(A_block, b_block, rcond=None)
                    except:
                        valid = False
                        break
                        
                    # Check if x is close to half-integers
                    x_scaled = x * 2
                    x_rounded = np.round(x_scaled)
                    if np.allclose(x_scaled, x_rounded, atol=1e-5):
                        W_sol[kp * n + ip, :] = x_rounded
                    else:
                        valid = False
                        break
                        
            if not valid:
                continue
                
            # Verify exact solution with integers
            # U_int, V_int are integers. W_sol is now integers (scaled by 2).
            # Check sum_r U[i,j,r] V[jp,k,r] W[kp,ip,r] == delta
            
            exact = True
            for i in range(n):
                for j in range(m):
                    for jp in range(m):
                        for k in range(p):
                            for kp in range(p):
                                for ip in range(n):
                                    lhs = np.dot(U_rand[i * m + j] * V_rand[jp * p + k], W_sol[kp * n + ip])
                                    rhs = 1 if (i == ip and j == jp and k == kp) else 0
                                    if lhs != rhs:
                                        exact = False
                                        break
                                if not exact: break
                            if not exact: break
                        if not exact: break
                    if not exact: break
                if not exact: break
                
            if exact:
                # Found solution for rank R
                # Convert W to half-integers for output
                W_list = (W_sol / 2.0).tolist()
                U_list = U_rand.tolist()
                V_list = V_rand.tolist()
                return {"U": U_list, "V": V_list, "W": W_list}
                
    # Fallback
    return {"U": U_school, "V": V_school, "W": W_school}
