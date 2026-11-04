import numpy as np
import time
import sys

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
    
    # Try to find lower rank decompositions
    deadline = time.time() + 140  # 140s limit with safety margin
    
    # Known lower bounds and upper bounds for matrix multiplication
    # For 2x3 * 3x4, we can try ranks from 19 down
    # Use a systematic search with structured random initializations
    
    for target_rank in [19, 20, 21]:
        if time.time() > deadline - 20:
            break
            
        # Try multiple random restarts
        max_attempts = 50 if target_rank == 19 else 100
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
                
            # Use a structured random approach with small integer entries
            # Scale by 2 to allow half-integers
            np.random.seed(42 + attempt * 1000 + target_rank)
            
            # Initialize with small random integers in {-2, -1, 0, 1, 2}
            # This gives more flexibility than {-1, 0, 1}
            U_rand = np.random.randint(-2, 3, size=(n * m, target_rank))
            V_rand = np.random.randint(-2, 3, size=(m * p, target_rank))
            
            # Solve for W block by block
            valid = True
            W_sol = np.zeros((p * n, target_rank))
            
            for kp in range(p):
                for ip in range(n):
                    # Build A_block: for each (i, j, jp, k), equation is
                    # sum_r U[i*m+j, r] * V[jp*p+k, r] * W[kp*n+ip, r] = delta(i==ip, j==jp, k==kp)
                    A_block = np.zeros((n * m * m * p, target_rank))
                    b_block = np.zeros(n * m * m * p)
                    
                    eq_idx = 0
                    for i in range(n):
                        for j in range(m):
                            for jp in range(m):
                                for k in range(p):
                                    u_val = U_rand[i * m + j]
                                    v_val = V_rand[jp * p + k]
                                    coeff = u_val * v_val
                                    A_block[eq_idx, :] = coeff
                                    
                                    if i == ip and j == jp and k == kp:
                                        b_block[eq_idx] = 1
                                    else:
                                        b_block[eq_idx] = 0
                                    eq_idx += 1
                                        
                    # Solve least squares
                    try:
                        x, residuals, rank, sv = np.linalg.lstsq(A_block, b_block, rcond=None)
                    except:
                        valid = False
                        break
                        
                    # Check if x is close to half-integers (within tolerance)
                    x_scaled = x * 2
                    x_rounded = np.round(x_scaled)
                    if not np.allclose(x_scaled, x_rounded, atol=1e-8):
                        valid = False
                        break
                        
                    W_sol[kp * n + ip, :] = x_rounded / 2.0
                    
                if not valid:
                    break
                    
            if not valid:
                continue
                
            # Verify exact solution with integer arithmetic
            U_int = U_rand.astype(np.int64)
            V_int = V_rand.astype(np.int64)
            W_int = np.round(W_sol * 2).astype(np.int64)
            
            # Fast verification using vectorized operations
            exact = True
            for i in range(n):
                for j in range(m):
                    u_row = U_int[i * m + j]  # shape (rank,)
                    for jp in range(m):
                        for k in range(p):
                            v_row = V_int[jp * p + k]  # shape (rank,)
                            coeff = u_row * v_row  # shape (rank,)
                            for kp in range(p):
                                for ip in range(n):
                                    w_row = W_int[kp * n + ip]  # shape (rank,)
                                    lhs = int(np.dot(coeff, w_row))
                                    rhs = 1 if (i == ip and j == jp and k == kp) else 0
                                    if lhs != rhs:
                                        exact = False
                                        break
                                if not exact:
                                    break
                            if not exact:
                                break
                        if not exact:
                            break
                    if not exact:
                        break
                if not exact:
                    break
            
            if exact:
                # Found a valid decomposition!
                U_list = U_int.tolist()
                V_list = V_int.tolist()
                W_list = (W_int / 2.0).tolist()
                return {"U": U_list, "V": V_list, "W": W_list}
                
    # Fallback to schoolbook
    return {"U": U_school, "V": V_school, "W": W_school}
