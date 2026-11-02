import time
import numpy as np
import random

DIMENSIONS = [2, 3, 8]

def construct():
    """
    Constructs a rank-minimal matrix multiplication decomposition for 2x3 * 3x8.
    Returns a dictionary with matrices U, V, W of shapes (n*m, rank), (m*p, rank), (p*n, rank).
    """
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin for return
    
    n, m, p = DIMENSIONS
    full_rank = n * m * p  # 48
    
    # Initialize best result as schoolbook
    best_rank = full_rank
    U_best = [[0] * full_rank for _ in range(n * m)]
    V_best = [[0] * full_rank for _ in range(m * p)]
    W_best = [[0] * full_rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_best[i * m + j][col] = 1
                V_best[j * p + k][col] = 1
                W_best[k * n + i][col] = 1
                col += 1
                
    best_result = {"U": U_best, "V": V_best, "W": W_best}
    
    if time.time() - start_time > deadline - 2:
        return best_result

    def try_find_decomposition(rank):
        if time.time() - start_time > deadline - 2:
            return None
            
        target_tensor = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    target_tensor[u_idx, v_idx, w_idx] = 1.0
        
        # We want U (nm x r), V (mp x r), W (pn x r)
        # T_{u,v,w} = sum_a U_{u,a} V_{v,a} W_{w,a}
        
        trials = 0
        max_trials = 1000
        
        # Pre-compute target flat matrix: T_flat (nm*mp x pn)
        T_flat = target_tensor.reshape(n * m * m * p, p * n)
        
        while trials < max_trials:
            if time.time() - start_time > deadline - 2:
                return None
            trials += 1
            
            # Generate sparse random U and V
            # Use density ~0.25 to keep the system manageable but diverse
            U_rand = np.zeros((n * m, rank))
            V_rand = np.zeros((m * p, rank))
            
            for a in range(rank):
                # Pick random non-zero entries
                # Ensure at least 1 non-zero per column to avoid trivial nulls
                num_u = max(1, int(n * m * 0.3))
                num_v = max(1, int(m * p * 0.3))
                
                idx_u = np.random.choice(n * m, size=num_u, replace=False)
                idx_v = np.random.choice(m * p, size=num_v, replace=False)
                
                U_rand[idx_u, a] = np.random.choice([-1, 1], size=num_u)
                V_rand[idx_v, a] = np.random.choice([-1, 1], size=num_v)
            
            # Construct B_matrix where column a is flatten(U_a * V_a^T)
            B_matrix = np.zeros((n * m * m * p, rank))
            for a in range(rank):
                M_a = np.outer(U_rand[:, a], V_rand[:, a]) # nm x mp
                B_matrix[:, a] = M_a.flatten()
                
            try:
                # Solve B_matrix * W_trans = T_flat
                # W_trans is (rank x pn)
                # We use least squares. If the residual is small, we have a solution.
                W_trans, residuals, rank_b, s = np.linalg.lstsq(B_matrix, T_flat, rcond=None)
                
                # Check residual
                residual = np.linalg.norm(B_matrix @ W_trans - T_flat)
                
                if residual < 1e-8:
                    # We have a solution!
                    # W is (pn x rank)
                    W_sol = W_trans.T
                    
                    # The task requires integer/half-integer entries.
                    # Random integer U, V often lead to rational W.
                    # Let's check if W_sol is close to half-integers or integers.
                    # If not, we might need to scale or reject.
                    # However, for a "nice" decomposition, if U, V are +/-1, 
                    # and the system is "generic enough" but has a solution, 
                    # W might be rational.
                    # The prompt says "integer/half-integer".
                    # Let's check if 2*W_sol is close to integer.
                    
                    W_check = W_sol * 2
                    if np.allclose(W_check, np.round(W_check), atol=1e-7):
                        # Valid solution with half-integer entries
                        U_sol = U_rand
                        V_sol = V_rand
                        W_final = W_sol
                        
                        # Convert to lists
                        U_list = U_sol.tolist()
                        V_list = V_sol.tolist()
                        W_list = W_final.tolist()
                        
                        return {"U": U_list, "V": V_list, "W": W_list}
                    else:
                        # Maybe it's integer?
                        if np.allclose(W_sol, np.round(W_sol), atol=1e-7):
                             U_sol = U_rand
                             V_sol = V_rand
                             W_final = W_sol
                             return {"U": U_sol.tolist(), "V": V_sol.tolist(), "W": W_final.tolist()}
                        
            except np.linalg.LinAlgError:
                pass
        
        return None

    # Try rank 47
    res_47 = try_find_decomposition(47)
    if res_47 is not None:
        return res_47
        
    # Try rank 46 if time permits
    if time.time() - start_time < deadline - 10:
        res_46 = try_find_decomposition(46)
        if res_46 is not None:
            return res_46
            
    # Try rank 45 if time permits
    if time.time() - start_time < deadline - 15:
        res_45 = try_find_decomposition(45)
        if res_45 is not None:
            return res_45

    return best_result
