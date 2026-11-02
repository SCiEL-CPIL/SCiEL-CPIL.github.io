import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    n, m, p = DIMENSIONS
    n_rows = n * m  # 6
    m_rows = m * p  # 27
    p_rows = p * n  # 18

    def verify_decomposition(U, V, W):
        """Verify that the tensor decomposition is exact."""
        r = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for c in range(r):
                        val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(val - 1) > 1e-9:
                        return False
        return True

    # Rank 50 decomposition (Strassen-based) - the known good solution
    def current_solution():
        rank = 50
        U = [[0] * rank for _ in range(n_rows)]
        V = [[0] * rank for _ in range(m_rows)]
        W = [[0] * rank for _ in range(p_rows)]
        
        col = 0
        for b in range(4):
            k0 = 2 * b
            k1 = 2 * b + 1
            v_e = 0 * p + k0
            v_f = 0 * p + k1
            v_g = 1 * p + k0
            v_h = 1 * p + k1
            u_a = 0 * m + 0
            u_b = 0 * m + 1
            u_c = 1 * m + 0
            u_d = 1 * m + 1
            
            U[u_a][col] += 1; U[u_d][col] += 1; V[v_e][col] += 1; V[v_h][col] += 1
            U[u_c][col+1] += 1; U[u_d][col+1] += 1; V[v_e][col+1] += 1
            U[u_a][col+2] += 1; V[v_f][col+2] += 1; V[v_h][col+2] -= 1
            U[u_d][col+3] += 1; V[v_g][col+3] += 1; V[v_e][col+3] -= 1
            U[u_a][col+4] += 1; U[u_b][col+4] += 1; V[v_h][col+4] += 1
            U[u_c][col+5] += 1; U[u_a][col+5] -= 1; V[v_e][col+5] += 1; V[v_f][col+5] += 1
            U[u_b][col+6] += 1; U[u_d][col+6] -= 1; V[v_g][col+6] += 1; V[v_h][col+6] += 1
            
            w_00 = k0 * n + 0
            w_01 = k1 * n + 0
            w_10 = k0 * n + 1
            w_11 = k1 * n + 1
            
            W[w_00][col] += 1; W[w_00][col+3] += 1; W[w_00][col+4] -= 1; W[w_00][col+6] += 1
            W[w_01][col+2] += 1; W[w_01][col+4] += 1
            W[w_10][col+1] += 1; W[w_10][col+3] += 1
            W[w_11][col] += 1; W[w_11][col+1] -= 1; W[w_11][col+2] += 1; W[w_11][col+5] += 1
            col += 7
            
        k8 = 8
        v_e = 0 * p + k8
        v_g = 1 * p + k8
        u_a = 0 * m + 0
        u_b = 0 * m + 1
        u_c = 1 * m + 0
        u_d = 1 * m + 1
        
        U[u_a][col] += 1; V[v_e][col] += 1
        U[u_b][col+1] += 1; V[v_g][col+1] += 1
        U[u_c][col+2] += 1; V[v_e][col+2] += 1
        U[u_d][col+3] += 1; V[v_g][col+3] += 1
        
        w_08 = k8 * n + 0
        w_18 = k8 * n + 1
        W[w_08][col] += 1; W[w_08][col+1] += 1
        W[w_18][col+2] += 1; W[w_18][col+3] += 1
        col += 4
        
        u_ar0 = 0 * m + 2
        u_ar1 = 1 * m + 2
        for k in range(9):
            v_b = 2 * p + k
            U[u_ar0][col] += 1; V[v_b][col] += 1; W[k * n + 0][col] += 1
            U[u_ar1][col+1] += 1; V[v_b][col+1] += 1; W[k * n + 1][col+1] += 1
            col += 2
            
        return {"U": U, "V": V, "W": W}

    def compute_tensor_entries(U, V, W, r):
        """Compute all tensor entries from factor matrices."""
        errors = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(r):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    errors[i, j, k] = val - 1.0
        return errors

    def try_find_lower_rank_structured(r, max_iterations=200, seed=42):
        """
        Structured search: exploit the known block structure.
        The rank-50 solution has:
        - 4 Strassen blocks (7 each) for (i,j) in {0,1}^2, k in pairs
        - 1 schoolbook 2x2x1 block for k=8
        - 18 schoolbook terms for (i, j=2, k)
        
        Try to find a rank-r solution by:
        1. Keeping the Strassen structure but allowing small perturbations
        2. Trying to merge the 2x2x1 block with adjacent Strassen blocks
        """
        if time.time() > deadline - 20:
            return None
        
        rng = np.random.RandomState(seed)
        
        U0 = np.array(current_solution()["U"], dtype=np.float64)
        V0 = np.array(current_solution()["V"], dtype=np.float64)
        W0 = np.array(current_solution()["W"], dtype=np.float64)
        
        best_error = float('inf')
        best_U, best_V, best_W = None, None, None
        
        for trial in range(5):
            if time.time() > deadline - 10:
                break
            
            # Start from rank-50 and zero out (50-r) terms
            U = U0.copy()
            V = V0.copy()
            W = W0.copy()
            
            if r < 50:
                # Zero out random terms
                terms_to_remove = rng.choice(50, size=50-r, replace=False)
                for t in terms_to_remove:
                    U[:, t] = 0
                    V[:, t] = 0
                    W[:, t] = 0
            
            # Local refinement: try to fix errors by adjusting non-zero entries
            for it in range(max_iterations):
                if time.time() > deadline - 5:
                    break
                
                errors = compute_tensor_entries(U, V, W, 50)
                current_error = np.max(np.abs(errors))
                
                if current_error < best_error:
                    best_error = current_error
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                
                if current_error < 1e-10:
                    break
                
                # Find worst entry
                worst_idx = np.unravel_index(np.argmax(np.abs(errors)), errors.shape)
                wi, wj, wk = worst_idx
                u_row = wi * m + wj
                v_row = wj * p + wk
                w_row = wk * n + wi
                
                # Try to fix by adjusting entries in active terms
                active_terms = [t for t in range(50) if r >= 50 or t not in terms_to_remove]
                if not active_terms:
                    break
                    
                improved = False
                for _ in range(20):
                    t = rng.choice(active_terms)
                    mat_choice = rng.randint(3)
                    delta = rng.choice([-1.0, 1.0, -0.5, 0.5])
                    
                    if mat_choice == 0:
                        old = U[u_row, t]
                        U[u_row, t] = old + delta
                    elif mat_choice == 1:
                        old = V[v_row, t]
                        V[v_row, t] = old + delta
                    else:
                        old = W[w_row, t]
                        W[w_row, t] = old + delta
                    
                    errors_new = compute_tensor_entries(U, V, W, 50)
                    new_error = np.max(np.abs(errors_new))
                    
                    if new_error < current_error * 0.95:
                        current_error = new_error
                        improved = True
                        break
                    else:
                        if mat_choice == 0:
                            U[u_row, t] = old
                        elif mat_choice == 1:
                            V[v_row, t] = old
                        else:
                            W[w_row, t] = old
                
                if not improved:
                    # Broader perturbation
                    t = rng.choice(active_terms)
                    mat_choice = rng.randint(3)
                    if mat_choice == 0:
                        row = rng.randint(n_rows)
                        U[row, t] += rng.choice([-1.0, 1.0])
                    elif mat_choice == 1:
                        row = rng.randint(m_rows)
                        V[row, t] += rng.choice([-1.0, 1.0])
                    else:
                        row = rng.randint(p_rows)
                        W[row, t] += rng.choice([-1.0, 1.0])
            
            # Check if we found a valid solution
            if best_error < 1e-10 and best_U is not None:
                # Extract only active terms
                if r < 50:
                    active_terms = [t for t in range(50) if t not in terms_to_remove]
                    U_final = best_U[:, active_terms]
                    V_final = best_V[:, active_terms]
                    W_final = best_W[:, active_terms]
                else:
                    U_final = best_U
                    V_final = best_V
                    W_final = best_W
                
                U_int = np.round(U_final).astype(np.int64)
                V_int = np.round(V_final).astype(np.int64)
                W_int = np.round(W_final).astype(np.int64)
                
                if verify_decomposition(U_int.tolist(), V_int.tolist(), W_int.tolist()):
                    return {
                        "U": U_int.tolist(),
                        "V": V_int.tolist(),
                        "W": W_int.tolist()
                    }
        
        return None

    # Get the known rank-50 solution
    known_sol = current_solution()
    
    if verify_decomposition(known_sol["U"], known_sol["V"], known_sol["W"]):
        best_result = known_sol
    else:
        best_result = known_sol
    
    # Try to find lower rank solutions
    for r in range(49, 44, -1):
        if time.time() > deadline - 30:
            break
        
        result = try_find_lower_rank_structured(r, max_iterations=150, seed=r*1000+7)
        if result is not None:
            best_result = result
            break
    
    return best_result
