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
                    if val != 1:
                        return False
        return True
    
    # Rank 50 decomposition (Strassen-based) - guaranteed fallback
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
    
    # Strategy: Fix W to a structured 0-1 sparse matrix, then solve for U and V
    # Key insight: If W[k*n+i, c] is 0 or 1, then the constraint becomes:
    # For each (i,j,k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # 
    # If we fix W, we can set up a linear system. For a given W, define for each c:
    # The active (i,j,k) triples are those where W[k*n+i, c] != 0
    # For each such triple, we need the product U[i*m+j,c] * V[j*p+k,c] to contribute
    # 
    # Better approach: Alternate between solving for U (given V,W) and V (given U,W)
    # When W is fixed and V is fixed, for each (i,j,k):
    #   sum_c U[i*m+j, c] * (V[j*p+k, c] * W[k*n+i, c]) = 1
    # This is linear in U! Similarly for V.
    
    def solve_given_W_V(W, V, r):
        """Solve for U given fixed W and V."""
        U = np.zeros((n_rows, r), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                # For each k, we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
                # This gives p equations for r unknowns (U[i*m+j, :])
                # But the equations are: for each k, dot(U[i*m+j,:], V[j*p+k,:] * W[k*n+i,:]) = 1
                # We can set up a least squares problem
                A = np.zeros((p, r))
                for k in range(p):
                    A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                # Solve A @ U[i*m+j, :] = ones(p)
                try:
                    sol, residuals, rank_A, sv = np.linalg.lstsq(A, np.ones(p), rcond=None)
                    U[i*m+j, :] = sol
                except:
                    pass
        return U
    
    def solve_given_W_U(W, U, r):
        """Solve for V given fixed W and U."""
        V = np.zeros((m_rows, r), dtype=np.float64)
        for j in range(m):
            for k in range(p):
                # For each i, we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
                A = np.zeros((n, r))
                for i in range(n):
                    A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                try:
                    sol, residuals, rank_A, sv = np.linalg.lstsq(A, np.ones(n), rcond=None)
                    V[j*p+k, :] = sol
                except:
                    pass
        return V
    
    def try_rank_with_fixed_W(r, max_iters=50, max_trials=10):
        """Try to find rank-r decomposition by fixing W structure and alternating solves."""
        if time.time() > deadline - 20:
            return None
        
        for trial in range(max_trials):
            if time.time() > deadline - 10:
                break
            
            # Initialize W with sparse 0-1 pattern
            # Each column of W should have a few non-zeros
            W = np.zeros((p_rows, r), dtype=np.float64)
            for c in range(r):
                # Choose 2-4 random positions for non-zeros
                num_nz = np.random.randint(2, 5)
                positions = np.random.choice(p_rows, size=num_nz, replace=False)
                for pos in positions:
                    W[pos, c] = 1.0
            
            # Initialize U and V randomly
            U = np.random.randn(n_rows, r) * 0.5
            V = np.random.randn(m_rows, r) * 0.5
            
            # Alternating least squares
            best_error = float('inf')
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            for it in range(max_iters):
                if time.time() > deadline - 5:
                    break
                
                # Solve for U given W, V
                U = solve_given_W_V(W, V, r)
                # Solve for V given W, U
                V = solve_given_W_U(W, U, r)
                
                # Compute error
                errors = []
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0
                            for c in range(r):
                                val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                            errors.append(val - 1)
                
                error = np.max(np.abs(errors))
                if error < best_error:
                    best_error = error
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if error < 1e-10:
                    break
            
            # Check if we found an integer/half-integer solution
            if best_error < 1e-10:
                # Round to nearest 0.5
                U_r = np.round(best_U * 2) / 2
                V_r = np.round(best_V * 2) / 2
                W_r = np.round(best_W * 2) / 2
                
                # Verify exactness
                if verify_decomposition(U_r.tolist(), V_r.tolist(), W_r.tolist()):
                    return {
                        "U": U_r.tolist(),
                        "V": V_r.tolist(),
                        "W": W_r.tolist()
                    }
        
        return None
    
    # Try structured approach for ranks below 50
    for r in range(49, 39, -1):
        if time.time() > deadline - 25:
            break
        res = try_rank_with_fixed_W(r, max_iters=30, max_trials=5)
        if res:
            return res
    
    # Fallback to rank 50
    return current_solution()
