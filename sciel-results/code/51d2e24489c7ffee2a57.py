import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 150.0
    
    n, m, p = 2, 4, 5
    
    # Try to find a rank-10 decomposition for 2x2x3 tensor
    # If successful, total rank = 7 (2x2x2) + 10 (2x2x3) = 17 per j-group, total 34
    # Otherwise fall back to rank 36
    
    # First, let's try a numerical search for the 2x2x3 rank-10 decomposition
    # The 2x2x3 tensor has 2*2*3 = 12 entries
    # We need 10 rank-1 terms: sum_{r=1}^{10} u_r ⊗ v_r ⊗ w_r
    
    # Let's parameterize:
    # U: 4x10 (i=0,1; j=0,1)
    # V: 6x10 (j=0,1; k=0,1,2)
    # W: 6x10 (k=0,1,2; i=0,1)
    
    # The constraint is: for all i,j,k: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
    
    # Let's try a simple approach: use the known structure and try to find it
    
    # Actually, let me try a different approach. Let's use the fact that
    # 2x2x3 can be decomposed as 2x2x2 (Strassen, rank 7) + 2x2x1 (rank 4) = rank 11
    # But we want rank 10.
    
    # Let me try to search numerically for a rank-10 solution.
    
    # Initialize with random values and use scipy optimization
    try:
        from scipy.optimize import minimize
        import sys
        
        # Target tensor: T[i,j,k] = 1 for all i in {0,1}, j in {0,1}, k in {0,1,2}
        # We want: sum_{r=0}^{9} U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
        
        # Flatten parameters: U (4x10), V (6x10), W (6x10)
        # Total: 40 + 60 + 60 = 160 parameters
        
        def loss(params):
            if time.time() > deadline - 10:
                return 1e10  # Early termination signal
            
            U = params[0:40].reshape(4, 10)
            V = params[40:100].reshape(6, 10)
            W = params[100:160].reshape(6, 10)
            
            error = 0.0
            for i in range(2):
                for j in range(2):
                    for k in range(3):
                        val = 0.0
                        for r in range(10):
                            val += U[i*2+j, r] * V[j*3+k, r] * W[k*2+i, r]
                        error += (val - 1.0) ** 2
            
            return error
        
        # Try a few random initializations
        best_loss = float('inf')
        best_params = None
        
        n_attempts = 5
        for attempt in range(n_attempts):
            if time.time() > deadline - 20:
                break
            
            np.random.seed(42 + attempt)
            params0 = np.random.randn(160) * 0.5
            
            result = minimize(loss, params0, method='Nelder-Mead', 
                            options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            
            if result.fun < best_loss:
                best_loss = result.fun
                best_params = result.x
        
        if best_loss < 1e-8 and best_params is not None:
            # Found a good solution! Extract the matrices
            U_223 = best_params[0:40].reshape(4, 10)
            V_223 = best_params[40:100].reshape(6, 10)
            W_223 = best_params[100:160].reshape(6, 10)
            
            # Round to nearest integer or half-integer
            def round_to_half(x):
                return np.round(x * 2) / 2
            
            U_223_r = round_to_half(U_223)
            V_223_r = round_to_half(V_223)
            W_223_r = round_to_half(W_223)
            
            # Verify the rounded solution
            error_r = 0.0
            for i in range(2):
                for j in range(2):
                    for k in range(3):
                        val = 0.0
                        for r in range(10):
                            val += U_223_r[i*2+j, r] * V_223_r[j*3+k, r] * W_223_r[k*2+i, r]
                        error_r += (val - 1.0) ** 2
            
            if error_r < 1e-6:
                # Great! Use rank 34 construction
                rank = 34
                U = [[0] * rank for _ in range(n * m)]
                V = [[0] * rank for _ in range(m * p)]
                W = [[0] * rank for _ in range(p * n)]
                
                def add_strassen(col_offset, j_offset, k_offset):
                    # Strassen for 2x2x2
                    U[0 * m + j_offset + 0][col_offset + 0] = 1
                    U[1 * m + j_offset + 1][col_offset + 0] = 1
                    V[(j_offset + 0) * p + k_offset + 0][col_offset + 0] = 1
                    V[(j_offset + 1) * p + k_offset + 1][col_offset + 0] = 1
                    W[(k_offset + 0) * n + 0][col_offset + 0] = 1
                    W[(k_offset + 1) * n + 1][col_offset + 0] = 1
                    
                    U[1 * m + j_offset + 0][col_offset + 1] = 1
                    U[1 * m + j_offset + 1][col_offset + 1] = 1
                    V[(j_offset + 0) * p + k_offset + 0][col_offset + 1] = 1
                    W[(k_offset + 0) * n + 1][col_offset + 1] = 1
                    W[(k_offset + 1) * n + 1][col_offset + 1] = -1
                    
                    U[0 * m + j_offset + 0][col_offset + 2] = 1
                    V[(j_offset + 0) * p + k_offset + 1][col_offset + 2] = 1
                    V[(j_offset + 1) * p + k_offset + 1][col_offset + 2] = -1
                    W[(k_offset + 1) * n + 0][col_offset + 2] = 1
                    W[(k_offset + 1) * n + 1][col_offset + 2] = 1
                    
                    U[1 * m + j_offset + 1][col_offset + 3] = 1
                    V[(j_offset + 0) * p + k_offset + 0][col_offset + 3] = -1
                    V[(j_offset + 1) * p + k_offset + 0][col_offset + 3] = 1
                    W[(k_offset + 0) * n + 0][col_offset + 3] = 1
                    W[(k_offset + 0) * n + 1][col_offset + 3] = 1
                    
                    U[0 * m + j_offset + 0][col_offset + 4] = 1
                    U[0 * m + j_offset + 1][col_offset + 4] = 1
                    V[(j_offset + 1) * p + k_offset + 1][col_offset + 4] = 1
                    W[(k_offset + 0) * n + 0][col_offset + 4] = -1
                    W[(k_offset + 1) * n + 0][col_offset + 4] = 1
                    
                    U[0 * m + j_offset + 0][col_offset + 5] = -1
                    U[1 * m + j_offset + 0][col_offset + 5] = 1
                    V[(j_offset + 0) * p + k_offset + 0][col_offset + 5] = 1
                    V[(j_offset + 0) * p + k_offset + 1][col_offset + 5] = 1
                    W[(k_offset + 1) * n + 1][col_offset + 5] = 1
                    
                    U[0 * m + j_offset + 1][col_offset + 6] = 1
                    U[1 * m + j_offset + 1][col_offset + 6] = -1
                    V[(j_offset + 1) * p + k_offset + 0][col_offset + 6] = 1
                    V[(j_offset + 1) * p + k_offset + 1][col_offset + 6] = 1
                    W[(k_offset + 0) * n + 0][col_offset + 6] = 1
                
                def add_223_decomp(col_offset, j_offset, k_offset):
                    # Add the 2x2x3 decomposition for k_offset, k_offset+1, k_offset+2
                    for r in range(10):
                        # U: i in {0,1}, j in {j_offset, j_offset+1}
                        U[0 * m + j_offset + 0][col_offset + r] = U_223_r[0, r]
                        U[0 * m + j_offset + 1][col_offset + r] = U_223_r[1, r]
                        U[1 * m + j_offset + 0][col_offset + r] = U_223_r[2, r]
                        U[1 * m + j_offset + 1][col_offset + r] = U_223_r[3, r]
                        
                        # V: j in {j_offset, j_offset+1}, k in {k_offset, k_offset+1, k_offset+2}
                        V[(j_offset + 0) * p + k_offset + 0][col_offset + r] = V_223_r[0, r]
                        V[(j_offset + 0) * p + k_offset + 1][col_offset + r] = V_223_r[1, r]
                        V[(j_offset + 0) * p + k_offset + 2][col_offset + r] = V_223_r[2, r]
                        V[(j_offset + 1) * p + k_offset + 0][col_offset + r] = V_223_r[3, r]
                        V[(j_offset + 1) * p + k_offset + 1][col_offset + r] = V_223_r[4, r]
                        V[(j_offset + 1) * p + k_offset + 2][col_offset + r] = V_223_r[5, r]
                        
                        # W: k in {k_offset, k_offset+1, k_offset+2}, i in {0,1}
                        W[(k_offset + 0) * n + 0][col_offset + r] = W_223_r[0, r]
                        W[(k_offset + 0) * n + 1][col_offset + r] = W_223_r[1, r]
                        W[(k_offset + 1) * n + 0][col_offset + r] = W_223_r[2, r]
                        W[(k_offset + 1) * n + 1][col_offset + r] = W_223_r[3, r]
                        W[(k_offset + 2) * n + 0][col_offset + r] = W_223_r[4, r]
                        W[(k_offset + 2) * n + 1][col_offset + r] = W_223_r[5, r]
                
                col = 0
                # J1 (j_offset=0)
                add_strassen(col, 0, 0); col += 7
                add_223_decomp(col, 0, 2); col += 10
                
                # J2 (j_offset=2)
                add_strassen(col, 2, 0); col += 7
                add_223_decomp(col, 2, 2); col += 10
                
                # Verify the full decomposition
                U_arr = np.array(U, dtype=float)
                V_arr = np.array(V, dtype=float)
                W_arr = np.array(W, dtype=float)
                
                # Check: for all i, j, k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for r in range(rank):
                                val += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                            max_err = max(max_err, abs(val - 1.0))
                
                if max_err < 1e-6:
                    return {"U": U, "V": W, "W": W}
                else:
                    # Fallback to rank 36
                    pass
        
        # Fallback: rank 36 construction
    except Exception:
        pass
    
    # Rank 36 construction (original)
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen(col_offset, j_offset, k_offset):
        U[0 * m + j_offset + 0][col_offset + 0] = 1
        U[1 * m + j_offset + 1][col_offset + 0] = 1
        V[(j_offset + 0) * p + k_offset + 0][col_offset + 0] = 1
        V[(j_offset + 1) * p + k_offset + 1][col_offset + 0] = 1
        W[(k_offset + 0) * n + 0][col_offset + 0] = 1
        W[(k_offset + 1) * n + 1][col_offset + 0] = 1
        
        U[1 * m + j_offset + 0][col_offset + 1] = 1
        U[1 * m + j_offset + 1][col_offset + 1] = 1
        V[(j_offset + 0) * p + k_offset + 0][col_offset + 1] = 1
        W[(k_offset + 0) * n + 1][col_offset + 1] = 1
        W[(k_offset + 1) * n + 1][col_offset + 1] = -1
        
        U[0 * m + j_offset + 0][col_offset + 2] = 1
        V[(j_offset + 0) * p + k_offset + 1][col_offset + 2] = 1
        V[(j_offset + 1) * p + k_offset + 1][col_offset + 2] = -1
        W[(k_offset + 1) * n + 0][col_offset + 2] = 1
        W[(k_offset + 1) * n + 1][col_offset + 2] = 1
        
        U[1 * m + j_offset + 1][col_offset + 3] = 1
        V[(j_offset + 0) * p + k_offset + 0][col_offset + 3] = -1
        V[(j_offset + 1) * p + k_offset + 0][col_offset + 3] = 1
        W[(k_offset + 0) * n + 0][col_offset + 3] = 1
        W[(k_offset + 0) * n + 1][col_offset + 3] = 1
        
        U[0 * m + j_offset + 0][col_offset + 4] = 1
        U[0 * m + j_offset + 1][col_offset + 4] = 1
        V[(j_offset + 1) * p + k_offset + 1][col_offset + 4] = 1
        W[(k_offset + 0) * n + 0][col_offset + 4] = -1
        W[(k_offset + 1) * n + 0][col_offset + 4] = 1
        
        U[0 * m + j_offset + 0][col_offset + 5] = -1
        U[1 * m + j_offset + 0][col_offset + 5] = 1
        V[(j_offset + 0) * p + k_offset + 0][col_offset + 5] = 1
        V[(j_offset + 0) * p + k_offset + 1][col_offset + 5] = 1
        W[(k_offset + 1) * n + 1][col_offset + 5] = 1
        
        U[0 * m + j_offset + 1][col_offset + 6] = 1
        U[1 * m + j_offset + 1][col_offset + 6] = -1
        V[(j_offset + 1) * p + k_offset + 0][col_offset + 6] = 1
        V[(j_offset + 1) * p + k_offset + 1][col_offset + 6] = 1
        W[(k_offset + 0) * n + 0][col_offset + 6] = 1
    
    def add_schoolbook_2x2x1(col_offset, j_offset, k_offset):
        U[0 * m + j_offset + 0][col_offset + 0] = 1
        V[(j_offset + 0) * p + k_offset][col_offset + 0] = 1
        W[k_offset * n + 0][col_offset + 0] = 1
        
        U[0 * m + j_offset + 1][col_offset + 1] = 1
        V[(j_offset + 1) * p + k_offset][col_offset + 1] = 1
        W[k_offset * n + 0][col_offset + 1] = 1
        
        U[1 * m + j_offset + 0][col_offset + 2] = 1
        V[(j_offset + 0) * p + k_offset][col_offset + 2] = 1
        W[k_offset * n + 1][col_offset + 2] = 1
        
        U[1 * m + j_offset + 1][col_offset + 3] = 1
        V[(j_offset + 1) * p + k_offset][col_offset + 3] = 1
        W[k_offset * n + 1][col_offset + 3] = 1
    
    col = 0
    add_strassen(col, 0, 0); col += 7
    add_strassen(col, 0, 2); col += 7
    add_schoolbook_2x2x1(col, 0, 4); col += 4
    
    add_strassen(col, 2, 0); col += 7
    add_strassen(col, 2, 2); col += 7
    add_schoolbook_2x2x1(col, 2, 4); col += 4
    
    return {"U": U, "V": V, "W": W}
