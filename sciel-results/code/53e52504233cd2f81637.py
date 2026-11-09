import numpy as np
import time
import random

DIMENSIONS = [4, 4, 5]

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
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school
    
    start_time = time.time()
    time_limit = 110  # Leave 10s margin
    
    # We want to find a rank < 80.
    # Known bounds suggest ~50-55.
    # Let's try ranks from 50 to 79.
    
    def verify(U, V, W, r):
        # Check if the decomposition is exact
        # U: (n*m, r), V: (m*p, r), W: (p*n, r)
        # Condition: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k
        # Note: The problem says "exactly decompose".
        # The entries are integer/half-integer.
        
        # We verify by checking all i, j, k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(r):
                        val += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def round_and_verify(U, V, W, r):
        # Round to nearest 0.5
        Ur = [[round(U[i][c] * 2) / 2.0 for c in range(r)] for i in range(n * m)]
        Vr = [[round(V[i][c] * 2) / 2.0 for c in range(r)] for i in range(m * p)]
        Wr = [[round(W[i][c] * 2) / 2.0 for c in range(r)] for i in range(p * n)]
        
        if verify(Ur, Vr, Wr, r):
            return Ur, Vr, Wr
        return None

    # Optimization function
    def optimize_rank(r, max_iters=5000, time_budget=5):
        # Flatten U, V, W into a single vector for optimization
        # Sizes: n*m*r, m*p*r, p*n*r
        size_u = n * m * r
        size_v = m * p * r
        size_w = p * n * r
        total_size = size_u + size_v + size_w
        
        # Random initialization
        x0 = np.random.randn(total_size) * 0.5
        
        def objective(x):
            if time.time() - start_time > time_limit:
                return 1e10
            u_flat = x[:size_u]
            v_flat = x[size_u:size_u + size_v]
            w_flat = x[size_u + size_v:]
            
            U = u_flat.reshape(n * m, r)
            V = v_flat.reshape(m * p, r)
            W = w_flat.reshape(p * n, r)
            
            error = 0.0
            # Calculate sum_c U[i,j,c] V[j,k,c] W[k,i,c] - 1
            # This is a tensor contraction.
            # For efficiency, we can use einsum or loops.
            # Given small sizes, loops are fine but numpy is faster.
            # T[i,j,k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
            # We want T[i,j,k] == 1.
            
            # Reshape to 3D tensors for einsum
            # U is (n, m, r)
            # V is (m, p, r)
            # W is (p, n, r) -- note index k*n+i means first index is k, second is i?
            # W row index is k*n + i. So W[k, i, c].
            
            U3 = U.reshape(n, m, r)
            V3 = V.reshape(m, p, r)
            W3 = W.reshape(p, n, r)
            
            # T[i, j, k] = sum_c U3[i, j, c] * V3[j, k, c] * W3[k, i, c]
            # Using einsum: 'ijc,jkc,kic->ijk'
            T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
            
            # Error is sum (T - 1)^2
            error = np.sum((T - 1.0) ** 2)
            return error

        # Use L-BFGS-B or Nelder-Mead
        # L-BFGS-B is faster for smooth problems
        try:
            from scipy.optimize import minimize
            res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': max_iters, 'maxfun': max_iters * 10})
            if res.success or res.fun < 1e-5:
                x_opt = res.x
                u_flat = x_opt[:size_u]
                v_flat = x_opt[size_u:size_u + size_v]
                w_flat = x_opt[size_u + size_v:]
                
                U = u_flat.reshape(n * m, r)
                V = v_flat.reshape(m * p, r)
                W = w_flat.reshape(p * n, r)
                
                # Convert to list of lists for rounding
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                
                rounded = round_and_verify(U_list, V_list, W_list, r)
                if rounded:
                    return rounded
        except Exception:
            pass
        return None

    # Try ranks from 50 up to 79
    for r in range(50, 80):
        if time.time() - start_time > time_limit - 5:
            break
            
        # Multiple restarts for each rank
        restarts = 0
        while time.time() - start_time < time_limit - 5:
            restarts += 1
            result = optimize_rank(r)
            if result:
                Ur, Vr, Wr = result
                # Update best
                if r < best_rank:
                    best_rank = r
                    best_result = {"U": Ur, "V": Vr, "W": Wr}
                # If we found a low rank, we can stop or try lower
                if r <= 52:
                    return best_result
            if restarts >= 3: # Limit restarts per rank to avoid getting stuck
                break
                
    return best_result
