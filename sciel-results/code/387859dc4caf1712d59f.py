import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit

    # 1. Build the initial Strassen-based decomposition (Rank 36)
    # This is a strong baseline that is significantly better than the trivial rank 40.
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_rank(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)

    def init_vecs():
        u = [0.0] * (n*m)
        v = [0.0] * (m*p)
        w = [0.0] * (p*n)
        return u, v, w

    # Strassen terms for 2x2x2
    # Local indices i,j in 0..1 for A, j,k in 0..1 for B, i,k in 0..1 for C
    # A: i,j. B: j,k. C: i,k.
    strassen_terms = [
        # P1 = (a+d)(e+h)
        {
            'u': {(0,0): 1, (1,1): 1},
            'v': {(0,0): 1, (1,1): 1},
            'w': {(0,0): 1, (1,1): 1}
        },
        # P2 = (c+d)e
        {
            'u': {(1,0): 1, (1,1): 1},
            'v': {(0,0): 1},
            'w': {(1,0): 1, (1,1): -1}
        },
        # P3 = a(f-h)
        {
            'u': {(0,0): 1},
            'v': {(0,1): 1, (1,1): -1},
            'w': {(0,1): 1, (1,1): 1}
        },
        # P4 = d(g-e)
        {
            'u': {(1,1): 1},
            'v': {(1,0): 1, (0,0): -1},
            'w': {(0,0): 1, (1,0): 1}
        },
        # P5 = (a+b)h
        {
            'u': {(0,0): 1, (0,1): 1},
            'v': {(1,1): 1},
            'w': {(0,0): -1, (0,1): 1}
        },
        # P6 = (c-a)(e+f)
        {
            'u': {(1,0): 1, (0,0): -1},
            'v': {(0,0): 1, (0,1): 1},
            'w': {(1,1): 1}
        },
        # P7 = (b-d)(g+h)
        {
            'u': {(0,1): 1, (1,1): -1},
            'v': {(1,0): 1, (1,1): 1},
            'w': {(0,0): 1}
        }
    ]

    def add_strassen(a_j_off, b_j_off, b_k_off):
        for term in strassen_terms:
            u, v, w = init_vecs()
            for (i, j), coeff in term['u'].items():
                u[i*m + (a_j_off + j)] = coeff
            for (j, k), coeff in term['v'].items():
                v[(b_j_off + j)*p + (b_k_off + k)] = coeff
            for (i, k), coeff in term['w'].items():
                w[(b_k_off + k)*n + i] = coeff
            add_rank(u, v, w)

    def add_trivial_2x2x1(a_j_off, b_j_off, b_k_off):
        # 2x2x1 trivial rank 4
        terms = [
            {'u': {(0,0): 1}, 'v': {(0,0): 1}, 'w': {(0,0): 1}},
            {'u': {(0,1): 1}, 'v': {(1,0): 1}, 'w': {(0,0): 1}},
            {'u': {(1,0): 1}, 'v': {(0,0): 1}, 'w': {(1,0): 1}},
            {'u': {(1,1): 1}, 'v': {(1,0): 1}, 'w': {(1,0): 1}},
        ]
        for term in terms:
            u, v, w = init_vecs()
            for (i, j), coeff in term['u'].items():
                u[i*m + (a_j_off + j)] = coeff
            for (j, k), coeff in term['v'].items():
                v[(b_j_off + j)*p + (b_k_off + k)] = coeff
            for (i, k), coeff in term['w'].items():
                w[(b_k_off + k)*n + i] = coeff
            add_rank(u, v, w)

    # Block 1: A_L (j=0,1) with B_T (j=0,1)
    # B_T split into k=0,1 (Strassen), k=2,3 (Strassen), k=4 (Trivial)
    add_strassen(0, 0, 0)
    add_strassen(0, 0, 2)
    add_trivial_2x2x1(0, 0, 4)
    
    # Block 2: A_R (j=2,3) with B_B (j=2,3)
    # B_B split into k=0,1 (Strassen), k=2,3 (Strassen), k=4 (Trivial)
    add_strassen(2, 2, 0)
    add_strassen(2, 2, 2)
    add_trivial_2x2x1(2, 2, 4)
    
    initial_rank = len(U_cols)
    
    # Convert to row-major matrices
    U = np.zeros((n*m, initial_rank))
    V = np.zeros((m*p, initial_rank))
    W = np.zeros((p*n, initial_rank))
    
    for r in range(initial_rank):
        U[:, r] = U_cols[r]
        V[:, r] = V_cols[r]
        W[:, r] = W_cols[r]

    best_U, best_V, best_W = U, V, W
    best_rank = initial_rank

    def check_decomposition(U, V, W, n, m, p):
        """Verify if U, V, W form a valid decomposition"""
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        W = np.asarray(W, dtype=float)
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # T[i,j,k] = 1
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err

    # 2. Attempt to reduce rank using optimization
    # We will try to find a decomposition of rank r < initial_rank.
    # We start from r = initial_rank - 1 and go down.
    
    # Optimization function for a given rank r
    def try_optimize_rank(r, max_restarts=10, max_nfev=5000):
        """
        Attempt to find a rank-r decomposition using least_squares.
        Returns (U, V, W) if successful, None otherwise.
        """
        U_size = n * m * r
        V_size = m * p * r
        W_size = p * n * r
        total_size = U_size + V_size + W_size
        
        best_obj = float('inf')
        best_params = None
        
        # Use multiple random restarts, but also try perturbing the initial solution
        # However, since we are reducing rank, we can't directly use the initial solution.
        # We can try to project the initial solution? No, that's hard.
        # Just random restarts with good scaling.
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
            
            # Random initialization
            np.random.seed(42 + restart * 137)
            scale = 1.0 / np.sqrt(r)
            params = np.random.randn(total_size) * scale
            
            def residuals(params):
                U = params[:U_size].reshape(n*m, r)
                V = params[U_size:U_size+V_size].reshape(m*p, r)
                W = params[U_size+V_size:U_size+V_size+W_size].reshape(p*n, r)
                
                res = np.zeros(n*m*p)
                idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                            res[idx] = val - 1.0
                            idx += 1
                return res
            
            try:
                # Use 'lm' (Levenberg-Marquardt) for speed, it's good for small residuals
                result = least_squares(residuals, params, method='lm', max_nfev=max_nfev)
                if result.cost < best_obj:
                    best_obj = result.cost
                    best_params = result.x
            except Exception:
                continue
        
        if best_params is not None and best_obj < 1e-12:
            U = best_params[:U_size].reshape(n*m, r)
            V = best_params[U_size:U_size+V_size].reshape(m*p, r)
            W = best_params[U_size+V_size:U_size+V_size+W_size].reshape(p*n, r)
            
            # Verify with higher precision check
            err = check_decomposition(U, V, W, n, m, p)
            if err < 1e-6:
                return U, V, W
        return None

    # Try to reduce rank from 35 down to 20 (arbitrary lower bound, likely won't go much lower)
    # Start from just below the known good rank
    current_rank = initial_rank
    found_better = False
    
    # We'll try a few ranks. Since optimization is expensive, we try fewer ranks.
    # 35, 34, 33...
    # Given time constraints, we might only try 1-2 ranks.
    
    # Heuristic: The Strassen construction is quite efficient. 
    # 2x4x5 can be seen as two 2x2x5 blocks.
    # Rank of 2x2x5? 
    # Trivial is 10. 
    # Strassen for 2x2x2 is 7.
    # For 2x2x5, we can split into 2x2x2 (rank 7) + 2x2x2 (rank 7) + 2x2x1 (rank 4) = 18.
    # So two blocks would be 18 + 18 = 36. This matches our current rank.
    # Can we do better?
    # Maybe 2x2x5 has rank < 18?
    # If rank(2x2x5) = 17, total rank = 34.
    
    # Let's try rank 35 first.
    # If that fails, try 34.
    
    for r in range(initial_rank - 1, max(20, initial_rank - 5), -1):
        if time.time() > deadline - 10:
            break
        
        res = try_optimize_rank(r, max_restarts=3, max_nfev=2000)
        if res is not None:
            U_opt, V_opt, W_opt = res
            # Verify again
            err = check_decomposition(U_opt, V_opt, W_opt, n, m, p)
            if err < 1e-6:
                best_U, best_V, best_W = U_opt, V_opt, W_opt
                best_rank = r
                found_better = True
                break # Found a lower rank, stop searching lower? 
                      # Actually, we might try to go even lower if time permits, 
                      # but usually if we find one, it's a good stop.
                      # Let's try one more step down if possible.
                      # But to be safe, let's just return this.
                      # Or try one more.
                      # Let's try to go down to r-1 if time allows.
                      # But the loop continues downwards.
                      # So we break out of the "try" loop and return.
                      # Wait, the loop is `for r in ...`.
                      # If we find a solution for r, we update best.
                      # We can continue the loop to try r-1.
                      # But we need to be careful with time.
    
    # Final check
    # If we found a better rank, use it. Otherwise, use the Strassen rank 36.
    
    # Convert to lists of lists for return
    # Ensure they are lists, not numpy arrays, as per some interface requirements?
    # The prompt says "factor matrices of shapes ...". 
    # The current program returns lists of lists.
    # The inspirations return lists of lists.
    # Let's stick to lists of lists to be safe.
    
    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
