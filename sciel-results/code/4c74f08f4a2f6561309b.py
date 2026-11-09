"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]
import time
_start_time = time.time()
TIME_LIMIT = 140.0  # 150s limit, 10s safety margin

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=6
    # U shape: (n*m, R) = (6, R)
    # V shape: (m*p, R) = (18, R)  
    # W shape: (p*n, R) = (12, R)
    
    def check_time():
        return (time.time() - _start_time) > TIME_LIMIT
    
    best_result = None
    best_rank = float('inf')
    
    def make_vec(size, idx_vals):
        vec = [0] * size
        for idx, val in idx_vals.items():
            vec[idx] = val
        return vec
    
    def build_from_terms(terms):
        rank = len(terms)
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        for c, (u,v,w) in enumerate(terms):
            for i in range(n*m):
                U[i][c] = u[i]
            for i in range(m*p):
                V[i][c] = v[i]
            for i in range(p*n):
                W[i][c] = w[i]
        return {"U": U, "V": V, "W": W}
    
    def verify_decomposition(U, V, W):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    # Construction 1: Trivial rank 36
    def trivial_decomposition():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Construction 2: Strassen on 2x2 blocks + separate j=2
    def strassen_based_decomposition():
        terms = []
        for i in range(n):
            for k in range(p):
                u_idx = i * m + 2
                v_idx = 2 * p + k
                w_idx = k * n + i
                terms.append((make_vec(n*m, {u_idx: 1}),
                             make_vec(m*p, {v_idx: 1}),
                             make_vec(p*n, {w_idx: 1})))
        
        for j_block in range(3):
            b00 = 0 * p + 2*j_block
            b01 = 0 * p + 2*j_block + 1
            b10 = 1 * p + 2*j_block
            b11 = 1 * p + 2*j_block + 1
            
            c00 = (2*j_block) * n + 0
            c01 = (2*j_block + 1) * n + 0
            c10 = (2*j_block) * n + 1
            c11 = (2*j_block + 1) * n + 1
            
            a00 = 0 * m + 0
            a01 = 0 * m + 1
            a10 = 1 * m + 0
            a11 = 1 * m + 1
            
            terms.append((make_vec(6, {a00:1, a11:1}),
                         make_vec(18, {b00:1, b11:1}),
                         make_vec(12, {c00:1, c11:1})))
            terms.append((make_vec(6, {a10:1, a11:1}),
                         make_vec(18, {b00:1}),
                         make_vec(12, {c10:1, c11:-1})))
            terms.append((make_vec(6, {a00:1}),
                         make_vec(18, {b01:1, b11:-1}),
                         make_vec(12, {c01:1, c11:1})))
            terms.append((make_vec(6, {a11:1}),
                         make_vec(18, {b10:1, b00:-1}),
                         make_vec(12, {c00:1, c10:1})))
            terms.append((make_vec(6, {a00:1, a01:1}),
                         make_vec(18, {b11:1}),
                         make_vec(12, {c00:-1, c01:1})))
            terms.append((make_vec(6, {a10:1, a00:-1}),
                         make_vec(18, {b00:1, b01:1}),
                         make_vec(12, {c11:1})))
            terms.append((make_vec(6, {a01:1, a11:-1}),
                         make_vec(18, {b10:1, b11:1}),
                         make_vec(12, {c00:1})))
        
        return build_from_terms(terms)
    
    # Construction 3: Try different blocking - use j=0,1,2 all together with smarter decomposition
    # For 2x3x6, try to find a rank < 33 decomposition
    def try_improved_decomposition():
        """Try to find a better decomposition using structured search"""
        import numpy as np
        
        # Strategy: Use the fact that we have 2 output rows.
        # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k]
        # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k]
        
        # Try a decomposition where we group B columns differently
        # or use a more efficient algorithm for 2x3x6
        
        # Approach: Use rank-7 for the 2x2 part but try to share more
        # across the 3 column-pairs of B
        
        # Alternative: Try to find if rank 30, 31, 32 is achievable
        # by solving a system of equations
        
        # For a given rank R, we need to find U (6xR), V (18xR), W (12xR)
        # such that for all (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
        
        # This is a nonlinear system. Let's try a structured approach.
        
        # Idea: Fix U and W, solve for V, or use iterative refinement
        # Better idea: Try specific structured patterns
        
        # Pattern 1: Use the same Strassen structure but try to merge some terms
        # by using the fact that j=2 terms can potentially be combined with Strassen terms
        
        # Let's try: for each of the 3 column-pairs, use 7 Strassen terms,
        # but allow the j=2 contributions to be absorbed into the Strassen terms
        
        # Actually, let's try a completely different approach:
        # Use a 2x3x6 specific decomposition
        
        # The tensor T has T[i,j,k]=1 for all i,j,k
        # We need rank R such that T = sum_c u_c ⊗ v_c ⊗ w_c
        
        # Lower bound: max(n*m, m*p, p*n) / something... 
        # Actually, the border rank lower bound is max(n*m, m*p, p*n) = 18
        # But actual rank is likely higher
        
        # Let's try to find a rank-32 or rank-31 decomposition by search
        # over structured decompositions
        
        # Try: 6 blocks of 5 terms each = 30? Or other configurations
        
        # Let me try a systematic search over small rank values
        # using the structure that each term has limited support
        
        for target_rank in range(min(best_rank - 1, 32), 17, -1):
            if check_time():
                break
            
            # Try to construct a valid decomposition at this rank
            # using a greedy/structured approach
            
            # Initialize with the Strassen-based terms and try to merge/modify
            # Start from rank-33 and try to reduce by 1, 2, 3...
            
            # Method: Take the rank-33 decomposition and try to express
            # one or more terms as combinations of others (if possible)
            
            # For now, try a random search with better initialization
            # based on the Strassen structure
            
            np.random.seed(target_rank * 1000 + 42)
            
            # Start from the Strassen decomposition and perturb
            base_terms = []
            for i in range(n):
                for k in range(p):
                    u_idx = i * m + 2
                    v_idx = 2 * p + k
                    w_idx = k * n + i
                    base_terms.append((make_vec(n*m, {u_idx: 1}),
                                     make_vec(m*p, {v_idx: 1}),
                                     make_vec(p*n, {w_idx: 1})))
            
            for j_block in range(3):
                b00 = 0 * p + 2*j_block
                b01 = 0 * p + 2*j_block + 1
                b10 = 1 * p + 2*j_block
                b11 = 1 * p + 2*j_block + 1
                
                c00 = (2*j_block) * n + 0
                c01 = (2*j_block + 1) * n + 0
                c10 = (2*j_block) * n + 1
                c11 = (2*j_block + 1) * n + 1
                
                a00 = 0 * m + 0
                a01 = 0 * m + 1
                a10 = 1 * m + 0
                a11 = 1 * m + 1
                
                base_terms.append((make_vec(6, {a00:1, a11:1}),
                                 make_vec(18, {b00:1, b11:1}),
                                 make_vec(12, {c00:1, c11:1})))
                base_terms.append((make_vec(6, {a10:1, a11:1}),
                                 make_vec(18, {b00:1}),
                                 make_vec(12, {c10:1, c11:-1})))
                base_terms.append((make_vec(6, {a00:1}),
                                 make_vec(18, {b01:1, b11:-1}),
                                 make_vec(12, {c01:1, c11:1})))
                base_terms.append((make_vec(6, {a11:1}),
                                 make_vec(18, {b10:1, b00:-1}),
                                 make_vec(12, {c00:1, c10:1})))
                base_terms.append((make_vec(6, {a00:1, a01:1}),
                                 make_vec(18, {b11:1}),
                                 make_vec(12, {c00:-1, c01:1})))
                base_terms.append((make_vec(6, {a10:1, a00:-1}),
                                 make_vec(18, {b00:1, b01:1}),
                                 make_vec(12, {c11:1})))
                base_terms.append((make_vec(6, {a01:1, a11:-1}),
                                 make_vec(18, {b10:1, b11:1}),
                                 make_vec(12, {c00:1})))
            
            # Try to reduce by removing some terms and adjusting others
            # This is complex; for now, just verify the base and return if better
            base_result = build_from_terms(base_terms)
            if verify_decomposition(base_result["U"], base_result["V"], base_result["W"]):
                base_rank = len(base_result["U"][0])
                if base_rank < best_rank:
                    return base_result
        
        return None
    
    # Try constructions
    for name, decomp_func in [("trivial", trivial_decomposition), 
                               ("strassen", strassen_based_decomposition)]:
        if check_time():
            break
        try:
            result = decomp_func()
            rank = len(result["U"][0])
            if verify_decomposition(result["U"], result["V"], result["W"]):
                if rank < best_rank:
                    best_rank = rank
                    best_result = result
        except Exception:
            pass
    
    # Try improved decomposition
    if not check_time() and best_rank > 28:
        try:
            import numpy as np
            from scipy.optimize import minimize
            
            def residual_from_factors(U, V, W):
                """Compute the residual for a given decomposition"""
                R = U.shape[1]
                res = np.zeros((n, m, p))
                for c in range(R):
                    res += np.einsum('i,j,k->ijk', U[:,c], V[:,c], W[:,c])
                return res - 1.0  # target is 1 everywhere
            
            def objective(params):
                R = len(params) // (6 + 18 + 12)
                U = params[:6*R].reshape(6, R)
                V = params[6*R:6*R+18*R].reshape(18, R)
                W = params[6*R+18*R:].reshape(12, R)
                res = residual_from_factors(U, V, W)
                return np.sum(res**2)
            
            def try_rank(R, seed=42):
                np.random.seed(seed)
                # Initialize with Strassen structure
                U_init = np.zeros((6, R))
                V_init = np.zeros((18, R))
                W_init = np.zeros((12, R))
                
                # Fill first 33 columns with Strassen structure
                for i in range(n):
                    for k in range(p):
                        col = i * 6 + k  # 0..11
                        if col < R:
                            U_init[i*m+2, col] = 1
                            V_init[2*p+k, col] = 1
                            W_init[k*n+i, col] = 1
                
                for j_block in range(3):
                    b00 = 0*p + 2*j_block
                    b01 = 0*p + 2*j_block + 1
                    b10 = 1*p + 2*j_block
                    b11 = 1*p + 2*j_block + 1
                    
                    c00 = (2*j_block)*n + 0
                    c01 = (2*j_block+1)*n + 0
                    c10 = (2*j_block)*n + 1
                    c11 = (2*j_block+1)*n + 1
                    
                    a00 = 0*m + 0
                    a01 = 0*m + 1
                    a10 = 1*m + 0
                    a11 = 1*m + 1
                    
                    strassen_terms = [
                        ({a00:1, a11:1}, {b00:1, b11:1}, {c00:1, c11:1}),
                        ({a10:1, a11:1}, {b00:1}, {c10:1, c11:-1}),
                        ({a00:1}, {b01:1, b11:-1}, {c01:1, c11:1}),
                        ({a11:1}, {b10:1, b00:-1}, {c00:1, c10:1}),
                        ({a00:1, a01:1}, {b11:1}, {c00:-1, c01:1}),
                        ({a10:1, a00:-1}, {b00:1, b01:1}, {c11:1}),
                        ({a01:1, a11:-1}, {b10:1, b11:1}, {c00:1}),
                    ]
                    
                    for t_idx, (u_dict, v_dict, w_dict) in enumerate(strassen_terms):
                        col = 12 + j_block*7 + t_idx
                        if col < R:
                            for idx, val in u_dict.items():
                                U_init[idx, col] = val
                            for idx, val in v_dict.items():
                                V_init[idx, col] = val
                            for idx, val in w_dict.items():
                                W_init[idx, col] = val
                
                # If R < 33, truncate
                if R < 33:
                    U_init = U_init[:, :R]
                    V_init = V_init[:, :R]
                    W_init = W_init[:, :R]
                
                params = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
                
                # Try to optimize
                result = minimize(objective, params, method='L-BFGS-B', 
                                 options={'maxiter': 50, 'ftol': 1e-12})
                
                if result.fun < 1e-10:
                    U_opt = result.x[:6*R].reshape(6, R)
                    V_opt = result.x[6*R:6*R+18*R].reshape(18, R)
                    W_opt = result.x[6*R+18*R:].reshape(12, R)
                    
                    # Round to nearest half-integer
                    U_round = np.round(U_opt * 2) / 2
                    V_round = np.round(V_opt * 2) / 2
                    W_round = np.round(W_opt * 2) / 2
                    
                    # Verify
                    res = residual_from_factors(U_round, V_round, W_round)
                    if np.max(np.abs(res)) < 1e-9:
                        return {"U": U_round.tolist(), "V": V_round.tolist(), "W": W_round.tolist()}
                
                # Try without rounding if exact solution found
                if result.fun < 1e-12:
                    return {"U": result.x[:6*R].reshape(6,R).tolist(), 
                            "V": result.x[6*R:6*R+18*R].reshape(18,R).tolist(), 
                            "W": result.x[6*R+18*R:].reshape(12,R).tolist()}
                
                return None
            
            for R in range(min(best_rank - 1, 32), 26, -1):
                if check_time():
                    break
                found = try_rank(R)
                if found is not None:
                    if verify_decomposition(found["U"], found["V"], found["W"]):
                        best_rank = R
                        best_result = found
                        break
        except Exception:
            pass
    
    if best_result is None:
        best_result = trivial_decomposition()
        best_rank = len(best_result["U"][0])
    
    return best_result
# EVOLVE-BLOCK-END
