import numpy as np
import time

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # 40s margin for safety
    
    n, m, p = DIMENSIONS
    
    # Build the structural rank-39 solution
    def build_structural(rank):
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        
        def add_term(u_indices, v_indices, w_indices):
            nonlocal col
            for idx in u_indices:
                U[idx][col] += 1
            for idx in v_indices:
                V[idx][col] += 1
            for idx in w_indices:
                W[idx][col] += 1
            col += 1

        def add_strassen_block(k_start):
            b00 = 0 * 7 + k_start
            b01 = 0 * 7 + k_start + 1
            b10 = 1 * 7 + k_start
            b11 = 1 * 7 + k_start + 1
            
            c00 = k_start * 2 + 0
            c01 = (k_start + 1) * 2 + 0
            c10 = k_start * 2 + 1
            c11 = (k_start + 1) * 2 + 1
            
            add_term([0, 4], [b00, b11], [c00, c11])
            add_term([3, 4], [b00], [c10, c11])
            W[c11][col-1] = -1
            
            add_term([0], [b01, b11], [c01, c11])
            V[b11][col-1] = -1
            
            add_term([4], [b10, b00], [c00, c10])
            V[b00][col-1] = -1
            
            add_term([0, 1], [b11], [c00, c01])
            W[c00][col-1] = -1
            
            add_term([3, 0], [b00, b01], [c11])
            U[0][col-1] = -1
            
            add_term([1, 4], [b10, b11], [c00])
            U[4][col-1] = -1

        add_strassen_block(0)
        add_strassen_block(2)
        add_strassen_block(4)
        
        b06 = 0 * 7 + 6
        b16 = 1 * 7 + 6
        c06 = 6 * 2 + 0
        c16 = 6 * 2 + 1
        
        add_term([0], [b06], [c06])
        add_term([1], [b16], [c06])
        add_term([3], [b06], [c16])
        add_term([4], [b16], [c16])
        
        for k in range(7):
            b2k = 2 * 7 + k
            c0k = k * 2 + 0
            c1k = k * 2 + 1
            add_term([2], [b2k], [c0k])
            add_term([5], [b2k], [c1k])
            
        return {"U": U, "V": V, "W": W}
    
    structural_result = build_structural(39)
    
    # Verify structural solution
    def verify_solution(result):
        U = np.array(result['U'])
        V = np.array(result['V'])
        W = np.array(result['W'])
        rank = U.shape[1]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    alpha = i * m + j
                    beta = j * p + k
                    gamma = k * n + i
                    val = np.sum(U[alpha, :] * V[beta, :] * W[gamma, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    if not verify_solution(structural_result):
        # Should never happen, but return anyway
        return structural_result
    
    # Prepare triples for search
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                triples.append((alpha, beta, gamma))
    
    num_eq = len(triples)
    
    # Try to find lower rank solutions
    best_result = structural_result
    best_rank = 39
    
    # Convert structural to numpy for initialization
    U_struct = np.array(structural_result['U'], dtype=float)
    V_struct = np.array(structural_result['V'], dtype=float)
    W_struct = np.array(structural_result['W'], dtype=float)
    
    # Try rank 38 first, then 37
    for target_rank in [38, 37]:
        if time.time() > deadline - 5:
            break
        
        num_vars = target_rank * (n * m + m * p + p * n)
        
        def residuals(params, R=target_rank):
            U = params[0 : R * n * m].reshape(R, n * m).T
            V = params[R * n * m : R * (n * m + m * p)].reshape(R, m * p).T
            W = params[R * (n * m + m * p) :].reshape(R, p * n).T
            
            res = np.zeros(num_eq)
            for idx, (a, b, g) in enumerate(triples):
                val = np.sum(U[a, :] * V[b, :] * W[g, :])
                res[idx] = val - 1.0
            return res
        
        # Use structural solution as initialization (pad/truncate to target rank)
        # Take first target_rank columns from structural
        init_params = np.concatenate([
            U_struct[:target_rank].flatten(),
            V_struct[:target_rank].flatten(),
            W_struct[:target_rank].flatten()
        ])
        
        # Also try random initialization
        attempts = 0
        max_attempts = 3
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            if attempt == 0:
                params0 = init_params.copy()
            else:
                params0 = np.random.randn(num_vars)
                params0 /= np.sqrt(num_vars)
            
            try:
                from scipy.optimize import least_squares
                sol = least_squares(residuals, params0, method='lm', max_nfev=300)
                
                if sol.success and sol.cost < 1e-6:
                    U = sol.x[0 : target_rank * n * m].reshape(target_rank, n * m).T
                    V = sol.x[target_rank * n * m : target_rank * (n * m + m * p)].reshape(target_rank, m * p).T
                    W = sol.x[target_rank * (n * m + m * p) :].reshape(target_rank, p * n).T
                    
                    # Round to half-integers
                    U_int = np.round(U * 2) / 2
                    V_int = np.round(V * 2) / 2
                    W_int = np.round(W * 2) / 2
                    
                    # Verify exactness
                    valid = True
                    for idx, (a, b, g) in enumerate(triples):
                        val = np.sum(U_int[a, :] * V_int[b, :] * W_int[g, :])
                        if abs(val - 1.0) > 1e-9:
                            valid = False
                            break
                    
                    if valid:
                        result = {
                            'U': U_int.tolist(),
                            'V': V_int.tolist(),
                            'W': W_int.tolist()
                        }
                        best_result = result
                        best_rank = target_rank
                        break  # Found valid solution for this rank, try next lower rank
            except Exception:
                pass
        
        if best_rank < 39:
            break  # Found improvement, no need to try lower ranks
    
    return best_result
# EVOLVE-BLOCK-END
