import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    N_U = n * m  # 6
    N_V = m * p  # 21
    N_W = p * n  # 14

    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill

    def get_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(N_U)]
        V = [[0] * rank for _ in range(N_V)]
        W = [[0] * rank for _ in range(N_W)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    best_result = get_schoolbook()

    def build_target_tensor():
        T = np.zeros((N_U, N_V, N_W), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i * m + j, j * p + k, k * n + i] = 1.0
        return T

    T_target = build_target_tensor()
    T_flat = T_target.flatten()

    def verify(U, V, W):
        T_calc = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_calc, T_target, atol=1e-9)

    def check_half_integer(U, V, W):
        # Check if entries are close to half-integers
        for mat in [U, V, W]:
            m_arr = np.array(mat)
            rounded = np.round(m_arr * 2) / 2.0
            if not np.allclose(m_arr, rounded, atol=1e-6):
                return False
        return True

    def attempt_rank(R, num_restarts=200, max_nfev=5000):
        if time.time() > deadline - 10:
            return None
            
        total_vars = (N_U + N_V + N_W) * R
        len_U = N_U * R
        len_V = N_V * R

        def residual(params):
            U = params[:len_U].reshape(N_U, R)
            V = params[len_U:len_U+len_V].reshape(N_V, R)
            W = params[len_U+len_V:].reshape(N_W, R)
            T_calc = np.einsum('ar,br,cr->abc', U, V, W)
            return T_calc.flatten() - T_flat
        
        def residual_norm(params):
            r = residual(params)
            return np.linalg.norm(r)

        best_sol = None
        best_cost = np.inf
        
        # Use scipy least_squares with LM method
        from scipy.optimize import least_squares
        
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                break
                
            # Random initialization
            # Scale is important. The tensor entries are 0 or 1.
            # If R is small, entries might need to be larger.
            # Let's try uniform [-1, 1] first, then maybe [-2, 2]
            x0 = np.random.uniform(-1, 1, total_vars)
            
            try:
                sol = least_squares(residual, x0, method='lm', max_nfev=max_nfev)
                
                if sol.cost < best_cost:
                    best_cost = sol.cost
                    best_sol = sol.x
                    
                    if sol.cost < 1e-10:
                        # Found a solution. Check if it's half-integer.
                        # If not, try to round.
                        x_rounded = np.round(best_sol * 2) / 2.0
                        res_rounded = residual(x_rounded)
                        if np.linalg.norm(res_rounded) < 1e-9:
                            U = x_rounded[:len_U].reshape(N_U, R)
                            V = x_rounded[len_U:len_U+len_V].reshape(N_V, R)
                            W = x_rounded[len_U+len_V:].reshape(N_W, R)
                            if verify(U, V, W):
                                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                        
                        # If rounding fails, maybe we can find a nearby half-integer solution?
                        # Or just return the float solution if the evaluator accepts it?
                        # Task says: "integer/half-integer entries".
                        # So we MUST return half-integers.
                        
            except Exception as e:
                continue
        
        return None

    # We know the schoolbook rank is 42.
    # Theoretical lower bound is max(n*m, m*p, p*n) = 21? No, that's for rectangular.
    # For matrix multiplication C = AB, the rank is at least max(n*m, m*p, p*n) is not correct.
    # The rank is at least the max of the dimensions of the factors? No.
    # The rank of the multiplication tensor M_{n,m,p} is at least max(n*m, m*p, p*n) is false.
    # It is at least n*m*p / max(n,m,p)? No.
    # A known lower bound is max(n*m, m*p, p*n) is not right either.
    # Actually, the rank is at least the dimension of the largest mode? No.
    # For 2x3x7, schoolbook is 42.
    # Let's try to find a rank smaller than 42.
    # Start from 28 (heuristic) up to 41.
    
    # Actually, let's try a specific structured search first.
    # If we can find a rank 30 solution, that's a big win.
    
    # Let's try ranks from 30 to 41.
    # Why 30? 42 is schoolbook. 28 is a common lower bound estimate for small cases.
    # Let's try 35 first as it's closer to 42 and might be easier to find?
    # No, let's try lower ranks first to maximize score improvement.
    
    # However, finding a lower rank is harder.
    # Let's try 35, 34, 33...
    
    # Actually, random restarts are very slow for high dimensions.
    # (6+21+14)*R = 41*R variables.
    # For R=30, 1230 variables.
    # LM method on 1230 variables is slow.
    
    # Let's try a different strategy:
    # Use the "block" structure.
    # 2x3x7 = 2x3x3 + 2x3x3 + 2x3x1
    # If we know rank(2x3x3) and rank(2x3x1), we can construct a solution.
    # rank(2x3x1) = 6 (schoolbook).
    # What is rank(2x3x3)?
    # Schoolbook for 2x3x3 is 18.
    # Can we do better?
    # Let's try to find rank(2x3x3) < 18.
    
    def find_2x3x3_decomposition(target_rank):
        nn, mm, pp = 2, 3, 3
        NU, NV, NW = nn*mm, mm*pp, pp*nn
        R = target_rank
        
        T_block = np.zeros((NU, NV, NW), dtype=np.float64)
        for i in range(nn):
            for j in range(mm):
                for k in range(pp):
                    T_block[i*mm+j, j*pp+k, k*nn+i] = 1.0
        T_block_flat = T_block.flatten()
        
        len_U = NU * R
        len_V = NV * R
        
        def residual_block(params):
            U = params[:len_U].reshape(NU, R)
            V = params[len_U:len_U+len_V].reshape(NV, R)
            W = params[len_U+len_V:].reshape(NW, R)
            T_calc = np.einsum('ar,br,cr->abc', U, V, W)
            return T_calc.flatten() - T_block_flat
            
        total_vars_block = (NU + NV + NW) * R
        num_restarts = 100
        
        from scipy.optimize import least_squares
        
        best_sol = None
        best_cost = np.inf
        
        for _ in range(num_restarts):
            if time.time() > deadline - 20:
                break
            x0 = np.random.uniform(-1, 1, total_vars_block)
            try:
                sol = least_squares(residual_block, x0, method='lm', max_nfev=2000)
                if sol.cost < best_cost:
                    best_cost = sol.cost
                    best_sol = sol.x
                    if sol.cost < 1e-10:
                        x_rounded = np.round(best_sol * 2) / 2.0
                        res_rounded = residual_block(x_rounded)
                        if np.linalg.norm(res_rounded) < 1e-9:
                            U = x_rounded[:len_U].reshape(NU, R)
                            V = x_rounded[len_U:len_U+len_V].reshape(NV, R)
                            W = x_rounded[len_U+len_V:].reshape(NW, R)
                            if verify(U, V, W):
                                return U, V, W
            except Exception:
                continue
        return None

    # Try to find a better 2x3x3 decomposition
    # Schoolbook is 18. Let's try 15, 16, 17.
    # If we find 15, total rank = 15+15+6 = 36.
    # If we find 14, total rank = 14+14+6 = 34.
    
    best_2x3x3 = None
    for r in [15, 16, 17]:
        if time.time() > deadline - 30:
            break
        res = find_2x3x3_decomposition(r)
        if res is not None:
            best_2x3x3 = res
            break

    if best_2x3x3 is not None:
        U11, V11, W11 = best_2x3x3
        R_block = U11.shape[1]
        R_total = R_block + R_block + 6
        
        U_full = np.zeros((N_U, R_total))
        V_full = np.zeros((N_V, R_total))
        W_full = np.zeros((N_W, R_total))
        
        # Block 1: k=0,1,2
        U_full[:, 0:R_block] = U11
        for j in range(3):
            for k in range(3):
                local_idx = j * 3 + k
                global_idx = j * 7 + k
                V_full[global_idx, 0:R_block] = V11[local_idx, :]
        W_full[:, 0:R_block] = W11
        
        # Block 2: k=3,4,5
        offset1 = R_block
        U_full[:, offset1:offset1+R_block] = U11
        for j in range(3):
            for k in range(3):
                local_idx = j * 3 + k
                global_idx = j * 7 + (k + 3)
                V_full[global_idx, offset1:offset1+R_block] = V11[local_idx, :]
        W_full[6:12, offset1:offset1+R_block] = W11
        
        # Block 3: k=6
        offset2 = R_block + R_block
        R_block3 = 6
        col = 0
        for j in range(3):
            # i=0
            U_full[0*3+j, offset2+col] = 1.0
            V_full[j*7+6, offset2+col] = 1.0
            W_full[6*2+0, offset2+col] = 1.0
            col += 1
            # i=1
            U_full[1*3+j, offset2+col] = 1.0
            V_full[j*7+6, offset2+col] = 1.0
            W_full[6*2+1, offset2+col] = 1.0
            col += 1
            
        if verify(U_full, V_full, W_full):
            return {
                "U": U_full.tolist(),
                "V": V_full.tolist(),
                "W": W_full.tolist()
            }

    # If block decomposition didn't improve, try direct search for ranks 35-41
    # This is a fallback.
    
    for R in [35, 36, 37, 38, 39, 40, 41]:
        if time.time() > deadline - 10:
            break
        res = attempt_rank(R, num_restarts=50, max_nfev=1000)
        if res is not None:
            return res

    return best_result
