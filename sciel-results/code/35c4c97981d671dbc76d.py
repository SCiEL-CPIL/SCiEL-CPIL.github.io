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
    deadline = 135.0

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

    # Strategy: Direct search for rank < 42
    # We will try to find a decomposition with rank R where R starts from 28 (block decomposition lower bound)
    # and goes up.
    # To make this robust, we use a grid search combined with local optimization.
    
    def attempt_rank(R):
        if time.time() > deadline - 10:
            return None
            
        total_vars = (N_U + N_V + N_W) * R
        # For R=28, total_vars = (6+21+14)*28 = 41*28 = 1148. Manageable.
        # For R=30, total_vars = 1230.
        
        # We'll use a custom optimization loop or least_squares.
        # least_squares is robust.
        
        def residual(params):
            len_U = N_U * R
            len_V = N_V * R
            U = params[:len_U].reshape(N_U, R)
            V = params[len_U:len_U+len_V].reshape(N_V, R)
            W = params[len_U+len_V:].reshape(N_W, R)
            T_calc = np.einsum('ar,br,cr->abc', U, V, W)
            return T_calc.flatten() - T_flat
        
        # Random restarts
        # Given the non-convexity, we need many restarts or good initialization.
        # A good initialization is to start with the schoolbook and perturb, 
        # or use the block structure as a hint.
        
        # Let's try to use the block structure (2x3x3 x 2 + 2x3x1) as a starting point if R >= 28.
        # If we found a 2x3x3 rank 11 decomposition, we could use it.
        # But since the previous code didn't reliably find it, let's try direct search.
        
        # Actually, a smarter initialization: 
        # Random initialization often fails for high dimensions.
        # Let's try a "greedy" or "iterative" approach?
        # Or just use least_squares with many restarts.
        
        num_restarts = 100
        best_sol = None
        best_cost = np.inf
        
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                break
                
            # Initialize
            # If R is close to schoolbook, we can start from schoolbook and remove columns?
            # No, let's just do random uniform.
            x0 = np.random.uniform(-2, 2, total_vars)
            
            try:
                from scipy.optimize import least_squares
                # max_nfev needs to be high enough
                sol = least_squares(residual, x0, method='lm', max_nfev=2000)
                
                if sol.cost < best_cost:
                    best_cost = sol.cost
                    best_sol = sol.x
                    
                    # Check if we found a solution
                    if sol.cost < 1e-12:
                        # Round to half-integers if possible
                        x_rounded = np.round(best_sol * 2) / 2.0
                        res_rounded = residual(x_rounded)
                        if np.linalg.norm(res_rounded) < 1e-9:
                            len_U = N_U * R
                            len_V = N_V * R
                            U = x_rounded[:len_U].reshape(N_U, R)
                            V = x_rounded[len_U:len_U+len_V].reshape(N_V, R)
                            W = x_rounded[len_U+len_V:].reshape(N_W, R)
                            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                        
                        # If rounding fails, return the float solution? 
                        # The task requires integer/half-integer entries.
                        # "integer/half-integer entries"
                        # So we MUST return half-integers.
                        # If we can't round, we failed.
                        
            except Exception as e:
                continue
        
        return None

    # Try ranks from 28 upwards
    # 28 is the theoretical minimum based on 11+11+6.
    # 29, 30...
    
    # Optimization: The space is huge. Random restarts might not find it.
    # Is there a better way?
    
    # Let's try a specific construction for 2x3x7.
    # Maybe we can find a rank 28 decomposition analytically?
    # If 2x3x3 has rank 11, we can construct it.
    # Let's try to hardcode a known rank 11 decomposition for 2x3x3 if possible?
    # Or try to find it with a smaller search space.
    
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
        
        def residual_block(params):
            len_U = NU * R
            len_V = NV * R
            U = params[:len_U].reshape(NU, R)
            V = params[len_U:len_U+len_V].reshape(NV, R)
            W = params[len_U+len_V:].reshape(NW, R)
            T_calc = np.einsum('ar,br,cr->abc', U, V, W)
            return T_calc.flatten() - T_block_flat
            
        total_vars_block = (NU + NV + NW) * R
        num_restarts = 50
        best_sol = None
        best_cost = np.inf
        
        for _ in range(num_restarts):
            if time.time() > deadline - 20:
                break
            x0 = np.random.uniform(-1, 1, total_vars_block)
            try:
                from scipy.optimize import least_squares
                sol = least_squares(residual_block, x0, method='lm', max_nfev=1000)
                if sol.cost < best_cost:
                    best_cost = sol.cost
                    best_sol = sol.x
                    if sol.cost < 1e-12:
                        x_rounded = np.round(best_sol * 2) / 2.0
                        res_rounded = residual_block(x_rounded)
                        if np.linalg.norm(res_rounded) < 1e-9:
                            len_U = NU * R
                            len_V = NV * R
                            U = x_rounded[:len_U].reshape(NU, R)
                            V = x_rounded[len_U:len_U+len_V].reshape(NV, R)
                            W = x_rounded[len_U+len_V:].reshape(NW, R)
                            return U, V, W
            except Exception:
                continue
        return None

    # Try to find 2x3x3 rank 11
    # If successful, we construct the full 2x3x7 rank 28 solution.
    
    U11, V11, W11 = None, None, None
    if time.time() < deadline - 30:
        res11 = find_2x3x3_decomposition(11)
        if res11 is not None:
            U11, V11, W11 = res11

    if U11 is not None:
        # Construct full 2x3x7 solution
        R_block = 11
        R_total = R_block + R_block + 6 # 28
        
        U_full = np.zeros((N_U, R_total))
        V_full = np.zeros((N_V, R_total))
        W_full = np.zeros((N_W, R_total))
        
        # Block 1: Cols 0-2
        U_full[:, 0:R_block] = U11
        for j in range(3):
            for k in range(3):
                local_idx = j * 3 + k
                global_idx = j * 7 + k
                V_full[global_idx, 0:R_block] = V11[local_idx, :]
        W_full[:, 0:R_block] = W11
        
        # Block 2: Cols 3-5
        offset1 = R_block
        U_full[:, offset1:offset1+R_block] = U11
        for j in range(3):
            for k in range(3):
                local_idx = j * 3 + k
                global_idx = j * 7 + (k + 3)
                V_full[global_idx, offset1:offset1+R_block] = V11[local_idx, :]
        # W rows for k=3,4,5 are 6..11
        W_full[6:12, offset1:offset1+R_block] = W11
        
        # Block 3: Col 6
        offset2 = R_block + R_block
        R_block3 = 6
        # 2x3x1 rank 6
        # U is 2x3, V is 3x1, W is 1x2
        # U_full[:, offset2:offset2+6]
        # V_full[:, offset2:offset2+6]
        # W_full[:, offset2:offset2+6]
        
        # Identity-like mapping for 2x3x1
        # a_ij * b_j -> c_i
        # Term: a_0j * b_j -> c_0 (j=0,1,2)
        # Term: a_1j * b_j -> c_1 (j=0,1,2)
        
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
            
        # Verify
        if verify(U_full, V_full, W_full):
            return {
                "U": U_full.tolist(),
                "V": V_full.tolist(),
                "W": W_full.tolist()
            }
            
    # If block decomposition failed, try direct search for rank 29, 30...
    # But random restarts are unlikely to succeed for R>28 if 28 is hard.
    # However, we can try a few more restarts for R=28 if block failed.
    
    if time.time() < deadline - 10:
        # Try R=28 directly with a different initialization strategy?
        # Maybe initialize with the block structure even if 2x3x3 wasn't perfectly found?
        # No, that's risky.
        
        # Try R=29
        # This is likely to fail too.
        
        pass

    return best_result
