import numpy as np
import time
import sys
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    N_U = n * m
    N_V = m * p
    N_W = p * n
    
    start_time = time.time()
    deadline = 140.0 # 10s margin

    # Helper to create schoolbook solution
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

    # Step 1: Find rank-11 decomposition for 2x3x3 using ALS
    def find_2x3x3_rank11():
        nn, mm, pp = 2, 3, 3
        NU, NV, NW = nn*mm, mm*pp, pp*nn
        R = 11
        
        # Target Tensor
        T = np.zeros((NU, NV, NW), dtype=np.float64)
        for i in range(nn):
            for j in range(mm):
                for k in range(pp):
                    alpha = i * mm + j
                    beta = j * pp + k
                    gamma = k * nn + i
                    T[alpha, beta, gamma] = 1.0
        T_flat = T.flatten()
        
        def compute_residual(U, V, W):
            T_calc = np.einsum('ar,br,cr->abc', U, V, W)
            return np.linalg.norm(T_calc.flatten() - T_flat)
        
        def solve_als(R, max_iters=200, tol=1e-10):
            best_U = None
            best_V = None
            best_W = None
            best_res = np.inf
            
            for restart in range(30):
                if time.time() > deadline - 25:
                    break
                
                # Random initialization
                U = np.random.randn(NU, R)
                V = np.random.randn(NV, R)
                W = np.random.randn(NW, R)
                
                for it in range(max_iters):
                    if time.time() > deadline - 25:
                        break
                    
                    # Solve for U: T_calc[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
                    # For fixed V, W: T[a,:,:] = U[a,:] @ (V * W.T).T
                    # Reshape: for each a, T[a,b,c] = sum_r U[a,r] * (V[b,r] * W[c,r])
                    # So T_reshaped[a, r] = U[a,r] * sum_{b,c} ... no, better:
                    # T[a,b,c] = sum_r U[a,r] * M1[b,c,r] where M1[b,c,r] = V[b,r]*W[c,r]
                    # This is tricky. Let's use unfolding.
                    
                    # Unfold along a: T_a[a, (b,c)] = sum_r U[a,r] * (V[:,r] * W[:,r].T).T.reshape(-1)
                    # Actually: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
                    # Fix V, W. For each a: T[a,:,:] = sum_r U[a,r] * outer(V[:,r], W[:,r])
                    # So T_a = U @ M where M[r, (b,c)] = V[b,r] * W[c,r]
                    M = np.einsum('br,cr->rbc', V, W).reshape(R, NV*NW)
                    T_a = T.reshape(NU, NV*NW)
                    U_new, _, _, _ = np.linalg.lstsq(M, T_a, rcond=None)
                    
                    # Solve for V: T[b,a,c] = sum_r V[b,r] * U[a,r] * W[c,r]
                    # Fix U, W. M[r, (a,c)] = U[a,r] * W[c,r]
                    M = np.einsum('ar,cr->rac', U, W).reshape(R, NU*NW)
                    T_b = T.transpose(1,0,2).reshape(NV, NU*NW)
                    V_new, _, _, _ = np.linalg.lstsq(M, T_b, rcond=None)
                    
                    # Solve for W: T[c,a,b] = sum_r W[c,r] * U[a,r] * V[b,r]
                    # Fix U, V. M[r, (a,b)] = U[a,r] * V[b,r]
                    M = np.einsum('ar,br->rab', U, V).reshape(R, NU*NV)
                    T_c = T.transpose(2,0,1).reshape(NW, NU*NV)
                    W_new, _, _, _ = np.linalg.lstsq(M, T_c, rcond=None)
                    
                    U, V, W = U_new, V_new, W_new
                    
                    res = compute_residual(U, V, W)
                    if res < best_res:
                        best_res = res
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    
                    if res < tol:
                        break
                
                if best_res < tol:
                    break
            
            if best_res > 1e-6:
                return None
            
            # Round to half-integers
            U_r = np.round(best_U * 2) / 2.0
            V_r = np.round(best_V * 2) / 2.0
            W_r = np.round(best_W * 2) / 2.0
            res_r = compute_residual(U_r, V_r, W_r)
            if res_r < 1e-9:
                return U_r, V_r, W_r
            
            # Try rounding with different strategies
            for scale in [2, 4]:
                U_r = np.round(best_U * scale) / float(scale)
                V_r = np.round(best_V * scale) / float(scale)
                W_r = np.round(best_W * scale) / float(scale)
                res_r = compute_residual(U_r, V_r, W_r)
                if res_r < 1e-9:
                    return U_r, V_r, W_r
            
            return None
        
        return solve_als(R)

    # Step 2: Get rank-6 decomposition for 2x3x1 (Trivial)
    def get_2x3x1_rank6():
        nn, mm, pp = 2, 3, 1
        NU, NV, NW = nn*mm, mm*pp, pp*nn
        R = 6
        U = np.zeros((NU, R))
        V = np.zeros((NV, R))
        W = np.zeros((NW, R))
        
        col = 0
        for j in range(mm):
            # Term a_0j * b_j -> c_0
            U[0*mm + j, col] = 1
            V[j*pp + 0, col] = 1 # V index j
            W[0*nn + 0, col] = 1 # W index 0
            col += 1
            # Term a_1j * b_j -> c_1
            U[1*mm + j, col] = 1
            V[j*pp + 0, col] = 1
            W[1*nn + 0, col] = 1
            col += 1
            
        return U, V, W

    # Attempt to find the rank-11 block, fallback to higher ranks
    U11, V11, W11 = None, None, None
    for try_rank in [11, 12, 13]:
        if time.time() > deadline - 30:
            break
        # Modify find function to accept rank
        def find_rank_R(R):
            nn, mm, pp = 2, 3, 3
            NU, NV, NW = nn*mm, mm*pp, pp*nn
            
            # Target Tensor
            T = np.zeros((NU, NV, NW), dtype=np.float64)
            for i in range(nn):
                for j in range(mm):
                    for k in range(pp):
                        alpha = i * mm + j
                        beta = j * pp + k
                        gamma = k * nn + i
                        T[alpha, beta, gamma] = 1.0
            T_flat = T.flatten()
            
            def compute_residual(U, V, W):
                T_calc = np.einsum('ar,br,cr->abc', U, V, W)
                return np.linalg.norm(T_calc.flatten() - T_flat)
            
            def solve_als(R, max_iters=200, tol=1e-10):
                best_U = None
                best_V = None
                best_W = None
                best_res = np.inf
                
                for restart in range(30):
                    if time.time() > deadline - 30:
                        break
                    
                    # Random initialization
                    U = np.random.randn(NU, R)
                    V = np.random.randn(NV, R)
                    W = np.random.randn(NW, R)
                    
                    for it in range(max_iters):
                        if time.time() > deadline - 30:
                            break
                        
                        # Solve for U
                        M = np.einsum('br,cr->rbc', V, W).reshape(R, NV*NW)
                        T_a = T.reshape(NU, NV*NW)
                        U_new, _, _, _ = np.linalg.lstsq(M, T_a, rcond=None)
                        
                        # Solve for V
                        M = np.einsum('ar,cr->rac', U, W).reshape(R, NU*NW)
                        T_b = T.transpose(1,0,2).reshape(NV, NU*NW)
                        V_new, _, _, _ = np.linalg.lstsq(M, T_b, rcond=None)
                        
                        # Solve for W
                        M = np.einsum('ar,br->rab', U, V).reshape(R, NU*NV)
                        T_c = T.transpose(2,0,1).reshape(NW, NU*NV)
                        W_new, _, _, _ = np.linalg.lstsq(M, T_c, rcond=None)
                        
                        U, V, W = U_new, V_new, W_new
                        
                        res = compute_residual(U, V, W)
                        if res < best_res:
                            best_res = res
                            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                        
                        if res < tol:
                            break
                    
                    if best_res < tol:
                        break
                
                if best_res > 1e-6:
                    return None
                
                # Round to half-integers
                U_r = np.round(best_U * 2) / 2.0
                V_r = np.round(best_V * 2) / 2.0
                W_r = np.round(best_W * 2) / 2.0
                res_r = compute_residual(U_r, V_r, W_r)
                if res_r < 1e-9:
                    return U_r, V_r, W_r
                
                for scale in [4, 8]:
                    U_r = np.round(best_U * scale) / float(scale)
                    V_r = np.round(best_V * scale) / float(scale)
                    W_r = np.round(best_W * scale) / float(scale)
                    res_r = compute_residual(U_r, V_r, W_r)
                    if res_r < 1e-9:
                        return U_r, V_r, W_r
                
                return None
            
            return solve_als(R)
        
        res_R = find_rank_R(try_rank)
        if res_R is not None:
            U11, V11, W11 = res_R
            break

    if U11 is not None:
        # Construct full 2x3x7 solution
        R11 = U11.shape[1]  # Actual rank found
        # Split B (3x7) into B1 (3x3), B2 (3x3), B3 (3x1)
        # C (2x7) into C1 (2x3), C2 (2x3), C3 (2x1)
        
        R_total = R11 + R11 + 6
        U_full = np.zeros((N_U, R_total))
        V_full = np.zeros((N_V, R_total))
        W_full = np.zeros((N_W, R_total))
        
        # Block 1: Cols 0-2 of B, Cols 0-2 of C
        U_full[:, 0:R11] = U11
        for j in range(3):
            for k in range(3):
                local_idx = j * 3 + k
                global_idx = j * 7 + k
                V_full[global_idx, 0:R11] = V11[local_idx, :]
        W_full[:, 0:R11] = W11
        
        # Block 2: Cols 3-5 of B, Cols 3-5 of C
        U_full[:, R11:2*R11] = U11
        for j in range(3):
            for k in range(3):
                local_idx = j * 3 + k
                global_idx = j * 7 + (k + 3)
                V_full[global_idx, R11:2*R11] = V11[local_idx, :]
        W_full[6:12, R11:2*R11] = W11
        
        # Block 3: Col 6 of B, Col 6 of C
        U6, V6, W6 = get_2x3x1_rank6()
        U_full[:, 2*R11:2*R11+6] = U6
        for j in range(3):
            local_idx = j
            global_idx = j * 7 + 6
            V_full[global_idx, 2*R11:2*R11+6] = V6[local_idx, :]
        W_full[12:14, 2*R11:2*R11+6] = W6
        
        # Verify
        T_calc = np.einsum('ar,br,cr->abc', U_full, V_full, W_full)
        
        # Build Target
        T_target = np.zeros((N_U, N_V, N_W), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_target[i*m+j, j*p+k, k*n+i] = 1.0
                    
        if np.allclose(T_calc, T_target, atol=1e-9):
            return {
                "U": U_full.tolist(),
                "V": V_full.tolist(),
                "W": W_full.tolist()
            }
            
    return best_result
# EVOLVE-BLOCK-END
