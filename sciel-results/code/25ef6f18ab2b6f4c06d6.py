"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # Safety margin before 120s limit
    
    # Build the true tensor T[alpha, beta, gamma] = 1 for multiplication
    dim_u = n * m   # 6
    dim_v = m * p   # 30
    dim_w = p * n   # 20
    
    T_true = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                T_true[alpha, beta, gamma] = 1.0
    
    # Helper: check if a decomposition exactly matches the tensor
    def check_decomposition(U, V, W, tol=1e-8):
        rank = U.shape[1]
        T_rec = np.zeros((dim_u, dim_v, dim_w))
        for c in range(rank):
            T_rec += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
        return np.allclose(T_rec, T_true, atol=tol, rtol=0)
    
    # Helper: round to nearest half-integer and check
    def round_and_check(U, V, W, tol=1e-8):
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        return check_decomposition(U_r, V_r, W_r, tol), U_r, V_r, W_r
    
    # Build the rank-55 Strassen decomposition
    def build_strassen():
        rank = 55
        U = np.zeros((dim_u, rank))
        V = np.zeros((dim_v, rank))
        W = np.zeros((dim_w, rank))
        
        col = 0
        # Part A: j=0
        for i in range(n):
            for k in range(p):
                u_idx = i * m + 0
                v_idx = 0 * p + k
                w_idx = k * n + i
                U[u_idx, col] = 1
                V[v_idx, col] = 1
                W[w_idx, col] = 1
                col += 1
            
        # Part B: j=1,2 - Strassen on 2x2 blocks
        for b in range(5):
            k0 = 2 * b
            k1 = 2 * b + 1
            
            w_00 = k0 * n + 0
            w_01 = k1 * n + 0
            w_10 = k0 * n + 1
            w_11 = k1 * n + 1
            
            u_11 = 0 * m + 1
            u_12 = 0 * m + 2
            u_21 = 1 * m + 1
            u_22 = 1 * m + 2
            
            v_11 = 1 * p + k0
            v_12 = 1 * p + k1
            v_21 = 2 * p + k0
            v_22 = 2 * p + k1
            
            # M1 = (a11+a22)(b11+b22)
            U[u_11, col] += 1; U[u_22, col] += 1
            V[v_11, col] += 1; V[v_22, col] += 1
            W[w_00, col] += 1; W[w_11, col] += 1
            col += 1
            # M2 = (a21+a22)b11
            U[u_21, col] += 1; U[u_22, col] += 1
            V[v_11, col] += 1
            W[w_10, col] += 1; W[w_11, col] += -1
            col += 1
            # M3 = a11(b12-b22)
            U[u_11, col] += 1
            V[v_12, col] += 1; V[v_22, col] += -1
            W[w_01, col] += 1; W[w_11, col] += 1
            col += 1
            # M4 = a22(b21-b11)
            U[u_22, col] += 1
            V[v_21, col] += 1; V[v_11, col] += -1
            W[w_00, col] += 1; W[w_10, col] += 1
            col += 1
            # M5 = (a11+a12)b22
            U[u_11, col] += 1; U[u_12, col] += 1
            V[v_22, col] += 1
            W[w_00, col] += -1; W[w_01, col] += 1
            col += 1
            # M6 = (a21-a11)(b11+b12)
            U[u_21, col] += 1; U[u_11, col] += -1
            V[v_11, col] += 1; V[v_12, col] += 1
            W[w_11, col] += 1
            col += 1
            # M7 = (a12-a22)(b21+b22)
            U[u_12, col] += 1; U[u_22, col] += -1
            V[v_21, col] += 1; V[v_22, col] += 1
            W[w_00, col] += 1
            col += 1
        
        return U, V, W
    
    U_s, V_s, W_s = build_strassen()
    
    # Verify Strassen is correct
    valid, _, _ = round_and_check(U_s, V_s, W_s)
    if not valid:
        # Fallback: schoolbook
        rank_sb = n * m * p
        U_sb = np.zeros((dim_u, rank_sb))
        V_sb = np.zeros((dim_v, rank_sb))
        W_sb = np.zeros((dim_w, rank_sb))
        col_s = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m+j, col_s] = 1
                    V_sb[j*p+k, col_s] = 1
                    W_sb[k*n+i, col_s] = 1
                    col_s += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    best_rank = 55
    best_U, best_V, best_W = U_s.copy(), V_s.copy(), W_s.copy()
    
    # Try ALS optimization for lower ranks
    # For each candidate rank, try to find a valid decomposition
    for target_rank in range(54, 40, -1):
        if time.time() > deadline - 5:
            break
        
        # Multiple random restarts
        n_restarts = 3
        for restart in range(n_restarts):
            if time.time() > deadline - 3:
                break
            
            # Initialize: take first target_rank columns from Strassen, add small noise
            np.random.seed(42 + restart)
            U_init = U_s[:, :target_rank].copy()
            V_init = V_s[:, :target_rank].copy()
            W_init = W_s[:, :target_rank].copy()
            
            # Add small perturbation to escape local minima
            U_init += np.random.randn(dim_u, target_rank) * 0.01
            V_init += np.random.randn(dim_v, target_rank) * 0.01
            W_init += np.random.randn(dim_w, target_rank) * 0.01
            
            # ALS optimization: minimize ||T - sum_r U_r ⊗ V_r ⊗ W_r||_F^2
            # Fix two, solve for the third
            n_iters = 200
            lr = 0.05
            
            for it in range(n_iters):
                if time.time() > deadline - 2:
                    break
                
                # Solve for U (fix V, W)
                # T = sum_r U_r ⊗ V_r ⊗ W_r
                # For each (alpha, beta, gamma): T[alpha,beta,gamma] = sum_r U[alpha,r]*V[beta,r]*W[gamma,r]
                # Fix V,W: for each r, U[alpha,r] should satisfy:
                # sum_r U[alpha,r] * V[beta,r] * W[gamma,r] = T[alpha,beta,gamma]
                # This is a linear system for U[:,r] given V,W
                # Reshape: T as (dim_u, dim_v*dim_w)
                # For each r: contribution is U[:,r] * (V[:,r] ⊗ W[:,r]).T
                # So T_flat = sum_r U[:,r] * (V[:,r] ⊗ W[:,r]).T
                # Let M = [V[:,0]⊗W[:,0], ..., V[:,R-1]⊗W[:,R-1]]  shape (dim_v*dim_w, R)
                # Then T_flat = U * M^T  => U = T_flat * (M^T)^{-1} if possible
                # More generally: U = T_flat @ pinv(M.T)
                
                T_flat = T_true.reshape(dim_u, dim_v * dim_w)
                
                # Build M for U update
                M_u = np.zeros((dim_v * dim_w, target_rank))
                for r in range(target_rank):
                    M_u[:, r] = np.outer(V_init[:, r], W_init[:, r]).flatten()
                
                # Solve U * M_u.T = T_flat => U = T_flat @ pinv(M_u.T)
                try:
                    U_new = T_flat @ np.linalg.pinv(M_u.T)
                except:
                    break
                
                # Solve for V (fix U, W)
                # T_flat2: reshape as (dim_v, dim_u*dim_w)
                T_flat2 = T_true.transpose(1, 0, 2).reshape(dim_v, dim_u * dim_w)
                M_v = np.zeros((dim_u * dim_w, target_rank))
                for r in range(target_rank):
                    M_v[:, r] = np.outer(U_init[:, r], W_init[:, r]).flatten()
                try:
                    V_new = T_flat2 @ np.linalg.pinv(M_v.T)
                except:
                    break
                
                # Solve for W (fix U, V)
                T_flat3 = T_true.transpose(2, 0, 1).reshape(dim_w, dim_u * dim_v)
                M_w = np.zeros((dim_u * dim_v, target_rank))
                for r in range(target_rank):
                    M_w[:, r] = np.outer(U_init[:, r], V_init[:, r]).flatten()
                try:
                    W_new = T_flat3 @ np.linalg.pinv(M_w.T)
                except:
                    break
                
                U_init = U_new
                V_init = V_new
                W_init = W_new
            
            # Check if the solution is valid after rounding
            valid_r, U_r, V_r, W_r = round_and_check(U_init, V_init, W_init, tol=1e-6)
            if valid_r:
                best_rank = target_rank
                best_U, best_V, best_W = U_r, V_r, W_r
                break  # Found valid lower rank, stop searching
        
        if best_rank < 55:
            break  # Found improvement, no need to try lower ranks
    
    # Return the best decomposition found
    if best_rank < 55:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
# EVOLVE-BLOCK-END
