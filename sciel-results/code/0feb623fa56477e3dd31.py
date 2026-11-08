import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    def build_strassen_36():
        rank = 36
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        def u_idx(i, j):
            return i * m + j
        
        def v_idx(j, k):
            return j * p + k
        
        def w_idx(k, i):
            return k * n + i
        
        col = 0
        
        def strassen_block(y_indices, z_indices, w_offsets):
            nonlocal col
            y00, y01, y10, y11 = y_indices
            z00, z01, z10, z11 = z_indices
            
            # M1 = (y00+y11)(z00+z11)
            U[u_idx(y00[0], y00[1])][col] += 1
            U[u_idx(y11[0], y11[1])][col] += 1
            V[v_idx(z00[0], z00[1])][col] += 1
            V[v_idx(z11[0], z11[1])][col] += 1
            W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] += 1
            W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] += 1
            col += 1
            
            # M2 = (y10+y11)z00
            U[u_idx(y10[0], y10[1])][col] += 1
            U[u_idx(y11[0], y11[1])][col] += 1
            V[v_idx(z00[0], z00[1])][col] += 1
            W[w_idx(w_offsets[2][0], w_offsets[2][1])][col] += 1
            W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] -= 1
            col += 1
            
            # M3 = y00(z01-z11)
            U[u_idx(y00[0], y00[1])][col] += 1
            V[v_idx(z01[0], z01[1])][col] += 1
            V[v_idx(z11[0], z11[1])][col] -= 1
            W[w_idx(w_offsets[1][0], w_offsets[1][1])][col] += 1
            W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] += 1
            col += 1
            
            # M4 = y11(z10-z00)
            U[u_idx(y11[0], y11[1])][col] += 1
            V[v_idx(z10[0], z10[1])][col] += 1
            V[v_idx(z00[0], z00[1])][col] -= 1
            W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] += 1
            W[w_idx(w_offsets[2][0], w_offsets[2][1])][col] += 1
            col += 1
            
            # M5 = (y00+y01)z11
            U[u_idx(y00[0], y00[1])][col] += 1
            U[u_idx(y01[0], y01[1])][col] += 1
            V[v_idx(z11[0], z11[1])][col] += 1
            W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] -= 1
            W[w_idx(w_offsets[1][0], w_offsets[1][1])][col] += 1
            col += 1
            
            # M6 = (y10-y00)(z00+z01)
            U[u_idx(y10[0], y10[1])][col] += 1
            U[u_idx(y00[0], y00[1])][col] -= 1
            V[v_idx(z00[0], z00[1])][col] += 1
            V[v_idx(z01[0], z01[1])][col] += 1
            W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] += 1
            col += 1
            
            # M7 = (y01-y11)(z10+z11)
            U[u_idx(y01[0], y01[1])][col] += 1
            U[u_idx(y11[0], y11[1])][col] -= 1
            V[v_idx(z10[0], z10[1])][col] += 1
            V[v_idx(z11[0], z11[1])][col] += 1
            W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] += 1
            col += 1
        
        # Block 1: A_L * B_11
        y1 = [(0,0), (0,1), (1,0), (1,1)]
        z1 = [(0,0), (0,1), (1,0), (1,1)]
        w1 = [(0,0), (1,0), (0,1), (1,1)]
        strassen_block(y1, z1, w1)
        
        # Block 2: A_R * B_21
        y2 = [(0,2), (0,3), (1,2), (1,3)]
        z2 = [(2,0), (2,1), (3,0), (3,1)]
        w2 = [(0,0), (1,0), (0,1), (1,1)]
        strassen_block(y2, z2, w2)
        
        # Block 3: A_L * B_12
        y3 = [(0,0), (0,1), (1,0), (1,1)]
        z3 = [(0,2), (0,3), (1,2), (1,3)]
        w3 = [(2,0), (3,0), (2,1), (3,1)]
        strassen_block(y3, z3, w3)
        
        # Block 4: A_R * B_22
        y4 = [(0,2), (0,3), (1,2), (1,3)]
        z4 = [(2,2), (2,3), (3,2), (3,3)]
        w4 = [(2,0), (3,0), (2,1), (3,1)]
        strassen_block(y4, z4, w4)
        
        # Block 5: A_L * B_15 (2x2 * 2x1)
        U[u_idx(0,0)][col] += 1; V[v_idx(0,4)][col] += 1; W[w_idx(4,0)][col] += 1; col += 1
        U[u_idx(0,1)][col] += 1; V[v_idx(1,4)][col] += 1; W[w_idx(4,0)][col] += 1; col += 1
        U[u_idx(1,0)][col] += 1; V[v_idx(0,4)][col] += 1; W[w_idx(4,1)][col] += 1; col += 1
        U[u_idx(1,1)][col] += 1; V[v_idx(1,4)][col] += 1; W[w_idx(4,1)][col] += 1; col += 1
        
        # Block 6: A_R * B_25 (2x2 * 2x1)
        U[u_idx(0,2)][col] += 1; V[v_idx(2,4)][col] += 1; W[w_idx(4,0)][col] += 1; col += 1
        U[u_idx(0,3)][col] += 1; V[v_idx(3,4)][col] += 1; W[w_idx(4,0)][col] += 1; col += 1
        U[u_idx(1,2)][col] += 1; V[v_idx(2,4)][col] += 1; W[w_idx(4,1)][col] += 1; col += 1
        U[u_idx(1,3)][col] += 1; V[v_idx(3,4)][col] += 1; W[w_idx(4,1)][col] += 1; col += 1
        
        assert col == rank, f"Expected {rank}, got {col}"
        return U, V, W
    
    U0, V0, W0 = build_strassen_36()
    
    # Pre-compute target indices
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target_indices.append((i * m + j, j * p + k, k * n + i))
    
    all_indices = [(a, b, c) for a in range(n*m) for b in range(m*p) for c in range(p*n)]
    target_set = set(target_indices)
    non_target_indices = [idx for idx in all_indices if idx not in target_set]
    
    def verify_solution(U, V, W, rank):
        """Verify that the decomposition is exact."""
        for a, b, c in target_indices:
            val = np.dot(U[a, :], V[b, :]) * W[c, :]
            if abs(val - 1.0) > 1e-8:
                return False
        for a, b, c in non_target_indices:
            val = np.dot(U[a, :], V[b, :]) * W[c, :]
            if abs(val) > 1e-8:
                return False
        return True
    
    def make_result(U, V, W, rank):
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    # Return rank-36 solution as fallback
    fallback_result = make_result(U0, V0, W0, 36)
    
    # Try to find rank 35
    for target_rank in [35, 34, 33]:
        if time.time() > deadline - 10:
            break
        
        # Strategy 1: Drop columns and optimize
        for drop_idx in range(36):
            if time.time() > deadline - 10:
                break
            
            U_init = np.delete(U0, drop_idx, axis=1).copy()
            V_init = np.delete(V0, drop_idx, axis=1).copy()
            W_init = np.delete(W0, drop_idx, axis=1).copy()
            
            if U_init.shape[1] < target_rank:
                pad_size = target_rank - U_init.shape[1]
                U_init = np.hstack([U_init, np.zeros((U_init.shape[0], pad_size))])
                V_init = np.hstack([V_init, np.zeros((V_init.shape[0], pad_size))])
                W_init = np.hstack([W_init, np.zeros((W_init.shape[0], pad_size))])
            
            def residuals(params, r):
                num_u = n * m * r
                num_v = m * p * r
                U_flat = params[:num_u]
                V_flat = params[num_u:num_u + num_v]
                W_flat = params[num_u + num_v:]
                
                U = U_flat.reshape((n * m, r))
                V = V_flat.reshape((m * p, r))
                W = W_flat.reshape((p * n, r))
                
                res = []
                for a, b, c in target_indices:
                    res.append(np.dot(U[a, :], V[b, :]) * W[c, :] - 1.0)
                for a, b, c in non_target_indices:
                    res.append(np.dot(U[a, :], V[b, :]) * W[c, :])
                
                return np.array(res)
            
            params0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                result = least_squares(residuals, params0, args=(target_rank,), method='trf', max_nfev=5000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
                if result.cost < 1e-10:
                    U_sol = result.x[:n*m*target_rank].reshape((n*m, target_rank))
                    V_sol = result.x[n*m*target_rank:n*m*target_rank+m*p*target_rank].reshape((m*p, target_rank))
                    W_sol = result.x[n*m*target_rank+m*p*target_rank:].reshape((p*n, target_rank))
                    
                    # Round to nearest half-integer
                    U_rounded = np.round(U_sol * 2) / 2
                    V_rounded = np.round(V_sol * 2) / 2
                    W_rounded = np.round(W_sol * 2) / 2
                    
                    if verify_solution(U_rounded, V_rounded, W_rounded, target_rank):
                        return make_result(U_rounded, V_rounded, W_rounded, target_rank)
            except Exception:
                pass
        
        # Strategy 2: Random initialization
        attempts = 0
        max_attempts = 30 if target_rank == 35 else 20
        while attempts < max_attempts and time.time() < deadline - 10:
            attempts += 1
            
            U_init = np.random.normal(0, 0.5, (n*m, target_rank))
            V_init = np.random.normal(0, 0.5, (m*p, target_rank))
            W_init = np.random.normal(0, 0.5, (p*n, target_rank))
            
            def residuals(params, r):
                num_u = n * m * r
                num_v = m * p * r
                U_flat = params[:num_u]
                V_flat = params[num_u:num_u + num_v]
                W_flat = params[num_u + num_v:]
                
                U = U_flat.reshape((n * m, r))
                V = V_flat.reshape((m * p, r))
                W = W_flat.reshape((p * n, r))
                
                res = []
                for a, b, c in target_indices:
                    res.append(np.dot(U[a, :], V[b, :]) * W[c, :] - 1.0)
                for a, b, c in non_target_indices:
                    res.append(np.dot(U[a, :], V[b, :]) * W[c, :])
                
                return np.array(res)
            
            params0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                result = least_squares(residuals, params0, args=(target_rank,), method='trf', max_nfev=3000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
                if result.cost < 1e-10:
                    U_sol = result.x[:n*m*target_rank].reshape((n*m, target_rank))
                    V_sol = result.x[n*m*target_rank:n*m*target_rank+m*p*target_rank].reshape((m*p, target_rank))
                    W_sol = result.x[n*m*target_rank+m*p*target_rank:].reshape((p*n, target_rank))
                    
                    U_rounded = np.round(U_sol * 2) / 2
                    V_rounded = np.round(V_sol * 2) / 2
                    W_rounded = np.round(W_sol * 2) / 2
                    
                    if verify_solution(U_rounded, V_rounded, W_rounded, target_rank):
                        return make_result(U_rounded, V_rounded, W_rounded, target_rank)
            except Exception:
                pass
    
    return fallback_result
