import numpy as np
from scipy.optimize import minimize
import time

def construct():
    n, m, p = 2, 2, 12
    start_time = time.time()
    deadline = 100  # seconds

    # Target: Rank 42 Strassen decomposition (Fallback)
    def get_strassen_42():
        rank = 42
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        strassen_coeffs = [
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # M1
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]), # M2
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]), # M3
            ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]), # M4
            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]), # M5
            ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]), # M6
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])  # M7
        ]
        
        r = 0
        for b in range(6):
            for s in range(7):
                u, v, w = strassen_coeffs[s]
                for i_loc in range(2):
                    for j_loc in range(2):
                        val = u[i_loc * 2 + j_loc]
                        if val != 0:
                            U[i_loc * m + j_loc][r] = val
                            
                for j_loc in range(2):
                    for k_loc in range(2):
                        val = v[j_loc * 2 + k_loc]
                        if val != 0:
                            V[j_loc * p + (2 * b + k_loc)][r] = val
                            
                for k_loc in range(2):
                    for i_loc in range(2):
                        val = w[k_loc * 2 + i_loc]
                        if val != 0:
                            W[(2 * b + k_loc) * n + i_loc][r] = val
                            
                r += 1
        return {"U": U, "V": V, "W": W}

    # Vectorized objective for 2x2x3 rank R
    def make_objective_2x2x3(R):
        dim_U = 4
        dim_V = 6
        dim_W = 6
        
        # Precompute index arrays
        i_idx = np.repeat(np.arange(2), 2*3)  # [0,0,0,1,1,1, 0,0,0,1,1,1, ...] for U
        j_idx = np.tile(np.repeat(np.arange(2), 3), 2)  # [0,0,0,1,1,1, 0,0,0,1,1,1, ...] for U
        # Actually, let's think in terms of the tensor equation:
        # T[i,j,k] = sum_s U[i*2+j, s] * V[j*3+k, s] * W[i*3+k, s]
        
        # For each s, we need to compute contributions to all (i,j,k)
        # Let's precompute the index patterns
        U_indices = np.array([i*2+j for i in range(2) for j in range(2)])  # [0,1,2,3]
        V_indices_by_jk = np.array([[j*3+k for k in range(3)] for j in range(2)])  # shape (2,3)
        W_indices_by_ik = np.array([[i*3+k for k in range(3)] for i in range(2)])  # shape (2,3)
        
        def objective(params):
            if time.time() > deadline:
                return 1e10
            U = np.reshape(params[:dim_U*R], (dim_U, R))
            V = np.reshape(params[dim_U*R:dim_U*R+dim_V*R], (dim_V, R))
            W = np.reshape(params[dim_U*R+dim_V*R:], (dim_W, R))
            
            T_calc = np.zeros((2, 2, 3))
            for s in range(R):
                u_s = U[:, s]  # shape (4,)
                v_s = V[:, s]  # shape (6,)
                w_s = W[:, s]  # shape (6,)
                
                for i in range(2):
                    for j in range(2):
                        u_val = u_s[i*2+j]
                        for k in range(3):
                            T_calc[i, j, k] += u_val * v_s[j*3+k] * w_s[i*3+k]
            
            T_target = np.ones((2, 2, 3))
            error = np.sum((T_calc - T_target)**2)
            return error
        
        return objective

    # Vectorized objective for 2x2x4 rank R
    def make_objective_2x2x4(R):
        dim_U = 4
        dim_V = 8
        dim_W = 8
        
        def objective(params):
            if time.time() > deadline:
                return 1e10
            U = np.reshape(params[:dim_U*R], (dim_U, R))
            V = np.reshape(params[dim_U*R:dim_U*R+dim_V*R], (dim_V, R))
            W = np.reshape(params[dim_U*R+dim_V*R:], (dim_W, R))
            
            T_calc = np.zeros((2, 2, 4))
            for s in range(R):
                u_s = U[:, s]
                v_s = V[:, s]
                w_s = W[:, s]
                
                for i in range(2):
                    for j in range(2):
                        u_val = u_s[i*2+j]
                        for k in range(4):
                            T_calc[i, j, k] += u_val * v_s[j*4+k] * w_s[i*4+k]
            
            T_target = np.ones((2, 2, 4))
            error = np.sum((T_calc - T_target)**2)
            return error
        
        return objective

    def search_2x2x3(R=7, num_restarts=50):
        """Search for rank-R decomposition of 2x2x3"""
        if time.time() > deadline:
            return None
            
        dim_U = 4
        dim_V = 6
        dim_W = 6
        objective = make_objective_2x2x3(R)
        
        best_error = float('inf')
        best_params = None
        
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
            
            # Try different initialization strategies
            if restart % 3 == 0:
                params = np.random.uniform(-1, 1, size=dim_U*R + dim_V*R + dim_W*R)
            elif restart % 3 == 1:
                params = np.random.normal(0, 1, size=dim_U*R + dim_V*R + dim_W*R)
            else:
                # Sparse initialization
                params = np.zeros(dim_U*R + dim_V*R + dim_W*R)
                num_nonzero = max(1, (dim_U*R + dim_V*R + dim_W*R) // 4)
                nonzero_idx = np.random.choice(len(params), num_nonzero, replace=False)
                params[nonzero_idx] = np.random.uniform(-1, 1, num_nonzero)
            
            res = minimize(objective, params, method='L-BFGS-B', 
                          options={'maxiter': 300, 'maxfun': 2000, 'ftol': 1e-10})
            
            if res.fun < best_error:
                best_error = res.fun
                best_params = res.x.copy()
                
            if best_error < 1e-10:
                break

        if best_error < 1e-8 and best_params is not None:
            U = np.reshape(best_params[:dim_U*R], (dim_U, R))
            V = np.reshape(best_params[dim_U*R:dim_U*R+dim_V*R], (dim_V, R))
            W = np.reshape(best_params[dim_U*R+dim_V*R:], (dim_W, R))
            
            # Try snapping to half-integers
            U_snapped = np.round(U * 2) / 2
            V_snapped = np.round(V * 2) / 2
            W_snapped = np.round(W * 2) / 2
            
            T_calc = np.zeros((2, 2, 3))
            for s in range(R):
                u_s = U_snapped[:, s]
                v_s = V_snapped[:, s]
                w_s = W_snapped[:, s]
                for i in range(2):
                    for j in range(2):
                        u_val = u_s[i*2+j]
                        for k in range(3):
                            T_calc[i, j, k] += u_val * v_s[j*3+k] * w_s[i*3+k]
            
            if np.max(np.abs(T_calc - np.ones((2,2,3)))) < 1e-9:
                # Construct full 2x2x12 decomposition using 4 blocks
                full_rank = R * 4
                full_U = [[0] * full_rank for _ in range(4)]
                full_V = [[0] * full_rank for _ in range(24)]
                full_W = [[0] * full_rank for _ in range(24)]
                
                for block in range(4):
                    offset = block * R
                    for s in range(R):
                        r = offset + s
                        for i in range(2):
                            for j in range(2):
                                full_U[i*2+j][r] = float(U_snapped[i*2+j, s])
                        for j in range(2):
                            for k in range(3):
                                full_V[j*12 + (block*3 + k)][r] = float(V_snapped[j*3+k, s])
                        for i in range(2):
                            for k in range(3):
                                full_W[(block*3 + k)*2 + i][r] = float(W_snapped[i*3+k, s])
                                
                return {"U": full_U, "V": full_V, "W": full_W}
        return None

    def search_2x2x4(R=9, num_restarts=30):
        """Search for rank-R decomposition of 2x2x4"""
        if time.time() > deadline:
            return None
            
        dim_U = 4
        dim_V = 8
        dim_W = 8
        objective = make_objective_2x2x4(R)
        
        best_error = float('inf')
        best_params = None
        
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
            
            if restart % 3 == 0:
                params = np.random.uniform(-1, 1, size=dim_U*R + dim_V*R + dim_W*R)
            elif restart % 3 == 1:
                params = np.random.normal(0, 1, size=dim_U*R + dim_V*R + dim_W*R)
            else:
                params = np.zeros(dim_U*R + dim_V*R + dim_W*R)
                num_nonzero = max(1, (dim_U*R + dim_V*R + dim_W*R) // 4)
                nonzero_idx = np.random.choice(len(params), num_nonzero, replace=False)
                params[nonzero_idx] = np.random.uniform(-1, 1, num_nonzero)
            
            res = minimize(objective, params, method='L-BFGS-B', 
                          options={'maxiter': 300, 'maxfun': 2000, 'ftol': 1e-10})
            
            if res.fun < best_error:
                best_error = res.fun
                best_params = res.x.copy()
                
            if best_error < 1e-10:
                break

        if best_error < 1e-8 and best_params is not None:
            U = np.reshape(best_params[:dim_U*R], (dim_U, R))
            V = np.reshape(best_params[dim_U*R:dim_U*R+dim_V*R], (dim_V, R))
            W = np.reshape(best_params[dim_U*R+dim_V*R:], (dim_W, R))
            
            U_snapped = np.round(U * 2) / 2
            V_snapped = np.round(V * 2) / 2
            W_snapped = np.round(W * 2) / 2
            
            T_calc = np.zeros((2, 2, 4))
            for s in range(R):
                u_s = U_snapped[:, s]
                v_s = V_snapped[:, s]
                w_s = W_snapped[:, s]
                for i in range(2):
                    for j in range(2):
                        u_val = u_s[i*2+j]
                        for k in range(4):
                            T_calc[i, j, k] += u_val * v_s[j*4+k] * w_s[i*4+k]
            
            if np.max(np.abs(T_calc - np.ones((2,2,4)))) < 1e-9:
                full_rank = R * 3
                full_U = [[0] * full_rank for _ in range(4)]
                full_V = [[0] * full_rank for _ in range(24)]
                full_W = [[0] * full_rank for _ in range(24)]
                
                for block in range(3):
                    offset = block * R
                    for s in range(R):
                        r = offset + s
                        for i in range(2):
                            for j in range(2):
                                full_U[i*2+j][r] = float(U_snapped[i*2+j, s])
                        for j in range(2):
                            for k in range(4):
                                full_V[j*12 + (block*4 + k)][r] = float(V_snapped[j*4+k, s])
                        for i in range(2):
                            for k in range(4):
                                full_W[(block*4 + k)*2 + i][r] = float(W_snapped[i*4+k, s])
                                
                return {"U": full_U, "V": full_V, "W": full_W}
        return None

    # Try searches with lower ranks first
    # 2x2x3 with rank 7 (Strassen-like)
    res = search_2x2x3(R=7, num_restarts=30)
    if res:
        return res
    
    # 2x2x3 with rank 8
    res = search_2x2x3(R=8, num_restarts=20)
    if res:
        return res
    
    # 2x2x4 with rank 9
    res = search_2x2x4(R=9, num_restarts=20)
    if res:
        return res
    
    # 2x2x4 with rank 10
    res = search_2x2x4(R=10, num_restarts=15)
    if res:
        return res

    # Fallback
    return get_strassen_42()
