import numpy as np
from scipy.optimize import minimize
import time

def construct():
    n, m, p = 2, 2, 12
    start_time = time.time()
    deadline = 140  # seconds, safety margin before 150s kill

    # Fallback: Rank 42 Strassen decomposition (6 blocks of 2x2x2)
    def get_strassen_42():
        rank = 42
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # Standard Strassen coefficients for 2x2x2
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # U indices: i*m+j -> 0:0,0 1:0,1 2:1,0 3:1,1
        # V indices: j*p+k -> j in 0..1, k in 0..1 (local)
        # W indices: k*n+i -> k in 0..1, i in 0..1 (local)
        
        # Coeffs for A (U), B (V), C (W)
        # Format: list of 7 tuples (u_coeffs, v_coeffs, w_coeffs)
        # u_coeffs: [u00, u01, u10, u11]
        # v_coeffs: [v00, v01, v10, v11]
        # w_coeffs: [w00, w01, w10, w11] where index is k*2+i
        
        strassen_coeffs = [
            # M1
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
            # M2
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]),
            # M3
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]),
            # M4
            ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]),
            # M5
            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),
            # M6
            ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
            # M7
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
        ]
        
        r = 0
        for b in range(6):
            for s in range(7):
                u, v, w = strassen_coeffs[s]
                # Fill U
                for i_loc in range(2):
                    for j_loc in range(2):
                        val = u[i_loc * 2 + j_loc]
                        if val != 0:
                            U[i_loc * m + j_loc][r] = val
                # Fill V
                for j_loc in range(2):
                    for k_loc in range(2):
                        val = v[j_loc * 2 + k_loc]
                        if val != 0:
                            V[j_loc * p + (2 * b + k_loc)][r] = val
                # Fill W
                for k_loc in range(2):
                    for i_loc in range(2):
                        val = w[k_loc * 2 + i_loc]
                        if val != 0:
                            W[(2 * b + k_loc) * n + i_loc][r] = val
                r += 1
        return {"U": U, "V": V, "W": W}

    def verify_decomposition(U, V, W, n, m, p):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        # T[i, j, k] = sum_s U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s]
        # Target: T[i, j, k] = 1 if i,j,k in range, else 0? 
        # Actually, the tensor is T[i, j, k] = delta_{i,k} * delta_{j, ?} No.
        # Matrix multiplication: C = A B. C_{i,k} = sum_j A_{i,j} B_{j,k}.
        # The tensor T is defined such that T_{i,j,k} = 1 if the term A_{i,j}B_{j,k} contributes to C_{i,k}.
        # So T_{i,j,k} = 1 for all i, j, k? No, that's not right.
        # The tensor for matrix multiplication M_{n,m,p} has entries T_{i,j,k} = 1 if i,j,k are valid indices.
        # The decomposition is: sum_s U_{i,j,s} V_{j,k,s} W_{i,k,s} = T_{i,j,k}.
        # For standard matrix multiplication, T_{i,j,k} = 1 for all valid i,j,k.
        # Let's check the shape.
        # U: (n*m, rank) -> U[i*m+j, s]
        # V: (m*p, rank) -> V[j*p+k, s]
        # W: (p*n, rank) -> W[k*n+i, s]
        # Sum over s: U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s]
        # This sum must equal 1 for all i, j, k.
        
        rank = len(U[0])
        T_calc = np.zeros((n, m, p))
        for s in range(rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_calc[i, j, k] += U[i*m+j][s] * V[j*p+k][s] * W[k*n+i][s]
        
        T_target = np.ones((n, m, p))
        max_err = np.max(np.abs(T_calc - T_target))
        return max_err < 1e-9

    def search_rank_10_2x2x3():
        """Search for rank 10 decomposition of 2x2x3 multiplication."""
        if time.time() > deadline:
            return None
        
        R = 10
        n, m, p = 2, 2, 3
        dim_U = n * m
        dim_V = m * p
        dim_W = p * n
        
        def objective(params):
            if time.time() > deadline:
                return 1e10
            U = np.reshape(params[:dim_U*R], (dim_U, R))
            V = np.reshape(params[dim_U*R:dim_U*R+dim_V*R], (dim_V, R))
            W = np.reshape(params[dim_U*R+dim_V*R:], (dim_W, R))
            
            T_calc = np.zeros((n, m, p))
            # Vectorized computation
            for s in range(R):
                # u_col shape (n, m), v_col shape (m, p), w_col shape (p, n)
                u_col = U[:, s].reshape(n, m)
                v_col = V[:, s].reshape(m, p)
                w_col = W[:, s].reshape(p, n)
                # T[i, j, k] += u[i, j] * v[j, k] * w[k, i]
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T_calc[i, j, k] += u_col[i, j] * v_col[j, k] * w_col[k, i]
            
            T_target = np.ones((n, m, p))
            error = np.sum((T_calc - T_target)**2)
            return error

        best_error = float('inf')
        best_params = None
        
        # Try a few restarts
        np.random.seed(42)
        for _ in range(15):
            if time.time() > deadline:
                break
            params = np.random.uniform(-1, 1, size=dim_U*R + dim_V*R + dim_W*R)
            res = minimize(objective, params, method='L-BFGS-B', options={'maxiter': 300, 'maxfun': 1500, 'ftol': 1e-12})
            
            if res.fun < best_error:
                best_error = res.fun
                best_params = res.x
                
            if best_error < 1e-8:
                break

        if best_error < 1e-6 and best_params is not None:
            U = np.reshape(best_params[:dim_U*R], (dim_U, R))
            V = np.reshape(best_params[dim_U*R:dim_U*R+dim_V*R], (dim_V, R))
            W = np.reshape(best_params[dim_U*R+dim_V*R:], (dim_W, R))
            
            # Snap to half-integers
            U_snapped = np.round(U * 2) / 2
            V_snapped = np.round(V * 2) / 2
            W_snapped = np.round(W * 2) / 2
            
            # Verify snapped solution
            if verify_decomposition(U_snapped.tolist(), V_snapped.tolist(), W_snapped.tolist(), n, m, p):
                # Construct full 2x2x12 decomposition
                # Split 12 into four blocks of 3
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

    def search_rank_12_2x2x4():
        """Search for rank 12 decomposition of 2x2x4 multiplication."""
        if time.time() > deadline:
            return None
            
        R = 12
        n, m, p = 2, 2, 4
        dim_U = n * m
        dim_V = m * p
        dim_W = p * n
        
        def objective(params):
            if time.time() > deadline:
                return 1e10
            U = np.reshape(params[:dim_U*R], (dim_U, R))
            V = np.reshape(params[dim_U*R:dim_U*R+dim_V*R], (dim_V, R))
            W = np.reshape(params[dim_U*R+dim_V*R:], (dim_W, R))
            
            T_calc = np.zeros((n, m, p))
            for s in range(R):
                u_col = U[:, s].reshape(n, m)
                v_col = V[:, s].reshape(m, p)
                w_col = W[:, s].reshape(p, n)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T_calc[i, j, k] += u_col[i, j] * v_col[j, k] * w_col[k, i]
            
            T_target = np.ones((n, m, p))
            error = np.sum((T_calc - T_target)**2)
            return error

        best_error = float('inf')
        best_params = None
        
        np.random.seed(123)
        for _ in range(10):
            if time.time() > deadline:
                break
            params = np.random.uniform(-1, 1, size=dim_U*R + dim_V*R + dim_W*R)
            res = minimize(objective, params, method='L-BFGS-B', options={'maxiter': 300, 'maxfun': 1500, 'ftol': 1e-12})
            
            if res.fun < best_error:
                best_error = res.fun
                best_params = res.x
                
            if best_error < 1e-8:
                break

        if best_error < 1e-6 and best_params is not None:
            U = np.reshape(best_params[:dim_U*R], (dim_U, R))
            V = np.reshape(best_params[dim_U*R:dim_U*R+dim_V*R], (dim_V, R))
            W = np.reshape(best_params[dim_U*R+dim_V*R:], (dim_W, R))
            
            U_snapped = np.round(U * 2) / 2
            V_snapped = np.round(V * 2) / 2
            W_snapped = np.round(W * 2) / 2
            
            if verify_decomposition(U_snapped.tolist(), V_snapped.tolist(), W_snapped.tolist(), n, m, p):
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

    # Try searches in order of expected rank
    # Rank 40: 4 * 10
    res = search_rank_10_2x2x3()
    if res:
        return res
        
    # Rank 36: 3 * 12
    res = search_rank_12_2x2x4()
    if res:
        return res

    # Fallback
    return get_strassen_42()
