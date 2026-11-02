import numpy as np
import time

def construct():
    n, m, p = 2, 3, 8
    dim_U = n * m  # 6
    dim_V = m * p  # 24
    dim_W = p * n  # 16
    
    t_start = time.time()
    deadline = t_start + 140.0
    
    # Build the target tensor
    target_tensor = np.zeros((dim_U, dim_V, dim_W))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target_tensor[i * m + j, j * p + k, k * n + i] = 1.0
    
    def build_rank44():
        rank = 44
        U = [[0] * rank for _ in range(dim_U)]
        V = [[0] * rank for _ in range(dim_V)]
        W = [[0] * rank for _ in range(dim_W)]
        col = 0
        
        # Part 1: j=0, 8 terms (one per k, covering both i values)
        for k in range(p):
            U[0 * m + 0][col] = 1
            U[1 * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + 0][col] = 1
            W[k * n + 1][col] = 1
            col += 1
        
        # Part 2: j=1,2 in 4 blocks of 2x2x2, 7 terms each = 28
        for b in range(4):
            k0 = 2 * b
            k1 = 2 * b + 1
            
            # P1: (A[0,1] + A[1,2]) * (B[1,k0] + B[2,k1])
            U[0 * m + 1][col] = 1
            U[1 * m + 2][col] = 1
            V[1 * p + k0][col] = 1
            V[2 * p + k1][col] = 1
            W[k0 * n + 0][col] = 1
            W[k1 * n + 1][col] = 1
            col += 1
            
            # P2: (A[1,1] + A[1,2]) * B[1,k0]
            U[1 * m + 1][col] = 1
            U[1 * m + 2][col] = 1
            V[1 * p + k0][col] = 1
            W[k0 * n + 1][col] = 1
            W[k1 * n + 1][col] = -1
            col += 1
            
            # P3: A[0,1] * (B[1,k1] - B[2,k1])
            U[0 * m + 1][col] = 1
            V[1 * p + k1][col] = 1
            V[2 * p + k1][col] = -1
            W[k1 * n + 0][col] = 1
            W[k1 * n + 1][col] = 1
            col += 1
            
            # P4: A[1,2] * (B[2,k0] - B[1,k0])
            U[1 * m + 2][col] = 1
            V[2 * p + k0][col] = 1
            V[1 * p + k0][col] = -1
            W[k0 * n + 0][col] = 1
            W[k0 * n + 1][col] = 1
            col += 1
            
            # P5: (A[0,1] + A[0,2]) * B[2,k1]
            U[0 * m + 1][col] = 1
            U[0 * m + 2][col] = 1
            V[2 * p + k1][col] = 1
            W[k0 * n + 0][col] = -1
            W[k1 * n + 0][col] = 1
            col += 1
            
            # P6: (A[1,1] - A[0,1]) * (B[1,k0] + B[1,k1])
            U[1 * m + 1][col] = 1
            U[0 * m + 1][col] = -1
            V[1 * p + k0][col] = 1
            V[1 * p + k1][col] = 1
            W[k1 * n + 1][col] = 1
            col += 1
            
            # P7: (A[0,2] - A[1,2]) * (B[2,k0] + B[2,k1])
            U[0 * m + 2][col] = 1
            U[1 * m + 2][col] = -1
            V[2 * p + k0][col] = 1
            V[2 * p + k1][col] = 1
            W[k0 * n + 0][col] = 1
            col += 1
        
        return {"U": U, "V": V, "W": W}
    
    def verify_solution(sol, tol=1e-10):
        U = np.array(sol['U']).T
        V = np.array(sol['V']).T
        W = np.array(sol['W']).T
        T = np.einsum('ur,vr,wr->uvw', U, V, W)
        return np.allclose(T, target_tensor, atol=tol)
    
    # Get the rank-44 solution as baseline
    sol44 = build_rank44()
    if not verify_solution(sol44):
        return sol44
    
    best_sol = sol44
    best_rank = 44
    
    # Try to find lower rank using least squares optimization
    def try_rank(target_rank, U0, V0, W0, max_nfev=800):
        if time.time() > deadline - 15:
            return None
        
        x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        
        def residuals(params):
            U = params[:target_rank * dim_U].reshape(target_rank, dim_U).T
            V = params[target_rank * dim_U : target_rank * (dim_U + dim_V)].reshape(target_rank, dim_V).T
            W = params[target_rank * (dim_U + dim_V):].reshape(target_rank, dim_W).T
            T = np.einsum('ur,vr,wr->uvw', U, V, W)
            return (T - target_tensor).flatten()
        
        try:
            import scipy.optimize as opt
            result = opt.least_squares(residuals, x0, method='trf', max_nfev=max_nfev,
                                       ftol=1e-14, xtol=1e-14, gtol=1e-14)
            if np.linalg.norm(result.fun) < 1e-8:
                U = result.x[:target_rank * dim_U].reshape(target_rank, dim_U).T
                V = result.x[target_rank * dim_U : target_rank * (dim_U + dim_V)].reshape(target_rank, dim_V).T
                W = result.x[target_rank * (dim_U + dim_V):].reshape(target_rank, dim_W).T
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        except:
            pass
        return None
    
    U44 = np.array(sol44['U']).T
    V44 = np.array(sol44['V']).T
    W44 = np.array(sol44['W']).T
    
    # Try rank 43 by dropping each term and optimizing
    for drop_idx in range(44):
        if time.time() > deadline - 30:
            break
        
        U_init = np.delete(U44, drop_idx, axis=1)
        V_init = np.delete(V44, drop_idx, axis=1)
        W_init = np.delete(W44, drop_idx, axis=1)
        
        sol = try_rank(43, U_init, V_init, W_init, max_nfev=600)
        if sol is not None and verify_solution(sol):
            best_sol = sol
            best_rank = 43
            break
    
    # If we found rank 43, try rank 42
    if best_rank == 43 and time.time() < deadline - 30:
        U43 = np.array(best_sol['U']).T
        V43 = np.array(best_sol['V']).T
        W43 = np.array(best_sol['W']).T
        
        for drop_idx in range(43):
            if time.time() > deadline - 20:
                break
            
            U_init = np.delete(U43, drop_idx, axis=1)
            V_init = np.delete(V43, drop_idx, axis=1)
            W_init = np.delete(W43, drop_idx, axis=1)
            
            sol = try_rank(42, U_init, V_init, W_init, max_nfev=500)
            if sol is not None and verify_solution(sol):
                best_sol = sol
                best_rank = 42
                break
    
    return best_sol
