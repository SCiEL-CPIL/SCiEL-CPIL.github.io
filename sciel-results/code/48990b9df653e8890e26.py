import numpy as np
import time

def construct():
    n, m, p = 2, 3, 8
    
    deadline = time.time() + 125
    
    def build_rank_44():
        rank = 44
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        
        col = 0
        # Part 1: A[:,0] (2x1) * B[0,:] (1x8) -> 2x8  (16 terms)
        for i in range(2):
            for k in range(8):
                U[i * 3 + 0][col] = 1.0
                V[0 * 8 + k][col] = 1.0
                W[k * 2 + i][col] = 1.0
                col += 1
            
        # Part 2: A[:,1:3] (2x2) * B[1:3,:] (2x8) -> 2x8  (28 terms via 4 Strassen blocks)
        for block in range(4):
            b00 = 1*8 + block*2
            b01 = 1*8 + block*2 + 1
            b10 = 2*8 + block*2
            b11 = 2*8 + block*2 + 1
            
            # 1. (a00+a11)(b00+b11) -> c00, c11
            U[1][col] = 1.0; U[5][col] = 1.0
            V[b00][col] = 1.0; V[b11][col] = 1.0
            W[(block*2)*2 + 0][col] = 1.0
            W[(block*2+1)*2 + 1][col] = 1.0
            col += 1
            
            # 2. (a10+a11)b00 -> c10, c11
            U[4][col] = 1.0; U[5][col] = 1.0
            V[b00][col] = 1.0
            W[(block*2)*2 + 1][col] = 1.0
            W[(block*2+1)*2 + 1][col] = -1.0
            col += 1
            
            # 3. a00(b01-b11) -> c01, c11
            U[1][col] = 1.0
            V[b01][col] = 1.0; V[b11][col] = -1.0
            W[(block*2+1)*2 + 0][col] = 1.0
            W[(block*2+1)*2 + 1][col] = 1.0
            col += 1
            
            # 4. a11(b10-b00) -> c00, c10
            U[5][col] = 1.0
            V[b10][col] = 1.0; V[b00][col] = -1.0
            W[(block*2)*2 + 0][col] = 1.0
            W[(block*2)*2 + 1][col] = 1.0
            col += 1
            
            # 5. (a00+a01)b11 -> c00, c01
            U[1][col] = 1.0; U[2][col] = 1.0
            V[b11][col] = 1.0
            W[(block*2)*2 + 0][col] = -1.0
            W[(block*2+1)*2 + 0][col] = 1.0
            col += 1
            
            # 6. (a10-a00)(b00+b01) -> c11
            U[4][col] = 1.0; U[1][col] = -1.0
            V[b00][col] = 1.0; V[b01][col] = 1.0
            W[(block*2+1)*2 + 1][col] = 1.0
            col += 1
            
            # 7. (a01-a11)(b10+b11) -> c00
            U[2][col] = 1.0; U[5][col] = -1.0
            V[b10][col] = 1.0; V[b11][col] = 1.0
            W[(block*2)*2 + 0][col] = 1.0
            col += 1
        
        return {"U": U, "V": V, "W": W}
    
    def verify_solution(sol):
        U = np.array(sol['U']).T
        V = np.array(sol['V']).T
        W = np.array(sol['W']).T
        T = np.einsum('ur,vr,wr->uvw', U, V, W)
        
        target = np.zeros((6, 24, 16))
        for i in range(2):
            for j in range(3):
                for k in range(8):
                    target[i*3+j, j*8+k, k*2+i] = 1.0
        
        return np.allclose(T, target, atol=1e-10)
    
    # Build the rank-44 solution first
    sol44 = build_rank_44()
    
    if not verify_solution(sol44):
        return sol44
    
    # Attempt to find rank-43 decomposition
    target_rank = 43
    U44 = np.array(sol44['U']).T  # shape (44, 6)
    V44 = np.array(sol44['V']).T  # shape (44, 24)
    W44 = np.array(sol44['W']).T  # shape (44, 16)
    
    target_tensor = np.zeros((6, 24, 16))
    for i in range(2):
        for j in range(3):
            for k in range(8):
                target_tensor[i*3+j, j*8+k, k*2+i] = 1.0
    
    best_norm = np.inf
    best_params = None
    
    # Try dropping each term and optimizing with L-BFGS-B
    for drop_idx in range(44):
        if time.time() > deadline:
            break
        
        U_init = np.delete(U44, drop_idx, axis=0)
        V_init = np.delete(V44, drop_idx, axis=0)
        W_init = np.delete(W44, drop_idx, axis=0)
        
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        def residuals(params):
            U = params[:target_rank * 6].reshape(target_rank, 6)
            V = params[target_rank * 6 : target_rank * 30].reshape(target_rank, 24)
            W = params[target_rank * 30 :].reshape(target_rank, 16)
            T = np.einsum('ur,vr,wr->uvw', U, V, W)
            return (T - target_tensor).flatten()
        
        try:
            import scipy.optimize as opt
            result = opt.minimize(
                lambda p: np.sum(residuals(p)**2), 
                x0, 
                method='L-BFGS-B', 
                options={'maxiter': 300, 'ftol': 1e-12}
            )
            if result.success:
                norm = np.sqrt(result.fun)
                if norm < 1e-8 and norm < best_norm:
                    best_norm = norm
                    best_params = result.x
        except:
            continue
    
    if best_params is not None and best_norm < 1e-8:
        U = best_params[:target_rank * 6].reshape(target_rank, 6).T
        V = best_params[target_rank * 6 : target_rank * 30].reshape(target_rank, 24).T
        W = best_params[target_rank * 30 :].reshape(target_rank, 16).T
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return sol44
