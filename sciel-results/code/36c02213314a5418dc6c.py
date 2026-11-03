import numpy as np
import time

def construct():
    n, m, p = 2, 3, 8
    
    def get_rank_44():
        rank = 44
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        # Part 1: A[:,0] (2x1) * B[0,:] (1x8) -> 2x8
        for i in range(2):
            for k in range(8):
                U[i * 3 + 0][col] = 1
                V[0 * 8 + k][col] = 1
                W[k * 2 + i][col] = 1
                col += 1
            
        # Part 2: A[:,1:3] (2x2) * B[1:3,:] (2x8) -> 2x8
        for block in range(4):
            b00 = 1*8 + block*2
            b01 = 1*8 + block*2 + 1
            b10 = 2*8 + block*2
            b11 = 2*8 + block*2 + 1
            
            # 1. (a00+a11)(b00+b11)
            U[1][col] = 1; U[5][col] = 1
            V[b00][col] = 1; V[b11][col] = 1
            W[(block*2)*2 + 0][col] = 1
            W[(block*2+1)*2 + 1][col] = 1
            col += 1
            
            # 2. (a10+a11)b00
            U[4][col] = 1; U[5][col] = 1
            V[b00][col] = 1
            W[(block*2)*2 + 1][col] = 1
            W[(block*2+1)*2 + 1][col] = -1
            col += 1
            
            # 3. a00(b01-b11)
            U[1][col] = 1
            V[b01][col] = 1; V[b11][col] = -1
            W[(block*2+1)*2 + 0][col] = 1
            W[(block*2+1)*2 + 1][col] = 1
            col += 1
            
            # 4. a11(b10-b00)
            U[5][col] = 1
            V[b10][col] = 1; V[b00][col] = -1
            W[(block*2)*2 + 0][col] = 1
            W[(block*2)*2 + 1][col] = 1
            col += 1
            
            # 5. (a00+a01)b11
            U[1][col] = 1; U[2][col] = 1
            V[b11][col] = 1
            W[(block*2)*2 + 0][col] = -1
            W[(block*2+1)*2 + 0][col] = 1
            col += 1
            
            # 6. (a10-a00)(b00+b01)
            U[4][col] = 1; U[1][col] = -1
            V[b00][col] = 1; V[b01][col] = 1
            W[(block*2+1)*2 + 1][col] = 1
            col += 1
            
            # 7. (a01-a11)(b10+b11)
            U[2][col] = 1; U[5][col] = -1
            V[b10][col] = 1; V[b11][col] = 1
            W[(block*2)*2 + 0][col] = 1
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
    
    deadline = time.time() + 110
    
    try:
        sol44 = get_rank_44()
        if not verify_solution(sol44):
            return sol44
        
        # Attempt rank 43 using structured optimization
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
        
        # Try dropping each term and optimizing with gradient descent
        for drop_idx in range(44):
            if time.time() > deadline - 10:
                break
            
            U_init = np.delete(U44, drop_idx, axis=0)
            V_init = np.delete(V44, drop_idx, axis=0)
            W_init = np.delete(W44, drop_idx, axis=0)
            
            # Initialize with small random perturbation to escape local minima
            np.random.seed(42 + drop_idx)
            U_opt = U_init + 0.01 * np.random.randn(target_rank, 6)
            V_opt = V_init + 0.01 * np.random.randn(target_rank, 24)
            W_opt = W_init + 0.01 * np.random.randn(target_rank, 16)
            
            # Use alternating least squares with gradient steps
            lr = 0.01
            for step in range(300):
                if time.time() > deadline - 5:
                    break
                
                # Compute current tensor
                T = np.einsum('ur,vr,wr->uvw', U_opt, V_opt, W_opt)
                residual = T - target_tensor
                
                # Update U: minimize ||sum_r U[:,r] outer V[:,r] outer W[:,r] - target||^2
                # Gradient w.r.t U: sum_r residual * (V[:,r] outer W[:,r])
                grad_U = np.einsum('uvw,vr,wr->ur', residual, V_opt, W_opt)
                U_opt -= lr * grad_U
                
                # Update V
                grad_V = np.einsum('uvw,ur,wr->vr', residual, U_opt, W_opt)
                V_opt -= lr * grad_V
                
                # Update W
                grad_W = np.einsum('uvw,ur,vr->wr', residual, U_opt, V_opt)
                W_opt -= lr * grad_W
                
                # Check convergence
                norm = np.linalg.norm(residual)
                if norm < best_norm:
                    best_norm = norm
                    best_params = (U_opt.copy(), V_opt.copy(), W_opt.copy())
                
                if norm < 1e-10:
                    break
                
                # Adaptive learning rate
                if step % 50 == 0:
                    lr *= 0.8
            
            if best_norm < 1e-8:
                break
        
        if best_params is not None and best_norm < 1e-8:
            U, V, W = best_params
            return {"U": U.T.tolist(), "V": V.T.tolist(), "W": W.T.tolist()}
            
    except Exception:
        pass
    
    return get_rank_44()
