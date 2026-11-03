import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    num_a = n * m  # 6
    num_b = m * p  # 27
    num_c = p * n  # 18
    
    def solve_2x3x1(rank_target=5, n_restarts=20, max_iters=500):
        best_err = float('inf')
        best = None
        
        # The tensor T for 2x3x1 has non-zero entries at (i,j,0,i) in the flattened space?
        # Indices: U is (2*3, r), V is (3*1, r), W is (1*2, r).
        # U index: i*3+j
        # V index: j*1+0 = j
        # W index: 0*2+i = i
        # We want sum_r U[i*3+j, r] V[j, r] W[i, r] = 1 for all i, j.
        # And 0 for other combinations? 
        # Actually, the tensor is ONLY non-zero for these (i,j) pairs.
        # The "other combinations" don't exist in the 2x3x1 case.
        # Wait, the rank-1 term u_ij v_j w_i contributes to C_i.
        # C_i = sum_j A_ij B_j.
        # The term A_ij B_j is multiplied by sum_r U_ij,r V_j,r W_i,r.
        # We want this sum to be 1 for all i,j.
        # So we just need sum_r U_ij,r V_j,r W_i,r = 1 for all i in {0,1}, j in {0,1,2}.
        
        for r in range(n_restarts):
            if time.time() > deadline - 10:
                break
            
            # Random initialization
            U = np.random.randn(6, rank_target) * 0.5
            V = np.random.randn(3, rank_target) * 0.5
            W = np.random.randn(2, rank_target) * 0.5
            
            # Flatten for optimization
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def objective(x):
                U_flat, V_flat, W_flat = x[:6*rank_target], x[6*rank_target:6*rank_target+3*rank_target], x[6*rank_target+3*rank_target:]
                U = U_flat.reshape(6, rank_target)
                V = V_flat.reshape(3, rank_target)
                W = W_flat.reshape(2, rank_target)
                
                err = 0.0
                for i in range(2):
                    for j in range(3):
                        val = np.dot(U[i*3+j], V[j] * W[i])
                        err += (val - 1.0)**2
                return err
            
            res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': max_iters, 'ftol': 1e-12})
            
            if res.fun < best_err:
                best_err = res.fun
                x_best = res.x
                U_best, V_best, W_best = x_best[:6*rank_target], x_best[6*rank_target:6*rank_target+3*rank_target], x_best[6*rank_target+3*rank_target:]
                best = (U_best.reshape(6, rank_target).copy(), V_best.reshape(3, rank_target).copy(), W_best.reshape(2, rank_target).copy())
                
            if best_err < 1e-12:
                break
        
        if best and best_err < 1e-9:
            return best
        return None

    # Try rank 5 first (total 45)
    res = solve_2x3x1(5, 20, 500)
    
    if res:
        U_block, V_block, W_block = res
        rank = 45
        U = np.zeros((6, rank))
        V = np.zeros((27, rank))
        W = np.zeros((18, rank))
        
        for k in range(9):
            offset = k * 5
            U[:, offset:offset+5] = U_block
            for j in range(3):
                V[j*9+k, offset:offset+5] = V_block[j]
            for i in range(2):
                W[k*2+i, offset:offset+5] = W_block[i]
                
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback to rank 54
    rank = 54
    U = [[0]*rank for _ in range(6)]
    V = [[0]*rank for _ in range(27)]
    W = [[0]*rank for _ in range(18)]
    col = 0
    for i in range(2):
        for j in range(3):
            for k in range(9):
                U[i*3+j][col] = 1
                V[j*9+k][col] = 1
                W[k*2+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
