import numpy as np
from scipy.optimize import least_squares
import time

def construct():
    n, m, p = 3, 3, 4
    start_time = time.time()
    
    # Dimensions
    dim_u = n * m  # 9
    dim_v = m * p  # 12
    dim_w = p * n  # 12
    
    # Target Tensor T (flattened indices a, b, c)
    # a in [0, 9), b in [0, 12), c in [0, 12)
    # T[a, b, c] = 1 if indices match, 0 otherwise.
    
    def get_target(a, b, c):
        # a = i*m + j => i = a//m, j = a%m
        i_a = a // m
        j_a = a % m
        
        # b = j'*p + k => j' = b%p, k = b//p
        j_b = b % p
        k_b = b // p
        
        # c = k'*n + i' => i' = c%n, k' = c//n
        i_c = c % n
        k_c = c // n
        
        if j_a == j_b and i_a == i_c and k_b == k_c:
            return 1.0
        return 0.0

    # Precompute target vector for all 1296 combinations
    # This is dense, but 1296 is small.
    targets = np.zeros((dim_u, dim_v, dim_w))
    for a in range(dim_u):
        for b in range(dim_v):
            for c in range(dim_w):
                targets[a, b, c] = get_target(a, b, c)
                
    # Flatten targets
    T_flat = targets.flatten()
    
    def residuals(params, U, V, W):
        # params are flattened U, V, W
        # U, V, W are shapes (dim, R)
        # Compute sum_r U[:,r] V[:,r] W[:,r]
        # This is a tensor contraction.
        # res[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        
        # Using np.einsum or loops
        # res = np.zeros((dim_u, dim_v, dim_w))
        # for r in range(R):
        #     res += np.outer(np.outer(U[:,r], V[:,r]), W[:,r])
        
        # Faster:
        # Temp = U @ V.T (dim_u x dim_v x R)? No.
        # Temp[a,b,r] = U[a,r] V[b,r]
        # Res[a,b,c] = sum_r Temp[a,b,r] W[c,r]
        
        # Let's use einsum
        # U: (a, r), V: (b, r), W: (c, r)
        # Result: (a, b, c)
        res = np.einsum('ar,br,cr->abc', U, V, W)
        return res.flatten() - T_flat

    def solve_for_rank(R, max_iters=50, restarts=10):
        # Initialize
        # Use random initialization
        # To help convergence, we can scale down
        
        for _ in range(restarts):
            U = np.random.randn(dim_u, R) * 0.5
            V = np.random.randn(dim_v, R) * 0.5
            W = np.random.randn(dim_w, R) * 0.5
            
            # Optionally, try to align with a known solution?
            # If we start from R=34 seed, we can drop 1 column.
            # But let's try random first.
            
            def fun(x):
                U_curr = x[0:dim_u*R].reshape(dim_u, R)
                V_curr = x[dim_u*R:dim_u*R+dim_v*R].reshape(dim_v, R)
                W_curr = x[-dim_w*R:].reshape(dim_w, R)
                return residuals(x, U_curr, V_curr, W_curr)
                
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                sol = least_squares(fun, x0, method='lm', max_nfev=1000)
                if sol.cost < 1e-6:
                    # Extract solution
                    U_sol = x[0:0] # placeholder
                    x_sol = sol.x
                    U_sol = x_sol[0:dim_u*R].reshape(dim_u, R)
                    V_sol = x_sol[dim_u*R:dim_u*R+dim_v*R].reshape(dim_v, R)
                    W_sol = x_sol[-dim_w*R:].reshape(dim_w, R)
                    
                    # Snap to half-integers
                    U_snap = np.round(U_sol * 2) / 2
                    V_snap = np.round(V_sol * 2) / 2
                    W_snap = np.round(W_sol * 2) / 2
                    
                    # Verify
                    res = np.einsum('ar,br,cr->abc', U_snap, V_snap, W_snap)
                    if np.allclose(res, targets, atol=1e-9):
                        return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
                        
            except Exception:
                pass
                
            if time.time() - start_time > 140:
                break
                
    # Try ranks from 33 down to 30
    for R in [33, 32, 31, 30]:
        res = solve_for_rank(R)
        if res:
            return res
        if time.time() - start_time > 145:
            break
            
    # Fallback to known rank 34 solution
    # Reconstruct the seed solution
    U = [[0] * 34 for _ in range(9)]
    V = [[0] * 34 for _ in range(12)]
    W = [[0] * 34 for _ in range(12)]
    
    col = 0
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        nonlocal col
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val
        col += 1

    # Strassen Left (A11 * B1L)
    U[0][col]=1; U[4][col]=1; V[0][col]=1; V[5][col]=1; W[0][col]=1; W[4][col]=1; col+=1
    U[3][col]=1; U[4][col]=1; V[0][col]=1; W[1][col]=1; W[4][col]=-1; col+=1
    U[0][col]=1; V[1][col]=1; V[5][col]=-1; W[3][col]=1; W[4][col]=1; col+=1
    U[4][col]=1; V[4][col]=1; V[0][col]=-1; W[0][col]=1; W[1][col]=1; col+=1
    U[0][col]=1; U[1][col]=1; V[5][col]=1; W[0][col]=-1; W[3][col]=1; col+=1
    U[3][col]=1; U[0][col]=-1; V[0][col]=1; V[1][col]=1; W[4][col]=1; col+=1
    U[1][col]=1; U[4][col]=-1; V[4][col]=1; V[5][col]=1; W[0][col]=1; col+=1
    
    # A12 * B2L
    U[2][col]=1; V[8][col]=1; W[0][col]=1; col+=1
    U[2][col]=1; V[9][col]=1; W[3][col]=1; col+=1
    U[5][col]=1; V[8][col]=1; W[1][col]=1; col+=1
    U[5][col]=1; V[9][col]=1; W[4][col]=1; col+=1
    
    # A21 * B1L
    U[6][col]=1; V[0][col]=1; W[2][col]=1; col+=1
    U[6][col]=1; V[1][col]=1; W[5][col]=1; col+=1
    U[7][col]=1; V[4][col]=1; W[2][col]=1; col+=1
    U[7][col]=1; V[5][col]=1; W[5][col]=1; col+=1
    
    # A22 * B2L
    U[8][col]=1; V[8][col]=1; W[2][col]=1; col+=1
    U[8][col]=1; V[9][col]=1; W[5][col]=1; col+=1
    
    # Strassen Right (A11 * B1R)
    U[0][col]=1; U[4][col]=1; V[2][col]=1; V[7][col]=1; W[6][col]=1; W[10][col]=1; col+=1
    U[3][col]=1; U[4][col]=1; V[2][col]=1; W[7][col]=1; W[10][col]=-1; col+=1
    U[0][col]=1; V[3][col]=1; V[7][col]=-1; W[9][col]=1; W[10][col]=1; col+=1
    U[4][col]=1; V[6][col]=1; V[2][col]=-1; W[6][col]=1; W[7][col]=1; col+=1
    U[0][col]=1; U[1][col]=1; V[7][col]=1; W[6][col]=-1; W[9][col]=1; col+=1
    U[3][col]=1; U[0][col]=-1; V[2][col]=1; V[3][col]=1; W[10][col]=1; col+=1
    U[1][col]=1; U[4][col]=-1; V[6][col]=1; V[7][col]=1; W[6][col]=1; col+=1
    
    # A12 * B2R
    U[2][col]=1; V[10][col]=1; W[6][col]=1; col+=1
    U[2][col]=1; V[11][col]=1; W[9][col]=1; col+=1
    U[5][col]=1; V[10][col]=1; W[7][col]=1; col+=1
    U[5][col]=1; V[11][col]=1; W[10][col]=1; col+=1
    
    # A21 * B1R
    U[6][col]=1; V[2][col]=1; W[8][col]=1; col+=1
    U[6][col]=1; V[3][col]=1; W[11][col]=1; col+=1
    U[7][col]=1; V[6][col]=1; W[8][col]=1; col+=1
    U[7][col]=1; V[7][col]=1; W[11][col]=1; col+=1
    
    # A22 * B2R
    U[8][col]=1; V[10][col]=1; W[8][col]=1; col+=1
    U[8][col]=1; V[11][col]=1; W[11][col]=1; col+=1
    
    return {"U": U, "V": V, "W": W}
