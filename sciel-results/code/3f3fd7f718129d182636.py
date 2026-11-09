import numpy as np
from scipy.optimize import minimize
import time
import random

DIMENSIONS = [3, 3, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # First, try to find a decomposition for 3x3x2
    n, m, p_small = 3, 3, 2
    
    # Target tensor for 3x3x2
    # Indices: i in 0..2, j in 0..2, k in 0..1
    # U index: i*m + j
    # V index: j*p_small + k
    # W index: k*n + i
    
    target = np.zeros((n*m, m*p_small, p_small*n))
    for i in range(n):
        for j in range(m):
            for k in range(p_small):
                u_idx = i * m + j
                v_idx = j * p_small + k
                w_idx = k * n + i
                target[u_idx, v_idx, w_idx] = 1.0
                
    # Function to check if a set of matrices U, V, W decomposes the target
    def check_decomp(U, V, W, target_tensor, tol=1e-9):
        # Reconstruct tensor
        # T[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]
        # Here i is (i,j), j is (j,k), k is (k,i)
        # We can use einsum
        # U shape (nm, R), V shape (mp, R), W shape (pn, R)
        # T shape (nm, mp, pn)
        T = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.allclose(T, target_tensor, atol=tol)
        
    # Function to try to find a decomposition of rank R for 3x3x2
    def find_decomp_332(R, max_restarts=10):
        for _ in range(max_restarts):
            if time.time() > deadline:
                return None
            
            # Random initialization
            U = np.random.randn(n*m, R) * 0.5
            V = np.random.randn(m*p_small, R) * 0.5
            W = np.random.randn(p_small*n, R) * 0.5
            
            # Flatten variables
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def objective(x):
                U_flat, V_flat, W_flat = x[:n*m*R], x[n*m*R:n*m*R+m*p_small*R], x[n*m*R+m*p_small*R:]
                U_curr = U_flat.reshape(n*m, R)
                V_curr = V_flat.reshape(m*p_small, R)
                W_curr = W_flat.reshape(p_small*n, R)
                
                T = np.einsum('ir,jr,kr->ijk', U_curr, V_curr, W_curr)
                error = np.sum((T - target)**2)
                return error
            
            res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10})
            
            if res.fun < 1e-8:
                # Found a solution
                U_flat, V_flat, W_flat = res.x[:n*m*R], res.x[n*m*R:n*m*R+m*p_small*R], res.x[n*m*R+m*p_small*R:]
                U_sol = U_flat.reshape(n*m, R)
                V_sol = V_flat.reshape(m*p_small, R)
                W_sol = W_flat.reshape(p_small*n, R)
                
                # Try to snap to half-integers
                # Scale by 2 to check for integers
                U_snap = np.round(U_sol * 2) / 2
                V_snap = np.round(V_sol * 2) / 2
                W_snap = np.round(W_sol * 2) / 2
                
                if check_decomp(U_snap, V_snap, W_snap, target):
                    return U_snap, V_snap, W_snap
                else:
                    # Maybe the solution is exact but not half-integer?
                    # The problem requires integer/half-integer entries.
                    # If the real solution is not close to half-integers, we can't use it directly.
                    # But for small tensors, solutions are often rational.
                    # Let's try to verify the real solution first.
                    if check_decomp(U_sol, V_sol, W_sol, target):
                         # It is a valid decomposition, but entries are not half-integers.
                         # We can try to find a rational approximation.
                         # For now, return None if not half-integer.
                         pass
                         
        return None

    # Try ranks for 3x3x2
    U_332 = V_332 = W_332 = None
    for R in range(10, 18):
        if time.time() > deadline:
            break
        res = find_decomp_332(R, max_restarts=5)
        if res:
            U_332, V_332, W_332 = res
            break

    if U_332 is not None:
        # Construct 3x3x8 decomposition
        # We have 4 blocks of 3x3x2
        # The 3x3x8 problem has p=8.
        # We can map k in 0..7 to k_block in 0..1 and block_id in 0..3
        # k = 2 * block_id + k_local?
        # Or k_local = k % 2, block_id = k // 2?
        # Let's say k = 2*b + l, where b in 0..3, l in 0..1.
        # Then we use the b-th copy of the 3x3x2 decomposition.
        
        n, m, p = DIMENSIONS
        R_total = 4 * len(U_332[0]) # 4 * R
        
        U = np.zeros((n*m, R_total))
        V = np.zeros((m*p, R_total))
        W = np.zeros((p*n, R_total))
        
        for b in range(4):
            # Copy the b-th block
            offset = b * len(U_332[0])
            
            # U is the same for all blocks (depends only on i, j)
            # Wait, U depends on i, j. It doesn't depend on k.
            # So U can be shared?
            # In the tensor decomposition, each rank component r has a vector u_r, v_r, w_r.
            # If we use the same u_r for all blocks, we need to ensure the v and w are zero for the other blocks.
            # So we can't just share U. We need separate columns for each block.
            # U[:, offset:offset+R] = U_332
            # V[:, offset:offset+R] needs to be filled only for the k's in this block.
            # V row index is j*p + k.
            # For block b, k = 2*b + l.
            # So V[j*p + 2*b + l, offset:offset+R] = V_332[j*2 + l, :]
            # W row index is k*n + i.
            # W[2*b + l, offset:offset+R] = W_332[l*n + i, :]?
            # W_332 index is k_local*n + i.
            # So W[(2*b + l)*n + i, offset:offset+R] = W_332[l*n + i, :]
            
            U[:, offset:offset+R] = U_332
            
            for j in range(m):
                for l in range(2):
                    k = 2 * b + l
                    V[j * p + k, offset:offset+R] = V_332[j * 2 + l, :]
            
            for l in range(2):
                for i in range(n):
                    k = 2 * b + l
                    W[k * n + i, offset:offset+R] = W_332[l * n + i, :]
                    
        # Verify
        target_338 = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    target_338[u_idx, v_idx, w_idx] = 1.0
        
        if check_decomp(U, V, W, target_338):
            # Convert to lists
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}

    # Fallback: Schoolbook
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
