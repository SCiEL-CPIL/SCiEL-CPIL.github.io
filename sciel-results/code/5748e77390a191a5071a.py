import numpy as np
import time

def construct():
    n, m, p = 2, 4, 4
    rank_known = 28
    
    # Build the target tensor T[x, y, z] = 1 if x=4i+k, y=4k+j, z=2j+i
    T = np.zeros((8, 16, 8), dtype=float)
    for i in range(n):
        for k in range(m):
            for j in range(p):
                x = i * m + k
                y = k * p + j
                z = j * n + i
                T[x, y, z] = 1
    
    # Build the known rank-28 Strassen solution
    def build_strassen_28():
        rank = 28
        U = np.zeros((8, rank))
        V = np.zeros((16, rank))
        W = np.zeros((8, rank))
        
        def add_block(col_offset, X_rows, Y_rows, Z_rows):
            # M1
            U[X_rows[0], col_offset + 0] += 1
            U[X_rows[3], col_offset + 0] += 1
            V[Y_rows[0], col_offset + 0] += 1
            V[Y_rows[3], col_offset + 0] += 1
            W[Z_rows[0], col_offset + 0] += 1
            W[Z_rows[3], col_offset + 0] += 1
            # M2
            U[X_rows[2], col_offset + 1] += 1
            U[X_rows[3], col_offset + 1] += 1
            V[Y_rows[0], col_offset + 1] += 1
            W[Z_rows[2], col_offset + 1] += 1
            W[Z_rows[3], col_offset + 1] -= 1
            # M3
            U[X_rows[0], col_offset + 2] += 1
            V[Y_rows[1], col_offset + 2] += 1
            V[Y_rows[3], col_offset + 2] -= 1
            W[Z_rows[1], col_offset + 2] += 1
            W[Z_rows[3], col_offset + 2] += 1
            # M4
            U[X_rows[3], col_offset + 3] += 1
            V[Y_rows[2], col_offset + 3] += 1
            V[Y_rows[0], col_offset + 3] -= 1
            W[Z_rows[0], col_offset + 3] += 1
            W[Z_rows[2], col_offset + 3] += 1
            # M5
            U[X_rows[0], col_offset + 4] += 1
            U[X_rows[1], col_offset + 4] += 1
            V[Y_rows[3], col_offset + 4] += 1
            W[Z_rows[0], col_offset + 4] -= 1
            W[Z_rows[1], col_offset + 4] += 1
            # M6
            U[X_rows[2], col_offset + 5] += 1
            U[X_rows[0], col_offset + 5] -= 1
            V[Y_rows[0], col_offset + 5] += 1
            V[Y_rows[1], col_offset + 5] += 1
            W[Z_rows[3], col_offset + 5] += 1
            # M7
            U[X_rows[1], col_offset + 6] += 1
            U[X_rows[3], col_offset + 6] -= 1
            V[Y_rows[2], col_offset + 6] += 1
            V[Y_rows[3], col_offset + 6] += 1
            W[Z_rows[0], col_offset + 6] += 1
        
        X1 = [0, 1, 4, 5]; Y1 = [0, 1, 4, 5]; Z1 = [0, 2, 1, 3]
        add_block(0, X1, Y1, Z1)
        X2 = [2, 3, 6, 7]; Y2 = [8, 9, 12, 13]; Z2 = [0, 2, 1, 3]
        add_block(7, X2, Y2, Z2)
        X3 = [0, 1, 4, 5]; Y3 = [2, 3, 6, 7]; Z3 = [4, 6, 5, 7]
        add_block(14, X3, Y3, Z3)
        X4 = [2, 3, 6, 7]; Y4 = [10, 11, 14, 15]; Z4 = [4, 6, 5, 7]
        add_block(21, X4, Y4, Z4)
        
        return U, V, W
    
    U28, V28, W28 = build_strassen_28()
    
    def check_solution(U, V, W, rank, tol=1e-9):
        # T_est[x,y,z] = sum_r U[x,r]*V[y,r]*W[z,r]
        # Use einsum for efficiency
        T_est = np.einsum('xr,yr,zr->xyz', U, V, W)
        return np.allclose(T_est, T, atol=tol)
    
    # Verify the known solution
    assert check_solution(U28, V28, W28, 28), "Known solution failed verification"
    
    best_rank = 28
    best_U, best_V, best_W = U28, V28, W28
    
    start_time = time.time()
    deadline = start_time + 130  # 20s margin before 150s kill
    
    # Try to find a lower rank solution using continuous optimization + snapping
    # Strategy: For target rank r < 28, initialize from a truncated/random perturbation
    # of the known solution, then do gradient descent on the smooth loss
    # L = sum_{x,y,z} (T_est[x,y,z] - T[x,y,z])^2
    # Then snap to nearest 0.5 and verify.
    
    for target_rank in range(27, 19, -1):
        if time.time() > deadline:
            break
        
        found = False
        # Try a few random initializations
        for restart in range(10):
            if time.time() > deadline:
                break
            
            # Initialize: take first target_rank columns of the known solution
            # and add small random perturbation
            U_try = U28[:, :target_rank].copy() + np.random.randn(8, target_rank) * 0.1
            V_try = V28[:, :target_rank].copy() + np.random.randn(16, target_rank) * 0.1
            W_try = W28[:, :target_rank].copy() + np.random.randn(8, target_rank) * 0.1
            
            # Gradient descent on smooth loss
            lr = 0.01
            for step in range(500):
                if time.time() > deadline:
                    break
                
                # Compute T_est
                T_est = np.einsum('xr,yr,zr->xyz', U_try, V_try, W_try)
                diff = T_est - T
                
                # Gradients:
                # dL/dU[x,r] = 2 * sum_{y,z} diff[x,y,z] * V[y,r] * W[z,r]
                # dL/dV[y,r] = 2 * sum_{x,z} diff[x,y,z] * U[x,r] * W[z,r]
                # dL/dW[z,r] = 2 * sum_{x,y} diff[x,y,z] * U[x,r] * V[y,r]
                
                dU = 2 * np.einsum('xyz,yr,zr->xr', diff, V_try, W_try)
                dV = 2 * np.einsum('xyz,xr,zr->yr', diff, U_try, W_try)
                dW = 2 * np.einsum('xyz,xr,yr->zr', diff, U_try, V_try)
                
                U_try -= lr * dU
                V_try -= lr * dV
                W_try -= lr * dW
                
                # Check if loss is small enough to try snapping
                loss = np.sum(diff**2)
                if loss < 1e-6:
                    break
            
            # Snap to nearest 0.5
            U_snap = np.round(U_try * 2) / 2
            V_snap = np.round(V_try * 2) / 2
            W_snap = np.round(W_try * 2) / 2
            
            if check_solution(U_snap, V_snap, W_snap, target_rank):
                best_rank = target_rank
                best_U, best_V, best_W = U_snap, V_snap, W_snap
                found = True
                break
        
        if found:
            break
    
    # Convert to lists of lists for the return interface
    def to_list(arr):
        return [[float(v) for v in row] for row in arr]
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
