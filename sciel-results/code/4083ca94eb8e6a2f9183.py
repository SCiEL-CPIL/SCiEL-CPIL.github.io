import numpy as np
from scipy.optimize import nnls
import time

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 2, 4, 8
    
    def verify(U, V, W, tol=1e-6):
        """Check if the decomposition is valid using vectorized numpy operations"""
        n, m, p = DIMENSIONS
        rank = U.shape[1]
        # Precompute index arrays
        i_idx = np.repeat(np.arange(n), m * p)
        j_idx = np.tile(np.repeat(np.arange(m), p), n)
        k_idx = np.tile(np.arange(p), n * m)
        
        u_idx = i_idx * m + j_idx
        v_idx = j_idx * p + k_idx
        w_idx = k_idx * n + i_idx
        
        # Compute reconstruction: rec[i,j,k] = sum_r U[u_idx[i,j,k], r] * V[v_idx[i,j,k], r] * W[w_idx[i,j,k], r]
        # Using einsum: for each (i,j,k), sum over r
        U_sel = U[u_idx, :]  # shape (n*m*p, rank)
        V_sel = V[v_idx, :]  # shape (n*m*p, rank)
        W_sel = W[w_idx, :]  # shape (n*m*p, rank)
        
        rec_flat = np.sum(U_sel * V_sel * W_sel, axis=1)  # shape (n*m*p,)
        rec = rec_flat.reshape(n, m, p)
        T = np.ones((n, m, p))
        max_err = np.max(np.abs(rec - T))
        return max_err < tol, max_err
    
    def build_strassen_rank56():
        """Build a rank-56 Strassen-based decomposition"""
        n, m, p = DIMENSIONS
        U_s = np.zeros((n * m, 56), dtype=float)
        V_s = np.zeros((m * p, 56), dtype=float)
        W_s = np.zeros((p * n, 56), dtype=float)
        
        # Strassen components for 2x2 matrix multiplication
        U_local = np.zeros((4, 7), dtype=float)
        V_local = np.zeros((4, 7), dtype=float)
        W_local = np.zeros((4, 7), dtype=float)
        
        # r=0 (M1)
        U_local[0][0] = 1; U_local[3][0] = 1
        V_local[0][0] = 1; V_local[3][0] = 1
        W_local[0][0] = 1
        W_local[3][0] = 1
        
        # r=1 (M2)
        U_local[2][1] = 1; U_local[3][1] = 1
        V_local[0][1] = 1
        W_local[2][1] = 1
        W_local[3][1] = -1
        
        # r=2 (M3)
        U_local[0][2] = 1
        V_local[1][2] = 1; V_local[3][2] = -1
        W_local[1][2] = 1
        W_local[3][2] = 1
        
        # r=3 (M4)
        U_local[3][3] = 1
        V_local[2][3] = 1; V_local[0][3] = -1
        W_local[0][3] = 1
        W_local[2][3] = 1
        
        # r=4 (M5)
        U_local[0][4] = 1; U_local[1][4] = 1
        V_local[3][4] = 1
        W_local[0][4] = -1
        W_local[1][4] = 1
        
        # r=5 (M6)
        U_local[2][5] = 1; U_local[0][5] = -1
        V_local[0][5] = 1; V_local[1][5] = 1
        W_local[3][5] = 1
        
        # r=6 (M7)
        U_local[1][6] = 1; U_local[3][6] = -1
        V_local[2][6] = 1; V_local[3][6] = 1
        W_local[0][6] = 1
        
        for j in range(4):
            for s in range(2):
                base_r = (j * 2 + s) * 7
                
                a_map = [0]*4
                if s == 0:
                    a_map[0] = 0*4 + 0
                    a_map[1] = 0*4 + 1
                    a_map[2] = 1*4 + 0
                    a_map[3] = 1*4 + 1
                else:
                    a_map[0] = 0*4 + 2
                    a_map[1] = 0*4 + 3
                    a_map[2] = 1*4 + 2
                    a_map[3] = 1*4 + 3
                    
                b_map = [0]*4
                if s == 0:
                    b_map[0] = 0*8 + (2*j)
                    b_map[1] = 0*8 + (2*j+1)
                    b_map[2] = 1*8 + (2*j)
                    b_map[3] = 1*8 + (2*j+1)
                else:
                    b_map[0] = 2*8 + (2*j)
                    b_map[1] = 2*8 + (2*j+1)
                    b_map[2] = 3*8 + (2*j)
                    b_map[3] = 3*8 + (2*j+1)
                    
                c_map = [0]*4
                c_map[0] = (2*j) * 2 + 0
                c_map[1] = (2*j+1) * 2 + 0
                c_map[2] = (2*j) * 2 + 1
                c_map[3] = (2*j+1) * 2 + 1

                for l in range(7):
                    r = base_r + l
                    for la in range(4):
                        val = U_local[la][l]
                        if val != 0:
                            U_s[a_map[la], r] += val
                    for lb in range(4):
                        val = V_local[lb][l]
                        if val != 0:
                            V_s[b_map[lb], r] += val
                    for lc in range(4):
                        val = W_local[lc][l]
                        if val != 0:
                            W_s[c_map[lc], r] += val
        
        return U_s, V_s, W_s
    
    # First, build and verify the rank-56 Strassen solution
    U_strassen, V_strassen, W_strassen = build_strassen_rank56()
    valid56, err56 = verify(U_strassen, V_strassen, W_strassen, tol=1e-6)
    
    if not valid56:
        # If Strassen construction has issues, fall back to schoolbook rank-64
        U_sb = np.zeros((n*m, n*m*p), dtype=float)
        V_sb = np.zeros((m*p, n*m*p), dtype=float)
        W_sb = np.zeros((p*n, n*m*p), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i*m*p + j*p + k
                    U_sb[i*m+j, r] = 1
                    V_sb[j*p+k, r] = 1
                    W_sb[k*n+i, r] = 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    # Try to find a lower rank solution using structured random search
    # Strategy: try rank 32, 36, 40, 44, 48, 52 with multiple random inits
    # Use vectorized gradient computation
    
    def random_init(rank, seed=42):
        rng = np.random.RandomState(seed)
        U = rng.randn(n * m, rank) * 0.5
        V = rng.randn(m * p, rank) * 0.5
        W = rng.randn(p * n, rank) * 0.5
        return U, V, W
    
    def compute_gradient(U, V, W):
        """Compute gradients using vectorized operations"""
        n, m, p = DIMENSIONS
        rank = U.shape[1]
        
        i_idx = np.repeat(np.arange(n), m * p)
        j_idx = np.tile(np.repeat(np.arange(m), p), n)
        k_idx = np.tile(np.arange(p), n * m)
        
        u_idx = i_idx * m + j_idx
        v_idx = j_idx * p + k_idx
        w_idx = k_idx * n + i_idx
        
        U_sel = U[u_idx, :]  # (n*m*p, rank)
        V_sel = V[v_idx, :]  # (n*m*p, rank)
        W_sel = W[w_idx, :]  # (n*m*p, rank)
        
        # rec = sum_r U_sel * V_sel * W_sel
        rec_flat = np.sum(U_sel * V_sel * W_sel, axis=1)
        err_flat = rec_flat - 1.0  # (n*m*p,)
        
        # Gradients:
        # dU[u_idx, r] += err * V_sel * W_sel
        # dV[v_idx, r] += err * U_sel * W_sel
        # dW[w_idx, r] += err * U_sel * V_sel
        
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        # For each of the n*m*p entries, accumulate gradients
        # err_flat[:, None] * V_sel * W_sel has shape (n*m*p, rank)
        # We need to scatter-add these to dU[u_idx, :]
        for idx in range(len(u_idx)):
            dU[u_idx[idx], :] += err_flat[idx] * V_sel[idx, :] * W_sel[idx, :]
            dV[v_idx[idx], :] += err_flat[idx] * U_sel[idx, :] * W_sel[idx, :]
            dW[w_idx[idx], :] += err_flat[idx] * U_sel[idx, :] * V_sel[idx, :]
        
        return dU, dV, dW, np.max(np.abs(err_flat))
    
    best_U, best_V, best_W = None, None, None
    best_err = err56
    
    # Try ranks from 32 up to 52
    for target_rank in [32, 36, 40, 44, 48, 52]:
        if time.time() - start_time > 120:
            break
        if target_rank >= 56:
            break
        
        for trial in range(8):
            if time.time() - start_time > 120:
                break
            
            seed = 100 + target_rank + trial * 10
            U, V, W = random_init(target_rank, seed=seed)
            
            # Gradient descent with adaptive learning rate
            lr = 0.01
            for it in range(150):
                if time.time() - start_time > 120:
                    break
                
                dU, dV, dW, cur_err = compute_gradient(U, V, W)
                
                # Adaptive learning rate
                lr = 0.01 / (1 + it * 0.01)
                
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                if it % 25 == 0 or it == 149:
                    _, cur_err = verify(U, V, W, tol=1e-6)
                    if cur_err < best_err:
                        best_err = cur_err
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    
                    if cur_err < 1e-6:
                        break
            
            # Final check
            _, cur_err = verify(U, V, W, tol=1e-6)
            if cur_err < best_err:
                best_err = cur_err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if best_err < 1e-6:
                break
        
        if best_err < 1e-6:
            break
    
    # Return best solution found
    if best_err < 1e-6 and best_U is not None:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Fallback to rank-56 Strassen solution
    return {"U": U_strassen.tolist(), "V": V_strassen.tolist(), "W": W_strassen.tolist()}
# EVOLVE-BLOCK-END
