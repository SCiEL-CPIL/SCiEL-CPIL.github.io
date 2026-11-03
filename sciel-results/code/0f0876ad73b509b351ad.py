import numpy as np
from scipy.optimize import nnls
import time

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 2, 4, 8
    
    def verify(U, V, W, tol=1e-6):
        """Check if the decomposition is valid using vectorized ops"""
        n, m, p = DIMENSIONS
        rank = U.shape[1]
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        # rec[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # Build rec efficiently
        rec = np.zeros((n, m, p))
        for r in range(rank):
            u_r = U[:, r].reshape(n, m)  # (n, m)
            v_r = V[:, r].reshape(m, p)  # (m, p)
            w_r = W[:, r].reshape(p, n)  # (p, n)
            # rec[i,j,k] += u_r[i,j] * v_r[j,k] * w_r[k,i]
            # = (u_r[:, :, None] * v_r[None, :, :] * w_r[None, :, :].T) summed over... no
            # Actually: for each i,j: u_r[i,j] * v_r[j,:] * w_r[:,i]
            # rec[i,j,k] = u_r[i,j] * v_r[j,k] * w_r[k,i]
            # This is: u_r[:, :, None] * v_r[None, :, :] * w_r.T[None, None, :] ... no
            # w_r is (p, n), so w_r[k,i]. We need w_r.T[i,k] = w_r[k,i]
            # So: rec[i,j,k] = u_r[i,j] * v_r[j,k] * w_r.T[i,k]
            # = (u_r[:, :, None] * v_r[None, :, :] * w_r.T[:, None, :]) ... no, w_r.T is (n,p)
            # w_r.T[i,k] where i in [0,n), k in [0,p)
            # So: rec = u_r[:, :, None] * v_r[None, :, :] * w_r.T[:, None, :]
            # u_r[:, :, None] is (n,m,1), v_r[None,:,:] is (1,m,p), w_r.T[:,None,:] is (n,1,p)
            # Broadcasting: (n,m,1) * (1,m,p) -> (n,m,p), then * (n,1,p) -> (n,m,p) ✓
            rec += u_r[:, :, None] * v_r[None, :, :] * w_r.T[:, None, :]
        
        max_err = np.max(np.abs(rec - 1.0))
        return max_err < tol, max_err

    def gradient_step(U, V, W, lr=0.01):
        """Compute gradient and update using vectorized ops"""
        n, m, p = DIMENSIONS
        rank = U.shape[1]
        
        # Compute rec and error
        rec = np.zeros((n, m, p))
        for r in range(rank):
            u_r = U[:, r].reshape(n, m)
            v_r = V[:, r].reshape(m, p)
            w_r = W[:, r].reshape(p, n)
            rec += u_r[:, :, None] * v_r[None, :, :] * w_r.T[:, None, :]
        
        err = rec - 1.0  # (n, m, p)
        
        # Gradients
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for r in range(rank):
            u_r = U[:, r].reshape(n, m)
            v_r = V[:, r].reshape(m, p)
            w_r = W[:, r].reshape(p, n)
            
            # dU[i*m+j, r] = sum_k err[i,j,k] * v_r[j,k] * w_r[k,i]
            # = sum_k err[i,j,k] * v_r[j,k] * w_r.T[i,k]
            # For fixed i: sum_{j,k} err[i,j,k] * v_r[j,k] * w_r.T[i,k] * e_{ij}... 
            # Actually: dU[i*m+j, r] = sum_k err[i,j,k] * v_r[j,k] * w_r.T[i,k]
            # Let's compute: for each (i,j), dU[i*m+j,r] = dot over k of err[i,j,:]*v_r[j,:]*w_r.T[i,:]
            # dU_flat[i*m+j, r] = sum_k err[i,j,k] * v_r[j,k] * w_r.T[i,k]
            
            # Vectorized: 
            # dU_r[i,j] = sum_k err[i,j,k] * v_r[j,k] * w_r.T[i,k]
            # = sum_k err[i,j,k] * (v_r[j,k] * w_r.T[i,k])
            # Let A[i,j,k] = err[i,j,k], B[j,k] = v_r[j,k], C[i,k] = w_r.T[i,k]
            # dU_r[i,j] = sum_k A[i,j,k] * B[j,k] * C[i,k]
            
            # dV[j*p+k, r] = sum_{i} err[i,j,k] * u_r[i,j] * w_r.T[i,k]
            # dW[k*n+i, r] = sum_{j} err[i,j,k] * u_r[i,j] * v_r[j,k]
            
            # Compute dU_r: (n, m)
            # dU_r[i,j] = sum_k err[i,j,k] * v_r[j,k] * w_r.T[i,k]
            tmp = err * v_r[None, :, :] * w_r.T[:, None, :]  # (n,m,p)
            dU_r = tmp.sum(axis=2)  # (n,m)
            dU[:, r] = dU_r.flatten()
            
            # dV_r[j,k] = sum_i err[i,j,k] * u_r[i,j] * w_r.T[i,k]
            tmp = err * u_r[:, :, None] * w_r.T[:, None, :]  # (n,m,p)
            dV_r = tmp.sum(axis=0)  # (m,p)
            dV[:, r] = dV_r.flatten()
            
            # dW_r[k,i] = sum_j err[i,j,k] * u_r[i,j] * v_r[j,k]
            # dW is indexed by k*n+i, so dW_r[k,i] = sum_j err[i,j,k] * u_r[i,j] * v_r[j,k]
            tmp = err * u_r[:, :, None] * v_r[None, :, :]  # (n,m,p)
            dW_r = tmp.sum(axis=1)  # (n,p) -> dW_r[i,k] = sum_j ...
            # But dW is indexed by k*n+i, so we need dW_r[k,i] = dW_r_T[i,k]
            dW[:, r] = dW_r.T.flatten()  # (p*n,) with index k*n+i
        
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
        return U, V, W, np.max(np.abs(err))

    # Build Strassen-based rank-56 solution as fallback
    U_strassen = np.zeros((n * m, 56), dtype=float)
    V_strassen = np.zeros((m * p, 56), dtype=float)
    W_strassen = np.zeros((p * n, 56), dtype=float)
    
    U_local = np.zeros((4, 7), dtype=float)
    V_local = np.zeros((4, 7), dtype=float)
    W_local = np.zeros((4, 7), dtype=float)
    
    U_local[0,0] = 1; U_local[3,0] = 1
    V_local[0,0] = 1; V_local[3,0] = 1
    W_local[0,0] = 1; W_local[3,0] = 1
    
    U_local[2,1] = 1; U_local[3,1] = 1
    V_local[0,1] = 1
    W_local[2,1] = 1; W_local[3,1] = -1
    
    U_local[0,2] = 1
    V_local[1,2] = 1; V_local[3,2] = -1
    W_local[1,2] = 1; W_local[3,2] = 1
    
    U_local[3,3] = 1
    V_local[2,3] = 1; V_local[0,3] = -1
    W_local[0,3] = 1; W_local[2,3] = 1
    
    U_local[0,4] = 1; U_local[1,4] = 1
    V_local[3,4] = 1
    W_local[0,4] = -1; W_local[1,4] = 1
    
    U_local[2,5] = 1; U_local[0,5] = -1
    V_local[0,5] = 1; V_local[1,5] = 1
    W_local[3,5] = 1
    
    U_local[1,6] = 1; U_local[3,6] = -1
    V_local[2,6] = 1; V_local[3,6] = 1
    W_local[0,6] = 1
    
    for j in range(4):
        for s in range(2):
            base_r = (j * 2 + s) * 7
            
            a_map = [0]*4
            if s == 0:
                a_map[0] = 0*4 + 0; a_map[1] = 0*4 + 1; a_map[2] = 1*4 + 0; a_map[3] = 1*4 + 1
            else:
                a_map[0] = 0*4 + 2; a_map[1] = 0*4 + 3; a_map[2] = 1*4 + 2; a_map[3] = 1*4 + 3
                
            b_map = [0]*4
            if s == 0:
                b_map[0] = 0*8 + (2*j); b_map[1] = 0*8 + (2*j+1)
                b_map[2] = 1*8 + (2*j); b_map[3] = 1*8 + (2*j+1)
            else:
                b_map[0] = 2*8 + (2*j); b_map[1] = 2*8 + (2*j+1)
                b_map[2] = 3*8 + (2*j); b_map[3] = 3*8 + (2*j+1)
                
            c_map = [0]*4
            c_map[0] = (2*j) * 2 + 0; c_map[1] = (2*j+1) * 2 + 0
            c_map[2] = (2*j) * 2 + 1; c_map[3] = (2*j+1) * 2 + 1

            for l in range(7):
                r = base_r + l
                for la in range(4):
                    val = U_local[la, l]
                    if val != 0:
                        U_strassen[a_map[la], r] += val
                for lb in range(4):
                    val = V_local[lb, l]
                    if val != 0:
                        V_strassen[b_map[lb], r] += val
                for lc in range(4):
                    val = W_local[lc, l]
                    if val != 0:
                        W_strassen[c_map[lc], r] += val
    
    # Verify Strassen solution
    valid, err = verify(U_strassen, V_strassen, W_strassen)
    
    # Try to find lower rank decompositions using gradient descent
    best_U, best_V, best_W = U_strassen.copy(), V_strassen.copy(), W_strassen.copy()
    best_err = err
    best_rank = 56
    
    # Try rank 32, 28, 24
    for target_rank in [32, 28, 24]:
        if time.time() - start_time > 100:
            break
        
        for trial in range(3):
            if time.time() - start_time > 110:
                break
            
            seed = 42 + trial * 1000 + target_rank
            rng = np.random.RandomState(seed)
            U = rng.randn(n * m, target_rank) * 0.1
            V = rng.randn(m * p, target_rank) * 0.1
            W = rng.randn(p * n, target_rank) * 0.1
            
            lr = 0.01
            for it in range(200):
                if time.time() - start_time > 115:
                    break
                
                U, V, W, max_err = gradient_step(U, V, W, lr)
                
                if it % 50 == 0 or it == 199:
                    valid, err = verify(U, V, W)
                    if err < best_err and valid:
                        best_err = err
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                        best_rank = target_rank
                    if err < 1e-6:
                        break
                if max_err < 1e-6:
                    break
            
            # Final check
            valid, err = verify(U, V, W)
            if err < best_err and valid:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                best_rank = target_rank
    
    if best_err < 1e-6:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Fallback to Strassen
    return {"U": U_strassen.tolist(), "V": V_strassen.tolist(), "W": W_strassen.tolist()}
# EVOLVE-BLOCK-END
