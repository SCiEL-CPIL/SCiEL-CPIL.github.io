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
            rec += u_r[:, :, None] * v_r[None, :, :] * w_r.T[:, None, :]
        
        max_err = np.max(np.abs(rec - 1.0))
        return max_err < tol, max_err

    def compute_residual(U, V, W):
        """Compute residual tensor"""
        n, m, p = DIMENSIONS
        rank = U.shape[1]
        rec = np.zeros((n, m, p))
        for r in range(rank):
            u_r = U[:, r].reshape(n, m)
            v_r = V[:, r].reshape(m, p)
            w_r = W[:, r].reshape(p, n)
            rec += u_r[:, :, None] * v_r[None, :, :] * w_r.T[:, None, :]
        return rec - 1.0

    def gradient_step(U, V, W, lr=0.01):
        """Compute gradient and update using vectorized ops"""
        n, m, p = DIMENSIONS
        rank = U.shape[1]
        
        err = compute_residual(U, V, W)
        
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for r in range(rank):
            u_r = U[:, r].reshape(n, m)
            v_r = V[:, r].reshape(m, p)
            w_r = W[:, r].reshape(p, n)
            
            tmp = err * v_r[None, :, :] * w_r.T[:, None, :]
            dU[:, r] = tmp.sum(axis=2).flatten()
            
            tmp = err * u_r[:, :, None] * w_r.T[:, None, :]
            dV[:, r] = tmp.sum(axis=0).flatten()
            
            tmp = err * u_r[:, :, None] * v_r[None, :, :]
            dW[:, r] = tmp.sum(axis=1).T.flatten()
        
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
        return U, V, W, np.max(np.abs(err))

    def build_strassen():
        """Build Strassen-based rank-56 solution"""
        U_s = np.zeros((n * m, 56), dtype=float)
        V_s = np.zeros((m * p, 56), dtype=float)
        W_s = np.zeros((p * n, 56), dtype=float)
        
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
                            U_s[a_map[la], r] += val
                    for lb in range(4):
                        val = V_local[lb, l]
                        if val != 0:
                            V_s[b_map[lb], r] += val
                    for lc in range(4):
                        val = W_local[lc, l]
                        if val != 0:
                            W_s[c_map[lc], r] += val
        
        return U_s, V_s, W_s

    # Build Strassen-based rank-56 solution as fallback
    U_strassen, V_strassen, W_strassen = build_strassen()
    valid, err = verify(U_strassen, V_strassen, W_strassen)
    
    best_U, best_V, best_W = U_strassen.copy(), V_strassen.copy(), W_strassen.copy()
    best_err = err
    best_rank = 56

    # Try to find lower rank decompositions using gradient descent
    # Use Strassen projection as initialization for better convergence
    for target_rank in [32, 28, 24, 20]:
        if time.time() - start_time > 100:
            break
        
        for trial in range(4):
            if time.time() - start_time > 112:
                break
            
            seed = 42 + trial * 1000 + target_rank
            rng = np.random.RandomState(seed)
            
            # Initialize from Strassen projection (better than random)
            if trial == 0:
                # Project Strassen to lower rank via SVD-like approach
                # Use random projection
                P = rng.randn(56, target_rank)
                U = U_strassen @ P
                V = V_strassen @ P
                W = W_strassen @ P
            else:
                # Random initialization with varying scales
                scale = 0.05 + trial * 0.02
                U = rng.randn(n * m, target_rank) * scale
                V = rng.randn(m * p, target_rank) * scale
                W = rng.randn(p * n, target_rank) * scale
            
            # Adaptive learning rate schedule
            base_lr = 0.005
            for it in range(300):
                if time.time() - start_time > 115:
                    break
                
                # Decay learning rate
                lr = base_lr * (0.99 ** (it // 30))
                U, V, W, max_err = gradient_step(U, V, W, lr)
                
                if it % 25 == 0 or it == 299:
                    valid, err = verify(U, V, W)
                    if err < best_err:
                        best_err = err
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                        best_rank = target_rank
                    if err < 1e-6:
                        break
                if max_err < 1e-6:
                    break
            
            # Final check
            valid, err = verify(U, V, W)
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                best_rank = target_rank
    
    # Final high-precision polish if we found a valid low-rank solution
    if best_err < 1e-6 and best_rank < 56:
        if time.time() - start_time < 118:
            U_p, V_p, W_p = best_U.copy(), best_V.copy(), best_W.copy()
            for it in range(50):
                if time.time() - start_time > 119:
                    break
                U_p, V_p, W_p, _ = gradient_step(U_p, V_p, W_p, 0.001)
                valid, err = verify(U_p, V_p, W_p)
                if err < best_err:
                    best_U, best_V, best_W = U_p.copy(), V_p.copy(), W_p.copy()
                    best_err = err
    
    if best_err < 1e-6:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Fallback to Strassen
    return {"U": U_strassen.tolist(), "V": V_strassen.tolist(), "W": W_strassen.tolist()}
# EVOLVE-BLOCK-END
