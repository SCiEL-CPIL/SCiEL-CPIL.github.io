"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    nU = n * m  # 6
    nV = m * p  # 9
    nW = p * n  # 6
    
    # Build target tensor T of shape (nU, nV, nW)
    T = np.zeros((nU, nV, nW))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0
    
    # Schoolbook solution as fallback (rank 18)
    rank_sb = n * m * p
    U_sb = np.zeros((nU, rank_sb))
    V_sb = np.zeros((nV, rank_sb))
    W_sb = np.zeros((nW, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U = U_sb.copy()
    best_V = V_sb.copy()
    best_W = W_sb.copy()
    
    def check_and_round(U, V, W, T, atol=1e-4):
        """Check if decomposition is valid and round to half-integers."""
        # Check if entries are close to half-integers
        for mat in [U, V, W]:
            rounded = np.round(mat * 2) / 2.0
            if np.max(np.abs(mat - rounded)) > atol:
                return False
        U_r = np.round(U * 2) / 2.0
        V_r = np.round(V * 2) / 2.0
        W_r = np.round(W * 2) / 2.0
        
        T_recon = np.einsum('ir,jr,kr->ijk', U_r, V_r, W_r)
        return np.allclose(T_recon, T, atol=1e-6)
    
    def optimize_rank(R, max_steps=3000, lr_init=0.05):
        """Try to find a rank-R decomposition using gradient descent."""
        if time.time() > deadline - 5:
            return False, None, None, None
        
        U = np.random.randn(nU, R) * 0.1
        V = np.random.randn(nV, R) * 0.1
        W = np.random.randn(nW, R) * 0.1
        
        lr = lr_init
        prev_err = np.inf
        
        for step in range(max_steps):
            if time.time() > deadline - 5:
                break
            
            # Compute reconstruction and error
            T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
            err_tensor = T_recon - T
            err = np.sum(err_tensor**2)
            
            if err < 1e-10:
                return True, U, V, W
            
            # Compute gradients
            dU = np.einsum('ijk,kr,jr->ir', err_tensor, W, V)
            dV = np.einsum('ijk,ir,kr->jr', err_tensor, U, W)
            dW = np.einsum('ijk,ir,jr->kr', err_tensor, U, V)
            
            # Update
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            # Adaptive learning rate
            if err > prev_err * 1.1:
                lr *= 0.5
            else:
                if err < prev_err * 0.9:
                    lr *= 1.05
                prev_err = err
            
            # Periodic check for half-integer convergence
            if step % 500 == 0 and step > 0:
                if check_and_round(U, V, W, T):
                    return True, U, V, W
        
        # Final check
        if check_and_round(U, V, W, T):
            return True, U, V, W
        return False, U, V, W
    
    # Try to find lower rank decompositions
    # Start from rank 17 down to 10
    for R in range(rank_sb - 1, 9, -1):
        if time.time() > deadline - 10:
            break
        
        found = False
        # Try multiple random restarts
        for restart in range(30):
            if time.time() > deadline - 5:
                break
            
            success, U, V, W = optimize_rank(R)
            if success:
                # Verify and store
                if check_and_round(U, V, W, T):
                    best_rank = R
                    best_U = np.round(U * 2) / 2.0
                    best_V = np.round(V * 2) / 2.0
                    best_W = np.round(W * 2) / 2.0
                    found = True
                    break
        
        if found:
            break
    
    # Convert to lists of half-integers
    best_U = np.round(best_U * 2) / 2.0
    best_V = np.round(best_V * 2) / 2.0
    best_W = np.round(best_W * 2) / 2.0
    
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    # Final verification
    T_check = np.zeros((n, m, p))
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for r in range(best_rank):
                    val += U_list[i*m + j][r] * V_list[j*p + k][r] * W_list[k*n + i][r]
                if abs(val - 1.0) > 1e-6:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to schoolbook
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
