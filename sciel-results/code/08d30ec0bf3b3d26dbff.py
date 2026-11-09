"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # Leave 5s safety margin
    
    # Known lower bounds: max(n*m, m*p, p*n) = max(9, 21, 21) = 21
    # Schoolbook gives 63. We'll search for better ranks.
    
    best_result = None
    best_rank = None
    
    # Try ranks from lower bound upward
    for rank in range(21, 40):
        if time.time() > deadline - 5:
            break
            
        # Try multiple random initializations for this rank
        max_restarts = 30 if rank <= 25 else 15
        if time.time() > deadline - 10:
            max_restarts = 5
            
        found = False
        for restart in range(max_restarts):
            if time.time() > deadline - 2:
                break
                
            try:
                result = _try_rank(rank, n, m, p, restart, deadline)
                if result is not None:
                    found = True
                    if best_result is None or rank < best_rank:
                        best_rank = rank
                        best_result = result
                    break
            except Exception:
                continue
        
        if found and best_rank == rank:
            continue
        elif not found:
            pass
    
    # If nothing found, return schoolbook
    if best_result is None:
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
    
    return best_result

def _try_rank(rank, n, m, p, restart_idx, deadline):
    """Try to find a rank-rank decomposition using random init + gradient descent."""
    np.random.seed(42 + restart_idx * 1000)
    
    # Initialize U, V, W randomly
    U = np.random.randn(n * m, rank)
    V = np.random.randn(m * p, rank)
    W = np.random.randn(p * n, rank)
    
    # Scale to reasonable magnitude
    scale = 1.0 / np.sqrt(rank)
    U *= scale
    V *= scale
    W *= scale
    
    # Use gradient descent to minimize ||tensor - sum_c U_c ⊗ V_c ⊗ W_c||
    lr = 0.01
    max_iter = 300
    
    for it in range(max_iter):
        if time.time() > deadline - 3:
            break
            
        # Compute current tensor estimate
        T_est = np.zeros((n, m, p))
        for c in range(rank):
            u_c = U[:, c].reshape(n, m)
            v_c = V[:, c].reshape(m, p)
            w_c = W[:, c].reshape(p, n)
            
            u3d = u_c[:, :, np.newaxis]
            v3d = v_c[np.newaxis, :, :]
            w3d = np.broadcast_to(w_c.T[np.newaxis, :, :], (n, 1, p))
            
            T_est += u3d * v3d * w3d
        
        # Target tensor: all ones
        T_target = np.ones((n, m, p))
        
        # Loss
        diff = T_est - T_target
        loss = np.sum(diff ** 2)
        
        # Check convergence
        if loss < 1e-10:
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}
        
        # Compute gradients
        dU = np.zeros((n * m, rank))
        dV = np.zeros((m * p, rank))
        dW = np.zeros((p * n, rank))
        
        for c in range(rank):
            u_c = U[:, c].reshape(n, m)
            v_c = V[:, c].reshape(m, p)
            w_c = W[:, c].reshape(p, n)
            
            D = diff * w_c.T[np.newaxis, np.newaxis, :]
            C_u = np.einsum('ijk,jk->ij', D, v_c)
            dU[c] = 2 * C_u.ravel()
            
            E = diff * u_c[:, :, np.newaxis]
            C_v = np.einsum('ijk,ki->jk', E, w_c)
            dV[c] = 2 * C_v.ravel()
            
            F = diff * v_c[np.newaxis, :, :]
            C_w = np.einsum('ijk,ij->ki', F, u_c)
            dW[c] = 2 * C_w.ravel()
        
        # Update
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
    
    # Final check
    T_est = np.zeros((n, m, p))
    for c in range(rank):
        u_c = U[:, c].reshape(n, m)
        v_c = V[:, c].reshape(m, p)
        w_c = W[:, c].reshape(p, n)
        u3d = u_c[:, :, np.newaxis]
        v3d = v_c[np.newaxis, :, :]
        w3d = np.broadcast_to(w_c.T[np.newaxis, :, :], (n, 1, p))
        T_est += u3d * v3d * w3d
    
    loss = np.sum((T_est - 1.0) ** 2)
    if loss < 1e-8:
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        return {"U": U_list, "V": V_list, "W": W_list}
    
    return None
# EVOLVE-BLOCK-END
