import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    # Build target tensor T of shape (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    
    # Schoolbook fallback
    rank_schoolbook = n * m * p
    U_fb = np.zeros((n*m, rank_schoolbook))
    V_fb = np.zeros((m*p, rank_schoolbook))
    W_fb = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m + j, col] = 1
                V_fb[j*p + k, col] = 1
                W_fb[k*n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_fb, V_fb, W_fb
    best_rank = rank_schoolbook
    
    # Try to find lower rank decompositions
    for target_rank in [32, 40, 48, 64, 80]:
        if time.time() > deadline - 20:
            break
        
        success = False
        for trial in range(3):
            if time.time() > deadline - 20:
                break
            
            np.random.seed(123 + trial * 7 + target_rank)
            U = np.random.randn(n*m, target_rank)
            V = np.random.randn(m*p, target_rank)
            W = np.random.randn(p*n, target_rank)
            
            prev_error = np.inf
            for iteration in range(50):
                if time.time() > deadline - 20:
                    break
                
                # Update U
                A_VW = np.zeros((m*p * p*n, target_rank))
                for r in range(target_rank):
                    A_VW[:, r] = np.outer(V[:, r], W[:, r]).flatten()
                
                for a in range(n*m):
                    t_a = T[a, :, :].flatten()
                    U[a, :], _, _, _ = np.linalg.lstsq(A_VW, t_a, rcond=None)
                
                # Update V
                A_UW = np.zeros((n*m * p*n, target_rank))
                for r in range(target_rank):
                    A_UW[:, r] = np.outer(U[:, r], W[:, r]).flatten()
                
                for b in range(m*p):
                    t_b = T[:, b, :].flatten()
                    V[b, :], _, _, _ = np.linalg.lstsq(A_UW, t_b, rcond=None)
                
                # Update W
                A_UV = np.zeros((n*m * m*p, target_rank))
                for r in range(target_rank):
                    A_UV[:, r] = np.outer(U[:, r], V[:, r]).flatten()
                
                for c in range(p*n):
                    t_c = T[:, :, c].flatten()
                    W[c, :], _, _, _ = np.linalg.lstsq(A_UV, t_c, rcond=None)
                
                # Check error
                error = np.abs(np.einsum('ar,br,cr->abc', U, V, W) - T).max()
                if error < 1e-8:
                    success = True
                    break
                
                # Check for stagnation
                if abs(prev_error - error) < 1e-12:
                    break
                prev_error = error
            
            if success:
                best_rank = target_rank
                best_U, best_V, best_W = U, V, W
                break
        
        if success:
            break
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
