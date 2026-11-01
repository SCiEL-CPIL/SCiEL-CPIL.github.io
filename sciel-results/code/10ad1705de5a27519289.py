"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Schoolbook fallback
    rank_sb = n * m * p
    U = np.zeros((n*m, rank_sb))
    V = np.zeros((m*p, rank_sb))
    W = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    best = {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    best_rank = rank_sb
    
    # Try to find lower rank solutions
    for target_rank in range(20, best_rank + 1, 4):
        if time.time() > deadline - 10:
            break
        
        found = False
        for seed in range(30):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(seed * 7919 + target_rank)
            
            U_t = np.random.randn(n*m, target_rank) * 0.5
            V_t = np.random.randn(m*p, target_rank) * 0.5
            W_t = np.random.randn(p*n, target_rank) * 0.5
            
            # Alternating least squares
            for it in range(50):
                if time.time() > deadline - 10:
                    break
                
                # Solve for W
                for k in range(p):
                    for i in range(n):
                        A = np.zeros((m, target_rank))
                        for j in range(m):
                            A[j] = U_t[i*m+j] * V_t[j*p+k]
                        b = np.ones(m)
                        W_t[k*n+i], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                
                # Solve for V
                for j in range(m):
                    for k in range(p):
                        A = np.zeros((n, target_rank))
                        for i in range(n):
                            A[i] = U_t[i*m+j] * W_t[k*n+i]
                        b = np.ones(n)
                        V_t[j*p+k], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                
                # Solve for U
                for i in range(n):
                    for j in range(m):
                        A = np.zeros((p, target_rank))
                        for k in range(p):
                            A[k] = V_t[j*p+k] * W_t[k*n+i]
                        b = np.ones(p)
                        U_t[i*m+j], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                
                # Check convergence
                max_res = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U_t[i*m+j], V_t[j*p+k] * W_t[k*n+i])
                            max_res = max(max_res, abs(1 - val))
                
                if max_res < 1e-10:
                    found = True
                    break
            
            if found:
                best = {'U': U_t.tolist(), 'V': V_t.tolist(), 'W': W_t.tolist()}
                best_rank = target_rank
                break  # Try lower rank
    
    return best
# EVOLVE-BLOCK-END
