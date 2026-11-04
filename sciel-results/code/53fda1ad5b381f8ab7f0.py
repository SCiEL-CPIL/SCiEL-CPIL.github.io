"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin
    
    best_rank = n * m * p
    best_result = None
    
    # Generate schoolbook solution first
    U_sb = np.zeros((n*m, best_rank))
    V_sb = np.zeros((m*p, best_rank))
    W_sb = np.zeros((p*n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    # Try to find lower rank
    for target_rank in [32, 33, 34, 35]:
        if time.time() > deadline:
            break
            
        found = False
        for attempt in range(100):
            if time.time() > deadline:
                break
            
            np.random.seed(attempt * 7919 + target_rank * 104729)
            U = np.random.randn(n*m, target_rank)
            V = np.random.randn(m*p, target_rank)
            W = np.random.randn(p*n, target_rank)
            
            # Precompute indices for efficiency
            # For each r, we need to compute contributions to all (i,j,k)
            # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            
            for iteration in range(500):
                if time.time() > deadline:
                    break
                
                # Compute errors vectorized
                errors = np.zeros((n, m, p))
                for r in range(target_rank):
                    # U[:, r] is shape (n*m,), V[:, r] is (m*p,), W[:, r] is (p*n,)
                    U_r = U[:, r].reshape(n, m)
                    V_r = V[:, r].reshape(m, p)
                    W_r = W[:, r].reshape(p, n)
                    # Contribution to (i,j,k): U_r[i,j] * V_r[j,k] * W_r[k,i]
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                errors[i,j,k] += U_r[i,j] * V_r[j,k] * W_r[k,i]
                errors -= 1
                
                max_err = np.max(np.abs(errors))
                if max_err < 1e-10:
                    found = True
                    break
                
                # Gradient descent
                lr = 0.005
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            err = errors[i,j,k]
                            if abs(err) < 1e-12:
                                continue
                            for r in range(target_rank):
                                U_r = U[:, r].reshape(n, m)
                                V_r = V[:, r].reshape(m, p)
                                W_r = W[:, r].reshape(p, n)
                                dU = V_r[j,k] * W_r[k,i]
                                dV = U_r[i,j] * W_r[k,i]
                                dW = U_r[i,j] * V_r[j,k]
                                U[i*m+j, r] -= lr * err * dU
                                V[j*p+k, r] -= lr * err * dV
                                W[k*n+i, r] -= lr * err * dW
                
                if time.time() > deadline:
                    break
            
            if found:
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                # Verify
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0
                            for r in range(target_rank):
                                val += U_rounded[i*m+j, r] * V_rounded[j*p+k, r] * W_rounded[k*n+i, r]
                            if abs(val - 1) > 1e-6:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    best_rank = target_rank
                    best_result = {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
                    break
        if best_rank == target_rank:
            break
    
    return best_result
# EVOLVE-BLOCK-END
