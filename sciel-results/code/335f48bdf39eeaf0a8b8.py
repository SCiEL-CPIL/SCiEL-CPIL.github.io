"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Schoolbook rank as fallback
    rank_schoolbook = n * m * p
    
    # Try to find lower rank solutions
    # The tensor equation: for all i,k,j1,j2:
    # sum_r U[i*m+j1, r] * V[j2*p+k, r] * W[k*n+i, r] = delta(j1, j2)
    
    # Let's try ranks from 29 down to see if we can find a solution
    # We'll use a randomized approach with local optimization
    
    best_rank = rank_schoolbook
    best_solution = None
    
    # Generate schoolbook solution first
    U_sb = np.zeros((n*m, rank_schoolbook))
    V_sb = np.zeros((m*p, rank_schoolbook))
    W_sb = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    best_solution = {'U': U_sb.tolist(), 'V': V_sb.tolist(), 'W': W_sb.tolist()}
    
    # Try to find lower rank solutions
    for target_rank in range(rank_schoolbook - 1, 20, -1):
        if time.time() > deadline - 10:
            break
        
        # Try multiple random initializations
        found = False
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
            
            # Random initialization
            np.random.seed(attempt * 1000 + target_rank)
            U = np.random.randn(n*m, target_rank) * 0.1
            V = np.random.randn(m*p, target_rank) * 0.1
            W = np.random.randn(p*n, target_rank) * 0.1
            
            # Gradient descent to minimize the error
            lr = 0.01
            for step in range(200):
                if time.time() > deadline - 3:
                    break
                    
                # Compute error and gradients
                # Error: for each (i,k,j1,j2), the sum should be delta(j1,j2)
                err = 0.0
                gU = np.zeros_like(U)
                gV = np.zeros_like(V)
                gW = np.zeros_like(W)
                
                for i in range(n):
                    for k in range(p):
                        for j1 in range(m):
                            for j2 in range(m):
                                target = 1.0 if j1 == j2 else 0.0
                                # Compute current value
                                val = 0.0
                                for r in range(target_rank):
                                    val += U[i*m+j1, r] * V[j2*p+k, r] * W[k*n+i, r]
                                diff = val - target
                                err += diff * diff
                                for r in range(target_rank):
                                    gU[i*m+j1, r] += 2 * diff * V[j2*p+k, r] * W[k*n+i, r]
                                    gV[j2*p+k, r] += 2 * diff * U[i*m+j1, r] * W[k*n+i, r]
                                    gW[k*n+i, r] += 2 * diff * U[i*m+j1, r] * V[j2*p+k, r]
                
                # Update
                U -= lr * gU
                V -= lr * gV
                W -= lr * gW
                
                if err < 1e-10:
                    found = True
                    break
            
            if found:
                # Verify exactly
                valid = True
                for i in range(n):
                    for k in range(p):
                        for j1 in range(m):
                            for j2 in range(m):
                                target = 1.0 if j1 == j2 else 0.0
                                val = 0.0
                                for r in range(target_rank):
                                    val += U[i*m+j1, r] * V[j2*p+k, r] * W[k*n+i, r]
                                if abs(val - target) > 1e-8:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    best_rank = target_rank
                    best_solution = {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
                    break
        
        if found:
            break
    
    if time.time() > deadline:
        return best_solution
    
    return best_solution
# EVOLVE-BLOCK-END
