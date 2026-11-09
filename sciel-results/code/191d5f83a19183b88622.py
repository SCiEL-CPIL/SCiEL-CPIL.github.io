"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [4, 5, 5]
DEADLINE = 120  # seconds, safety margin from 150s limit

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    nm = n * m  # 20
    mp = m * p  # 25
    pn = p * n  # 20
    
    # Target tensor: T[i,j,k] = 1 for all valid (i,j,k)
    # We need: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    best_result = None
    best_rank = nm * mp  # schoolbook = 100
    
    # Try ranks from lower bound upward
    # Lower bound for 4x5x5: at least max(n,m,p) = 5, but realistically higher
    # Try a range of ranks
    target_ranks = [70, 75, 80, 85, 90, 95, 100]
    
    for target_rank in target_ranks:
        if time.time() - start_time > DEADLINE - 5:
            break
        
        # Try multiple random initializations
        for attempt in range(50):
            if time.time() - start_time > DEADLINE - 5:
                break
            
            # Initialize with small random integers
            U = np.random.randint(-2, 3, size=(nm, target_rank))
            V = np.random.randint(-2, 3, size=(mp, target_rank))
            W = np.random.randint(-2, 3, size=(pn, target_rank))
            
            # Iterative refinement: try to satisfy tensor constraints
            # For each (i,j,k), we need sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
            success = False
            for iteration in range(100):
                if time.time() - start_time > DEADLINE - 5:
                    break
                
                # Check current error
                max_err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                            err = abs(val - 1)
                            if err > max_err:
                                max_err = err
                
                if max_err < 1e-10:
                    success = True
                    break
                
                # Simple gradient-like update: adjust one random entry
                # Pick a random (i,j,k) with high error
                i = random.randint(0, n-1)
                j = random.randint(0, m-1)
                k = random.randint(0, p-1)
                
                r = random.randint(0, target_rank-1)
                
                # Try adjusting U[i*m+j, r]
                current_val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                needed = 1 - current_val
                if V[j*p+k, r] * W[k*n+i, r] != 0:
                    delta = needed / (V[j*p+k, r] * W[k*n+i, r])
                    U[i*m+j, r] += int(round(delta))
                
                # Occasionally adjust V or W too
                if random.random() < 0.3:
                    r = random.randint(0, target_rank-1)
                    current_val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    needed = 1 - current_val
                    if U[i*m+j, r] * W[k*n+i, r] != 0:
                        delta = needed / (U[i*m+j, r] * W[k*n+i, r])
                        V[j*p+k, r] += int(round(delta))
                elif random.random() < 0.3:
                    r = random.randint(0, target_rank-1)
                    current_val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    needed = 1 - current_val
                    if U[i*m+j, r] * V[j*p+k, r] != 0:
                        delta = needed / (U[i*m+j, r] * V[j*p+k, r])
                        W[k*n+i, r] += int(round(delta))
            
            if success:
                # Verify
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                            if abs(val - 1) > 1e-10:
                                valid = False
                                break
                    if not valid:
                        break
                if valid:
                    best_rank = target_rank
                    best_result = {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist()
                    }
                    break
        
        if best_result is not None:
            break
    
    if best_result is None:
        # Fall back to schoolbook
        rank = nm * mp
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        best_result = {"U": U, "V": V, "W": W}
    
    return best_result
# EVOLVE-BLOCK-END
