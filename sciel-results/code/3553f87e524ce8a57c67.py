"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import itertools
import random
import time

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Shapes
    U_rows = n * m  # 6
    V_rows = m * p  # 21
    W_rows = p * n  # 14
    
    # The tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # Condition: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Try to find minimum rank decomposition with integer/half-integer entries
    # Scale by 2: let U2=2U, V2=2V, W2=2W (integers)
    # Then sum_c U2[i*m+j,c] * V2[j*p+k,c] * W2[k*n+i,c] = 8
    
    deadline = time.time() + 140  # safety margin
    
    # Start with schoolbook rank 42
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # Schoolbook
    U = np.zeros((U_rows, best_rank))
    V = np.zeros((V_rows, best_rank))
    W = np.zeros((W_rows, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    best_U, best_V, best_W = U, V, W
    
    # Try to find lower rank decompositions
    # Use random search with local optimization
    for rank in range(best_rank - 1, 30, -1):
        if time.time() > deadline:
            break
        
        found = False
        attempts = 0
        max_attempts = 5000
        
        while attempts < max_attempts and not found:
            attempts += 1
            if time.time() > deadline:
                break
            
            # Random initialization with small values
            # Use values from {-2, -1, 0, 1, 2} for scaled (integer) values
            # So actual values are {-1, -0.5, 0, 0.5, 1}
            U2 = np.random.randint(-2, 3, (U_rows, rank))
            V2 = np.random.randint(-2, 3, (V_rows, rank))
            W2 = np.random.randint(-2, 3, (W_rows, rank))
            
            # Check if this satisfies the equations
            # sum_c U2[i*m+j,c] * V2[j*p+k,c] * W2[k*n+i,c] should equal 8
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for c in range(rank):
                            s += U2[i*m+j, c] * V2[j*p+k, c] * W2[k*n+i, c]
                        if s != 8:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if valid:
                found = True
                # Convert back to original scale
                U = U2 / 2.0
                V = V2 / 2.0
                W = W2 / 2.0
                best_rank = rank
                best_U, best_V, best_W = U, V, W
                break
        
        if found:
            # Continue trying lower ranks
            continue
        else:
            # If we didn't find at this rank, try a few more with different strategies
            # Try structured initialization
            for strategy in range(3):
                if time.time() > deadline:
                    break
                for _ in range(1000):
                    if time.time() > deadline:
                        break
                    # Structured: sparse random
                    U2 = np.zeros((U_rows, rank), dtype=int)
                    V2 = np.zeros((V_rows, rank), dtype=int)
                    W2 = np.zeros((W_rows, rank), dtype=int)
                    
                    # Place some non-zero entries
                    for c in range(rank):
                        # Randomly choose which entries are non-zero
                        if random.random() < 0.7:
                            U2[i*m+j, c] = random.choice([-2, -1, 1, 2])
                        if random.random() < 0.5:
                            V2[j*p+k, c] = random.choice([-2, -1, 1, 2])
                        if random.random() < 0.5:
                            W2[k*n+i, c] = random.choice([-2, -1, 1, 2])
                    
                    valid = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                s = 0
                                for c in range(rank):
                                    s += U2[i*m+j, c] * V2[j*p+k, c] * W2[k*n+i, c]
                                if s != 8:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                    if valid:
                        found = True
                        U = U2 / 2.0
                        V = V2 / 2.0
                        W = W2 / 2.0
                        best_rank = rank
                        best_U, best_V, best_W = U, V, W
                        break
                if found:
                    break
        
        if found:
            continue
        else:
            # If we can't find at this rank, stop trying lower ranks (heuristic)
            # Actually, let's try a few more ranks down
            break
    
    # Convert to lists for the return format
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
