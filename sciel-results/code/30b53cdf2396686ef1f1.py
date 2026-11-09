import numpy as np
import time
import random

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target: C_{i,k} = sum_j A_{i,j} B_{j,k}
    # Tensor T_{i,j,k} = 1 if multiplying A_{i,j} and B_{j,k} contributes to C_{i,k}
    # Decomposition: T_{i,j,k} = sum_c U_{i,c} V_{j,c} W_{k,c}
    # Note: The problem statement says W[k*n+i, r]. 
    # Let's re-read carefully: "output factor follows the official trace convention W[k*n+i,r]"
    # Standard tensor decomposition for matrix multiplication:
    # M_{i,k} = sum_j A_{i,j} B_{j,k}
    # This is a bilinear map. The tensor is over indices (i, j, k).
    # The term for (i, j, k) is 1.
    # Decomposition: M_{i,k} = sum_c (sum_j A_{i,j} U_{j,c}) (V_{c,k} ?) 
    # Usually: M_{i,k} = sum_c L_c(A) R_c(B)
    # Where L_c is a linear form on A and R_c is a linear form on B.
    # Let L_c(A) = sum_j U_{j,c} A_{i,j} ? No, U is a matrix.
    # The interface specifies:
    # U: (n*m, rank) -> U[i*m+j, c]
    # V: (m*p, rank) -> V[j*p+k, c]
    # W: (p*n, rank) -> W[k*n+i, c]
    
    # This structure looks like the decomposition of the multiplication tensor itself, not the bilinear map directly in the standard L/R sense, but rather the 3-way tensor.
    # The equation is: sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1 for all i, j, k?
    # Wait, if I multiply U, V, W directly as defined, the indices don't contract to a single scalar for a given (i,k) without summing over j.
    # Let's check the "official trace convention".
    # Usually, for matrix multiplication $C = AB$, the tensor $T$ has $T_{i,j,k} = 1$ if $i,j,k$ are valid indices for the term $A_{i,j} B_{j,k}$ contributing to $C_{i,k}$.
    # A rank-1 tensor is a product $u_{i,j} v_{j,k} w_{k,i}$? No.
    # A rank-1 tensor is $U_{i,j} V_{k,l} W_{m,n}$? No.
    # A rank-1 tensor in this context is of the form $a_i b_j c_k$.
    # So the decomposition is $1 = \sum_c U_{i*m+j, c} V_{j*p+k, c} W_{k*n+i, c}$?
    # This implies that for a specific triplet $(i,j,k)$, the value is determined.
    # But $j$ appears in all three indices.
    # $U$ depends on $i,j$. $V$ depends on $j,k$. $W$ depends on $k,i$.
    # This is the standard definition of the matrix multiplication tensor rank.
    # We need $\sum_c U[i, j, c] V[j, k, c] W[k, i, c] = 1$ for all valid $i, j, k$.
    
    # Let's verify with n=1, m=1, p=1.
    # U[0,0,c], V[0,0,c], W[0,0,c]. Product is 1. Rank 1. Correct.
    # n=2, m=2, p=2. Rank 7.
    
    # Goal: Find min rank r such that there exist U (10xr), V (30xr), W (12xr) 
    # where for all i in [0,n), j in [0,m), k in [0,p):
    # sum_{c=0}^{r-1} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # We will try to find a decomposition with rank < 60.
    # Given the time constraint, we will use a randomized local search for small ranks.
    # If it fails, we return the schoolbook rank 60.
    
    start_time = time.time()
    time_limit = 145.0  # Safety margin
    
    # Schoolbook solution as fallback
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank), dtype=float)
        V = np.zeros((m*p, rank), dtype=float)
        W = np.zeros((p*n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Helper to check validity
    def check(U, V, W):
        # U: n*m x r, V: m*p x r, W: p*n x r
        # Convert to numpy for speed
        U_np = np.array(U)
        V_np = np.array(V)
        W_np = np.array(W)
        r = U_np.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U_np[i*m+j, c] * V_np[j*p+k, c] * W_np[k*n+i, c]
                    if abs(s - 1.0) > 1e-6:
                        return False
        return True

    # Random search for better rank
    # Try ranks from 14 to 40. 
    # Using integers or half-integers.
    
    best_result = get_schoolbook()
    
    for target_rank in range(14, 40):
        if time.time() - start_time > time_limit - 5:
            break
            
        # Try a few restarts for this rank
        found = False
        restarts = 10
        for _ in range(restarts):
            if time.time() - start_time > time_limit - 5:
                break
                
            # Initialize with random small integers
            # Heuristic: entries should be small
            U = np.random.randint(-1, 2, size=(n*m, target_rank))
            V = np.random.randint(-1, 2, size=(m*p, target_rank))
            W = np.random.randint(-1, 2, size=(p*n, target_rank))
            
            # Local search: flip signs or add 1/-1 to random entries
            steps = 200
            for step in range(steps):
                if time.time() - start_time > time_limit - 5:
                    break
                
                # Calculate error
                err = 0.0
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for c in range(target_rank):
                                s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                            if s != 1:
                                err += abs(s - 1)
                                if abs(s - 1) > 1e-6:
                                    valid = False
                if valid:
                    # Found a valid decomposition
                    # Ensure it's integer/half-integer (it is, since we started with int and modified by int)
                    # Return it
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
                if err == 0:
                    # This shouldn't happen if valid is false, but just in case
                    continue

                # Move: pick a random variable and entry, change by +/- 1
                # Weight changes based on which matrix
                var = np.random.randint(3)
                if var == 0:
                    idx = np.random.randint(n*m)
                    c = np.random.randint(target_rank)
                    delta = np.random.choice([-1, 1])
                    U[idx, c] += delta
                elif var == 1:
                    idx = np.random.randint(m*p)
                    c = np.random.randint(target_rank)
                    delta = np.random.choice([-1, 1])
                    V[idx, c] += delta
                else:
                    idx = np.random.randint(p*n)
                    c = np.random.randint(target_rank)
                    delta = np.random.choice([-1, 1])
                    W[idx, c] += delta
    
    return best_result
# EVOLVE-BLOCK-END
