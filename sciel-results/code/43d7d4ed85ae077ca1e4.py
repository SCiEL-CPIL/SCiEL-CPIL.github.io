"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_schoolbook = n * m * p  # 80
    
    # Try to find a lower rank decomposition
    # The tensor T[i,j,k] = 1 if we're computing C[i,k] += A[i,j]*B[j,k]
    # We want to write T as sum of r rank-1 tensors
    
    # For small cases, we can use a greedy/random approach
    
    # First, let's establish the schoolbook as baseline
    rank = rank_schoolbook
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Try to improve using random restarts and local search
    # The idea: try different rank values and see if we can find a valid decomposition
    
    best_U, best_V, best_W, best_rank = U, V, W, rank
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Try ranks from a lower bound up to schoolbook
    # Lower bound: at least max(n*m, m*p, n*p) = 20, but likely higher
    # For 4x4x5, let's try from 20 upward
    
    for test_rank in range(20, rank + 1):
        if time.time() > deadline:
            break
            
        # Try random initialization with this rank
        for attempt in range(5):
            if time.time() > deadline:
                break
                
            U_try = np.random.randn(n * m, test_rank)
            V_try = np.random.randn(m * p, test_rank)
            W_try = np.random.randn(p * n, test_rank)
            
            # Check if this gives an exact decomposition
            # The tensor equation: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] should equal 1 for all i,j,k
            # Actually, we need: for each (i,j,k), sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
            # And for "wrong" combinations, it should be 0? No, the tensor is defined only on the valid indices.
            
            # Actually, the matrix multiplication tensor T has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
            # Wait, that's not right. The tensor for matrix multiplication is:
            # T[(i,j),(j,k),(i,k)] = 1, meaning C[i,k] = sum_j A[i,j]*B[j,k]
            # So the tensor has non-zero entries only when the middle indices match.
            
            # Let me reformulate: the tensor T has dimensions (n*m) x (m*p) x (n*p)
            # T[a,b,c] = 1 if a = i*m+j, b = j'*p+k, c = i'*n+k' where j=j' and i=i' and k=k'
            # Otherwise T[a,b,c] = 0
            
            # So we need: sum_c U[a,c]*V[b,c]*W[c,c] = T[a,b,c] for all a,b,c
            
            # This is a system of equations. Let's check if our random guess works.
            
            # For efficiency, let's only check the "on-support" entries first
            valid_count = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        val = np.sum(U_try[a, :] * V_try[b, :] * W_try[c, :])
                        if abs(val - 1.0) > 1e-6:
                            valid_count += 1
                            break
                    else:
                        continue
                    break
                else:
                    continue
                break
            else:
                # Check some off-support entries to ensure they're zero
                off_support_ok = True
                for _ in range(100):
                    a = np.random.randint(0, n * m)
                    b = np.random.randint(0, m * p)
                    c = np.random.randint(0, p * n)
                    # Check if this is an on-support entry
                    i1 = a // m
                    j1 = a % m
                    j2 = b // p
                    k2 = b % p
                    i2 = c % n
                    k1 = c // n
                    if j1 == j2 and i1 == i2 and k1 == k2:
                        continue  # on support, skip
                    val = np.sum(U_try[a, :] * V_try[b, :] * W_try[c, :])
                    if abs(val) > 1e-6:
                        off_support_ok = False
                        break
                
                if off_support_ok:
                    best_U, best_V, best_W, best_rank = U_try, V_try, W_try, test_rank
                    break  # Found a valid decomposition at this rank, try lower
        
        # If we found a decomposition at this rank, we can stop (since we're going upward)
        if best_rank == test_rank:
            break
    
    # Convert to lists of floats
    result = {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
    return result
# EVOLVE-BLOCK-END
