import time
import numpy as np
import random

DIMENSIONS = [3, 5, 7]

def _validate_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Validate that U, V, W form a valid decomposition for n×m × m×p matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    if U.shape[1] != V.shape[1] or V.shape[1] != W.shape[1]:
        return False
    r = U.shape[1]
    if r == 0:
        return False
    
    # Check that entries are integer or half-integer
    for mat in [U, V, W]:
        for val in mat.flatten():
            if abs(val * 2 - round(val * 2)) > tol:
                return False
    
    # Reconstruct the tensor and check
    # T[i*m+j, k*n+i'] should be delta_{i,i'} * delta_{j,k} for the multiplication
    # Actually, the tensor for matrix multiplication:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has T[i,j,k] = 1 for all i,j,k
    # And the decomposition is: T = sum_r U_r ⊗ V_r ⊗ W_r
    # where U_r is a vector of length n*m, V_r of length m*p, W_r of length p*n
    
    # For the multiplication to be correct:
    # sum_r U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    # Wait, that's not right. Let me think again.
    
    # The tensor for matrix multiplication M[n,m,p] is defined as:
    # T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # No, that's not right either.
    
    # Actually, the matrix multiplication tensor is:
    # T is a 3-way tensor where T[a, b, c] = 1 if a corresponds to (i,j) in A, 
    # b corresponds to (j,k) in B, and c corresponds to (i,k) in C, with the same j.
    # So T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k, and 0 otherwise.
    
    # The rank-r decomposition means:
    # T = sum_{r=1}^{rank} u_r ⊗ v_r ⊗ w_r
    # where u_r is in R^{n*m}, v_r in R^{m*p}, w_r in R^{p*n}
    
    # So for each (a, b, c) = (i*m+j, j*p+k, k*n+i):
    # sum_r U[a,r] * V[b,r] * W[c,r] = 1
    # And for all other (a, b, c):
    # sum_r U[a,r] * V[b,r] * W[c,r] = 0
    
    # Let's check this
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = 0.0
                for r in range(r):
                    val += U[a, r] * V[b, r] * W[c, r]
                if abs(val - 1.0) > tol:
                    return False
    
    # Check that non-matching entries are zero
    # Sample some non-matching entries
    for _ in range(min(1000, r)):
        a = random.randint(0, n * m - 1)
        b = random.randint(0, m * p - 1)
        c = random.randint(0, p * n - 1)
        i_a = a // m
        j_a = a % m
        j_b = b // p
        k_b = b % p
        k_c = c // n
        i_c = c % n
        # This is a matching entry if j_a == j_b and i_a == i_c
        if j_a != j_b or i_a != i_c:
            val = 0.0
            for rr in range(r):
                val += U[a, rr] * V[b, rr] * W[c, rr]
            if abs(val) > tol:
                return False
    
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    timeout = 110.0  # 10s margin for safety
    
    # 1. Schoolbook Decomposition (Rank 105)
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    best_result = {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
    best_rank = rank_school
    
    # 2. Attempt to find lower rank
    # Try ranks from 104 down to 95 with a time budget
    for target_rank in range(rank_school - 1, 94, -1):
        if time.time() - start_time > timeout:
            break
        
        # Try a few random restarts
        max_attempts = 5
        found = False
        for attempt in range(max_attempts):
            if time.time() - start_time > timeout:
                break
            
            # Initialize with schoolbook and randomly zero out some entries
            U = np.random.choice([-1, 0, 0.5, 1, 1.5, 2], size=(n * m, target_rank))
            V = np.random.choice([-1, 0, 0.5, 1, 1.5, 2], size=(m * p, target_rank))
            W = np.random.choice([-1, 0, 0.5, 1, 1.5, 2], size=(p * n, target_rank))
            
            if _validate_decomposition(U, V, W, n, m, p):
                best_result = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                best_rank = target_rank
                found = True
                break
        
        if found:
            break
    
    return best_result
