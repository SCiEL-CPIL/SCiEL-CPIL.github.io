import numpy as np
import time
import random

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # We know schoolbook is 36.
    # We suspect rank 23 is possible.
    # We will search for ranks from 20 to 35.
    
    for r in range(20, 36):
        if time.time() > deadline - 5:
            break
            
        # Try to find a decomposition of rank r
        found = False
        for attempt in range(100):
            if time.time() > deadline - 5:
                break
                
            random.seed(r * 1000 + attempt)
            # Initialize U and V with small random integers
            # U shape: (n*m, r) = (9, r)
            # V shape: (m*p, r) = (12, r)
            U = np.random.randint(-2, 3, size=(n * m, r))
            V = np.random.randint(-2, 3, size=(m * p, r))
            
            # We need to solve for W.
            # W shape: (p*n, r) = (12, r)
            # For each k in 0..p-1, we solve for W[:, k*n:(k+1)*n]
            
            W = np.zeros((p * n, r))
            valid_for_all_k = True
            
            for k in range(p):
                # Build linear system M * W_k = b
                # W_k is a vector of size 3*r (flattened W[0..2, k, :])
                # Actually, W is indexed by [k*n + i, c].
                # So for fixed k, the unknowns are W[k*n + i, c] for i=0..2, c=0..r-1.
                # Total 3*r unknowns.
                
                # Equations: for each i in 0..2, j1 in 0..2, j2 in 0..2:
                # sum_c W[k*n+i, c] * U[i*m+j1, c] * V[j2*p+k, c] = delta(j1, j2)
                
                num_eqs = n * m * m  # 3 * 3 * 3 = 27
                num_unk = n * r      # 3 * r
                
                M = np.zeros((num_eqs, num_unk))
                b = np.zeros(num_eqs)
                
                eq_idx = 0
                for i in range(n):
                    for j1 in range(m):
                        for j2 in range(m):
                            # b[eq_idx] = 1 if j1 == j2 else 0
                            b[eq_idx] = 1.0 if j1 == j2 else 0.0
                            
                            for c in range(r):
                                for i_prime in range(n):
                                    if i_prime == i:
                                        # Column index in M: i_prime * r + c
                                        col_idx = i_prime * r + c
                                        M[eq_idx, col_idx] = U[i * m + j1, c] * V[j2 * p + k, c]
                            eq_idx += 1
                
                # Solve M * W_k_flat = b
                # Use least squares or direct solve if square/underdetermined
                try:
                    # np.linalg.lstsq works for underdetermined systems
                    W_k_flat, residuals, rank, sv = np.linalg.lstsq(M, b, rcond=None)
                    
                    # Check if the solution is exact (residuals should be 0)
                    # Since we are using integers for U,V, and solving linear eqs, 
                    # the solution W might be rational.
                    # We need to check if the decomposition is EXACT.
                    # The problem requires integer/half-integer entries.
                    # If W has large denominators, it might not be "integer/half-integer".
                    # However, the task says "integer/half-integer entries".
                    # If the solution has denominators, we might need to scale.
                    # But scaling U,V,W doesn't change the rank.
                    # We can just return the rational numbers if they are simple fractions.
                    
                    # Let's check if the solution satisfies the equations exactly.
                    if np.allclose(M @ W_k_flat, b, atol=1e-9):
                        # Reshape W_k_flat to (n, r) and place in W
                        W_k = W_k_flat.reshape((n, r))
                        W[k * n:(k + 1) * n, :] = W_k
                    else:
                        valid_for_all_k = False
                        break
                except Exception:
                    valid_for_all_k = False
                    break
            
            if valid_for_all_k:
                # Verify the full decomposition
                if _verify(U, V, W):
                    # Convert to lists
                    return {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist()
                    }
        
    # Fall back to schoolbook
    return _schoolbook()

def _verify(U, V, W):
    n, m, p = DIMENSIONS
    r = U.shape[1]
    # Check if sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] == 1 for all i,j,k
    # Wait, the condition is:
    # sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] A[i,j]) * ...
    # The tensor condition is:
    # For all i, j1, j2, k:
    # sum_c W[k*n+i, c] U[i*m+j1, c] V[j2*p+k, c] == delta(j1, j2)
    
    for k in range(p):
        for i in range(n):
            for j1 in range(m):
                for j2 in range(m):
                    s = 0.0
                    for c in range(r):
                        s += W[k * n + i, c] * U[i * m + j1, c] * V[j2 * p + k, c]
                    target = 1.0 if j1 == j2 else 0.0
                    if abs(s - target) > 1e-9:
                        return False
    return True

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
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
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
