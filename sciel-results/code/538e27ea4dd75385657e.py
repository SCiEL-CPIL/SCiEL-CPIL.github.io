import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    num_a = n * m  # 16
    num_b = m * p  # 32
    num_c = p * n  # 32
    
    # Build the tensor T: T[a, b, c] = 1 if a = i*m + j, b = j'*p + k, c = k'*n + i, and j == j' and i == i' (wait, no)
    # Actually: c = k_out * n + i_out. a = i_in * m + j_in. b = j_in' * p + k_in.
    # T[a, b, c] = 1 if i_in == i_out and k_in == k_out and j_in == j_in'.
    # So for a given c = k_out * n + i_out, the non-zero (a, b) are:
    # a = i_out * m + j, b = j * p + k_out, for j = 0..m-1.
    # T[a, b, c] = 1 for those, 0 otherwise.
    
    # We'll try ranks from a lower bound up to num_a * num_b * num_c (but that's huge).
    # Lower bound for 4x4x8: I'll start from 48.
    # But to ensure we finish in 120s, I'll try a few ranks.
    
    import random
    import time
    
    start_time = time.time()
    timeout = 110  # seconds
    
    # Try ranks from 48 to 128
    for R in range(48, 129):
        if time.time() - start_time > timeout:
            break
            
        # Try many random initializations
        num_trials = 200
        for trial in range(num_trials):
            if time.time() - start_time > timeout:
                break
                
            # Random U, V with entries in [-2, 2] rationals
            U = np.random.randint(-2, 3, size=(num_a, R))
            V = np.random.randint(-2, 3, size=(num_b, R))
            
            # For each c, solve M_c * W_c = ones
            W = np.zeros((num_c, R))
            valid = True
            for c in range(num_c):
                i_out = c % n  # c = k_out * n + i_out, so i_out = c % n, k_out = c // n
                k_out = c // n
                
                # M_c is m x R, where M_c[j, t] = U[i_out * m + j, t] * V[j * p + k_out, t]
                M_c = np.zeros((m, R))
                for j in range(m):
                    a_idx = i_out * m + j
                    b_idx = j * p + k_out
                    M_c[j, :] = U[a_idx, :] * V[b_idx, :]
                
                rhs = np.ones(m)
                
                # Solve M_c * W_c = rhs
                try:
                    # Use least squares and check residual
                    W_c, residuals, rank_m, s = np.linalg.lstsq(M_c, rhs, rcond=None)
                    # Check exactness
                    residual = np.linalg.norm(M_c @ W_c - rhs)
                    if residual > 1e-9:
                        valid = False
                        break
                    W[c, :] = W_c
                except:
                    valid = False
                    break
            
            if not valid:
                continue
                
            # Now verify the full decomposition
            # Check that for all (a, b, c), sum_t U[a,t] V[b,t] W[c,t] == T[a,b,c]
            # We only need to check the non-zero and zero cases.
            # Non-zero: for all i, k, j: a = i*m+j, b = j*p+k, c = k*n+i -> sum should be 1
            ok = True
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        a_idx = i * m + j
                        b_idx = j * p + k
                        c_idx = k * n + i
                        val = np.dot(U[a_idx, :], V[b_idx, :] * W[c_idx, :])
                        if abs(val - 1.0) > 1e-9:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if not ok:
                continue
                
            # Also check that for other (a,b,c), the sum is 0.
            # This is expensive, but we can check a sample or all.
            # Total number of (a,b,c) triples is num_a * num_b * num_c = 16*32*32 = 16384.
            # For each, we compute the dot product. This is 16384 * R operations.
            # For R=48, that's 786,432 operations, which is fine.
            for a_idx in range(num_a):
                for b_idx in range(num_b):
                    for c_idx in range(num_c):
                        # Determine if T[a_idx, b_idx, c_idx] should be 1 or 0
                        i_in = a_idx // m
                        j_in = a_idx % m
                        j_in_prime = b_idx // p
                        k_in = b_idx % p
                        k_out = c_idx // n
                        i_out = c_idx % n
                        
                        expected = 1.0 if (i_in == i_out and k_in == k_out and j_in == j_in_prime) else 0.0
                        val = np.dot(U[a_idx, :], V[b_idx, :] * W[c_idx, :])
                        if abs(val - expected) > 1e-9:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if not ok:
                continue
                
            # Success! Return the decomposition
            # Convert to lists
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}
        
        # If we didn't find for this R, continue
        print(f"Rank {R}: not found")
    
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
