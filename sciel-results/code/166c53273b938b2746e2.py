import numpy as np
import time

TARGET_DEADLINE = time.time() + 110  # Safety margin

def _verify(U, V, W, n, m, p):
    """
    Verify if U, V, W form a valid decomposition.
    U: (n*m, R)
    V: (m*p, R)
    W: (p*n, R)
    """
    R = U.shape[1]
    # Reconstruct tensor
    # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
    # We check against the target tensor T_target
    # T_target[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i else 0
    
    # Efficient check:
    # For each r, the term is u_r (outer) v_r (outer) w_r (outer)
    # We can compute the sum for specific (a,b,c) or check consistency.
    
    # Since R is small (<=60), we can iterate.
    # But n*m*m*p*p*n is large.
    # We can check random samples or specific structures.
    # Better: Check that for all i, k, the sum over r of W[k*n+i, r] * (U[:, r] * V[:, r]) equals the target slice.
    
    # Target slice for C[i, k]:
    # Non-zero at a = i*m + j, b = j*p + k. Value 1.
    
    for i in range(n):
        for k in range(p):
            c_idx = k * n + i
            # Vector of coefficients for A and B
            # We need sum_r W[c_idx, r] * U[a, r] * V[b, r] = 1 if a=i*m+j, b=j*p+k else 0
            
            # Construct the vector of products P_r = U[:, r] * V[:, r] ? No.
            # We need to check the bilinear form.
            
            # Let's just check the non-zero entries and a few zero entries.
            # Non-zero: j in 0..m-1
            for j in range(m):
                a_idx = i * m + j
                b_idx = j * p + k
                val = 0.0
                for r in range(R):
                    val += W[c_idx, r] * U[a_idx, r] * V[b_idx, r]
                if abs(val - 1.0) > 1e-9:
                    return False, val - 1.0
            
            # Check a zero entry: a = i*m + j, b = j'*p + k where j != j'
            # Or a = i'*m + j, b = j*p + k where i != i' (but c fixes i)
            # c fixes i. So a must have row i.
            # If we pick a with row i, and b with col k, but different j?
            # If j != j', T is 0.
            if m > 1:
                j_test = (j + 1) % m if j < m - 1 else 0
                # Ensure j_test != j
                if j_test == j:
                    j_test = (j + 1) % m
                a_idx = i * m + j
                b_idx = j_test * p + k
                val = 0.0
                for r in range(R):
                    val += W[c_idx, r] * U[a_idx, r] * V[b_idx, r]
                if abs(val) > 1e-9:
                    return False, val
    return True, 0.0

def construct():
    n, m, p = 2, 3, 10
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Trivial decomposition: Rank = n * m * p
    R = n * m * p
    U = np.zeros((nm, R))
    V = np.zeros((mp, R))
    W = np.zeros((pn, R))
    
    for i in range(n):
        for k in range(p):
            for j in range(m):
                r = (i * p + k) * m + j
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                
                U[a_idx, r] = 1.0
                V[b_idx, r] = 1.0
                W[c_idx, r] = 1.0
                
    return {
        'U': U.tolist(),
        'V': V.tolist(),
        'W': W.tolist()
    }
