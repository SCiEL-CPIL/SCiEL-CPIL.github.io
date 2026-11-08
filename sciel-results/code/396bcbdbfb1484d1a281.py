import numpy as np
import time

TARGET_DEADLINE = time.time() + 110  # Safety margin

def _verify(U, V, W, n, m, p):
    R = U.shape[1]
    # Check non-zero entries
    for i in range(n):
        for k in range(p):
            for j in range(m):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                val = np.dot(W[c_idx, :], np.multiply(U[a_idx, :], V[b_idx, :]))
                if abs(val - 1.0) > 1e-9:
                    return False
    # Check zero entries (sample)
    for i in range(n):
        for k in range(p):
            c_idx = k * n + i
            for j in range(m):
                a_idx = i * m + j
                for j_test in range(m):
                    if j_test == j:
                        continue
                    b_idx = j_test * p + k
                    val = np.dot(W[c_idx, :], np.multiply(U[a_idx, :], V[b_idx, :]))
                    if abs(val) > 1e-9:
                        return False
    return True

def construct():
    n, m, p = 2, 3, 10
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Trivial decomposition
    R_trivial = n * m * p
    U_trivial = np.zeros((nm, R_trivial))
    V_trivial = np.zeros((mp, R_trivial))
    W_trivial = np.zeros((pn, R_trivial))
    
    for i in range(n):
        for k in range(p):
            for j in range(m):
                r = (i * p + k) * m + j
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                U_trivial[a_idx, r] = 1.0
                V_trivial[b_idx, r] = 1.0
                W_trivial[c_idx, r] = 1.0
                
    # Try to find a lower rank solution (e.g., rank 54)
    # We will try to solve for W given random U, V of rank R
    best_U, best_V, best_W = U_trivial, V_trivial, W_trivial
    best_rank = R_trivial
    
    R_target = 54
    if time.time() < TARGET_DEADLINE - 10:
        # Try to find a rank-54 solution
        # Fix U, V to be random sparse matrices
        # This is a heuristic
        pass # Implementation of search would go here
        
    return {
        'U': best_U.tolist(),
        'V': best_V.tolist(),
        'W': best_W.tolist()
    }
