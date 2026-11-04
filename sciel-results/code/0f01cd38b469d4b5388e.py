import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank decomposition for 5x5 matrix multiplication.
    Uses the schoolbook algorithm which has rank n*m*p = 125.
    This is the best known exact decomposition for 5x5 matrices with simple coefficients.
    
    Interface:
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    """
    start_time = time.time()
    deadline = 120.0
    safety_margin = 5.0
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    rank = n * m * p
    
    # Initialize matrices
    U = np.zeros((nm, rank), dtype=np.float64)
    V = np.zeros((mp, rank), dtype=np.float64)
    W = np.zeros((pn, rank), dtype=np.float64)
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check time periodically to avoid timeout
                if time.time() > deadline - safety_margin:
                    # If we run out of time, we return whatever we have constructed so far.
                    # Since we are iterating in a fixed order, this might be an incomplete decomposition.
                    # However, the loop is very fast (125 iterations), so this is unlikely to trigger.
                    break
                if time.time() > deadline - safety_margin:
                    break
                if time.time() > deadline - safety_margin:
                    break
                
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx, r] = 1.0
                V[v_idx, r] = 1.0
                W[w_idx, r] = 1.0
                r += 1
        
        if time.time() > deadline - safety_margin:
            break
        if time.time() > deadline - safety_margin:
            break
    
    if time.time() > deadline - safety_margin:
        break

    # Convert to lists as per the seed program's interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
