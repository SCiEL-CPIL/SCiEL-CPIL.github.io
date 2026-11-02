import numpy as np
import time

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition
    # Rank = n * m * p
    rank = n * m * p
    
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                U[i * m + j, col] = 1
                # V index: j*p + k
                V[j * p + k, col] = 1
                # W index: k*n + i
                W[k * n + i, col] = 1
                col += 1
                
    # Convert to lists if required by interface, though numpy arrays are usually accepted
    # The prompt says "factor matrices of shapes ... with integer/half-integer entries"
    # It doesn't explicitly forbid numpy arrays, but the seed uses lists.
    # To be safe, I'll return lists or numpy arrays?
    # "Keep the function names and signatures ... return interface EXACTLY"
    # The seed returns a dict of lists.
    # I will return a dict of lists to be safe, or numpy arrays if the evaluator accepts them.
    # Given "Python standard library, NumPy and SciPy are available", numpy is likely fine.
    # However, mixing might be risky if the evaluator expects lists.
    # I will stick to lists to match the seed's interface exactly.
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
