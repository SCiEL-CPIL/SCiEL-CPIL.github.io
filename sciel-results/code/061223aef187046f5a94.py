import numpy as np

def construct():
    n, m, p = 2, 3, 6
    rank = n * m * p
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # We can fill these using index arithmetic
    # Index in U: i*m + j
    # Index in V: j*p + k
    # Index in W: k*n + i
    
    # We iterate over the rank dimension (which corresponds to i, j, k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m * p + j * p + k
                U[i * m + j, r] = 1
                V[j * p + k, r] = 1
                W[k * n + i, r] = 1
                
    # Convert to lists as required by the interface? 
    # The interface says "factor matrices of shapes ... with integer/half-integer entries".
    # The return type in the example was a dict of lists of lists.
    # I should check if numpy arrays are accepted or if I must convert.
    # "Keep the function names and signatures ... return interface EXACTLY as in the current program"
    # Current program returns {"U": U, "V": V, "W": W} where U, V, W are lists of lists.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
