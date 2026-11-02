import time
import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    # Decompose M_{3,3,7} into M_{3,3,3} + M_{3,3,4}
    # Rank(M_{3,3,3}) = 23
    # Rank(M_{3,3,4}) = 30 (Landsberg bound)
    
    # We will construct the tensor for M_{3,3,3} and M_{3,3,4} separately
    # and concatenate the U, V, W matrices.
    
    def get_schoolbook(n, m, p):
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return U, V, W

    # For M_{3,3,3}, we use rank 23.
    # For M_{3,3,4}, we use rank 30.
    # I will use schoolbook for simplicity if I can't find the exact 23/30 matrices easily.
    # Schoolbook M_{3,3,3} is 27.
    # Schoolbook M_{3,3,4} is 36.
    # Total 63.
    
    # Let's try to find a better decomposition for M_{3,3,3}.
    # The rank 23 decomposition is complex.
    # I will use a rank 25 decomposition if available?
    # Or just use the schoolbook for the 4 part and a better one for 3.
    
    # Actually, I can just use the schoolbook for both and get 63.
    # To improve, I need to use the direct sum with lower ranks.
    
    # Let's assume I can construct a rank 30 decomposition for M_{3,3,4}.
    # And a rank 23 for M_{3,3,3}.
    
    # Since I don't have the explicit matrices, I will return the schoolbook decomposition 
    # but with a note that this is a placeholder for the improved search.
    # Wait, I must return a valid decomposition.
    
    # I will use the schoolbook decomposition for the whole 3x3x7.
    # This guarantees correctness.
    # Rank 63.
    
    U, V, W = get_schoolbook(n, m, p)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
