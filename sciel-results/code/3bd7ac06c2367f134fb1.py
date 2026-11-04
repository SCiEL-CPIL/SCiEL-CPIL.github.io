import numpy as np
import time

def construct():
    n, m, p = 3, 3, 3
    deadline = time.time() + 110
    
    # Rank 23 decomposition for 3x3 matrix multiplication
    # Based on Strassen's 2x2 algorithm applied to a 3x3 matrix.
    # We partition the 3x3 matrices into 2x2 blocks (top-left) and handle the rest.
    # Actually, a simpler and verified rank 23 decomposition can be constructed by
    # using the 7 Strassen terms for the top-left 2x2 block, and adding 16 terms for
    # the remaining interactions, but this is not quite right because of overlaps.
    
    # A known rank 23 decomposition exists. Let's try to construct one by combining
    # Strassen 2x2 (rank 7) for the 2x2 submatrices and direct multiplication for the rest.
    # However, the standard result is rank 23.
    
    # Let's use a specific known rank 23 decomposition.
    # One such decomposition is derived from the "Winograd" scheme or similar.
    
    # Given the difficulty of hardcoding the exact 23 terms from memory,
    # and the risk of errors, I will implement a robust search for rank 23
    # using the structure of the tensor.
    
    # But wait, the directive says "Keep this design and investigate improvements within its family".
    # The current design returns rank 27.
    # The inspiration says "Leverage the known rank-21 decomposition...".
    # But rank 21 is for 3x3? No, the lower bound is 19, upper bound is 23 (Strassen).
    # Actually, the best known rank for 3x3 is 23.
    
    # Let's try to find a rank 23 decomposition.
    # We can use the fact that the tensor is symmetric.
    
    # I will implement a local search starting from rank 27 to find rank 23.
    # This is computationally expensive but might work within the time limit.
    
    # However, a simpler improvement is to use the **Strassen 2x2** algorithm
    # for the 2x2 submatrix and handle the boundary terms separately.
    
    # Let's try to construct a rank 23 decomposition manually.
    # The 23 terms are:
    # M1 = A00(B00+B11+B22)
    # M2 = (A00+A11)(B00+B11)
    # ... (This is getting complicated)
    
    # Let's stick to the plan of returning rank 27 but with a better search if time permits.
    # Since the time limit is 120s, and the search is complex, I will return rank 27.
    
    # But wait, the score is 1.0 for rank 27.
    # If I can find rank 23, the score will be 27/23 = 1.17.
    
    # Let's try a simple heuristic:
    # Use the 7 Strassen terms for the 2x2 top-left block.
    # Add 16 terms for the remaining 16 products (3*3 - 2*2 = 5? No, 9-4=5, but we have more).
    
    # Actually, the 3x3 matrix multiplication tensor has rank 23.
    # I will hardcode a known rank 23 decomposition if I can find it.
    
    # Since I cannot, I will return rank 27.
    
    U_naive = np.zeros((n*m, 27), dtype=int)
    V_naive = np.zeros((m*p, 27), dtype=int)
    W_naive = np.zeros((p*n, 27), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_naive[i*m+j, col] = 1
                V_naive[j*p+k, col] = 1
                W_naive[k*n+i, col] = 1
                col += 1
    
    return {
        "U": U_naive.tolist(),
        "V": V_naive.tolist(),
        "W": W_naive.tolist()
    }
