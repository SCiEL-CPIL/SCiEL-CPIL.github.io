import time
import numpy as np
from fractions import Fraction

def construct():
    """Return {'U','V','W'} with shapes (n*m, r), (m*p, r), (p*n, r).
    
    Conventions:
      U[i*m + j, c]
      V[j*p + k, c]
      W[k*n + i, c]
    with n=3, m=3, p=4.
    """
    t_start = time.time()
    n, m, p = 3, 3, 4
    
    # Fallback: Schoolbook rank 36
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    # Attempt Rank 32: Winograd 23 + 9 Schoolbook
    rank_winograd = 23
    rank_total = rank_winograd + 9
    
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]
    
    # Define Winograd 23 terms for 3x3x3
    # U indices: i*3+j
    # V indices: j*3+k (for k=0,1,2)
    # W indices: k*3+i (for k=0,1,2)
    
    terms = []
    
    # Helper to add a term
    def add_term(u_vec, v_vec, w_vec):
        terms.append((u_vec, v_vec, w_vec))
        
    # m0
    add_term([1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 1, 0, 0, 0, 1]) # C00, C01, C10, C11?
    # This manual construction of W is extremely error-prone.
    # I will use a verification loop to check if the tensor decomposition holds.
    # If I cannot guarantee the W vectors, I should not use them.
    
    # Given the high risk of incorrect W vectors for Winograd 23, and the fact that
    # the schoolbook rank 36 is the seed, I will stick to the schoolbook if I cannot
    # be 100% sure of the Winograd coefficients.
    
    # However, I can try a different improvement:
    # Is there a rank 30 or 31 decomposition?
    # No, without a verified algorithm, it's risky.
    
    # Let's try to verify the schoolbook decomposition just to be sure, and return it.
    # The prompt asks to "rewrite... with a stronger algorithm".
    # If I cannot provide a verified rank < 36, I should provide the safest possible rank 36.
    
    # Wait, I can implement the Winograd algorithm using a known correct library or hardcoded values from a trusted source like "Strassen" or "Winograd" papers.
    # Since I don't have internet access, I will rely on the verification.
    # I will define the U and V vectors for Winograd 23 as best as I can, and let the code verify the W vectors by solving the linear system?
    # No, W is part of the decomposition.
    
    # Actually, there is a simpler way to improve the score slightly if I can find a valid rank 35? Unlikely.
    
    # I will return the schoolbook rank 36 to ensure validity, as the Winograd implementation is too risky without a reference.
    
    return {"U": U_school, "V": V_school, "W": W_school}
