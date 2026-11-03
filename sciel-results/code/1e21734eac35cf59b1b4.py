import time
import numpy as np

DIMENSIONS = [4, 4, 5]

def _strassen_2x2_terms():
    # Returns list of 7 tuples (u, v, w) of length 4
    # u[i*2+j], v[j*2+k], w[k*2+i]
    terms = []
    # 1
    u1 = [1, 0, 1, 0] # A00, A11
    v1 = [1, 0, 1, 0] # B00, B11
    w1 = [1, 0, 1, 0] # C00, C11
    terms.append((u1, v1, w1))
    # ... (as in seed)
    return terms

def _construct_4x4x4_strassen():
    # Returns U, V, W for 4x4x4 part (rank 49)
    # U: 16x49, V: 16x49 (only first 4 cols of B used), W: 16x49 (only first 4 rows of C used)
    # Actually V is 20x49, W is 20x49.
    # We will build the 49 terms.
    
    s2_terms = _strassen_2x2_terms()
    
    # The 7 block formulas
    # M1 = (A00+A11)(B00+B11)
    # A00 is A[0:2, 0:2], A11 is A[2:4, 2:4]
    # We need to express A00+A11 as a linear form in A.
    # Let's represent the block matrices as 4x4 matrices with zeros elsewhere.
    
    # Block A00: rows 0,1 cols 0,1
    # Block A11: rows 2,3 cols 2,3
    
    # For each of the 7 block products, we define the linear forms for the "left" and "right" 2x2 matrices.
    # Then we apply the 2x2 Strassen decomposition to the product of these two 2x2 matrices.
    
    # This is getting complicated to do symbolically in code without a CAS.
    # Instead, I will just use the 80-rank schoolbook if I can't get 49 quickly.
    # But 49 is doable.
    
    # Let's try a different approach:
    # Use the 28-rank decomposition if I can find it.
    # Actually, let's just use the 65-rank one (49 + 16).
    
    pass
