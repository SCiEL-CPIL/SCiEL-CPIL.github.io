import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # Part 1: Strassen for 4x4x4
    # Rank 49
    # We will build U1, V1, W1 for the first 4 columns of B and C
    
    U1 = []
    V1 = []
    W1 = []
    
    # Strassen 2x2x2 decomposition
    # Inputs A (2x2), B (2x2), Output C (2x2)
    # Local indices for A: 0,1,2,3 (i*m+j)
    # Local indices for B: 0,1,2,3 (j*p+k)
    # Local indices for C: 0,1,2,3 (k*n+i)
    
    # Strassen 2x2x2 has 7 multiplications
    # We need to define the 7 rank-1 terms for 2x2x2
    # This is standard Strassen.
    
    # Let's hardcode the Strassen 2x2x2 decomposition
    # A = [[a11, a12], [a21, a22]]
    # B = [[b11, b12], [b21, b22]]
    # C = A B
    
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 - M2 + M3 - M4
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M5 + M6 + M7
    
    # We need to represent each Mi as a sum of rank-1 tensors?
    # No, each Mi is a 2x2x2 multiplication itself.
    # So we need to decompose each Mi using the 7-term Strassen for 2x2?
    # No, Mi is a product of two 2x2 matrices.
    # So we use the 7-term Strassen for 2x2x2 to compute each Mi.
    # This results in 7 * 7 = 49 terms for 4x4x4.
    
    # This is getting complex to code from scratch correctly.
    # Alternative: Use the known rank 49 decomposition.
    
    # Let's just use the schoolbook for the whole thing if Strassen is too hard to implement correctly in the time.
    # But 96 is the seed. 81 is a significant improvement.
    
    # Let's try to implement the 4x4x4 Strassen.
    
    pass
