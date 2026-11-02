"""
Rank-minimal decomposition for 2x2 by 2x10 matrix multiplication.

The tensor for this multiplication is a direct sum of 5 independent 2x2x2 
matrix multiplication tensors. The rank of 2x2x2 is 7 (Strassen).
Thus, the minimal rank is 5 * 7 = 35.

This program constructs the rank-35 decomposition directly.
"""
import time

DIMENSIONS = [2, 2, 10]

def construct():
    n, m, p = DIMENSIONS
    rank = 35
    
    # Initialize matrices with zeros
    # U: (n*m, rank) = (4, 35)
    # V: (m*p, rank) = (20, 35)
    # W: (p*n, rank) = (20, 35)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen factors for 2x2 matrix multiplication.
    # A indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # B indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # C indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # Note: The W vector order in the code below maps to:
    # C[0] -> C_{11} (row 0, col 0)
    # C[1] -> C_{21} (row 0, col 1) -- Wait, let's re-verify the W index mapping.
    
    # From analysis:
    # W[4*t]   -> k=4t   -> p=2t, r=0 -> C[2t, 0]
    # W[4*t+1] -> k=4t+1 -> p=2t, r=1 -> C[2t, 1]
    # W[4*t+2] -> k=4t+2 -> p=2t+1, r=0 -> C[2t+1, 0]
    # W[4*t+3] -> k=4t+3 -> p=2t+1, r=1 -> C[2t+1, 1]
    
    # So the local C vector [c0, c1, c2, c3] adds to:
    # c0 * C[2t, 0]
    # c1 * C[2t, 1]
    # c2 * C[2t+1, 0]
    # c3 * C[2t+1, 1]
    
    # Standard Strassen outputs:
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    
    # We need to map these to the local indices:
    # Local C11 is index 0.
    # Local C21 is index 1.
    # Local C12 is index 2.
    # Local C22 is index 3.
    
    # So for each Pi, we define a vector [coeff_C11, coeff_C21, coeff_C12, coeff_C22]
    
    strassen_terms = [
        # P1: (A11+A22)(B11+B22)
        # A: [1, 0, 0, 1]
        # B: [1, 0, 0, 1]
        # C: C11+=1, C22+=1 -> [1, 0, 0, 1]
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        
        # P2: (A21+A22)B11
        # A: [0, 0, 1, 1]
        # B: [1, 0, 0, 0]
        # C: C21+=1, C22-=1 -> [0, 1, 0, -1]
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]),
        
        # P3: A11(B12-B22)
        # A: [1, 0, 0, 0]
        # B: [0, 1, 0, -1]
        # C: C12+=1, C22+=1 -> [0, 0, 1, 1]
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]),
        
        # P4: A22(B21-B11)
        # A: [0, 0, 0, 1]
        # B: [-1, 0, 1, 0]
        # C: C11+=1, C21+=1 -> [1, 1, 0, 0]
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]),
        
        # P5: (A11+A12)B22
        # A: [1, 1, 0, 0]
        # B: [0, 0, 0, 1]
        # C: C11-=1, C12+=1 -> [-1, 0, 1, 0]
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),
        
        # P6: (A21-A11)(B11+B12)
        # A: [-1, 0, 1, 0]
        # B: [1, 1, 0, 0]
        # C: C22+=1 -> [0, 0, 0, 1]
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        
        # P7: (A12-A22)(B21+B22)
        # A: [0, 1, 0, -1]
        # B: [0, 0, 1, 1]
        # C: C11+=1 -> [1, 0, 0, 0]
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]
    
    col = 0
    for t in range(5):
        for a_vec, b_vec, c_vec in strassen_terms:
            # Fill U (A part)
            for i in range(4):
                U[i][col] = a_vec[i]
            
            # Fill V (B part)
            # B11 -> 2*t
            # B12 -> 2*t+1
            # B21 -> 10+2*t
            # B22 -> 10+2*t+1
            V[2*t][col] = b_vec[0]
            V[2*t+1][col] = b_vec[1]
            V[10+2*t][col] = b_vec[2]
            V[10+2*t+1][col] = b_vec[3]
            
            # Fill W (C part)
            # C11 -> 4*t
            # C21 -> 4*t+1
            # C12 -> 4*t+2
            # C22 -> 4*t+3
            W[4*t][col] = c_vec[0]
            W[4*t+1][col] = c_vec[1]
            W[4*t+2][col] = c_vec[2]
            W[4*t+3][col] = c_vec[3]
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
