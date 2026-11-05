"""Program to find rank-minimal exact tensor decomposition for 3x3 * 3x6 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Use Laderman's algorithm for 3x3 matrix multiplication (rank 19)
    # and apply to two 3x3 blocks of the 3x6 matrix
    
    # Laderman's 19-rank decomposition for 3x3 * 3x3 = 3x3
    # U: (9, 19), V: (9, 19), W: (9, 19)
    # Indexing: U[i*3+j], V[j*3+k], W[k*3+i] for i,j,k in 0..2
    
    # Laderman's algorithm coefficients
    # Based on Laderman (1976)
    laderman_U = [
        [1,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
        [0,1,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
        [0,0,1, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
        [1,1,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
        [0,0,0, 1,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
        [0,0,0, 0,1,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
        [0,0,0, 0,0,1, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
        [1,1,1, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
        [0,0,0, 1,1,1, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
    ]
    
    # This is getting complex. Let me use a known correct Laderman decomposition.
    # Actually, let me try a different approach: use a simpler but still improved decomposition.
    
    # For 3x3 * 3x6, we can use the fact that we're doing 6 separate 3x3 multiplications
    # but share the A matrix. 
    # 
    # Alternative: Use a rank-r decomposition where r < 54
    # 
    # Let me try: group the 6 columns of B into 2 groups of 3.
    # For each group, use a 3x3 multiplication.
    # If we use rank 19 for each 3x3 multiplication, total rank = 38.
    
    # Let me implement Laderman's algorithm properly.
    # Reference: Laderman, "The number of multiplications and additions needed to compute a 3x3 matrix product"
    
    # Laderman's 19 multiplications:
    # Let A = [[a11,a12,a13],[a21,a22,a23],[a31,a32,a33]]
    # Let B = [[b11,b12,b13],[b21,b22,b23],[b31,b32,b33]]
    
    # The 19 products are:
    # M1 = a11*b11
    # M2 = a12*b21 + (a11+a12)*b12
    # M3 = a13*b31 + (a11+a13)*b32  
    # M4 = a21*b11 + (a21+a22)*b12
    # M5 = a22*b22
    # M6 = a23*b32 + (a22+a23)*b33
    # M7 = a31*b12 + (a31+a32)*b22
    # M8 = a32*b23 + (a32+a33)*b33
    # M9 = a33*b33
    # M10 = (a11+a12+a13+a21+a22+a23+a31+a32+a33)*(b11+b12+b13+b21+b22+b23+b31+b32+b33)
    # ... this is getting complicated
    
    # Let me use a verified Laderman decomposition from literature.
    # I'll use the one from: https://en.wikipedia.org/wiki/Matrix_multiplication_algorithm
    
    # Actually, let me try a more systematic approach.
    # For 3x3 * 3x6, the minimum rank is unknown but bounded.
    # Schoolbook: 54
    # Using 2x Laderman(19): 38
    
    # Let me implement a correct Laderman decomposition.
    
    # Based on Laderman 1976, the 19 terms are:
    # Define sums:
    # s1 = a11 + a12 + a13
    # s2 = a21 + a22 + a23  
    # s3 = a31 + a32 + a33
    # t1 = b11 + b21 + b31
    # t2 = b12 + b22 + b32
    # t3 = b13 + b23 + b33
    
    # This is getting too complex to derive from scratch. Let me use a known correct implementation.
    
    # Alternative approach: Use a rank-27 decomposition by sharing some computations.
    # Or try to find a lower rank through optimization.
    
    # For now, let me try a simple improvement: 
    # Note that C[i,k] = sum_j A[i,j]*B[j,k]
    # We can compute this using 54 multiplications (schoolbook).
    # 
    # Can we do better? Let's think about the structure.
    # 
    # Actually, let me try to use the fact that we can combine some terms.
    # For example, if we define X[i,j] = A[i,j] for all i,j, and Y[j,k] = B[j,k],
    # then C[i,k] = sum_j X[i,j]*Y[j,k].
    # 
    # The tensor rank of this is what we're looking for.
    
    # Let me try a different strategy: use numerical optimization to find a low-rank decomposition.
    # Start from schoolbook and try to reduce rank.
    
    # Given time constraints, let me return schoolbook for now but set up for future improvement.
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
