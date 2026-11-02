"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import time
import sys

def construct():
    start_time = time.time()
    DEADLINE = 110.0  # safety margin before 120s limit

    n, m, p = 4, 4, 9

    # Try to find a better decomposition using Winograd-like transforms
    # For 4x4x9: split into 4x4x4 + 4x4x4 + 4x4x1
    # Winograd for 4x4: rank 32 (using 4-point DFT-like structure)
    # Winograd for 4x1: rank 4 (just 4 multiplications)
    
    # Actually, let me use a simpler approach:
    # The 4x4x4 matrix multiplication can be decomposed using
    # the 4-point Winograd Fourier transform.
    
    # For the 4x4x4 case, we can use the fact that:
    # C = A * B where A is 4x4, B is 4x4
    # Using 4-point DFT: DFT4 * A * DFT4_inv gives diagonal in frequency domain
    
    # The 4-point DFT matrix is:
    # [1  1   1  1 ]
    # [1  i  -1 -i ]
    # [1 -1  1 -1 ]
    # [1 -i -1  i ]
    
    # But we need integer/half-integer entries. Let's use a different approach.
    
    # Actually, for exact decomposition with integer entries, let's use:
    # The standard result that R(4,4,4) = 32
    
    # Let me construct the rank-32 decomposition for 4x4x4 using
    # a known construction based on the 4-point Winograd algorithm.
    
    # The 4-point Winograd transform for convolution uses:
    # G matrix (4x4), M matrix (4x4), F matrix (4x4)
    # where G transforms A, M does pointwise mult, F transforms result
    
    # For matrix multiplication 4x4x4:
    # We view it as computing C[i,j] = sum_k A[i,k] * B[k,j]
    
    # One known rank-32 construction uses the tensor product structure.
    # Actually, let's just use a simple approach:
    # Split the 4x4x9 into three 4x4x3 blocks and handle each.
    
    # For 4x4x3: rank is at most 48 (schoolbook)
    # But we can do better.
    
    # Let me just use the schoolbook decomposition which is guaranteed correct.
    # Rank = 4*4*9 = 144
    
    # Actually, wait. Let me try a smarter approach.
    # We can decompose 4x4x9 as:
    # - 4x4x4 (rank 32 with Winograd)
    # - 4x4x4 (rank 32 with Winograd) 
    # - 4x4x1 (rank 4 with simple)
    # Total: 32 + 32 + 4 = 68
    
    # Let me implement the rank-32 Winograd for 4x4x4.
    # The key insight is that 4x4x4 matrix mult has rank 32.
    
    # Here's a construction using the Kronecker product:
    # 4x4x4 = (2x2x2) ⊗ (2x2x2) in some sense
    # R(2,2,2) = 7
    # But 7*7 = 49 > 32, so that's not right.
    
    # Actually R(4,4,4) = 32 is a known result.
    # Let me use a direct construction.
    
    # The simplest correct approach is schoolbook: rank 144.
    # Let me check if I can do better with a simple block decomposition.
    
    # Split B (4x9) into B1 (4x4), B2 (4x4), B3 (4x1)
    # C = A*B1 + A*B2 + A*B3
    # Each A*Bk is 4x4 result
    
    # For A*B where both are 4x4, schoolbook gives 16 terms per (i,j) pair
    # Total 16*16 = 256? No, that's wrong.
    # For 4x4x4, schoolbook is 4*4*4 = 64 terms.
    
    # So total schoolbook for 4x4x9 is 4*4*9 = 144.
    
    # Let me just use schoolbook to be safe and correct.
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
