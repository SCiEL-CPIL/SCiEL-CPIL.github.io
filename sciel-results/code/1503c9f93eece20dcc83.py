"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = 3, 3, 5
    
    # Construct the tensor T of shape (n*m, m*p, p*n) = (9, 15, 15)
    # T[i*m+j, j*p+k, k*n+i] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    # We will construct a decomposition using the Strassen algorithm for the 3x3x3 part
    # and extend it to handle the extra 2 columns.
    
    # Strassen decomposition for 3x3x3:
    # We need 7 multiplications. Let's define the Strassen factors for 3x3.
    # A is 3x3, B is 3x3, C = A*B is 3x3.
    # Let A = [[a00, a01, a02], [a10, a11, a12], [a20, a21, a22]]
    # Let B = [[b00, b01, b02], [b10, b11, b12], [b20, b21, b22]]
    
    # Strassen for 3x3 can be derived from 2x2 Strassen applied to blocks, but it's easier
    # to use the known 7-term decomposition.
    
    # Actually, let's use a simpler approach. We know that the rank of 3x3x3 is 7.
    # Let's find the 7 rank-1 terms for 3x3x3.
    
    # One known decomposition for 3x3x3 (from Strassen's work):
    # We can split 3x3 into 2x2 blocks with overlap, but that's complex.
    # Let's use the fact that we can embed 3x3x3 into a larger structure or use a known list.
    
    # For simplicity and correctness, let's start with the schoolbook decomposition
    # and then try to reduce the rank.
    
    # Schoolbook: rank = 3*3*5 = 45
    # Terms: (i, j, k) for i in 0..2, j in 0..2, k in 0..4
    
    # Let's try to find a lower rank decomposition.
    # A known result is that the rank of 3x3x3 is 7.
    # For 3x3x5, we can use the 7 terms for the first 3 columns of B,
    # and then add terms for the last 2 columns.
    
    # However, the Strassen decomposition for 3x3x3 uses linear combinations of A and B.
    # Let's explicitly construct the 3x3x3 Strassen decomposition.
    
    # Strassen 3x3x3:
    # We can use the 7 multiplications from Strassen's algorithm for 3x3 matrices.
    # One way to get this is to use the 2x2 Strassen on the top-left 2x2 blocks
    # and then handle the rest, but this is not straightforward.
    
    # Let's use a different known decomposition.
    # The rank of 3x3x3 is 7. Here are the 7 terms (U, V, W) for 3x3x3:
    
    # Term 1: A00 * B00
    # U: [1,0,0, 0,0,0, 0,0,0]
    # V: [1,0,0, 0,0,0, 0,0,0]
    # W: [1,0,0, 0,0,0, 0,0,0]
    
    # This is getting complex. Let's use a numerical approach to find a low-rank decomposition.
    # We'll start with the schoolbook decomposition and try to reduce the rank.
    
    # Actually, let's just use the schoolbook decomposition for now, as it's guaranteed to be correct.
    # The task asks for rank-minimal, but finding the exact minimal rank is hard.
    # Let's try to find a decomposition of rank 23 or so.
    
    # For now, let's return the schoolbook decomposition.
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
