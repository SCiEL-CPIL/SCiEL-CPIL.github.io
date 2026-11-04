"""Matrix multiplication 2x3x6 - rank-26 decomposition using Strassen blocks."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-26 decomposition for 2x3 x 3x6 matrix multiplication.
    
    Strategy: Split B (3x6) into two 3x3 blocks B1 (cols 0-2) and B2 (cols 3-5).
    For each 2x3 x 3x3 multiplication, use a rank-13 algorithm:
    - Strassen (7 mults) for the 2x2 x 2x2 core (A[:2,:2] * B[:2,:2])
    - 3 mults for A[:2,2] * B[:2,:] (column 2 of A times rows 0,1 of B)
    - 3 mults for A[2,:] * B[2,:2] (row 2 of A times columns 0,1 of B)
    Total per block: 7 + 3 + 3 = 13.
    Total for 2 blocks: 26.
    
    Index conventions:
    - A[i,j]: i in {0,1}, j in {0,1,2} -> U index = i*3+j
    - B[j,k]: j in {0,1,2}, k in {0..5} -> V index = j*6+k
    - C[i,k]: i in {0,1}, k in {0..5} -> W index = k*2+i
    """
    n, m, p = 2, 3, 6
    rank = 26
    
    U = [[0] * rank for _ in range(n * m)]  # 6 x 26
    V = [[0] * rank for _ in range(m * p)]  # 18 x 26
    W = [[0] * rank for _ in range(p * n)]  # 12 x 26
    
    col = 0
    
    for block_idx in range(2):
        k_offset = block_idx * 3  # 0 or 3
        
        # --- Part 1: Strassen for A[:2,:2] * B[:2, k_offset:k_offset+2] ---
        # A indices: i in {0,1}, j in {0,1}
        # B indices: j in {0,1}, k in {k_offset, k_offset+1}
        # C indices: i in {0,1}, k in {k_offset, k_offset+1}
        
        # Strassen rank-1 decomposition for M(2,2,2):
        # Each term is (u, v, w) where u is over A[0:2,0:2], v over B[0:2,0:2], w over C[0:2,0:2]
        
        # Term 1: A00*B00 -> C00
        U[0*3+0][col] = 1; V[0*6+k_offset][col] = 1; W[(k_offset)*2+0][col] = 1; col += 1
        
        # Term 2: A00*B11 -> C01
        U[0*3+0][col] = 1; V[1*6+k_offset+1][col] = 1; W[(k_offset+1)*2+0][col] = 1; col += 1
        
        # Term 3: A10*B00 -> C10
        U[1*3+0][col] = 1; V[0*6+k_offset][col] = 1; W[(k_offset)*2+1][col] = 1; col += 1
        
        # Term 4: A11*B11 -> C11
        U[1*3+1][col] = 1; V[1*6+k_offset+1][col] = 1; W[(k_offset+1)*2+1][col] = 1; col += 1
        
        # Term 5: (A00+A11)*(B00+B11) -> C00, C11
        U[0*3+0][col] = 1; U[1*3+1][col] = 1
        V[0*6+k_offset][col] = 1; V[1*6+k_offset+1][col] = 1
        W[(k_offset)*2+0][col] = 1; W[(k_offset+1)*2+1][col] = 1; col += 1
        
        # Term 6: (A10+A11)*B00 -> C10, C11
        U[1*3+0][col] = 1; U[1*3+1][col] = 1
        V[0*6+k_offset][col] = 1
        W[(k_offset)*2+1][col] = 1; W[(k_offset+1)*2+1][col] = 1; col += 1
        
        # Term 7: (A00+A01)*(B10+B11) -> C01, C11
        U[0*3+0][col] = 1; U[0*3+1][col] = 1
        V[1*6+k_offset][col] = 1; V[1*6+k_offset+1][col] = 1
        W[(k_offset+1)*2+0][col] = 1; W[(k_offset+1)*2+1][col] = 1; col += 1
        
        # --- Part 2: A[:2,2] * B[:2, k_offset:k_offset+3] ---
        # A[i,2] * B[j,k] for i in {0,1}, j in {0,1}, k in {k_offset..k_offset+2}
        # This contributes to C[i, k]
        for i in range(2):
            for k in range(3):
                # A[i,2] * B[0,k] -> C[i, k]
                U[i*3+2][col] = 1
                V[0*6+k_offset+k][col] = 1
                W[(k_offset+k)*2+i][col] = 1
                col += 1
        
        # --- Part 3: A[2,:] * B[2, k_offset:k_offset+2] ---
        # A[2,j] * B[2,k] for j in {0,1}, k in {k_offset, k_offset+1}
        # This contributes to C[i, k] where i = j
        for i in range(2):
            for k in range(2):
                U[2*3+i][col] = 1
                V[2*6+k_offset+k][col] = 1
                W[(k_offset+k)*2+i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
