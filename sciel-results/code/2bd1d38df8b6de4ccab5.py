import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Laderman decomposition for 4x4x4
    # U, V, W are 16x23
    # This is the 23-term algorithm for 4x4 matrix multiplication
    # Reference: Laderman (1976)
    
    # I will use the known Laderman factors.
    # Since I cannot reliably recall all 23x16 entries from memory, 
    # I will construct a valid rank 44 decomposition using Strassen blocks.
    
    # Structure:
    # A (3x4) = [A_top (2x4); A_bottom (1x4)]
    # B (4x4)
    # C_top (2x4) = A_top * B
    # C_bottom (1x4) = A_bottom * B
    
    # For A_top * B (2x4 * 4x4 -> 2x4):
    # Split B into 2x2 blocks: B11, B12, B21, B22
    # Split A_top into A1 (2x2), A2 (2x2)
    # C_top = [A1*B11 + A2*B21, A1*B12 + A2*B22]
    # This uses 4 multiplications of 2x2 matrices.
    # Each 2x2 multiplication uses Strassen (rank 7).
    # Total rank for top part: 4 * 7 = 28.
    
    # For A_bottom * B (1x4 * 4x4 -> 1x4):
    # This is a dot product for each column.
    # Rank is 1 * 4 * 4 = 16.
    
    # Total rank: 28 + 16 = 44.
    
    rank = 44
    U = [[0] * rank for _ in range(n * m)] # 12 x 44
    V = [[0] * rank for _ in range(m * p)] # 16 x 44
    W = [[0] * rank for _ in range(p * n)] # 12 x 44
    
    # Strassen factors for 2x2x2
    # U_s, V_s, W_s are 4x7
    U_s = [
        [1, 0, 0, 0, 0, 0, 0], # A00
        [0, 0, 0, 0, 1, 0, 0], # A01
        [0, 0, 0, 1, 0, 0, 0], # A10
        [0, 1, 0, 0, 0, 0, 0], # A11
        [0, 0, 0, 0, 0, 0, 0], # placeholder
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0]
    ]
    # Correct Strassen 2x2 factors:
    # M1 = A00 B00
    # M2 = A10 (B01 + B11)
    # M3 = (A00 + A01) B10
    # M4 = A11 (B10 - B00)
    # M5 = (A01 + A11) (B00 - B10)
    # M6 = (A00 + A10) (B11 - B01)
    # M7 = A10 B11
    
    # This is getting complicated to hardcode correctly without verification.
    
    # I will fall back to the schoolbook 48 to ensure correctness, 
    # as the prompt requires a valid decomposition.
    # However, 48 is the baseline.
    
    # Let's try a simple optimization:
    # Use Strassen for the 2x2 blocks in the 2x4 part.
    
    return _schoolbook(n, m, p)

def _schoolbook(n, m, p):
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
