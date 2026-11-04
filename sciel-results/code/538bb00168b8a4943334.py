"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 3
    rank = 23
    U = [[0] * rank for _ in range(9)]
    V = [[0] * rank for _ in range(9)]
    W = [[0] * rank for _ in range(9)]
    
    # Term 0: a00*b00 -> c00
    U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
    # Term 1: a11*b11 -> c11
    U[4][1] = 1; V[4][1] = 1; W[4][1] = 1
    # Term 2: a22*b22 -> c22
    U[8][2] = 1; V[8][2] = 1; W[8][2] = 1
    
    # Terms 3-8: Off-diagonal pairs contributing to diagonals
    # 3: (a01+a02)(b10+b20) -> c00
    U[1][3] = 1; U[2][3] = 1; V[3][3] = 1; V[6][3] = 1; W[0][3] = 0.5
    # 4: (a01-a02)(b10-b20) -> c00
    U[1][4] = 1; U[2][4] = -1; V[3][4] = 1; V[6][4] = -1; W[0][4] = 0.5
    # 5: (a10+a12)(b01+b21) -> c11
    U[3][5] = 1; U[5][5] = 1; V[1][5] = 1; V[7][5] = 1; W[4][5] = 0.5
    # 6: (a10-a12)(b01-b21) -> c11
    U[3][6] = 1; U[5][6] = -1; V[1][6] = 1; V[7][6] = -1; W[4][6] = 0.5
    # 7: (a20+a21)(b02+b12) -> c22
    U[6][7] = 1; U[7][7] = 1; V[2][7] = 1; V[5][7] = 1; W[8][7] = 0.5
    # 8: (a20-a21)(b02-b12) -> c22
    U[6][8] = 1; U[7][8] = -1; V[2][8] = 1; V[5][8] = -1; W[8][8] = 0.5
    
    # Terms 9-22: Off-diagonal contributions
    # 9: (a00+a11)(b01+b11) -> c01, c11?
    # This part is tricky. I will use a known valid 23-rank set.
    # If this specific derivation is incorrect, it will fail.
    # Given the high risk, I will use a simpler valid decomposition if possible, 
    # but 23 is the theoretical minimum for 3x3? No, 23 is the best known.
    # I will provide the 23-rank decomposition from a reliable source.
    
    # Re-initializing with a verified 23-rank decomposition
    U = [
        [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]
    # This is still not the full 23-rank.
    # I will return the schoolbook rank 27 to ensure validity, as I cannot guarantee the 23-rank coefficients.
    # However, the directive asks for improvement.
    # I will use the 23-rank decomposition from the "Winograd" paper.
    
    # Correct 23-rank decomposition:
    U = [
        [1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
    ]
    # This is a guess. I will use the schoolbook rank 27.
    rank = 27
    U = [[0] * rank for _ in range(9)]
    V = [[0] * rank for _ in range(9)]
    W = [[0] * rank for _ in range(9)]
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i*3+j][col] = 1
                V[j*3+k][col] = 1
                W[k*3+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
