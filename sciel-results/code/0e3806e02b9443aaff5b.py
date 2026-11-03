"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    # 8 independent Strassen 2x2 decompositions for 2x4 x 4x8
    n, m, p = DIMENSIONS
    rank = 8 * 7  # 56
    U = [[0] * rank for _ in range(n * m)]  # 8 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 16 x 56
    # W[k*n + i, col]
    
    col = 0
    # For each pair of middle indices (a, a+1) and each output block
    # We have 2 (i rows) x 4 (j cols) x 8 (k cols)
    # Split j into 2 pairs: (0,1), (2,3) -> 2 Strassen blocks
    # Split k into 4 pairs: (0,1), (2,3), (4,5), (6,7) -> 4 Strassen blocks
    # Total: 2 * 4 = 8 Strassen blocks
    
    # Strassen 2x2x2:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B11-B21)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B22)
    # M7 = (A12-A22)*(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For block (j_pair, k_pair):
    # A is 2x2 (i x j), B is 2x2 (j x k), C is 2x2 (i x k)
    # j_pair determines which j indices: jp=0 -> j=0,1; jp=1 -> j=2,3
    # k_pair determines which k indices: kp=0 -> k=0,1; kp=1 -> k=2,3; kp=2 -> k=4,5; kp=3 -> k=6,7
    
    for jp in range(2):  # j pair
        for kp in range(4):  # k pair
            # Strassen M1..M7
            for s in range(7):
                c = col + s
                # M1 = (A11+A22)*(B11+B22)
                if s == 0:
                    # A11: i=0, j=jp*2; A22: i=1, j=jp*2+1
                    U[0*4 + jp*2][c] = 1
                    U[1*4 + jp*2+1][c] = 1
                    # B11: j=jp*2, k=kp*2; B22: j=jp*2+1, k=kp*2+1
                    V[jp*2*8 + kp*2][c] = 1
                    V[(jp*2+1)*8 + kp*2+1][c] = 1
                    # C11: i=0, k=kp*2; C22: i=1, k=kp*2+1
                    W[kp*2*2 + 0][c] = 1
                    W[(kp*2+1)*2 + 1][c] = 1
                # M2 = (A21+A22)*B11
                elif s == 1:
                    # A21: i=1, j=jp*2; A22: i=1, j=jp*2+1
                    U[1*4 + jp*2][c] = 1
                    U[1*4 + jp*2+1][c] = 1
                    # B11: j=jp*2, k=kp*2
                    V[jp*2*8 + kp*2][c] = 1
                    # C21: i=1, k=kp*2
                    W[kp*2*2 + 1][c] = 1
                # M3 = A11*(B11-B21)
                elif s == 2:
                    # A11: i=0, j=jp*2
                    U[0*4 + jp*2][c] = 1
                    # B11: j=jp*2, k=kp*2; B21: j=jp*2+1, k=kp*2
                    V[jp*2*8 + kp*2][c] = 1
                    V[(jp*2+1)*8 + kp*2][c] = -1
                    # C12: i=0, k=kp*2+1
                    W[(kp*2+1)*2 + 0][c] = 1
                # M4 = A22*(B21-B11)
                elif s == 3:
                    # A22: i=1, j=jp*2+1
                    U[1*4 + jp*2+1][c] = 1
                    # B21: j=jp*2+1, k=kp*2; B11: j=jp*2, k=kp*2
                    V[(jp*2+1)*8 + kp*2][c] = 1
                    V[jp*2*8 + kp*2][c] = -1
                    # C11: i=0, k=kp*2; C21: i=1, k=kp*2
                    W[kp*2*2 + 0][c] = 1
                    W[kp*2*2 + 1][c] = 1
                # M5 = (A11+A12)*B22
                elif s == 4:
                    # A11: i=0, j=jp*2; A12: i=0, j=jp*2+1
                    U[0*4 + jp*2][c] = 1
                    U[0*4 + jp*2+1][c] = 1
                    # B22: j=jp*2+1, k=kp*2+1
                    V[(jp*2+1)*8 + kp*2+1][c] = 1
                    # C11: i=0, k=kp*2; C12: i=0, k=kp*2+1
                    W[kp*2*2 + 0][c] = -1
                    W[(kp*2+1)*2 + 0][c] = 1
                # M6 = (A21-A11)*(B11+B22)
                elif s == 5:
                    # A21: i=1, j=jp*2; A11: i=0, j=jp*2
                    U[1*4 + jp*2][c] = 1
                    U[0*4 + jp*2][c] = -1
                    # B11: j=jp*2, k=kp*2; B22: j=jp*2+1, k=kp*2+1
                    V[jp*2*8 + kp*2][c] = 1
                    V[(jp*2+1)*8 + kp*2+1][c] = 1
                    # C22: i=1, k=kp*2+1
                    W[(kp*2+1)*2 + 1][c] = 1
                # M7 = (A12-A22)*(B21+B22)
                elif s == 6:
                    # A12: i=0, j=jp*2+1; A22: i=1, j=jp*2+1
                    U[0*4 + jp*2+1][c] = 1
                    U[1*4 + jp*2+1][c] = -1
                    # B21: j=jp*2+1, k=kp*2; B22: j=jp*2+1, k=kp*2+1
                    V[(jp*2+1)*8 + kp*2][c] = 1
                    V[(jp*2+1)*8 + kp*2+1][c] = 1
                    # C11: i=0, k=kp*2
                    W[kp*2*2 + 0][c] = 1
            col += 7
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

