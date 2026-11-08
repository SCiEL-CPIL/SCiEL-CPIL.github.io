"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()
TIME_LIMIT = 120.0

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    
    # Schoolbook decomposition of rank n*m*p
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
    
    # Embedded Strassen decomposition for 2x4 x 4x4
    # Partition A (2x4) into A = [A11 A12; A21 A22] where each is 1x2 (but we need 2x2 blocks for Strassen)
    # Actually, let's think of it differently.
    # A is 2x4, B is 4x4.
    # We can pad A to 4x4 by adding zero rows: A_pad = [A; 0]
    # Then A_pad * B is 4x4, and the top 2 rows are A*B.
    # We can use a rank-28 decomposition for 4x4x4 matrix multiplication.
    # The standard Strassen algorithm for 4x4x4 uses 28 multiplications.
    # However, since the bottom 2 rows of A are zero, we might be able to save some multiplications.
    
    # Let's use the known rank-28 decomposition for 4x4x4.
    # The Strassen algorithm for 4x4 matrices:
    # Partition into 2x2 blocks of 2x2 matrices.
    # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For our case, A_pad = [A; 0] where A is 2x4.
    # A_pad = [[A11, A12], [0, 0]] where A11, A12 are 2x2 blocks.
    # So A21 = 0, A22 = 0.
    # M1 = (A11 + 0)(B11 + B22) = A11(B11 + B22)
    # M2 = (0 + 0)B11 = 0
    # M3 = A11(B12 - B22)
    # M4 = 0(B21 - B11) = 0
    # M5 = (A11 + A12)B22
    # M6 = (0 - A11)(B11 + B12) = -A11(B11 + B12)
    # M7 = (A12 - 0)(B21 + B22) = A12(B21 + B22)
    # C11 = M1 + 0 - M5 + M7 = A11(B11 + B22) - (A11 + A12)B22 + A12(B21 + B22)
    #     = A11*B11 + A11*B22 - A11*B22 - A12*B22 + A12*B21 + A12*B22
    #     = A11*B11 + A12*B21
    # C12 = M3 + M5 = A11(B12 - B22) + (A11 + A12)B22 = A11*B12 - A11*B22 + A11*B22 + A12*B22 = A11*B12 + A12*B22
    # C21 = M2 + M4 = 0
    # C22 = M1 - M2 + M3 + M6 = A11(B11 + B22) - 0 + A11(B12 - B22) - A11(B11 + B12)
    #     = A11*B11 + A11*B22 + A11*B12 - A11*B22 - A11*B11 - A11*B12 = 0
    
    # So we only need 5 multiplications: M1, M3, M5, M6, M7 (M2 and M4 are zero)
    # Wait, M1, M3, M5, M6, M7 are 5 multiplications of 2x2 matrices.
    # Each 2x2 multiplication can be done with 8 scalar multiplications (schoolbook) or 7 (Strassen).
    # But we want a tensor decomposition, so we need to express this as U, V, W.
    
    # Actually, let's just use the full 4x4x4 rank-28 decomposition and see if it works.
    # The rank-28 decomposition for 4x4x4 is well-known.
    
    # Let's construct the U, V, W matrices for the rank-28 decomposition.
    # Each entry is a 2x2 matrix multiplication, which can be decomposed into 8 scalar multiplications.
    # But that would give rank 28*8 = 224, which is worse than 32.
    
    # So we need a different approach. Let's think about the problem more carefully.
    # A is 2x4, B is 4x4. We want to compute C = A*B, which is 2x4.
    # This is equivalent to computing 8 scalar products, each of length 4.
    # The schoolbook rank is 32.
    
    # Can we do better? Let's try to use the structure of the problem.
    # A = [a11 a12 a13 a14; a21 a22 a23 a24]
    # B = [b11 b12 b13 b14; b21 b22 b23 b24; b31 b32 b33 b34; b41 b42 b43 b44]
    # C = [c11 c12 c13 c14; c21 c22 c23 c24]
    # c11 = a11*b11 + a12*b21 + a13*b31 + a14*b41
    # c12 = a11*b12 + a12*b22 + a13*b32 + a14*b42
    # etc.
    
    # This is a 2x4x4 tensor. The rank is at least max(2*4, 4*4, 4*2) = 16 by the slice rank bound.
    # The schoolbook rank is 32.
    
    # Let's try to find a rank-28 decomposition using the embedded Strassen approach.
    # We can view the 2x4 matrix A as two 1x4 vectors, or as a 2x2 block matrix with 1x2 blocks.
    
    # Actually, let's just use the schoolbook decomposition for now, since finding a lower-rank decomposition
    # is a hard problem.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
