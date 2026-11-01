"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # safety margin before 150s kill
    
    # Strassen-based rank 14 decomposition for 2x2 * 2x4 matrix multiplication
    # U has shape (n*m, rank) = (4, 14)
    # V has shape (m*p, rank) = (8, 14)  
    # W has shape (p*n, rank) = (8, 14)
    # W[k*n+i, r] follows the trace convention
    
    n, m, p = 2, 2, 4
    rank = 14
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Strassen's algorithm applied to compute A (2x2) * B (2x4)
    # We split B into two 2x2 blocks: B_left = B[:, 0:2], B_right = B[:, 2:4]
    # Compute C_left = A * B_left and C_right = A * B_right using Strassen
    # Then combine: C[:, 0:2] = C_left, C[:, 2:4] = C_right
    
    # For each 2x2 block multiplication, Strassen uses 7 products:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # Results:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For B_left (columns 0,1):
    # B_left[0,0] = B[0,0], B_left[0,1] = B[0,1], B_left[1,0] = B[1,0], B_left[1,1] = B[1,1]
    # For B_right (columns 2,3):
    # B_right[0,0] = B[0,2], B_right[0,1] = B[0,3], B_right[1,0] = B[1,2], B_right[1,1] = B[1,3]
    
    # Total: 7 products for left + 7 products for right = 14 rank-1 terms
    
    # Define the 14 rank-1 factors
    # Each term c has U[:,c], V[:,c], W[:,c]
    
    # Left block (columns 0,1) - 7 terms
    # M1_left = (A00+A11)*(B[0,0]+B[1,1])
    c = 0
    U[0][c] = 1.0; U[3][c] = 1.0  # A00, A11
    V[0][c] = 1.0; V[5][c] = 1.0  # B[0,0], B[1,1]
    # Result contributes to C00 and C11 of left block
    # C00_left = A00*B[0,0] + A01*B[1,0] -> index (0,0) in C, which is k*n+i = 0*2+0 = 0
    # C11_left = A10*B[0,1] + A11*B[1,1] -> index (1,1) in C, which is k*n+i = 1*2+1 = 3
    # From M1: contributes +1 to both C00 and C11
    W[0][c] = 1.0  # C[0,0]
    W[3][c] = 1.0  # C[1,1]
    
    # M2_left = (A10+A11)*B[0,0]
    c = 1
    U[2][c] = 1.0; U[3][c] = 1.0  # A10, A11
    V[0][c] = 1.0  # B[0,0]
    # Contributes to C10_left (C[1,0]) and subtracts from C11_left (C[1,1])
    # C10_left = A10*B[0,0] + A11*B[1,0]
    # C11_left = A10*B[0,1] + A11*B[1,1]
    # From M2: +1 to C10, -1 to C11
    W[2][c] = 1.0   # C[1,0]
    W[3][c] = -1.0  # C[1,1]
    
    # M3_left = A00*(B[0,1]-B[1,1])
    c = 2
    U[0][c] = 1.0  # A00
    V[1][c] = 1.0; V[5][c] = -1.0  # B[0,1], -B[1,1]
    # Contributes to C01_left (C[0,1]) and C11_left (C[1,1])
    # C01_left = A00*B[0,1] + A01*B[1,1]
    # C11_left = A10*B[0,1] + A11*B[1,1]
    # From M3: +1 to C01, +1 to C11
    W[1][c] = 1.0   # C[0,1]
    W[3][c] = 1.0   # C[1,1]
    
    # M4_left = A11*(B[1,0]-B[0,0])
    c = 3
    U[3][c] = 1.0  # A11
    V[3][c] = 1.0; V[0][c] = -1.0  # B[1,0], -B[0,0]
    # Contributes to C00_left (C[0,0]) and C10_left (C[1,0])
    # C00_left = A00*B[0,0] + A01*B[1,0]
    # C10_left = A10*B[0,0] + A11*B[1,0]
    # From M4: +1 to C00, +1 to C10
    W[0][c] = 1.0   # C[0,0]
    W[2][c] = 1.0   # C[1,0]
    
    # M5_left = (A00+A01)*B[1,1]
    c = 4
    U[0][c] = 1.0; U[1][c] = 1.0  # A00, A01
    V[5][c] = 1.0  # B[1,1]
    # Subtracts from C00_left (C[0,0]), adds to C01_left (C[0,1])
    # From M5: -1 to C00, +1 to C01
    W[0][c] = -1.0  # C[0,0]
    W[1][c] = 1.0   # C[0,1]
    
    # M6_left = (A10-A00)*(B[0,0]+B[0,1])
    c = 5
    U[2][c] = 1.0; U[0][c] = -1.0  # A10, -A00
    V[0][c] = 1.0; V[1][c] = 1.0   # B[0,0], B[0,1]
    # Adds to C11_left (C[1,1])
    # From M6: +1 to C11
    W[3][c] = 1.0   # C[1,1]
    
    # M7_left = (A01-A11)*(B[1,0]+B[1,1])
    c = 6
    U[1][c] = 1.0; U[3][c] = -1.0  # A01, -A11
    V[3][c] = 1.0; V[5][c] = 1.0   # B[1,0], B[1,1]
    # Adds to C00_left (C[0,0])
    # From M7: +1 to C00
    W[0][c] = 1.0   # C[0,0]
    
    # Right block (columns 2,3) - 7 terms
    # M1_right = (A00+A11)*(B[0,2]+B[1,3])
    c = 7
    U[0][c] = 1.0; U[3][c] = 1.0  # A00, A11
    V[2][c] = 1.0; V[7][c] = 1.0  # B[0,2], B[1,3]
    # Result contributes to C[0,2] and C[1,3]
    # C[0,2] = A00*B[0,2] + A01*B[1,2]
    # C[1,3] = A10*B[0,3] + A11*B[1,3]
    # From M1_right: +1 to C[0,2], +1 to C[1,3]
    W[2][c] = 1.0   # C[0,2]
    W[7][c] = 1.0   # C[1,3]
    
    # M2_right = (A10+A11)*B[0,2]
    c = 8
    U[2][c] = 1.0; U[3][c] = 1.0  # A10, A11
    V[2][c] = 1.0  # B[0,2]
    # Contributes to C[1,2] and subtracts from C[1,3]
    # From M2_right: +1 to C[1,2], -1 to C[1,3]
    W[4][c] = 1.0   # C[1,2]
    W[7][c] = -1.0  # C[1,3]
    
    # M3_right = A00*(B[0,3]-B[1,3])
    c = 9
    U[0][c] = 1.0  # A00
    V[3][c] = 1.0; V[7][c] = -1.0  # B[0,3], -B[1,3]
    # Contributes to C[0,3] and C[1,3]
    # From M3_right: +1 to C[0,3], +1 to C[1,3]
    W[3][c] = 1.0   # C[0,3]
    W[7][c] = 1.0   # C[1,3]
    
    # M4_right = A11*(B[1,2]-B[0,2])
    c = 10
    U[3][c] = 1.0  # A11
    V[5][c] = 1.0; V[2][c] = -1.0  # B[1,2], -B[0,2]
    # Contributes to C[0,2] and C[1,2]
    # From M4_right: +1 to C[0,2], +1 to C[1,2]
    W[2][c] = 1.0   # C[0,2]
    W[4][c] = 1.0   # C[1,2]
    
    # M5_right = (A00+A01)*B[1,3]
    c = 11
    U[0][c] = 1.0; U[1][c] = 1.0  # A00, A01
    V[7][c] = 1.0  # B[1,3]
    # Subtracts from C[0,2], adds to C[0,3]
    # From M5_right: -1 to C[0,2], +1 to C[0,3]
    W[2][c] = -1.0  # C[0,2]
    W[3][c] = 1.0   # C[0,3]
    
    # M6_right = (A10-A00)*(B[0,2]+B[0,3])
    c = 12
    U[2][c] = 1.0; U[0][c] = -1.0  # A10, -A00
    V[2][c] = 1.0; V[3][c] = 1.0   # B[0,2], B[0,3]
    # Adds to C[1,3]
    # From M6_right: +1 to C[1,3]
    W[7][c] = 1.0   # C[1,3]
    
    # M7_right = (A01-A11)*(B[1,2]+B[1,3])
    c = 13
    U[1][c] = 1.0; U[3][c] = -1.0  # A01, -A11
    V[5][c] = 1.0; V[7][c] = 1.0   # B[1,2], B[1,3]
    # Adds to C[0,2]
    # From M7_right: +1 to C[0,2]
    W[2][c] = 1.0   # C[0,2]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

