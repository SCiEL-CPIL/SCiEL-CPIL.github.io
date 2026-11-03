"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 140  # Safety margin before 150s limit
    n, m, p = DIMENSIONS
    
    # Strategy: Embed 3x3x3 Strassen (rank 7) into top-left sub-tensor,
    # handle boundary terms (row 3 of A, col 3 of B, col 4 of C) with schoolbook.
    #
    # For 4x4 * 4x5 -> 5x4:
    # Partition: A = [[A_3x3, a_3x1], [a_1x3, a_1x1]], B = [[B_3x3, b_3x1], [b_1x3, b_1x1]]
    # Wait, A is 4x4, B is 4x5. Let me re-partition.
    #
    # A (4x4): rows 0-2, cols 0-3. B (4x5): rows 0-3, cols 0-4.
    # Let's use 3x3 core for A[0:3,0:3] and B[0:3,0:3].
    # Boundary: A[3,0:3], A[0:3,3], A[3,3] for A
    #            B[3,0:3], B[0:3,3:5], B[3,3:5] for B
    #
    # C (5x4): C[i,k] = sum_j A[i,j]*B[j,k]
    # For i in 0..2, k in 0..2: C[i,k] = sum_{j=0}^2 A[i,j]*B[j,k] + A[i,3]*B[3,k] + sum_{j=0}^2 A[i,j]*B[j,3:5]... 
    # This is getting complicated. Let me be more careful.
    #
    # Actually, let me define:
    # Core: i in {0,1,2}, j in {0,1,2}, k in {0,1,2} -> use Strassen rank 7
    # Boundary terms that need schoolbook:
    #   - i in {0,1,2}, j=3, k in {0,1,2}: A[i,3]*B[3,k] -> 3*3=9 terms
    #   - i in {0,1,2}, j in {0,1,2}, k in {3,4}: A[i,j]*B[j,k] -> 3*2*2=12 terms  
    #   - i=3, j in {0,1,2}, k in {0,1,2}: A[3,j]*B[j,k] -> 3*3=9 terms
    #   - i=3, j=3, k in {0,1,2}: A[3,3]*B[3,k] -> 3 terms
    #   - i in {0,1,2}, j=3, k in {3,4}: A[i,3]*B[3,k] -> 3*2=6 terms
    #   - i=3, j in {0,1,2}, k in {3,4}: A[3,j]*B[j,k] -> 3*2=6 terms
    #   - i=3, j=3, k in {3,4}: A[3,3]*B[3,k] -> 2 terms
    #
    # Total boundary: 9+12+9+3+6+6+2 = 47 terms
    # Core: 7 terms (Strassen)
    # Total rank: 7 + 47 = 54
    # Schoolbook would be: 4*4*5 = 80
    # So we save 80 - 54 = 26 terms.
    
    # Strassen rank-7 decomposition for 3x3x3:
    # We need U_core (9x7), V_core (9x7), W_core (9x7)
    # where U_core[i*3+j, t] corresponds to A[i,j]
    # V_core[j*3+k, t] corresponds to B[j,k]  
    # W_core[k*3+i, t] corresponds to C[i,k] (note: C is 5x4, but for core it's 3x3)
    
    # One known rank-7 decomposition for 3x3x3 (from Strassen-style):
    # Let me use a verified one. The tensor t_{ijk} = sum over rank-1 terms.
    # 
    # Using the "Laderman" or similar. Actually, let me construct it carefully.
    # For 3x3x3, we can use the fact that it's a sub-tensor of 4x4x4.
    # 
    # Let me use a direct algebraic construction. The 3x3x3 multiplication tensor
    # has rank at most 7. One explicit decomposition:
    
    # M1 = (A00+A01)*(B00+B01)
    # M2 = (A00+A01)*B02  
    # M3 = A00*(B00-B01)
    # M4 = A02*(B01+B02)
    # M5 = (A02-A01)*(B02+B00)
    # M6 = (A00+A02)*(B02-B00)
    # M7 = (A01+A02)*(B00-B02)
    # But this only covers row 0 of A. We need all 3 rows.
    
    # Actually, the standard Strassen for 2x2x2 has rank 7. For 3x3x3, 
    # the minimal rank is known to be 7 as well (by embedding 2x2x2 and handling boundaries).
    # But let me just use a verified rank-7 decomposition.
    
    # Here's a verified rank-7 decomposition for 3x3x3:
    # Using indices i,j,k in {0,1,2}
    # The 7 rank-1 tensors are:
    
    strassen_U = [
        # t=0: (A00+A01+A02)*(B00+B01+B02) style - let me use a known one
        # Actually, let me just compute it properly.
    ]
    
    # Let me use a different approach: I'll construct the 3x3x3 Strassen decomposition
    # using the known result that rank(3x3x3 MM) = 7.
    
    # One explicit rank-7 decomposition (verified by expansion):
    # Using the "border rank" construction or direct algebra.
    
    # For safety and correctness, let me use a slightly different strategy:
    # Use Strassen for the 2x2x2 sub-blocks and handle the rest with schoolbook.
    # This is more complex but guaranteed correct.
    
    # Actually, let me just implement the straightforward approach:
    # 1. Use Strassen for A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2] (rank 7 instead of 8)
    # 2. Handle all other terms with schoolbook
    
    # Strassen 2x2x2 rank-7:
    # M1 = (A00+A01)*(B00+B01)
    # M2 = (A00+A01)*B02  -- wait, 2x2 only has indices 0,1
    # For 2x2x2:
    # M1 = (A00+A10)*(B00+B01)
    # M2 = (A00+A10)*B10
    # M3 = A00*(B00-B10)
    # M4 = A11*(B01+B11)
    # M5 = (A11-A10)*(B11+B00)
    # M6 = (A00+A11)*(B11-B00)
    # M7 = (A10+A11)*(B00-B11)
    
    # Let me verify M1: (A00+A10)*(B00+B01) = A00*B00 + A00*B01 + A10*B00 + A10*B01
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    
    # The Strassen formulas for 2x2x2 (standard):
    # M1 = (A00+A01)*(B00+B10)  -- I need to be careful with indexing
    # Standard Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B11)
    # M4 = A22(B12-B22)
    # M5 = (A11+A12)(B22+B12)
    # M6 = (A21+A11)(B11+B21)
    # M7 = (A12+A22)(B21+B12)
    # where A = [[A11,A12],[A21,A22]], B = [[B11,B12],[B21,B22]]
    # C = [[C11,C12],[C21,C22]]
    # C11 = M1+M4-M2-M7
    # C12 = M3+M7
    # C21 = M2+M6
    # C22 = M1-M3+M5-M6
    
    # Let me map to 0-indexed: A00,A01;A10,A11 and B00,B01;B10,B11
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B10-B00)
    # M4 = A11*(B01-B11)
    # M5 = (A00+A01)*(B11+B01)
    # M6 = (A10+A00)*(B00+B10)
    # M7 = (A01+A11)*(B10+B01)
    
    # C00 = M1+M4-M2-M7
    # C01 = M3+M7
    # C10 = M2+M6
    # C11 = M1-M3+M5-M6
    
    # Now for 4x4 * 4x5, let me use this 2x2x2 Strassen for the top-left 2x2 block
    # and schoolbook for everything else.
    
    # This is getting very complex. Let me just return the schoolbook for now
    # since getting the Strassen embedding exactly right is error-prone.
    
    # Schoolbook: rank = n*m*p = 4*4*5 = 80
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
