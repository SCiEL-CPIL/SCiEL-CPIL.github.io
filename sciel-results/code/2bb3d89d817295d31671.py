"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import itertools
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Schoolbook as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_school
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    # Try to find better decompositions
    # Strategy: Use block structure with Strassen 2x2x2 (rank 7)
    
    # For 4x4x6, split into blocks of size 2x2
    # A: 4x4 -> 4 blocks of 2x2
    # B: 4x6 -> 6 blocks of 2x2
    # C: 4x6 -> 6 blocks of 2x2 (output)
    
    # C_block[a,b] = sum over shared block index of A_block[a,c] * B_block[c,b]
    # where a in {0,1} (row block of A/C), c in {0,1} (col block of A / row block of B), b in {0,1,2} (col block of B/C)
    
    # So we have 2*1*3 = 6 output blocks, each = sum of 2 products of 2x2 matrices
    # Total 12 scalar 2x2x2 multiplications
    
    # Use Strassen for each 2x2x2: rank 7
    # But share factors where possible
    
    # Strassen 2x2x2 decomposition:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # For our problem, we have 12 2x2x2 products. Let's index them:
    # Product (a, c, b) where a in {0,1}, c in {0,1}, b in {0,1,2}
    # A_block(a,c), B_block(c,b), C_block(a,b) += A_block(a,c) * B_block(c,b)
    
    # For each of the 12 products, apply Strassen.
    # Total rank = 12 * 7 = 84. Worse than schoolbook 96? No, 84 < 96.
    
    # Let's construct this.
    # Each 2x2x2 Strassen product contributes 7 rank-1 terms.
    # For product (a,c,b):
    #   A-side: A_block(a,c) = A[2a:2a+2, 2c:2c+2]
    #   B-side: B_block(c,b) = B[2c:2c+2, 2b:2b+2]
    #   C-side: C_block(a,b) = C[2a:2a+2, 2b:2b+2]
    
    # Strassen factors for product (a,c,b):
    # M1: A_side = A11+A22 of A_block, B_side = B11+B22 of B_block
    # etc.
    
    # Let's build U, V, W matrices.
    # U has shape (n*m, rank) = (16, rank)
    # V has shape (m*p, rank) = (24, rank)
    # W has shape (p*n, rank) = (24, rank)
    
    # For each Strassen term in each 2x2x2 product, we add entries to U, V, W.
    
    # Strassen coefficients:
    # For a 2x2x2 product with A blocks (A00,A01,A10,A11) and B blocks (B00,B01,B10,B11):
    # M1: A_coef = {A00:1, A11:1}, B_coef = {B00:1, B11:1}, C_coef = {C00:1, C11:-1}... 
    # Wait, let me be more careful.
    
    # Standard Strassen:
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B10-B11)
    # P4 = A11(B01-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # For each P_i, the A-side is a linear combination of A's 4 entries, B-side is a linear combo of B's 4 entries, and C-side gets contributions to C's 4 entries.
    
    # Let me define the Strassen terms as (A_coef_dict, B_coef_dict, C_coef_dict)
    # where A_coef_dict maps (row,col) in 2x2 A to coefficient, etc.
    
    strassen_terms = [
        # (A_coefs, B_coefs, C_coefs)
        # P1
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # P2
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1}),
        # P3
        ({(0,0):1}, {(1,0):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        # P4
        ({(1,1):1}, {(0,1):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        # P5
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        # P6
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # P7
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1, (1,1):-1}),
    ]
    
    # Now build the full decomposition.
    # For each (a, c, b) in 2x2x2 product space (a:0-1, c:0-1, b:0-2):
    #   A_block = A[2a:2a+2, 2c:2c+2]
    #   B_block = B[2c:2c+2, 2b:2b+2]
    #   C_block = C[2a:2a+2, 2b:2b+2]
    #   Apply 7 Strassen terms
    
    rank = 12 * 7  # 84
    U = [[0] * rank for _ in range(n * m)]  # 16 x 84
    V = [[0] * rank for _ in range(m * p)]  # 24 x 84
    W = [[0] * rank for _ in range(p * n)]  # 24 x 84
    
    col = 0
    for a in range(2):
        for c in range(2):
            for b in range(3):
                for (a_coefs, b_coefs, c_coefs) in strassen_terms:
                    # A-side: for each (ar, ac) in a_coefs, U[2a+ar, 2c+ac] += coef
                    for (ar, ac), coef in a_coefs.items():
                        U[2*a + ar][2*c + ac] = coef  # Wait, U index is i*m+j = i*4+j
                        # i = 2a+ar, j = 2c+ac
                        # U[i*m + j] = U[i*4 + j]
                    
                    # Actually let me redo: U[i*m+j, col] where i is row of A, j is col of A
                    # A is n x m = 4 x 4, so i in 0-3, j in 0-3
                    # A_block(a,c) = A[2a:2a+2, 2c:2c+2], so global (i,j) = (2a+ar, 2c+ac)
                    
                    # Reset and redo properly
                    for (ar, ac), coef in a_coefs.items():
                        gi = 2*a + ar  # global row in A
                        gj = 2*c + ac  # global col in A
                        U[gi * m + gj][col] += coef
                    
                    for (br, bc), coef in b_coefs.items():
                        # B is m x p = 4 x 6, B_block(c,b) = B[2c:2c+2, 2b:2b+2]
                        gi = 2*c + br  # global row in B
                        gj = 2*b + bc  # global col in B
                        V[gi * p + gj][col] += coef
                    
                    for (cr, cc), coef in c_coefs.items():
                        # C is p x n = 6 x 4, C_block(a,b) = C[2a:2a+2, 2b:2b+2]
                        # W index: k*n+i where k is row of C (=col of B), i is col of C (=row of A)
                        # C[k, i] where k = 2b+cc (col of B = row of C), i = 2a+cr (row of A = col of C)
                        # W[k*n + i] = W[(2b+cc)*4 + (2a+cr)]
                        gi = 2*b + cc  # k
                        gj = 2*a + cr  # i
                        W[gi * n + gj][col] += coef
                    
                    col += 1
    
    # Verify: rank should be 84
    assert col == rank, f"col={col}, rank={rank}"
    
    # Check if this is valid
    # Verify by checking the tensor equation
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For matrix mult: T[i,j,k] = 1 if the (i,j) of A and (j,k) of B produce (k,i) of C
    # i.e., T[i,j,k] = 1 for all valid (i,j,k)
    
    # Let's verify a few entries
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # T[i,j,k] should be 1
                total = 0
                for r in range(rank):
                    total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if total != 1:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if valid:
        best_rank = rank
        best_U = U
        best_V = V
        best_W = W
    else:
        # Fall back to schoolbook
        pass
    
    # Check deadline
    if time.time() > deadline:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
