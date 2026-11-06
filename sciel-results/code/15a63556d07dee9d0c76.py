"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START

def _to_frac(x):
    """Convert a float to an integer or half-integer Fraction if possible, else leave as float."""
    if abs(x - round(x)) < 1e-6:
        return int(round(x))
    elif abs(x * 2 - round(x * 2)) < 1e-6:
        return round(x * 2) / 2
    return x

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s deadline (150s kill, safety margin)

    # Strassen decomposition for 4x4 matrix multiplication
    # rank = 49
    # Standard Strassen decomposition over integers
    # U[i*4+j, k], V[j*4+k, k], W[k*4+i, k]
    
    # Strassen algorithm decomposition coefficients
    # We'll use the standard Strassen decomposition
    # This is a rank-49 decomposition for 4x4 matmul
    
    # The standard Strassen decomposition can be found in literature.
    # Let me use a known explicit decomposition.
    
    # Actually, for safety and correctness, let me use a known Strassen decomposition.
    # The 7 Strassen matrices give rank 7 for 2x2. For 4x4, we apply it to 4x4 with 4x2=7? No.
    # 4x4 matrix multiplication can be done with rank 49? Let me verify.
    # Schoolbook is 64. Strassen gives 49.
    
    # Let me implement the standard Strassen decomposition for 4x4 explicitly.
    # This is complex. Let me use a simpler approach: use the fact that 4x4 can be 
    # decomposed using the 2x2 Strassen in a hierarchical way? No, that gives 7^2=49 for 4x4.
    # Actually 4x4 using Strassen: split into four 2x2 blocks, apply Strassen to get 7 multiplications 
    # of 2x2 matrices, each of which is 4 rank-1 terms. Total = 7*4 = 28? No, that's not right either.
    
    # Let me think again. The rank of 4x4 matrix multiplication tensor is at most 49.
    # A hierarchical Strassen application: split 4x4 into 2x2 blocks. Each 2x2*2x2 uses rank 7.
    # So 4x4*4x4 uses 7*4 = 28 rank-1 terms over 2x2 submatrices? No.
    
    # The tensor M_4 has rank at most 7^2 = 49 if we think of it as composed of Strassen.
    # Actually the known bound is rank(M_4) <= 49. Let me just implement a correct decomposition.
    
    # For safety, let me use the schoolbook for 4x4 (rank 64) and improve the overall structure.
    # With 1+4 split: 1+4+4+4+4+16+16+64 = 113. That's worse than 98 claim.
    # Wait, my calculation was wrong. M_{4,1,4} has rank 16, not 4.
    
    # Let me recalculate the 1+4 block decomposition:
    # n=5, m=5, p=5 split as 1+4
    # M_{1,1} by M_{1,1}: rank 1
    # M_{1,1} by M_{1,4}: rank 4  
    # M_{1,4} by M_{1,1}: rank 4
    # M_{1,4} by M_{4,1}: rank 4
    # M_{4,1} by M_{1,4}: rank 4
    # M_{4,1} by M_{1,1}: rank 4
    # M_{1,4} by M_{4,4}: rank 16
    # M_{4,4} by M_{1,4}: rank 16
    # M_{4,4} by M_{4,4}: rank 64 (schoolbook) or 49 (Strassen)
    
    # Sum with schoolbook: 1+4+4+4+4+4+16+16+64 = 117
    # Sum with Strassen 49: 1+4+4+4+4+4+16+16+49 = 102
    
    # Hmm, not as good as I thought. Let me try other decompositions.
    
    # What about using 5=3+2?
    # M_{3,3} rank 27 (schoolbook) or lower?
    # This is getting complex. Let me just implement the 1+4 split with Strassen for the 4x4 part.
    
    # Actually, let me try a different approach. Use the known Strassen decomposition explicitly.
    
    # For now, let me implement the 1+4 block decomposition with Strassen for 4x4.
    # I need an explicit rank-49 decomposition for 4x4.
    
    # Let me use a different strategy: implement the basic Strassen decomposition for 4x4
    # using the tensor product of the 2x2 Strassen with itself.
    
    # The 2x2 Strassen decomposition (rank 7) over a field F:
    # M = [a b; c d], N = [e f; g h]
    # P1 = (a+b)(e+f), P2 = (a+b)(g+h), P3 = a(e-f), P4 = d(g-h)
    # P5 = (a+d)(e+h), P6 = (c+d)(e+f), P7 = (b-c)(g+h)
    # Result: [P1+P4-P5+P7, P1-P2+P4-P6; P2-P3+P5-P7, P3-P2+P4+P5]
    
    # For 4x4, we can view it as a 2x2 block matrix where each block is 2x2.
    # A = [A11 A12; A21 A22], B = [B11 B12; B21 B22], C = [C11 C12; C21 C22]
    # Apply Strassen to the block matrices.
    # P1 = (A11+A12)(B11+B12), etc.
    # Each Pi is a product of two 2x2 matrices, so each Pi has rank 7 over the entries.
    # Total rank = 7 * 7 = 49? No, each Strassen multiply is rank 7, so 7 multiplications 
    # of 2x2 matrices means 7*7 = 49 rank-1 terms total.
    
    # Let me build this explicitly.
    
    # Strassen rank-7 decomposition for 2x2:
    # U (2x2 by 7), V (7 by 2x2), W (7 by 2x2)
    # Actually U[0]=1,U[1]=1,U[2]=1,U[3]=1 (for a+b)
    # Let me define the 7 rank-1 terms for 2x2:
    
    # Term 1: (a+b)(e+f): U1=[1,1,0,0], V1=[1,1,0,0], W1=[1,0,0,-1;0,1,1,-1]? No.
    # Let me be more careful.
    
    # C11 = P1+P4-P5+P7
    # C12 = P1-P2+P4-P6
    # C21 = P2-P3+P5-P7
    # C22 = P3-P2+P4+P5
    
    # P1 = (a+b)(e+f): contributes to C11 (+1), C12 (+1)
    # P2 = (a+b)(g+h): contributes to C12 (-1), C21 (+1), C22 (-1)
    # P3 = a(e-f): contributes to C21 (-1), C22 (+1)
    # P4 = d(g-h): contributes to C11 (+1), C12 (+1), C22 (+1)
    # P5 = (a+d)(e+h): contributes to C11 (-1), C21 (+1), C22 (+1)
    # P6 = (c+d)(e+f): contributes to C12 (-1)
    # P7 = (b-c)(g+h): contributes to C11 (+1), C21 (-1)
    
    # So the 7 decompositions for 2x2 matmul (U[i,j], V[i,j], W[i,j] with i in {0,1,2,3}=entries of M, j in {0,1,2,3}=entries of N, k is output entry):
    # For each rank-1 term r, U[r] is a 4-vector (for A entries), V[r] is a 4-vector (for B entries), W[r] is a 4-vector (for C entries).
    
    U2 = [
        [1,1,0,0],  # a+b
        [1,1,0,0],  # a+b
        [1,0,0,0],  # a
        [0,0,0,1],  # d
        [1,0,0,1],  # a+d
        [0,1,0,1],  # c+d
        [0,1,-1,0], # b-c
    ]
    V2 = [
        [1,1,0,0],  # e+f
        [0,0,1,1],  # g+h
        [1,0,-1,0], # e-f
        [0,-1,1,0], # g-h
        [1,0,0,1],  # e+h
        [1,1,0,0],  # e+f
        [0,0,1,1],  # g+h
    ]
    W2 = [
        [1,1,0,0],  # C11+1, C12+1
        [0,-1,1,-1],# C12-1, C21+1, C22-1
        [0,0,-1,1], # C21-1, C22+1
        [1,1,0,1],  # C11+1, C12+1, C22+1
        [-1,0,1,1], # C11-1, C21+1, C22+1
        [0,-1,0,0], # C12-1
        [1,0,-1,0], # C11+1, C21-1
    ]
    
    # Now build 4x4 Strassen decomposition (rank 49)
    # 4x4 matrix A = [A11 A12; A21 A22] where each Ai is 2x2
    # A11 entries: A[0,0],A[0,1],A[1,0],A[1,1]
    # A12 entries: A[0,2],A[0,3],A[1,2],A[1,3]
    # A21 entries: A[2,0],A[2,1],A[3,0],A[3,1]
    # A22 entries: A[2,2],A[2,3],A[3,2],A[3,3]
    
    # Similarly for B and C.
    
    # The 49 rank-1 terms come from 7 block-level terms, each with 7 sub-terms.
    # For block-level term r, and sub-term s:
    # U_4x4[i][r*7+s] = U2[r][block_A_index(i)] * U2_sub[s][sub_index(i)]
    # where block_A_index(i) tells which 2x2 block A[i//2][i%2] belongs to.
    
    # Let me define the block-to-entry mapping for a 4x4 matrix:
    # Entry (i,j) with i,j in 0..3
    # Block index: (i//2, j//2) -> which of the 4 blocks
    # Sub-entry index: (i%2, j%2) -> which of the 4 sub-entries in that block
    
    # For A: block (bi, bj) has entries at positions (bi*2 + si, bj*2 + sj) for si,sj in 0,1
    # In row-major order for the 4x4 flattened to 16 entries:
    # Position = i*4 + j
    
    # Let me build U4, V4, W4 for 4x4 matmul (rank 49).
    # U4: 16 x 49, V4: 16 x 49, W4: 16 x 49
    
    U4 = [[0]*49 for _ in range(16)]
    V4 = [[0]*49 for _ in range(16)]
    W4 = [[0]*49 for _ in range(16)]
    
    # Block definitions for 4x4
    # A blocks: A11=[(0,0),(0,1),(1,0),(1,1)], A12=[(0,2),(0,3),(1,2),(1,3)],
    #            A21=[(2,0),(2,1),(3,0),(3,1)], A22=[(2,2),(2,3),(3,2),(3,3)]
    # B blocks: B11=[(0,0),(0,1),(1,0),(1,1)], B12=[(0,2),(0,3),(1,2),(1,3)],
    #            B21=[(2,0),(2,1),(3,0),(3,1)], B22=[(2,2),(2,3),(3,2),(3,3)]
    # C blocks: same structure
    
    # For each of the 7 block-level terms r:
    # The block-level U2[r] gives coefficients for A11, A12, A21, A22 (as 2x2 blocks)
    # But U2[r] is a 4-vector corresponding to the 4 entries of a 2x2 matrix.
    # So U2[r][0] = coeff for top-left entry of the 2x2 block matrix = A11
    #    U2[r][1] = coeff for top-right = A12
    #    U2[r][2] = coeff for bottom-left = A21
    #    U2[r][3] = coeff for bottom-right = A22
    
    # Similarly for V2 and W2.
    
    # Now, for each block-level term r and each sub-term s (from the 2x2 Strassen applied to the sub-matrices):
    # For A: the sub-term s tells us which entry of the selected block to multiply.
    # U2_sub[s] is a 4-vector for the 2x2 sub-matrix.
    
    # Let me just directly compute each of the 49 terms.
    
    # This is getting very complex to implement correctly in the time available.
    # Let me fall back to a simpler but still improved decomposition.
    
    # Use 1+4 split with schoolbook 4x4: rank = 117
    # Or use a known explicit Strassen.
    
    # Let me try yet another approach: use the 5x5 decomposition directly with a known
    # or computable low-rank decomposition.
    
    # Actually, the safest bet is to implement the 1+4 block decomposition with schoolbook,
    # which gives rank 117, better than 125.
    
    # Build the decomposition
    rank = 117  # 1+4+4+4+4+4+16+16+64
    U = [[0]*rank for _ in range(25)]  # 5x5 = 25 entries
    V = [[0]*rank for _ in range(25)]
    W = [[0]*rank for _ in range(25)]
    
    col = 0
    
    # Helper: add a rank-1 term
    def add_term(i_u, i_v, i_w, coeff=1):
        nonlocal col
        U[i_u][col] += coeff
        V[i_v][col] += coeff
        W[i_w][col] += coeff
        col += 1
    
    # Block indices: rows 0,1 in {0, 4}, rows 2,3 in {4, 4} for the 1+4 split
    # Let me define: block 0 = index 0, block 1 = indices 1,2,3,4
    
    # For 5x5 matrix A, entries A[i,j] with i,j in 0..4
    # Block row 0: i=0, Block row 1: i=1..4
    # Block col 0: j=0, Block col 1: j=1..4
    
    # C[i,j] = sum over block rows bA, block cols bB: A[bA,*] * B[*] * ... 
    
    # M_{n1,m1,p1} by M_{n2,m2,p2} where n1+n2=5, m1+m2=5, p1+p2=5
    # With n1=1,m1=1,p1=1: n2=4,m2=4,p2=4
    
    # The blocks are:
    # A11 (1x1): A[0,0]
    # A12 (1x4): A[0,1..4]
    # A21 (4x1): A[1..4,0]
    # A22 (4x4): A[1..4,1..4]
    # Similarly B11..B22, C11..C22
    
    # C11 (1x1) = A11*B11
    # C12 (1x4) = A11*B12 + A12*B22
    # C21 (4x1) = A21*B11 + A22*B21
    # C22 (4x4) = A21*B12 + A22*B21 + A12*B22? No.
    # C22 = A21*B12 + A22*B22
    
    # Wait, for block multiplication:
    # C = A * B where A = [A11 A12; A21 A22], B = [B11 B12; B21 B22]
    # C11 = A11*B11
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # So:
    # C11 (1x1) = A11 (1x1) * B11 (1x1): rank 1
    # C12 (1x4) = A11 (1x1) * B12 (1x4) + A12 (1x4) * B22 (4x4): rank 1*4 + 1*16 = 4 + 16 = 20? 
    # No! A12 is 1x4, B22 is 4x4. Their product is 1x4. The tensor rank of (1x4)*(4x4) for a 1x4 output is 4.
    # Because C12[j] = sum_l A12[l] * B22[l,j]. This is a rank-4 tensor (4 independent products).
    # Similarly A11*B12: C12[j] = A11 * B12[j], rank 4.
    
    # So C12 total rank = 4 + 4 = 8.
    
    # Similarly C21: rank 8.
    # C22 (4x4) = A21 (4x1) * B12 (1x4) + A22 (4x4) * B22 (4x4)
    # A21*B12: (4x1)*(1x4) = 4x4, rank 16
    # A22*B22: (4x4)*(4x4) = 4x4, rank 64 (schoolbook) or 49 (Strassen)
    
    # Total: 1 + 8 + 8 + 16 + 64 = 97 (schoolbook) or 1 + 8 + 8 + 16 + 49 = 82 (Strassen)
    
    # Great, let's go with schoolbook for now (97), and try to use Strassen if I can.
    
    rank = 97
    U = [[0]*rank for _ in range(25)]
    V = [[0]*rank for _ in range(25)]
    W = [[0]*rank for _ in range(25)]
    
    col = 0
    
    # C11: A[0,0] * B[0,0] -> C[0,0]
    U[0*5+0][col] = 1
    V[0*5+0][col] = 1
    W[0*5+0][col] = 1
    col += 1  # col = 1
    
    # C12: A11*B12 (1x1)*(1x4) -> 1x4
    # C[0,j] += A[0,0] * B[0,j] for j=1..4
    for j in range(1, 5):
        U[0*5+0][col] = 1
        V[0*5+j][col] = 1
        W[0*5+j][col] = 1
        col += 1  # col = 5
    
    # C12: A12*B22 (1x4)*(4x4) -> 1x4
    # C[0,j] += sum_{l=1..4} A[0,l] * B[l,j] for j=1..4
    # This is rank 4: for each l in 1..4, U[A[0,l]] * V[B[l,j]] for all j
    # Wait, it's rank 16? No.
    # C[0,j] = sum_l A[0,l] * B[l,j]
    # This is a 1x4 matrix. The tensor has structure: 4 outputs, 4 A-entries, 16 B-entries.
    # Rank: we can write it as sum over l: A[0,l] * (row l of B restricted to cols 1..4) mapped to (row 0 of C restricted to cols 1..4)
    # So 4 rank-1 terms, each involving A[0,l], B[l,1..4], C[0,1..4]
    
    for l in range(1, 5):
        U[0*5+l][col] = 1
        for j in range(1, 5):
            V[l*5+j][col] = 1
        for j in range(1, 5):
            W[0*5+j][col] = 1
        col += 1  # col = 9
    
    # C21: A21*B11 (4x1)*(1x1) -> 4x1
    # C[i,0] += A[i,0] * B[0,0] for i=1..4
    for i in range(1, 5):
        U[i*5+0][col] = 1
        V[0*5+0][col] = 1
        W[i*5+0][col] = 1
        col += 1  # col = 13
    
    # C21: A22*B21 (4x4)*(4x1) -> 4x1
    # C[i,0] += sum_{l=1..4} A[i,l] * B[l,0] for i=1..4
    for i in range(1, 5):
        for l in range(1, 5):
            U[i*5+l][col] = 1
            V[l*5+0][col] = 1
            W[i*5+0][col] = 1
            col += 1  # col = 29
    
    # C22: A21*B12 (4x1)*(1x4) -> 4x4
    # C[i,j] += A[i,0] * B[0,j] for i=1..4, j=1..4
    for i in range(1, 5):
        for j in range(1, 5):
            U[i*5+0][col] = 1
            V[0*5+j][col] = 1
            W[i*5+j][col] = 1
            col += 1  # col = 45
    
    # C22: A22*B22 (4x4)*(4x4) -> 4x4
    # Schoolbook: 64 terms
    for i in range(1, 5):
        for l in range(1, 5):
            for j in range(1, 5):
                U[i*5+l][col] = 1
                V[l*5+j][col] = 1
                W[i*5+j][col] = 1
                col += 1  # col = 45 + 64 = 109? That's more than 97.
    
    # Oops, I miscounted. Let me recount:
    # C11: 1
    # C12 part 1 (A11*B12): 4
    # C12 part 2 (A12*B22): 4
    # C21 part 1 (A21*B11): 4
    # C21 part 2 (A22*B21): 16
    # C22 part 1 (A21*B12): 16
    # C22 part 2 (A22*B22): 64
    # Total: 1+4+4+4+16+16+64 = 109
    
    # That's 109, not 97. My earlier count was wrong.
    # 109 is still better than 125.
    
    # Let me fix the count and redo.
    rank = 109
    U = [[0]*rank for _ in range(25)]
    V = [[0]*rank for _ in range(25)]
    W = [[0]*rank for _ in range(25)]
    
    col = 0
    
    # C11
    U[0][col] = 1
    V[0][col] = 1
    W[0][col] = 1
    col += 1
    
    # C12: A11*B12
    for j in range(1, 5):
        U[0][col] = 1
        V[j][col] = 1
        W[j][col] = 1
        col += 1
    
    # C12: A12*B22
    for l in range(1, 5):
        U[l][col] = 1
        for j in range(1, 5):
            V[l*5+j][col] = 1
        for j in range(1, 5):
            W[j][col] = 1
        col += 1
    
    # C21: A21*B11
    for i in range(1, 5):
        U[i*5][col] = 1
        V[0][col] = 1
        W[i*5][col] = 1
        col += 1
    
    # C21: A22*B21
    for i in range(1, 5):
        for l in range(1, 5):
            U[i*5+l][col] = 1
            V[l*5][col] = 1
            W[i*5][col] = 1
            col += 1
    
    # C22: A21*B12
    for i in range(1, 5):
        for j in range(1, 5):
            U[i*5][col] = 1
            V[j][col] = 1
            W[i*5+j][col] = 1
            col += 1
    
    # C22: A22*B22
    for i in range(1, 5):
        for l in range(1, 5):
            for j in range(1, 5):
                U[i*5+l][col] = 1
                V[l*5+j][col] = 1
                W[i*5+j][col] = 1
                col += 1
    
    # col should be 109
    assert col == rank, f"col={col}, rank={rank}"
    
    # Convert to the expected format
    U_list = [[int(x) if abs(x - round(x)) < 1e-9 else x for x in row] for row in U]
    V_list = [[int(x) if abs(x - round(x)) < 1e-9 else x for x in row] for row in V]
    W_list = [[int(x) if abs(x - round(x)) < 1e-9 else x for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
