"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    import time
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 3, 5, 7
    
    # Split contraction dim m=5 into m1=3, m2=2
    # Split output dim p=7 into p1=3, p2=4
    m1, m2 = 3, 2
    p1, p2 = 3, 4
    
    # Winograd 3x3x3 fast matrix multiplication (rank 23)
    # For computing C[i,k] = sum_j A[i,j] * B[j,k] for i,k in [0,2], j in [0,2]
    # Winograd algorithm for 3x3:
    # m = 2*3 - 1 = 5? No, for 3x3 it's different.
    # Actually Winograd for 3x3 uses rank 23.
    # Let me use the known Winograd 3x3 decomposition.
    
    # Winograd 3x3x3 tensor decomposition (23 terms)
    # Reference: Winograd's algorithm for 3x3 matrix multiplication
    # The 23 bilinear forms for 3x3 matrix mult
    
    # Standard Winograd 3x3 coefficients:
    # Input transforms for A (3x3 -> 5x3? No, for 3x3 it's 3x3 -> 3x3 with specific coeffs)
    # Actually for 3x3 Winograd, we use:
    # m = 2*3-1 = 5 intermediate values per row? No.
    # Let me use the explicit 23-term decomposition.
    
    # Winograd 3x3: rank 23
    # The decomposition uses the following structure:
    # We compute (A * B)[i,k] = sum_j A[i,j]*B[j,k]
    # Using 23 bilinear terms.
    
    # Here's a known Winograd 3x3 decomposition:
    # It's based on the tensor decomposition of 3x3 matrix mult with 23 terms.
    
    # Let me construct it explicitly. The Winograd 3x3 algorithm:
    # For each of the 23 terms t=0..22:
    #   U vector: 3x3=9 entries (for A)
    #   V vector: 3x3=9 entries (for B)  
    #   W vector: 3x3=9 entries (for C)
    #   C += U_t * V_t * W_t (outer product style)
    
    # Known Winograd 3x3 coefficients (from literature):
    # This is the standard 23-term decomposition
    
    # Let me use a verified Winograd 3x3 decomposition:
    # From Winograd (1979) "Arithmetic Complexity of Computer Systems"
    
    # The 23 bilinear forms for 3x3 matrix multiplication:
    winograd_3x3 = [
        # (U_coeffs for A[3x3], V_coeffs for B[3x3], W_coeffs for C[3x3])
        # Each is a tuple of 9 floats (row-major)
    ]
    
    # Let me build the Winograd 3x3 decomposition properly.
    # Using the standard construction:
    # For 3x3, we use the tensor rank 23 decomposition.
    
    # Actually, let me use a simpler but correct approach:
    # Use Strassen-style for the 3x3 block, or just verify with explicit terms.
    
    # Let me use the known result: 3x3 matrix mult has tensor rank 23.
    # Here's one explicit decomposition (I'll construct it carefully):
    
    # Winograd 3x3 algorithm coefficients:
    # The algorithm computes C = A*B where A, B, C are 3x3.
    # It uses 23 multiplications.
    
    # Standard Winograd 3x3:
    # Let's define it via the input/output transforms.
    
    # For Winograd 3x3:
    # A (3x3) is transformed to M (5x3) via some matrix
    # B (3x3) is transformed to N (5x3) via some matrix  
    # Then 15 intermediate products... no that's not right for rank 23.
    
    # Let me just use the explicit 23-term list from known references.
    # I'll construct the Winograd 3x3 decomposition term by term.
    
    # Known Winograd 3x3 decomposition (23 terms):
    # Each term is (u, v, w) where u,v,w are in R^9 (flattened 3x3 matrices)
    # C = sum_t u_t * v_t * w_t (in the sense of tensor decomposition)
    
    # From the literature, here is one valid decomposition:
    # I'll use the construction from "Fast Matrix Multiplication" by Winograd
    
    # Let me use a different approach: build the decomposition algorithmically.
    # For 3x3, we can use the fact that it's related to the 2x2 Strassen in a structured way.
    
    # Actually, let me just use schoolbook for now and focus on getting the block structure right,
    # then optimize the 3x3 block.
    
    # For safety and correctness, let me use a verified approach:
    # Block structure:
    # A = [A11 A12] where A11 is 3x3, A12 is 3x2
    # B = [B11; B21] where B11 is 3x7, B21 is 2x7
    # C = A11*B11 + A12*B21
    
    # For A11*B11: 3x3 * 3x7 = 3x7
    #   Split B11 = [B11a B11b] where B11a is 3x3, B11b is 3x4
    #   A11*B11a: 3x3 * 3x3 -> use Winograd rank 23
    #   A11*B11b: 3x3 * 3x4 -> schoolbook rank 3*3*4 = 36
    # For A12*B21: 3x2 * 2x7 -> schoolbook rank 3*2*7 = 42
    
    # Total: 23 + 36 + 42 = 101
    # Hmm, that's worse than 99. Let me reconsider.
    
    # Alternative: 
    # C = A*B, A is 3x5, B is 5x7
    # Split A = [A1 A2] where A1 is 3x3, A2 is 3x2
    # Split B = [B1; B2] where B1 is 3x7, B2 is 2x7
    # C = A1*B1 + A2*B2
    
    # A1*B1: 3x3 * 3x7
    #   Split B1 = [B1a B1b] where B1a is 3x3, B1b is 3x4
    #   A1*B1a: 3x3*3x3, Winograd rank 23
    #   A1*B1b: 3x3*3x4, schoolbook rank 36
    # A2*B2: 3x2*2x7, schoolbook rank 42
    # Total: 23+36+42 = 101. Worse.
    
    # Let me try: use Winograd for the full 3x3 part differently.
    # What if I split B into 3+2 in the output dimension too?
    # B is 5x7. Split columns into 3+4: B = [B_left(5x3) B_right(5x4)]
    # A is 3x5. Split rows of B (contraction) into 3+2: 
    # B_left = [B_left_top(3x3); B_left_bot(2x3)]
    # B_right = [B_right_top(3x4); B_right_bot(2x4)]
    
    # C_left(3x3) = A_top(3x3)*B_left_top(3x3) + A_bot(3x2)*B_left_bot(2x3)
    # C_right(3x4) = A_top(3x3)*B_right_top(3x4) + A_bot(3x2)*B_right_bot(2x4)
    
    # For C_left:
    #   A_top*B_left_top: 3x3*3x3, Winograd 23
    #   A_bot*B_left_bot: 3x2*2x3, schoolbook 18
    # For C_right:
    #   A_top*B_right_top: 3x3*3x4, schoolbook 36
    #   A_bot*B_right_bot: 3x2*2x4, schoolbook 24
    # Total: 23+18+36+24 = 101. Still worse.
    
    # The problem is that Winograd 23 saves 6 vs schoolbook 27 for 3x3*3x3,
    # but the cross terms eat into that.
    
    # Let me reconsider. Schoolbook total: 3*5*7 = 105
    # Current best with 6 Strassen blocks: 99 (saves 6)
    
    # With Winograd: save 6 on one 3x3*3x3 block.
    # But we need to check if the block structure allows isolating a 3x3*3x3.
    
    # A is 3x5, B is 5x7.
    # If I take the first 3 columns of A and first 3 rows of B, that's A[:,0:3] * B[0:3,:]
    # This is 3x3 * 3x3 = 3x3 contribution to C.
    # The remaining terms:
    # A[:,3:5] * B[0:3,:] : 3x2 * 3x3 = 3x3, schoolbook 18
    # A[:,0:3] * B[3:5,:] : 3x3 * 2x7 = 3x7, schoolbook 42  
    # A[:,3:5] * B[3:5,:] : 3x2 * 2x4 = 3x4, schoolbook 24
    
    # Wait, B is 5x7. B[0:3,:] is 3x7, B[3:5,:] is 2x7.
    # A[:,0:3] is 3x3, A[:,3:5] is 3x2.
    
    # C = A[:,0:3]*B[0:3,:] + A[:,3:5]*B[3:5,:]
    #    = (3x3)(3x7) + (3x2)(2x7)
    
    # For the first term: A[:,0:3]*B[0:3,:]
    #   Split B[0:3,:] into B[0:3,0:3] (3x3) and B[0:3,3:7] (3x4)
    #   A[:,0:3]*B[0:3,0:3]: 3x3*3x3, Winograd 23
    #   A[:,0:3]*B[0:3,3:7]: 3x3*3x4, schoolbook 36
    # For the second term: A[:,3:5]*B[3:5,:]
    #   3x2*2x7, schoolbook 42
    
    # Total: 23 + 36 + 42 = 101. Still 101.
    
    # The issue: Winograd only saves 6 (27->23), but we're still doing 36+42=78 schoolbook
    # vs 105-27=78 schoolbook in the pure schoolbook case. So net save is 6, giving 99.
    # Same as Strassen!
    
    # So Winograd 3x3 gives same rank as 6 Strassen blocks: 99.
    # No improvement. Let me stick with the Strassen approach which already scores 1.06.
    
    # Let me implement the 6 Strassen blocks approach properly.
    # Split A (3x5) and B (5x7) into blocks:
    # A = [A11 A12 A13] where A11 is 3x2, A12 is 3x2, A13 is 3x1
    # B = [B11 B12 B13 B14 B15 B16 B17] ... no, B is 5x7.
    
    # Better: partition the 5-dim contraction into 2+2+1 and 7-dim output into 2+2+2+1
    # A: 3x(2+2+1) -> A1(3x2), A2(3x2), A3(3x1)
    # B: (2+2+1)x7 -> B1(2x7), B2(2x7), B3(1x7)
    
    # C = A1*B1 + A2*B2 + A3*B3
    # For A1*B1: 3x2*2x7. Split into 2x2 blocks where possible.
    #   A1 = [A11 A12] (3x2), B1 = [B11 B12 B13 B14 B15 B16 B17] no...
    #   B1 is 2x7. Split columns into 2+2+2+1: B1 = [B1a(2x2) B1b(2x2) B1c(2x2) B1d(2x1)]
    #   A1*B1a: 3x2*2x2. Use Strassen? 3x2*2x2 can't directly use 2x2 Strassen.
    
    # This is getting complicated. Let me use a different strategy.
    
    # Use Strassen on 2x2x2 sub-blocks where both dimensions allow it.
    # A is 3x5, B is 5x7.
    # For a 2x2 Strassen block, we need 2x2 submatrices of A and B.
    # A has 3 rows, B has 5 rows (contraction) and 7 cols.
    
    # Place Strassen blocks at positions where we can get 2x2 A-block and 2x2 B-block:
    # A rows: 0-1, 1-2 (two possible 2-row windows)
    # B contraction rows: 0-1, 1-2, 2-3, 3-4 (four possible 2-row windows)
    # B output cols: 0-1, 1-2, 2-3, 3-4, 4-5, 5-6 (six possible 2-col windows)
    
    # Each Strassen block covers a 2x2 A-block, a 2x2 B-block (contraction x output),
    # and contributes to a 2x2 C-block.
    
    # A Strassen block uses 7 terms instead of 8, saving 1.
    # We need to place non-overlapping Strassen blocks.
    
    # A rows: can use rows (0,1) or (1,2) or (0,1,2) but Strassen needs exactly 2.
    # Let's use A rows (0,1) for some blocks and (1,2) for others, but they overlap at row 1.
    # For non-overlapping, we can only use one 2-row window from A's 3 rows.
    
    # Hmm, this is the constraint. With 3 rows in A, we can only fit one 2-row Strassen block
    # in the A dimension without overlap. But we can have multiple Strassen blocks using the same
    # A rows but different B columns.
    
    # So: fix A rows (0,1). For B, use contraction rows (0,1) and various output col pairs.
    # A(0:2, j1:j2) * B(j1:j2, k1:k2) for various j,k.
    
    # Let me define:
    # A_block = A[0:2, :] (2x5)
    # For each pair of contraction indices (j, j+1) and output cols (k, k+1):
    #   If we use A[0:2, j:j+2] and B[j:j+2, k:k+2], that's a 2x2*2x2 Strassen block.
    
    # But A[0:2, j:j+2] changes with j. So for j=0: A[0:2,0:2], j=1: A[0:2,1:3], etc.
    # These are different A submatrices, so they're independent terms.
    
    # So we can have Strassen blocks for:
    # (A rows 0-1, B contraction 0-1, B output 0-1)
    # (A rows 0-1, B contraction 0-1, B output 1-2)
    # ... etc.
    
    # But wait, each Strassen block uses specific entries of A and B. The tensor decomposition
    # is global, so we need to make sure the terms don't conflict.
    
    # Actually, in tensor decomposition, each term is a rank-1 tensor. We can have
    # multiple Strassen blocks as long as we account for all the bilinear forms correctly.
    
    # Let me think differently. The total tensor has 3*5*7 = 105 entries.
    # Schoolbook uses 105 rank-1 terms.
    # Each Strassen block replaces 8 schoolbook terms with 7 Strassen terms, saving 1.
    # We need to find how many non-overlapping 2x2x2 sub-tensors we can embed.
    
    # A 2x2x2 sub-tensor corresponds to choosing 2 rows of A, 2 contraction indices, 2 output cols.
    # The sub-tensor is the 8 entries C[i,k] for i in {i0,i1}, k in {k0,k1} that come from
    # A[i,j]*B[j,k] for j in {j0,j1}.
    
    # For these 8 entries to form a complete 2x2x2 multiplication tensor, we need all 8
    # (i,j,k) combinations. That's automatically satisfied if we pick 2 i's, 2 j's, 2 k's.
    
    # The constraint is that different Strassen blocks should use disjoint sets of (i,j,k) triples.
    # Actually no - in tensor decomposition, the terms can share indices as long as the
    # overall sum is correct. But for the "block replacement" approach, we need the blocks
    # to be on disjoint sets of output entries (i,k) to avoid double-counting.
    
    # Each Strassen block covers a 2x2 output block (4 entries of C).
    # With 3x7 = 21 output entries, we can fit at most floor(21/4) = 5 complete 2x2 blocks
    # without overlap, but with 3 rows we can only do 2x2 blocks using rows 0-1 or 1-2.
    
    # Using rows 0-1: we can cover output cols (0,1), (2,3), (4,5) -> 3 blocks, covering 12 entries
    # Remaining: row 2 (7 entries) + col 6 for rows 0-1 (2 entries) = 9 entries, schoolbook.
    # Total: 3*7 + 9*1 = 21 + 9 = 30? No, that's not right.
    
    # Let me count differently. Each Strassen block saves 1 term.
    # If we place 6 Strassen blocks, we save 6, getting 105-6=99.
    
    # The inspiration says 6 Strassen blocks work. Let me trust that and implement it.
    
    # Strategy: Place 6 Strassen 2x2x2 blocks on disjoint (i,j,k) subsets.
    # A has 3 rows, B has 5 contraction rows, 7 output cols.
    # 
    # Block 1: i={0,1}, j={0,1}, k={0,1}
    # Block 2: i={0,1}, j={0,1}, k={2,3}
    # Block 3: i={0,1}, j={2,3}, k={0,1}
    # Block 4: i={0,1}, j={2,3}, k={2,3}
    # Block 5: i={1,2}, j={0,1}, k={4,5}
    # Block 6: i={1,2}, j={2,3}, k={4,5}
    
    # Check for overlaps in (i,j,k) triples:
    # Block 1: (0,0,0),(0,0,1),(0,1,0),(0,1,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1)
    # Block 2: (0,0,2),(0,0,3),(0,1,2),(0,1,3),(1,0,2),(1,0,3),(1,1,2),(1,1,3)
    # Block 3: (0,2,0),(0,2,1),(0,3,0),(0,3,1),(1,2,0),(1,2,1),(1,3,0),(1,3,1)
    # Block 4: (0,2,2),(0,2,3),(0,3,2),(0,3,3),(1,2,2),(1,2,3),(1,3,2),(1,3,3)
    # Block 5: (1,0,4),(1,0,5),(1,1,4),(1,1,5),(2,0,4),(2,0,5),(2,1,4),(2,1,5)
    # Block 6: (1,2,4),(1,2,5),(1,3,4),(1,3,5),(2,2,4),(2,2,5),(2,3,4),(2,3,5)
    
    # Check: Block 3 has (1,2,0) and (1,3,0). Block 6 has (1,2,4) and (1,3,4). No overlap.
    # Block 5 has (1,0,4) and (1,1,4). Block 6 has (1,2,4) and (1,3,4). No overlap.
    # All blocks seem to have disjoint (i,j,k) sets. 
    
    # Total covered: 6*8 = 48 entries. Remaining: 105-48 = 57 schoolbook terms.
    # Total rank: 6*7 + 57 = 42 + 57 = 99. 
    
    # Now implement the Strassen decomposition for each block.
    
    def add_strassen_block(U, V, W, col, i0, j0, k0):
        """Add a Strassen 2x2x2 block. i0,j0,k0 are starting indices."""
        # Strassen 2x2: 7 bilinear forms
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
        
        # A is 2x2: A[i0:i0+2, j0:j0+2] in original A (n x m)
        # B is 2x2: B[j0:j0+2, k0:k0+2] in original B (m x p)
        # C is 2x2: C[i0:i0+2, k0:k0+2] in original C (n x p)
        
        # A indices in U: row = i*m + j
        # B indices in V: row = j*p + k
        # C indices in W: row = k*n + i (trace convention W[k*n+i, r])
        
        # A11 = A[i0, j0], A12 = A[i0, j0+1]
        # A21 = A[i0+1, j0], A22 = A[i0+1, j0+1]
        # B11 = B[j0, k0], B12 = B[j0, k0+1]
        # B21 = B[j0+1, k0], B22 = B[j0+1, k0+1]
        
        # C11 = C[i0, k0], C12 = C[i0, k0+1]
        # C21 = C[i0+1, k0], C22 = C[i0+1, k0+1]
        
        a11 = (i0, j0)
        a12 = (i0, j0+1)
        a21 = (i0+1, j0)
        a22 = (i0+1, j0+1)
        b11 = (j0, k0)
        b12 = (j0, k0+1)
        b21 = (j0+1, k0)
        b22 = (j0+1, k0+1)
        c11 = (i0, k0)
        c12 = (i0, k0+1)
        c21 = (i0+1, k0)
        c22 = (i0+1, k0+1)
        
        # M1 = (A11+A22)(B11+B22)
        U[a11[0]*m+a11[1], col] += 1
        U[a22[0]*m+a22[1], col] += 1
        V[b11[0]*p+b11[1], col] += 1
        V[b22[0]*p+b22[1], col] += 1
        W[c11[1]*n+c11[0], col] += 1
        W[c22[1]*n+c22[0], col] += 1
        col += 1
        
        # M2 = (A21+A22)B11
        U[a21[0]*m+a21[1], col] += 1
        U[a22[0]*m+a22[1], col] += 1
        V[b11[0]*p+b11[1], col] += 1
        W[c21[1]*n+c21[0], col] += 1
        col += 1
        
        # M3 = A11(B21-B22)
        U[a11[0]*m+a11[1], col] += 1
        V[b21[0]*p+b21[1], col] += 1
        V[b22[0]*p+b22[1], col] -= 1
        W[c12[1]*n+c12[0], col] += 1
        W[c22[1]*n+c22[0], col] += 1
        col += 1
        
        # M4 = A22(B12-B11)
        U[a22[0]*m+a22[1], col] += 1
        V[b12[0]*p+b12[1], col] += 1
        V[b11[0]*p+b11[1], col] -= 1
        W[c11[1]*n+c11[0], col] += 1
        W[c21[1]*n+c21[0], col] += 1
        col += 1
        
        # M5 = (A11+A12)B22
        U[a11[0]*m+a11[1], col] += 1
        U[a12[0]*m+a12[1], col] += 1
        V[b22[0]*p+b22[1], col] += 1
        W[c11[1]*n+c11[0], col] -= 1
        W[c12[1]*n+c12[0], col] += 1
        col += 1
        
        # M6 = (A21-A11)(B11+B12)
        U[a21[0]*m+a21[1], col] += 1
        U[a11[0]*m+a11[1], col] -= 1
        V[b11[0]*p+b11[1], col] += 1
        V[b12[0]*p+b12[1], col] += 1
        W[c22[1]*n+c22[0], col] += 1
        col += 1
        
        # M7 = (A12-A22)(B21+B22)
        U[a12[0]*m+a12[1], col] += 1
        U[a22[0]*m+a22[1], col] -= 1
        V[b21[0]*p+b21[1], col] += 1
        V[b22[0]*p+b22[1], col] += 1
        W[c11[1]*n+c11[0], col] += 1
        col += 1
        
        return col
    
    # Initialize matrices
    U = np.zeros((n * m, 99), dtype=float)
    V = np.zeros((m * p, 99), dtype=float)
    W = np.zeros((p * n, 99), dtype=float)
    
    col = 0
    
    # Add 6 Strassen blocks
    strassen_blocks = [
        (0, 0, 0),  # i0=0, j0=0, k0=0
        (0, 0, 2),  # i0=0, j0=0, k0=2
        (0, 2, 0),  # i0=0, j0=2, k0=0
        (0, 2, 2),  # i0=0, j0=2, k0=2
        (1, 0, 4),  # i0=1, j0=0, k0=4
        (1, 2, 4),  # i0=1, j0=2, k0=4
    ]
    
    for (i0, j0, k0) in strassen_blocks:
        col = add_strassen_block(U, V, W, col, i0, j0, k0)
    
    # Add schoolbook terms for remaining (i,j,k) triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check if this (i,j,k) is covered by a Strassen block
                covered = False
                for (i0, j0, k0) in strassen_blocks:
                    if i in (i0, i0+1) and j in (j0, j0+1) and k in (k0, k0+1):
                        covered = True
                        break
                if not covered:
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
    
    # Verify correctness
    # The tensor T[i,j,k] = delta (in the sense that C[i,k] = sum_j A[i,j]*B[j,k])
    # Our decomposition should give: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    # Actually, the tensor for matrix mult is T[i,j,k] = 1 for all valid i,j,k (since C[i,k] = sum_j A[i,j]B[j,k])
    # Wait no. The tensor is T such that C = T(A,B) where T[i,j,k] = 1 means the (i,k) entry gets A[i,j]*B[j,k].
    # In the decomposition, sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] should equal T[i,j,k] = 1 for all i,j,k.
    
    # Let's verify:
    verified = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(col):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - 1.0) > 1e-10:
                    verified = False
                    break
            if not verified:
                break
    
    if not verified:
        # Fall back to schoolbook
        U = np.zeros((n * m, n*m*p), dtype=float)
        V = np.zeros((m * p, n*m*p), dtype=float)
        W = np.zeros((p * n, n*m*p), dtype=float)
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, column] = 1.0
                    V[j * p + k, column] = 1.0
                    W[k * n + i, column] = 1.0
                    column += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Trim to actual rank
    U = U[:, :col]
    V = V[:, :col]
    W = W[:, :col]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
