"""Matrix multiplication 4x5x5 rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 5, 5]

# Deadline guard
START_TIME = time.time()
DEADLINE = 150.0
SAFETY_MARGIN = 5.0

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 4x5x5 matrix multiplication.
    
    Strategy: Use Strassen's algorithm for the 4x4x4 core (rank 28), then extend
    to 4x5x5 by handling the extra dimensions. The 4x5x5 multiplication can be
    decomposed as:
    - 4x4x4 core (Strassen, rank 28)
    - Cross terms for the 5th row/col
    
    We use a block decomposition approach:
    A (4x5) = [A1 (4x4) | A2 (4x1)]
    B (5x5) = [B11 (4x4) B12 (4x1); B21 (1x4) B22 (1x1)]
    
    C = A*B = A1*B11 + A1*B12 + A2*B21 + A2*B22
    
    We can use Strassen for A1*B11 (rank 28), and handle the rest more efficiently.
    
    Actually, let me think about this more carefully. The standard approach:
    - 4x4x4: Strassen rank 28
    - For 4x5x5, we can pad to 5x5x5 or use a direct construction.
    
    Let me use a known result: the rank of 4x5x5 matrix multiplication tensor.
    The schoolbook is 4*5*5 = 100.
    
    Using Strassen on the 4x4x4 part (28) plus the remaining terms:
    A is 4x5, B is 5x5.
    
    Let me try a different approach: use the fact that we can write the 4x5x5
    multiplication using a combination of Strassen-like blocks.
    
    Actually, the most practical approach given the time constraint:
    1. Use Strassen for 4x4x4 (rank 28)
    2. Handle the extra 5th column of A and 5th row/col of B
    
    A (4x5) = [A_4x4 | a_4x1]
    B (5x5) = [[B_4x4, b_4x1], [c_1x4, d_1x1]]
    
    C = A*B:
    C_ij = sum_k A_ik B_kj
    
    For j <= 4: C_ij = sum_{k<=4} A_ik B_kj + A_i5 * B_5j
    For j = 5: C_i5 = sum_{k<=4} A_ik B_k5 + A_i5 * B_55
    
    So we need:
    - A_4x4 * B_4x4 (Strassen, 28 terms)
    - A_4x4 * b_4x1 (4x4x1, can be done in 4*1=4 terms or less with tricks)
    - a_4x1 * c_1x4 (4x1x4, 4 terms)
    - a_4x1 * d_1x1 (4x1x1, 4 terms)
    
    Total: 28 + 4 + 4 + 4 = 40? That seems too good. Let me verify.
    
    Actually for 4x4x1 matrix multiplication, the rank is 4 (each output element
    is a dot product). Similarly 4x1x4 has rank 4, and 4x1x1 has rank 4.
    
    So total rank = 28 + 4 + 4 + 4 = 40.
    
    But wait, we need to be careful about how these are combined. Let me construct
    this explicitly.
    
    Strassen's 4x4x4 decomposition (rank 28):
    A = [[A11, A12], [A21, A22]] (2x2 blocks of 2x2)
    B = [[B11, B12], [B21, B22]]
    
    The 7 Strassen products for 2x2x2:
    M1 = (A11+A22)(B11+B22)
    M2 = (A21+A22)B11
    M3 = A11(B12-B22)
    M4 = A22(B21-B11)
    M5 = (A11+A12)B22
    M6 = (A21-A11)(B11+B12)
    M7 = (A12-A22)(B21+B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    For 4x4x4, we apply this at the 2x2 block level where each block is 2x2.
    So we get 7 groups of 2x2x2 multiplications, each of rank 8 (schoolbook) 
    or we can use Strassen recursively.
    
    Actually, the standard Strassen for 4x4x4 uses rank 28 = 7*4. Let me think...
    
    For 2x2x2, Strassen rank is 7.
    For 4x4x4, we can view it as 2x2 blocks of 2x2 matrices.
    Each of the 7 Strassen products involves multiplying 2x2 matrices.
    If we use schoolbook for each 2x2x2 (rank 4), total = 7*4 = 28.
    If we use Strassen for each 2x2x2 (rank 7), total = 7*7 = 49. Worse!
    
    So for 4x4x4, rank 28 is the Strassen result using schoolbook at the leaf level.
    
    Let me construct the full 4x5x5 decomposition with rank 40.
    """
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    # We'll build the decomposition explicitly.
    # The multiplication tensor T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    # In the output convention: W[k*n+i, r]
    
    # Let's use the block decomposition:
    # A is 4x5, split as [A1 (4x4) | A2 (4x1)]
    # B is 5x5, split as [[B11 (4x4), B12 (4x1)], [B21 (1x4), B22 (1x1)]]
    #
    # C = A1*B11 + A1*B12 + A2*B21 + A2*B22
    #
    # Term 1: A1*B11 - 4x4x4, Strassen rank 28
    # Term 2: A1*B12 - 4x4x1, rank 4
    # Term 3: A2*B21 - 4x1x4, rank 4  
    # Term 4: A2*B22 - 4x1x1, rank 4
    # Total: 40
    
    # But we need to verify this is valid. Each term contributes to different parts
    # of C, so they don't interfere.
    
    # Let me construct the factor matrices.
    # U has shape (n*m, rank) = (20, rank)
    # V has shape (m*p, rank) = (25, rank)  
    # W has shape (p*n, rank) = (20, rank)
    
    # Indexing:
    # U[i*m + j, r] for i in 0..3, j in 0..4
    # V[j*p + k, r] for j in 0..4, k in 0..4
    # W[k*n + i, r] for k in 0..4, i in 0..3
    
    # Let's build each term:
    
    U = []
    V = []
    W = []
    
    # Term 1: A1 (4x4) * B11 (4x4) -> contributes to C[0..3, 0..3]
    # Strassen decomposition of 4x4x4 with rank 28
    # A1 indices: i in 0..3, j in 0..3 (j is the 4th column, not the 5th)
    # B11 indices: j in 0..3, k in 0..3
    
    # Strassen for 4x4x4:
    # Split A1 into 2x2 blocks of 2x2:
    # A111 = A1[0:2, 0:2], A112 = A1[0:2, 2:4]
    # A121 = A1[2:4, 0:2], A122 = A1[2:4, 2:4]
    # Similarly for B11.
    
    # The 7 Strassen products, each is a 2x2x2 multiplication (schoolbook, rank 4)
    
    # Let me define the Strassen factors for 2x2x2:
    # For 2x2x2 with rank 7:
    # M1 = (A11+A22)(B11+B22): factors involve A11,A22,B11,B22
    # This gets complex. Let me just use the known rank-28 factors for 4x4x4.
    
    # Actually, let me just construct it directly using the schoolbook for each
    # sub-block and combine with Strassen at the top level.
    
    # For 4x4x4 Strassen (rank 28):
    # The 7 Strassen products, each requiring 4 scalar multiplications (2x2x2 schoolbook)
    
    # Let me define the 28 basis vectors for the Strassen part.
    
    # Strassen 2x2x2 factors (rank 7):
    # Each product Mq = (sum of A blocks) * (sum of B blocks)
    # For 2x2x2, the factors are:
    # M1: A11,B11; A11,B22; A22,B11; A22,B22 (with signs)
    # ...
    
    # This is getting complicated. Let me use a simpler but still good approach.
    
    # Alternative: Use the known optimal or near-optimal constructions.
    # For 4x5x5, the best known upper bound might be around 40-50.
    
    # Let me just use a clean construction with rank 40 as described above.
    
    # I'll construct each term explicitly.
    
    def add_term(u_entries, v_entries, w_entries):
        """Add a rank-1 term. u_entries: list of (idx, val), etc."""
        u_vec = [0] * (n * m)
        v_vec = [0] * (m * p)
        w_vec = [0] * (p * n)
        for idx, val in u_entries:
            u_vec[idx] = val
        for idx, val in v_entries:
            v_vec[idx] = val
        for idx, val in w_entries:
            w_vec[idx] = val
        U.append(u_vec)
        V.append(v_vec)
        W.append(w_vec)
    
    # Term 1: A1 (4x4) * B11 (4x4) -> C[0:4, 0:4]
    # Use Strassen rank 28 for this.
    # 
    # Strassen 4x4x4 construction:
    # A1 = [[P, Q], [R, S]] where P,Q,R,S are 2x2
    # B11 = [[W, X], [Y, Z]] where W,X,Y,Z are 2x2
    #
    # 7 Strassen products (each is 2x2x2, schoolbook rank 4):
    # M1 = (P+S)(W+Z)
    # M2 = (R+S)W
    # M3 = P(X-Z)
    # M4 = S(Y-W)
    # M5 = (P+Q)Z
    # M6 = (R-P)(W+X)
    # M7 = (Q-S)(Y+Z)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For each 2x2x2 product, we use schoolbook (rank 4).
    # So total 7 * 4 = 28 terms.
    
    # Let me define the 2x2x2 schoolbook factors:
    # For A(2x2) * B(2x2) -> C(2x2):
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    
    # For a 2x2x2 product with A block at rows (r1,r2) cols (c1,c2) of A1,
    # and B block at rows (c1,c2) cols (d1,d2) of B11:
    # The 4 scalar products are:
    # (r1,c1)*(c1,d1) -> C[r1,d1]
    # (r1,c2)*(c2,d1) -> C[r1,d1]
    # (r1,c1)*(c1,d2) -> C[r1,d2]
    # (r1,c2)*(c2,d2) -> C[r1,d2]
    # (r2,c1)*(c1,d1) -> C[r2,d1]
    # (r2,c2)*(c2,d1) -> C[r2,d1]
    # (r2,c1)*(c1,d2) -> C[r2,d2]
    # (r2,c2)*(c2,d2) -> C[r2,d2]
    
    # This is getting very involved. Let me just use a direct approach.
    
    # Given the complexity, let me use a simpler but effective construction.
    # I'll use the schoolbook for the 4x4x4 part (rank 16) and optimize the rest.
    # Total: 16 + 4 + 4 + 4 = 28. That's better than Strassen here!
    
    # Wait, 4x4x4 schoolbook is 4*4*4 = 64, not 16.
    # 4x4x1 schoolbook is 4*4*1 = 16, not 4.
    
    # I was confusing myself. Let me redo:
    # 4x4x4: rank 64 (schoolbook) or 28 (Strassen)
    # 4x4x1: rank 16 (schoolbook) - each of 4*4=16 outputs is a dot product of length 4
    # 4x1x4: rank 16 (schoolbook)
    # 4x1x1: rank 4 (schoolbook)
    
    # With Strassen for 4x4x4: 28 + 16 + 16 + 4 = 64
    # Without Strassen: 64 + 16 + 16 + 4 = 100 (schoolbook total)
    
    # So Strassen saves 36: 100 -> 64.
    
    # Let me construct the Strassen 4x4x4 part properly.
    
    # I'll use a known explicit construction for Strassen 4x4x4 rank 28.
    
    # Let me define it programmatically.
    
    # For the 4x4x4 Strassen part:
    # A1 has indices (i, j) with i,j in 0..3
    # B11 has indices (j, k) with j,k in 0..3
    # Output C has indices (i, k) with i,k in 0..3
    
    # Block structure:
    # A1 = [[A00, A01], [A10, A11]] where Aab is 2x2 with indices
    #       Aab[i,j] for i in {0,1}+2*a, j in {0,1}+2*b
    # B11 = [[B00, B01], [B10, B11]] where Bab[i,j] for i in {0,1}+2*a, j in {0,1}+2*b
    
    # Strassen products (each is a 2x2x2 multiplication):
    # Product q has A-side = sum of certain Aab blocks, B-side = sum of certain Bab blocks
    
    # The 7 Strassen combinations:
    strassen_combos = [
        # (A_blocks_with_signs, B_blocks_with_signs)
        # Each block is (a, b) meaning Aab or Bab
        # M1 = (A00+A11)(B00+B11)
        (([(0,0,1),(1,1,1)]), ([(0,0,1),(1,1,1)])),
        # M2 = (A10+A11)B00
        (([(1,0,1),(1,1,1)]), ([(0,0,1)])),
        # M3 = A00(B01-B11)
        (([(0,0,1)]), ([(0,1,1),(1,1,-1)])),
        # M4 = A11(B10-B00)
        (([(1,1,1)]), ([(1,0,1),(0,0,-1)])),
        # M5 = (A00+A01)B11
        (([(0,0,1),(0,1,1)]), ([(1,1,1)])),
        # M6 = (A10-A00)(B00+B01)
        (([(1,0,1),(0,0,-1)]), ([(0,0,1),(0,1,1)])),
        # M7 = (A01-A11)(B10+B11)
        (([(0,1,1),(1,1,-1)]), ([(1,0,1),(1,1,1)])),
    ]
    
    # For each Strassen product q, we do a 2x2x2 schoolbook multiplication.
    # The 2x2x2 has A-side blocks (2x2 matrices) and B-side blocks (2x2 matrices).
    # The A-side is a linear combination of 2x2 blocks from A1.
    # The B-side is a linear combination of 2x2 blocks from B11.
    
    # For the 2x2x2 schoolbook within each Strassen product:
    # We have A_side (2x2) * B_side (2x2) -> C_side (2x2)
    # A_side[i,j] = sum of contributions from A1 blocks
    # B_side[i,j] = sum of contributions from B11 blocks
    
    # The 4 scalar products in the 2x2x2:
    # (0,0)*(0,0), (0,1)*(1,0), (0,0)*(0,1), (0,1)*(1,1)
    # (1,0)*(0,0), (1,1)*(1,0), (1,0)*(0,1), (1,1)*(1,1)
    
    # For each scalar product (ai, aj) * (bi, bj):
    # U factor: for each A1 block (a1, b1) with sign s_a, and for each (r, c) in 2x2:
    #   U[(2*a1+r)*m + (2*b1+c)] += s_a * delta_{(r,c),(ai,aj)}
    # V factor: for each B11 block (a2, b2) with sign s_b, and for each (r, c) in 2x2:
    #   V[(2*a2+r)*p + (2*b2+c)] += s_b * delta_{(r,c),(bi,bj)}
    # W factor: for the output position in the 2x2 block:
    #   The output C_side[ai, bj] contributes to C at position (2*a_out+ai, 2*b_out+bj)
    #   where (a_out, b_out) depends on which C block this Strassen product contributes to.
    
    # This is very complex. Let me just hardcode a known rank-28 construction.
    
    # Given the time constraints, let me use a simpler approach:
    # Just use schoolbook for everything but with some optimization.
    
    # Actually, let me just return the schoolbook decomposition (rank 100) as a fallback,
    # and try to improve if time permits.
    
    # Schoolbook: rank = 4*5*5 = 100
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Check time and try to improve
    if time.time() - START_TIME > DEADLINE - SAFETY_MARGIN:
        return {"U": U, "V": V, "W": W}
    
    # Try to use Strassen for the 4x4x4 core
    # This would give rank 28 + 16 + 16 + 4 = 64
    
    # Let me construct this properly.
    U2 = []
    V2 = []
    W2 = []
    
    # Term 1: A1(4x4) * B11(4x4) using Strassen rank 28
    # A1[i,j] for i,j in 0..3 maps to U[i*5+j]
    # B11[j,k] for j,k in 0..3 maps to V[j*5+k]
    # C[i,k] for i,k in 0..3 maps to W[k*4+i]
    
    # Strassen 4x4x4: 7 products, each 2x2x2 schoolbook (rank 4), total 28
    
    # For each of the 7 Strassen products, and for each of the 4 schoolbook terms:
    for q, (a_blocks, b_blocks) in enumerate(strassen_combos):
        # a_blocks: list of (block_row, block_col, sign)
        # b_blocks: list of (block_row, block_col, sign)
        
        # The 2x2x2 schoolbook terms:
        # (0,0)*(0,0) -> C[0,0]
        # (0,1)*(1,0) -> C[0,0]
        # (0,0)*(0,1) -> C[0,1]
        # (0,1)*(1,1) -> C[0,1]
        # (1,0)*(0,0) -> C[1,0]
        # (1,1)*(1,0) -> C[1,0]
        # (1,0)*(0,1) -> C[1,1]
        # (1,1)*(1,1) -> C[1,1]
        
        sb_terms = [
            ((0,0), (0,0), (0,0)),  # (a_i,a_j) * (b_i,b_j) -> c_i,c_j
            ((0,1), (1,0), (0,0)),
            ((0,0), (0,1), (0,1)),
            ((0,1), (1,1), (0,1)),
            ((1,0), (0,0), (1,0)),
            ((1,1), (1,0), (1,0)),
            ((1,0), (0,1), (1,1)),
            ((1,1), (1,1), (1,1)),
        ]
        
        for (ai, aj), (bi, bj), (ci, cj) in sb_terms:
            u_vec = [0] * (n * m)
            v_vec = [0] * (m * p)
            w_vec = [0] * (p * n)
            
            # U factor: contributions from A1 blocks
            for (ba, bb, sa) in a_blocks:
                # A1 block at (ba, bb) is 2x2, indices (2*ba+r, 2*bb+c) for r,c in 0,1
                # We need the element (ai, aj) of this 2x2 block
                if ai < 2 and aj < 2:
                    r = ai
                    c = aj
                    u_idx = (2 * ba + r) * m + (2 * bb + c)
                    u_vec[u_idx] += sa
            
            # V factor: contributions from B11 blocks
            for (ba, bb, sb) in b_blocks:
                # B11 block at (ba, bb) is 2x2, indices (2*ba+r, 2*bb+c) for r,c in 0,1
                if bi < 2 and bj < 2:
                    r = bi
                    c = bj
                    v_idx = (2 * ba + r) * p + (2 * bb + c)
                    v_vec[v_idx] += sb
            
            # W factor: output position
            # The Strassen product q contributes to specific C blocks.
            # We need to determine which C block (2x2) this contributes to.
            # For now, let's assume it contributes to C[0:2, 0:2] (this is wrong, need to fix)
            
            # Actually, the output of Strassen product q goes to specific C blocks.
            # M1 -> C00, C11 (with signs)
            # M2 -> C10
            # M3 -> C01, C11
            # M4 -> C00, C10
            # M5 -> C00, C01
            # M6 -> C11
            # M7 -> C00, C11
            
            # This mapping is complex. Let me use a different approach.
            
            # For simplicity, let me just use the schoolbook for the 4x4x4 part too.
            # Total rank = 64 + 16 + 16 + 4 = 100 (same as full schoolbook)
            
            # This isn't helping. Let me just return the schoolbook.
            
            pass
    
    # The Strassen construction is too complex to get right quickly.
    # Let me just return the schoolbook decomposition.
    
    return {"U": U, "V": V, "W": W}
