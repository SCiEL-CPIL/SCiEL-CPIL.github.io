"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Use the known Strassen decomposition for 2x2 matrix multiplication (rank 7).
    # For 6x6, we can view it as a block matrix of 3x3 blocks of size 2x2.
    # However, the standard Strassen algorithm applies to the whole 2x2 block matrix.
    # A direct way to get a lower rank than schoolbook (216) is to use the
    # tensor product structure if we can find a low-rank decomposition for smaller blocks.
    
    # Actually, the simplest improvement is to use the known rank of matrix multiplication.
    # For n=m=p=6, the best known upper bound is via algorithms like Bini et al. or others.
    # But implementing complex decompositions is error-prone.
    
    # Let's try a simple block decomposition.
    # 6x6 can be seen as 3x3 blocks of 2x2.
    # If we use schoolbook for the 3x3 blocks (27 multiplications) and Strassen for each 2x2 (7 mults),
    # total rank = 27 * 7 = 189.
    # Alternatively, use Strassen for the 6x6 directly? No, Strassen is for powers of 2.
    # We can pad to 8x8 and use Strassen recursively, but the rank would be higher.
    
    # Let's implement the 27 * 7 = 189 rank decomposition.
    # We need the U, V, W for the 2x2 Strassen decomposition.
    # The 2x2 matrices are:
    # A = [[a11, a12], [a21, a22]]
    # B = [[b11, b12], [b21, b22]]
    # C = A * B
    
    # Strassen's 7 products:
    # m1 = (a11 + a22) * (b11 + b22)
    # m2 = (a21 + a22) * b11
    # m3 = a11 * (b12 - b22)
    # m4 = a22 * (b21 - b11)
    # m5 = (a11 + a12) * b22
    # m6 = (a21 - a11) * (b11 + b12)
    # m7 = (a12 - a22) * (b21 + b22)
    
    # C11 = m1 + m4 - m5 + m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m2 + m3 + m6
    
    # The U matrix (for A) has shape (4, 7) where rows correspond to a11, a12, a21, a22.
    # The V matrix (for B) has shape (4, 7) where rows correspond to b11, b12, b21, b22.
    # The W matrix (for C) has shape (4, 7) where rows correspond to c11, c12, c21, c22.
    
    # Let's define the Strassen matrices:
    # U (4x7):
    # m1: a11 + a22 -> [1, 0, 0, 1]
    # m2: a21 + a22 -> [0, 0, 1, 1]
    # m3: a11       -> [1, 0, 0, 0]
    # m4: a22       -> [0, 0, 0, 1]
    # m5: a11 + a12 -> [1, 1, 0, 0]
    # m6: a21 - a11 -> [-1, 0, 1, 0]
    # m7: a12 - a22 -> [0, 1, 0, -1]
    
    # V (4x7):
    # m1: b11 + b22 -> [1, 0, 0, 1]
    # m2: b11       -> [1, 0, 0, 0]
    # m3: b12 - b22 -> [0, 1, 0, -1]
    # m4: b21 - b11 -> [-1, 0, 1, 0]
    # m5: b22       -> [0, 0, 0, 1]
    # m6: b11 + b12 -> [1, 1, 0, 0]
    # m7: b21 + b22 -> [0, 0, 1, 1]
    
    # W (4x7):
    # c11: m1 + m4 - m5 + m7 -> [1, 0, 0, 1, -1, 0, 1]
    # c12: m3 + m5           -> [0, 0, 1, 0, 1, 0, 0]
    # c21: m2 + m4           -> [0, 1, 0, 1, 0, 0, 0]
    # c22: m1 - m2 + m3 + m6 -> [1, -1, 1, 0, 0, 1, 0]
    
    import numpy as np
    
    U_strassen = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=int)
    
    V_strassen = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=int)
    
    W_strassen = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=int)
    
    # Now, for 6x6, we use 3x3 blocks of 2x2.
    # The total rank is 3*3 * 7 = 63? No, that's not right.
    # The block structure is:
    # A is 6x6, divided into 3x3 blocks of 2x2.
    # B is 6x6, divided into 3x3 blocks of 2x2.
    # C is 6x6, divided into 3x3 blocks of 2x2.
    # C_ij = sum_k A_ik * B_kj
    # Each A_ik * B_kj is a 2x2 * 2x2 = 2x2 multiplication, which takes 7 products.
    # So for each (i, j, k) triple, we have 7 products.
    # Total number of (i, j, k) triples is 3*3*3 = 27.
    # Total rank = 27 * 7 = 189.
    
    rank = 189
    U = np.zeros((36, rank), dtype=int)
    V = np.zeros((36, rank), dtype=int)
    W = np.zeros((36, rank), dtype=int)
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                # For each (i, k) in A, (k, j) in B, (i, j) in C
                # The block A_ik is at rows (2*i, 2*i+1), columns (2*k, 2*k+1)
                # The block B_kj is at rows (2*k, 2*k+1), columns (2*j, 2*j+1)
                # The block C_ij is at rows (2*i, 2*i+1), columns (2*j, 2*j+1)
                
                for s in range(7):
                    # U entries for A_ik
                    for u_idx in range(4):
                        U[2*i + u_idx, 2*k + (0 if u_idx < 2 else 1) + col] = 0 # This is getting complicated
                    # Let's do it more carefully.
                    # A_ik is a 2x2 block. Its elements are at:
                    # (2*i, 2*k), (2*i, 2*k+1), (2*i+1, 2*k), (2*i+1, 2*k+1)
                    # These correspond to rows 0, 1, 2, 3 of U_strassen.
                    # So U[2*i + r, col] = U_strassen[r, s] for r in 0..3
                    # But we need to map the local 2x2 indices to global 6x6 indices.
                    
                    # Let's restart the indexing.
                    # A is 6x6. A_ik is the block at row i, col k.
                    # Element (r, c) of A_ik is at global position (2*i+r, 2*k+c).
                    # U_strassen has rows corresponding to (0,0), (0,1), (1,0), (1,1) of the 2x2 block.
                    # So U[2*i+r, 2*k+c, col] = U_strassen[r*2+c, s]? No, U_strassen is 4x7.
                    # The 4 rows of U_strassen correspond to the 4 elements of the 2x2 block.
                    # Row 0: (0,0), Row 1: (0,1), Row 2: (1,0), Row 3: (1,1).
                    
                    for r in range(2):
                        for c in range(2):
                            local_idx = r * 2 + c
                            U[2*i + r, 2*k + c, col] = U_strassen[local_idx, s]
                    
                    # Similarly for V and B_kj
                    for r in range(2):
                        for c in range(2):
                            local_idx = r * 2 + c
                            V[2*k + r, 2*j + c, col] = V_strassen[local_idx, s]
                    
                    # And W and C_ij
                    for r in range(2):
                        for c in range(2):
                            local_idx = r * 2 + c
                            W[2*i + r, 2*j + c, col] = W_strassen[local_idx, s]
                    
                    col += 1
    
    # Convert to lists of lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
