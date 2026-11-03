DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # We will build the decomposition recursively.
    # Base case: M_1,2,2 rank 4
    # M_2,2,2 rank 7
    # M_3,2,2 rank 11 (7+4)
    # M_3,4,2 rank 22 (11+11)
    # M_3,4,3 rank 34 (22+12)
    # M_3,4,4 rank 46 (34+12)
    # M_3,4,5 rank 58 (46+12)
    
    # To be more efficient, we can just use the schoolbook for the last step
    # and Strassen for the 2x2 blocks.
    
    # Let's construct M_3,4,5 directly using the block structure.
    # A is 3x4, B is 4x5.
    # Split A into A1 (3x2) and A2 (3x2).
    # Split B into B1 (2x5) and B2 (2x5).
    # C = A1 B1 + A2 B2.
    # Rank(A1 B1) + Rank(A2 B2).
    # A1 B1 is M_3,2,5.
    # A2 B2 is M_3,2,5.
    
    # So we need Rank(M_3,2,5).
    # M_3,2,5 can be split into M_3,2,2 + M_3,2,2 + M_3,2,1.
    # Rank(M_3,2,2) = 11.
    # Rank(M_3,2,1) = 6.
    # Rank(M_3,2,5) = 11 + 11 + 6 = 28.
    # Total Rank = 28 + 28 = 56.
    
    # This is better than 58.
    
    # Let's verify Rank(M_3,2,2) = 11.
    # It is M_2,2,2 (rank 7) + M_1,2,2 (rank 4).
    # The supports are disjoint in the A dimension (rows).
    # So yes, 11.
    
    # Rank(M_3,2,1) = 6.
    # A is 3x2, B is 2x1. C is 3x1.
    # C_i = A_i0 B_0 + A_i1 B_1.
    # 3 outputs, each sum of 2 products.
    # No sharing possible between different i?
    # B_0, B_1 are shared.
    # C_0 = a0 B_0 + b0 B_1
    # C_1 = a1 B_0 + b1 B_1
    # C_2 = a2 B_0 + b2 B_1
    # This is 6 multiplications.
    # Can we share?
    # Let S = B_0 + B_1, D = B_0 - B_1.
    # C_i = A_i0 (S+D)/2 + A_i1 (S-D)/2 = (A_i0+A_i1)S/2 + (A_i0-A_i1)D/2.
    # This requires 2 multiplications per i?
    # (A_i0+A_i1) S and (A_i0-A_i1) D.
    # Still 2 per i. Total 6.
    # So Rank(M_3,2,1) = 6.
    
    # So Rank(M_3,2,5) = 11 + 11 + 6 = 28.
    # Rank(M_3,4,5) = 28 + 28 = 56.
    
    return {"U": None, "V": None, "W": None} # Placeholder
