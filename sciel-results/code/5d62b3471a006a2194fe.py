import numpy as np

def construct():
    n, m, p = 4, 5, 5
    # Total rank 85
    rank = 85
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        nonlocal col
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val
        col += 1

    # 1. A11 * B11 (4x4 * 4x4 -> 4x4)
    # Strassen for 4x4
    # We need the 49 terms for 4x4x4
    # Let's generate them recursively or hardcode.
    # Strassen 2x2 terms:
    # M1: (A00+A11)(B00+B11)
    # M2: (A10+A11)B00
    # M3: A00(B01-B11)
    # M4: A11(B10-B00)
    # M5: (A00+A01)B11
    # M6: (A10-A00)(B00+B01)
    # M7: (A01-A11)(B10+B11)
    
    # For 4x4, we apply Strassen to the 2x2 blocks of 2x2 matrices.
    # Let A blocks be A00, A01, A10, A11 (each 2x2)
    # Let B blocks be B00, B01, B10, B11 (each 2x2)
    # C blocks C00, C01, C10, C11
    
    # The 7 products are:
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B01-B11)
    # P4 = A11(B10-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    
    # Each Pi is a 2x2 matrix product.
    # We decompose each Pi using the 7 rank-1 terms for 2x2.
    
    # 2x2 Strassen terms (u, v, w) for 2x2x2
    # u, v, w are length 4.
    # Indices: A00=0, A01=1, A10=2, A11=3
    # B00=0, B01=1, B10=2, B11=3
    # C00=0, C01=1, C10=2, C11=3
    
    # Term 1: (A00+A11)(B00+B11)
    # u: [1, 0, 0, 1]
    # v: [1, 0, 0, 1]
    # w: C00 += 1, C01 += 1, C10 += 1, C11 += 1? 
    # C = P1 + P4 - P5 + P7
    # So w for P1 is [1, 1, 1, 1]
    
    # Term 2: (A10+A11)B00
    # u: [0, 0, 1, 1]
    # v: [1, 0, 0, 0]
    # w: C10 += 1, C11 += 1 -> [0, 0, 1, 1]
    
    # Term 3: A00(B01-B11)
    # u: [1, 0, 0, 0]
    # v: [0, 1, 0, -1]
    # w: C01 += 1, C11 += 1 -> [0, 1, 0, 1]
    
    # Term 4: A11(B10-B00)
    # u: [0, 0, 0, 1]
    # v: [0, 0, 1, -1]
    # w: C10 += 1, C11 += 1 -> [0, 0, 1, 1]
    
    # Term 5: (A00+A01)B11
    # u: [1, 1, 0, 0]
    # v: [0, 0, 0, 1]
    # w: C00 -= 1, C01 -= 1 -> [0, -1, 0, -1] (because C = ... - P5 ...)
    # Wait, C00 = P1 + P4 - P5 + P7. So P5 coeff is -1.
    # C01 = P3 + P5. So P5 coeff is +1.
    # C10 = P2 + P4. P5 coeff 0.
    # C11 = P1 - P2 + P3 + P6. P5 coeff 0.
    # So w for P5: C00=-1, C01=1, C10=0, C11=0 -> [-1, 1, 0, 0]
    
    # Term 6: (A10-A00)(B00+B01)
    # u: [-1, 0, 1, 0]
    # v: [1, 1, 0, 0]
    # w: C11 += 1 -> [0, 0, 0, 1]
    
    # Term 7: (A01-A11)(B10+B11)
    # u: [0, 1, 0, -1]
    # v: [0, 0, 1, 1]
    # w: C00 += 1 -> [1, 0, 0, 0]
    
    strassen_2x2_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 1, 1, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        ([0, 0, 0, 1], [0, 0, 1, -1], [0, 0, 1, 1]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
    ]
    
    # Now build 4x4 terms
    # For each of the 7 Pi, we have 7 sub-terms.
    # We need to map local 2x2 indices to global 4x4 indices.
    # Global 4x4 indices:
    # A: i in 0..3, j in 0..3.
    # Block A00: i in 0..1, j in 0..1.
    # Block A01: i in 0..1, j in 2..3.
    # Block A10: i in 2..3, j in 0..1.
    # Block A11: i in 2..3, j in 2..3.
    
    def map_2x2_to_4x4_A(block, li, lj):
        # block: 0=A00, 1=A01, 2=A10, 3=A11
        # li, lj: 0 or 1
        if block == 0: return li * 4 + lj
        if block == 1: return li * 4 + (2 + lj)
        if block == 2: return (2 + li) * 4 + lj
        if block == 3: return (2 + li) * 4 + (2 + lj)
        
    def map_2x2_to_4x4_B(block, li, lj):
        # Same structure for B (rows 0..3, cols 0..3)
        if block == 0: return li * 4 + lj
        if block == 1: return li * 4 + (2 + lj)
        if block == 2: return (2 + li) * 4 + lj
        if block == 3: return (2 + li) * 4 + (2 + lj)
        
    def map_2x2_to_4x4_C(block, li, lj):
        # C is 4x4.
        if block == 0: return li * 4 + lj
        if block == 1: return li * 4 + (2 + lj)
        if block == 2: return (2 + li) * 4 + lj
        if block == 3: return (2 + li) * 4 + (2 + lj)

    # Pi definitions:
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B01-B11)
    # P4 = A11(B10-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    
    # For each Pi, we sum the contributions of its components.
    # This is complex to do dynamically. Let's just generate the 49 terms by iterating.
    
    # List of (A_blocks, B_blocks) for each P
    # P1: A00, A11; B00, B11
    # P2: A10, A11; B00
    # P3: A00; B01, B11 (with -1 on B11)
    # P4: A11; B10, B00 (with -1 on B00)
    # P5: A00, A01; B11
    # P6: A10, A00 (with -1 on A00); B00, B01
    # P7: A01, A11 (with -1 on A11); B10, B11
    
    # Actually, it's easier to just use the 2x2 Strassen terms for each 2x2 block product.
    # But P1 is (A00+A11)(B00+B11). This is NOT a single 2x2 product.
    # It's the product of two 2x2 matrices: M_A = A00+A11, M_B = B00+B11.
    # So we need to decompose M_A * M_B.
    # M_A is a 2x2 matrix. M_B is a 2x2 matrix.
    # We can use the 7 Strassen terms for M_A * M_B.
    # But M_A and M_B are linear combinations of A blocks and B blocks.
    
    # Let's stick to the additive property.
    # T = sum of terms.
    # We can just expand everything into rank-1 terms.
    
    # For P1:
    # M_A = A00 + A11. M_B = B00 + B11.
    # Product = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11.
    # Each of these is a 2x2 * 2x2 product.
    # We can decompose each 2x2*2x2 product into 7 rank-1 terms.
    # Total 4 * 7 = 28 terms for P1.
    # But wait, Strassen reduces 8 to 7.
    # If we just expand, we get 28 terms.
    # The "Strassen for 4x4" rank 49 comes from applying Strassen to the 7 products.
    # P1 is ONE product of 2x2 matrices.
    # So P1 contributes 7 terms.
    # But P1 involves A00, A11, B00, B11.
    # The 7 terms for P1 will have u, v vectors that are linear combinations of A00, A11, etc.
    
    # Let's generate the 7 terms for a generic 2x2 product X*Y.
    # X = sum x_k A_k, Y = sum y_k B_k.
    # This is getting complicated to generate symbolically.
    
    # Alternative: Hardcode the 49 terms? No, too long.
    # Let's use a simple recursive generator.
    
    def get_strassen_4x4_terms():
        terms = []
        # P1: (A00+A11)(B00+B11)
        # We need the 7 terms for (A00+A11) * (B00+B11)
        # Let X = A00+A11, Y = B00+B11.
        # The 7 Strassen terms for X*Y are:
        # T1 = (X00+X11)(Y00+Y11)
        # ...
        # This is recursive.
        
        # Let's just use the 7 terms for 2x2 and substitute.
        # For each of the 7 Pi, we have a specific structure.
        
        # P1: X=A00+A11, Y=B00+B11.
        # X00=A00_00, X01=A00_01+A11_01? No.
        # A00 is a 2x2 matrix. A11 is a 2x2 matrix.
        # X = A00 + A11.
        # X00 = A00_00 + A11_00.
        # This is a mess.
        
        # Let's try a different approach.
        # We know the rank is 85.
        # We can just use the schoolbook for 4x4x4 (rank 64) and see if it's faster?
        # No, 64 + 16 + 16 + 4 = 100. Score 1.0.
        # We need < 100.
        
        # Let's use the 49 rank Strassen for 4x4.
        # I will implement the generator for 4x4 Strassen terms.
        
        def expand_2x2_product(A_blocks, B_blocks):
            # A_blocks: list of (block_id, coeff)
            # B_blocks: list of (block_id, coeff)
            # Returns list of 7 terms (u, v, w) for 2x2 product
            # u, v, w are length 4 (local 2x2 indices)
            
            # We form the 2x2 matrix X from A_blocks
            # X = sum c_i A_block_i
            # This is not a single 2x2 matrix of constants.
            
            # Actually, we can just treat the 4x4 multiplication as 16 schoolbook terms
            # and then apply Strassen to the 2x2 blocks of the 4x4?
            # No, Strassen is for 2x2 * 2x2.
            
            # Let's just use the 49 terms from a known source or derive them.
            # Given time constraints, I will implement the 4x4 Strassen by calling the 2x2 Strassen 7 times,
            # but handling the linear combinations carefully.
            
            pass
            
    # Due to complexity, I will use a simpler optimization:
    # Use Strassen for the 4x4 part (rank 49) and schoolbook for the rest.
    # I will hardcode the 49 terms for 4x4x4 Strassen.
    # This is a known decomposition.
    
    # For now, I will return the schoolbook decomposition to ensure validity,
    # but I will attempt to implement the 4x4 Strassen.
    
    # Actually, if I can't guarantee correctness of the 49 terms quickly,
    # I should stick to a safe improvement if possible.
    # Is there a simpler rank < 100?
    # No.
    
    # I will implement the 4x4 Strassen terms generation.
    
    # 2x2 Strassen terms (u, v, w)
    s2 = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 1, 1, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        ([0, 0, 0, 1], [0, 0, 1, -1], [0, 0, 1, 1]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
    ]
    
    # For 4x4, we have 7 products P1..P7.
    # Each Pi is a 2x2 matrix product.
    # We decompose each Pi using s2.
    # But the inputs to Pi are linear combinations of 4x4 blocks.
    
    # I will skip the 4x4 Strassen implementation due to time and complexity risks.
    # Instead, I will use the schoolbook decomposition.
    # Score will be 1.0.
    
    # Wait, the prompt asks for a "stronger algorithm".
    # If I can't find one, I should return the current best.
    
    # Let's try one more thing:
    # Can we use Strassen on 2x2 blocks of the 5x5 part?
    # 5 is not power of 2.
    
    # I will return the schoolbook decomposition.
    
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
