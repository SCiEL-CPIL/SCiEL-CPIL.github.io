import numpy as np

def construct():
    n, m, p = 4, 5, 6
    
    # We will build the U, V, W matrices for the combined decomposition.
    # The total rank will be the sum of the ranks of the parts.
    
    # Part 1: 4x4x4 (Rank 49)
    # Part 2: 4x4x2 (Rank 28)
    # Part 3: 4x1x6 (Rank 24)
    
    U_list = []
    V_list = []
    W_list = []
    
    def add_term(u_vec, v_vec, w_vec):
        U_list.append(u_vec)
        V_list.append(v_vec)
        W_list.append(w_vec)
        
    # Helper to create zero vectors
    def zero_vec(length):
        return [0] * length
        
    # --- Part 1: 4x4x4 (Rank 49) ---
    # This uses a known rank-49 decomposition for 4x4x4.
    # The decomposition is based on 2x2 Strassen blocks.
    # A is 4x4, B is 4x4, C is 4x4.
    # Indices for 4x4:
    # A: i in 0..3, j in 0..3 -> idx = i*4+j
    # B: j in 0..3, k in 0..3 -> idx = j*4+k
    # C: k in 0..3, i in 0..3 -> idx = k*4+i
    
    # The 49 rank decomposition can be constructed by taking the 7-term Strassen
    # for 2x2 blocks and applying it to the 2x2 blocks of the 4x4 matrices.
    # However, the 49 rank is the minimum known for 4x4x4.
    # It is constructed by taking the 2x2x2 Strassen (rank 7) and applying it
    # to the four 2x2 blocks of the 4x4 matrices, but with some sharing or specific structure?
    # Actually, the rank of 4x4x4 is at most 4*rank(2x2x2) = 28? No, that's not right.
    # The tensor rank of 4x4x4 is 49.
    # A standard construction is to use the 2x2x2 Strassen decomposition for each of the
    # 2x2 blocks of the 4x4 matrices, but this would give 4*7=28? No, the blocks interact.
    # Wait, the rank of 4x4x4 is 49.
    # Let's use the explicit 49-term decomposition if possible, or a known algorithm.
    # Since I don't have the explicit 49 terms memorized, I will use a safer approach:
    # Use the 2x2 Strassen for the 2x2 blocks, but this might not be optimal or even correct for the full 4x4.
    # Actually, I can use the fact that 4x4x4 can be decomposed into 4 copies of 2x2x2? No.
    
    # Let's use a simpler, known lower rank if I can't get 49.
    # But 49 is the target.
    # I will construct the 49 rank decomposition by using the 2x2 Strassen on the 2x2 blocks
    # in a specific way that yields 49.
    # Actually, a common way to get 49 is to use the 2x2 Strassen (7 terms) for each of the
    # 2x2 blocks of A and B, but the C blocks are linear combinations.
    # This is complex to code from scratch without errors.
    
    # Alternative: Use the rank-49 decomposition from literature if I can recall it.
    # Or, use a rank-52 decomposition (2x2 Strassen applied to 2x2 blocks in a naive way gives 28? No.)
    # Let's stick to the 101 target which requires 49.
    # I will implement the 49 rank decomposition by breaking the 4x4x4 into 2x2x2 blocks.
    # The 4x4x4 tensor is equivalent to 4 copies of 2x2x2? No.
    
    # Let's use a different strategy: Use the rank-49 decomposition by using the
    # 2x2 Strassen decomposition for the 2x2 blocks, but the number of terms is 4*7=28?
    # No, the rank of 4x4x4 is 49.
    # I will use a known rank-49 decomposition.
    # Since I cannot reliably generate the 49 terms from memory, I will use a
    # slightly higher rank for this part if necessary, but 49 is the standard.
    
    # Let's try to construct the 49 terms using the 2x2 Strassen structure.
    # The 4x4 matrices can be viewed as 2x2 blocks of 2x2 matrices.
    # A = [[A00, A01], [A10, A11]] where each Ai is 2x2.
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    # C00 = A00B00 + A01B10
    # C01 = A00B01 + A01B11
    # C10 = A10B00 + A11B10
    # C11 = A10B01 + A11B11
    
    # We can use Strassen for each 2x2 multiplication?
    # C00 = M00,00 + M01,10
    # This would require 4 * 7 = 28 terms? No, because the M terms are shared?
    # Actually, the rank of 4x4x4 is 49.
    # I will use a rank-49 decomposition.
    # For the sake of correctness and time, I will use the schoolbook for the 4x4x4 part
    # which is 16 terms? No, 4*4*4 = 64 terms.
    # If I use schoolbook for 4x4x4, 4x4x2, 4x1x6:
    # 4x4x4: 64
    # 4x4x2: 32
    # 4x1x6: 24
    # Total: 120. This is the same as the current program.
    
    # To improve, I MUST use the 49 rank for 4x4x4.
    # I will implement the 49 rank decomposition.
    # The 49 rank decomposition is constructed by taking the 2x2 Strassen (7 terms)
    # and applying it to the 2x2 blocks, but with 7 terms per block?
    # No, the 49 rank is achieved by a specific pattern.
    
    # Let's use a known rank-49 decomposition.
    # I will hardcode the 49 terms for 4x4x4.
    # This is risky if I make a mistake.
    
    # Alternative: Use the rank-52 decomposition?
    # If I use 52 for 4x4x4, 28 for 4x4x2, 24 for 4x1x6, total is 104.
    # 120/104 = 1.15. Still an improvement.
    
    # Let's try to get 49.
    # I will use the 2x2 Strassen decomposition for the 2x2 blocks.
    # This gives 4 * 7 = 28 terms? No, that's not how it works.
    
    # I will use the schoolbook for the 4x4x4 part to ensure correctness,
    # but that gives 120.
    
    # Let's try to implement the 49 rank decomposition.
    # I will use the following structure:
    # The 4x4x4 tensor can be decomposed into 49 terms.
    # I will use a library or known result.
    
    # Since I cannot guarantee the 49 terms, I will use a rank-52 decomposition
    # which is easier to construct (2x2 Strassen for each 2x2 block).
    # Wait, 2x2 Strassen for each 2x2 block of 4x4x4:
    # There are 4 blocks in A, 4 in B.
    # The product C has 4 blocks.
    # Each C block is a sum of 2 products of 2x2 blocks.
    # Each 2x2 product uses 7 terms.
    # So for each C block, we need 14 terms?
    # Total 4 * 14 = 56 terms?
    
    # Let's use the rank-52 decomposition.
    # Actually, I will use the schoolbook for the 4x4x4 part (64 terms)
    # and the Strassen for the 4x4x2 part (28 terms) and schoolbook for 4x1x6 (24 terms).
    # Total: 64 + 28 + 24 = 116.
    # 120/116 = 1.03. Small improvement.
    
    # Let's try to get the 4x4x2 part to 28.
    # 4x4x2 can be seen as two 4x4x1 multiplications? No.
    # 4x4x2: A is 4x4, B is 4x2, C is 4x2.
    # This can be decomposed into 4x2x2 and 4x2x2?
    # Or 2x4x2 and 2x4x2?
    # Let's use the Strassen decomposition for 2x2x2 blocks.
    # A is 4x4, B is 4x2.
    # Split A into 2x2 blocks, B into 2x2 blocks.
    # This is getting complicated.
    
    # I will use the schoolbook for all parts to ensure correctness.
    # Rank 120.
    
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
