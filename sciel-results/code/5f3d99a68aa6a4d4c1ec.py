import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def _time_remaining(deadline):
    """Helper to check time remaining."""
    return deadline - time.time()

def construct():
    start_time = time.time()
    deadline = start_time + 120  # Hard deadline for safety
    
    n, m, p = DIMENSIONS
    
    # Dimensions
    dim_A = n * m  # 16
    dim_B = m * p  # 24
    dim_C = p * n  # 24
    
    # We will start with the schoolbook decomposition (Rank = n*m*p = 96)
    # Actually, schoolbook rank for n x m times m x p is n*m*p.
    # For 4x4 x 4x6, rank is 4*4*6 = 96.
    
    # Let's try to improve using known fast matrix multiplication schemes.
    # For 4x4, Strassen gives rank 49 (vs 64).
    # For 4x6, we can split into 4x4 and 4x2.
    
    # Strategy 1: Just use schoolbook to ensure a valid baseline.
    # Rank = 96.
    # Score = 72 / 96? No, score is seed/value. Seed is likely the schoolbook or a simple bound.
    # The prompt says "score is the seed/value".
    # Usually seed is the trivial solution rank.
    # Trivial rank for 4x4*4x6 is 4*4*6 = 96.
    # If I return 96, score = 96/96 = 1.0.
    # If I return less, score > 1.0.
    
    # Let's implement the 49 + 28 split mentioned in the previous code attempt.
    # Rank 49 for 4x4x4 core.
    # Rank 28 for 4x4 x 4x2 strip.
    # Total Rank = 77.
    # This was measured to give a score of 1.247 in a previous run (according to inspirations).
    # So 77 is a solid target.
    
    # Let's verify the validity of the previous 77-rank construction.
    # The previous code failed (score 0.0000). Why?
    # The inspiration 2 says: "The previous program failed because the `rank` variable was set to 63, but the logic inside the loop actually generates 77 columns... This ca [uses numpy, time...]"
    # Actually, the previous code I see in "CURRENT PROGRAM" has `rank = 49 + 28` (which is 77) and generates columns accordingly.
    # Wait, the CURRENT PROGRAM in the prompt is the one with score 0.0000.
    # Let's debug the CURRENT PROGRAM logic mentally.
    
    # The CURRENT PROGRAM logic:
    # Part 1: 4x4x4 using Strassen.
    # Part 2: 4x4x2 using 4 separate 2x2x1 Strassen? No, it uses 4 separate 2x2x2 Strassen blocks?
    # In Part 2, it does:
    # P1 = A00 * B0 (where B0 is 2x2 part of B? No, B is 4x2).
    # B is 4x2. B[0:2, :] is 2x2. B[2:4, :] is 2x2.
    # So B0 is 2x2, B1 is 2x2.
    # It decomposes A00 (2x2) * B0 (2x2) -> C0 (2x2) using Strassen (rank 7).
    # It decomposes A01 (2x2) * B1 (2x2) -> C0 (2x2) using Strassen (rank 7).
    # It decomposes A10 (2x2) * B0 (2x2) -> C1 (2x2) using Strassen (rank 7).
    # It decomposes A11 (2x2) * B1 (2x2) -> C1 (2x2) using Strassen (rank 7).
    # Total rank for strip = 4 * 7 = 28.
    # This seems correct.
    
    # So why did it fail?
    # Maybe an index error?
    # Let's check indices in Current Program.
    # A is 4x4. Index = row * 4 + col.
    # B is 4x2. Index = row * 2 + col.
    # C is 4x2. Index = row * 2 + col. (Output factor follows W[k*n+i, r] -> W[col*n + row, r] ?
    # "W[k*n+i,r]" where C is 4x2.
    # Standard C[i,j]. i in 0..3, j in 0..5.
    # Here output is 4x2. So i in 0..3 (rows of A), j in 0..1 (cols of B).
    # Wait, matrix multiply: A (4x4) * B (4x6) = C (4x6).
    # Task says: "multiplying a 4 by 4 matrix by a 4 by 6 matrix".
    # So A is 4x4, B is 4x6. C is 4x6.
    # BUT, the task description says: "output factor follows the official trace convention W[k*n+i,r]".
    # And shapes: U (n*m, rank), V (m*p, rank), W (p*n, rank).
    # n=4, m=4, p=6.
    # U size: 4*4 = 16.
    # V size: 4*6 = 24.
    # W size: 6*4 = 24.
    #
    # The tensor is T_{i, k, j} corresponding to C_{i, j} = sum_k A_{i, k} B_{k, j}.
    # U_{(i,k), r}, V_{(k,j), r}, W_{(j,i), r}.
    # Note the order in W: (p*n) means index is j*n + i? Or i*p + j?
    # "W[k*n+i,r]" suggests index = k*n + i.
    # If k is column index (0..5) and i is row index (0..3).
    # So W_{j*4 + i, r}.
    
    # Let's check the previous code's W indexing.
    # `W = np.zeros((p*n, rank))` -> size 24.
    # In Part 1 (4x4x4 part), C is 4x4. j is 0..3. i is 0..3.
    # Code: `global_idx = global_row * 4 + global_col`.
    # `global_row` corresponds to `br*2+r` which is row of A (i).
    # `global_col` corresponds to `bc*2+c` which is col of B (j).
    # So `global_idx` = i * 4 + j.
    # But the contract says W[k*n+i] = W[j*4+i].
    # So the previous code likely had W indexed as [i*4+j] instead of [j*4+i].
    # This would make the decomposition INVALID.
    
    # So, I need to construct a valid decomposition with the correct indexing.
    # And I should try to get a rank lower than 77 if possible, or at least ensure 77 works.
    # Rank 77 is known to be valid for 4x4*4x6 if constructed correctly.
    
    # Can we do better than 77?
    # 4x4x4 rank is 49 (Strassen).
    # 4x4x2 rank?
    # We can view 4x4x2 as (4x2) x (2x2) x (2x2)? No.
    # It's (4x4) x (4x2).
    # We can split A into 2x2 blocks.
    # A = [A00 A01; A10 A11].
    # B = [B0; B1] where B0 is 2x2, B1 is 2x2.
    # C = A * B = [A00 B0 + A01 B1; A10 B0 + A11 B1].
    # This is a sum of 4 products of 2x2 matrices.
    # Each product of 2x2 matrices has rank 7 (Strassen).
    # So total rank = 4 * 7 = 28.
    # Total rank = 49 + 28 = 77.
    
    # Is there a more efficient decomposition for 4x4x2?
    # Maybe combine some terms?
    # C_top = A00 B0 + A01 B1
    # C_bot = A10 B0 + A11 B1
    # This looks like (A00, A01) * [B0; B1] ?? No, dimensions don't match for simple block multiply.
    # A00 is 2x2, B0 is 2x2. Product is 2x2.
    # A01 is 2x2, B1 is 2x2. Product is 2x2.
    # Sum is 2x2.
    # These are independent products that need to be added.
    # If we just sum the decompositions, rank is sum of ranks.
    # Can we share some rank-1 components?
    # Unlikely to reduce significantly without a specific algebraic identity.
    # The bound for [4,4,2] is often cited as around 20-22?
    # Let's check known values.
    # R(4,4,1) = 16 (rank of vector). No, R(n, 1, 1) is 1. R(1,1,1)=1.
    # R(4,4,1) is the rank of the matrix multiplication tensor for 4x4 * 4x1.
    # This is just multiplying a 4x4 matrix by a 4x1 vector.
    # Rank is 16 (schoolbook) or can it be less?
    # For n x n * n x 1, rank is n^2.
    # So R(4,4,1) = 16.
    # For 4x4 * 4x2, we can view it as two 4x4 * 4x1 multiplications.
    # Rank <= 16 + 16 = 32.
    # My Strassen split gave 28. So 28 is better than 32.
    # Can we do better than 28?
    # Maybe 4x4x2 has rank 24?
    # If we can get rank 24 for the strip, total rank = 49 + 24 = 73.
    # That would be an improvement over 77.
    
    # Let's try to find a better decomposition for 4x4x2.
    # Or maybe split differently?
    # Split 4x4*4x6 into four 2x2*2x2 blocks?
    # A = [A00 A01 A02 A03; ...] No, 4x4.
    # A is 2x2 blocks of size 2x2.
    # B is 2x3 blocks of size 2x2? No, B is 4x6.
    # B rows 0-1, cols 0-1: B00.
    # B rows 0-1, cols 2-3: B01.
    # B rows 0-1, cols 4-5: B02.
    # B rows 2-3, cols 0-1: B10.
    # B rows 2-3, cols 2-3: B11.
    # B rows 2-3, cols 4-5: B12.
    
    # C = A * B.
    # C is 4x6.
    # C00 = A00 B00 + A01 B10
    # C01 = A00 B01 + A01 B11
    # C02 = A00 B02 + A01 B12
    # C10 = A10 B00 + A11 B10
    # C11 = A10 B01 + A11 B11
    # C12 = A10 B02 + A11 B12
    
    # Each block Ci_j is a sum of two 2x2x2 multiplications.
    # We can decompose each 2x2x2 multiplication using Strassen (rank 7).
    # But we have 12 terms here. 12 * 7 = 84.
    # Plus, we need to combine them.
    # However, we can share the A side?
    # A00 is used in C00, C01, C02.
    # A01 is used in C00, C01, C02.
    # A10 is used in C10, C11, C12.
    # A11 is used in C10, C11, C12.
    
    # Let's group by A blocks.
    # A00 * [B00, B01, B02] = [A00 B00, A00 B01, A00 B02]
    # This is A00 (2x2) * B_top (2x6).
    # Rank of 2x2 * 2x6?
    # We can split B_top into three 2x2 blocks.
    # Rank = 3 * R(2,2,2) = 3 * 7 = 21?
    # Or can we do better for 2x2 * 2x6?
    # B_top = [B00, B01, B02].
    # A00 B_top = [A00 B00, A00 B01, A00 B02].
    # We can use Strassen on each column pair?
    # Actually, 2x2 * 2x6 is equivalent to 3 multiplications of 2x2 * 2x2.
    # If we just sum the ranks, it's 21.
    # Same for A01, A10, A11.
    # Total rank = 4 * 21 = 84.
    # This is worse than 77.
    
    # The 77 strategy (49 + 28) is better.
    # 4x4x4 part: Rank 49.
    # 4x4x2 part: Rank 28.
    
    # Let's stick to 77 but ensure correctness.
    # And maybe try to optimize the 4x4x2 part.
    # Is there a known rank for 4x4x2?
    # A search might help, but I can't search.
    # I know that R(4,4,2) is at most 28 (as derived).
    # Is it lower?
    # R(2,2,2) = 7.
    # R(4,4,2) <= 4 * 7 = 28.
    # Could it be 24?
    # If we can combine the four 2x2x2 products:
    # P1 = A00 B0
    # P2 = A01 B1
    # P3 = A10 B0
    # P4 = A11 B1
    # C_top = P1 + P2
    # C_bot = P3 + P4
    # Note B0 is used in P1 and P3.
    # B1 is used in P2 and P4.
    # This structure suggests we might not save rank by sharing, because the products are distinct.
    # However, if we look at (A00 A01) * [B0; B1]? No.
    
    # Let's implement the 77 solution correctly.
    # Key: Correct W indexing.
    # W index = j * n + i = j * 4 + i.
    
    def get_strassen_2x2():
        U = np.zeros((4, 7), dtype=int)
        V = np.zeros((4, 7), dtype=int)
        W = np.zeros((4, 7), dtype=int)
        
        # M1 = (A00+A11)(B00+B11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2 = (A10+A11)B00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1
        
        # M3 = A00(B01-B11)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        
        # M4 = A11(B10-B00)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        # M5 = (A00+A01)B11
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # M6 = (A10-A00)(B00+B01)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # M7 = (A01-A11)(B10+B11)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W

    U2, V2, W2 = get_strassen_2x2()
    
    rank = 77
    U = np.zeros((16, rank), dtype=int)
    V = np.zeros((24, rank), dtype=int)
    W = np.zeros((24, rank), dtype=int)
    
    # --- Part 1: 4x4x4 (Rank 49) ---
    # Uses Strassen on 4x4 blocks.
    # L_coeffs and R_coeffs as in previous code.
    
    L_coeffs = [
        [1, 0, 0, 1], 
        [0, 0, 1, 1], 
        [1, 0, 0, 0], 
        [0, 0, 0, 1], 
        [1, 1, 0, 0], 
        [-1, 0, 1, 0],
        [0, 1, 0, -1] 
    ]
    
    R_coeffs = [
        [1, 0, 0, 1], 
        [1, 0, 0, 0], 
        [0, 1, 0, -1],
        [0, 0, 1, -1], 
        [0, 0, 0, 1], 
        [1, 1, 0, 0], 
        [0, 0, 1, 1]  
    ]
    
    C_coeffs = [
        [1, 0, 0, 1, -1, 0, 1], 
        [0, 0, 1, 0, 1, 0, 0],  
        [0, 1, 0, 1, 0, 0, 0],  
        [1, -1, 1, 0, 0, 1, 0]  
    ]
    
    for k in range(7):
        base_col = k * 7
        
        # U part (A 4x4)
        for block_idx in range(4):
            coeff = L_coeffs[k][block_idx]
            if coeff != 0:
                br, bc = divmod(block_idx, 2)
                for r in range(2):
                    for c in range(2):
                        global_row = br * 2 + r
                        global_col = bc * 2 + c
                        global_idx_A = global_row * 4 + global_col
                        local_idx = r * 2 + c
                        for s in range(7):
                            U[global_idx_A, base_col + s] += coeff * U2[local_idx, s]
        
        # V part (B 4x4 part)
        # B is 4x6. We are using B[:, 0:4] here.
        # B index = row * 6 + col.
        for block_idx in range(4):
            coeff = R_coeffs[k][block_idx]
            if coeff != 0:
                br, bc = divmod(block_idx, 2)
                for r in range(2):
                    for c in range(2):
                        global_row = br * 2 + r
                        global_col = bc * 2 + c
                        global_idx_B = global_row * 6 + global_col
                        local_idx = r * 2 + c
                        for s in range(7):
                            V[global_idx_B, base_col + s] += coeff * V2[local_idx, s]
        
        # W part (C 4x4 part)
        # C is 4x6. We are using C[:, 0:4] here.
        # W index = col * 4 + row.
        for c_block_idx in range(4):
            coeff = C_coeffs[c_block_idx][k]
            if coeff != 0:
                br, bc = divmod(c_block_idx, 2)
                for r in range(2):
                    for c in range(2):
                        global_row = br * 2 + r
                        global_col = bc * 2 + c
                        global_idx_W = global_col * 4 + global_row
                        local_idx = r * 2 + c
                        for s in range(7):
                            W[global_idx_W, base_col + s] += coeff * W2[local_idx, s]

    # --- Part 2: 4x4x2 (Rank 28) ---
    # B[:, 4:6] and C[:, 4:6].
    base_col = 49
    
    # Products:
    # P1 = A00 * B0 (B0 = B[0:2, 4:6])
    # P2 = A01 * B1 (B1 = B[2:4, 4:6])
    # P3 = A10 * B0
    # P4 = A11 * B1
    
    # U part
    # P1: A00
    for r in range(2):
        for c in range(2):
            global_idx_A = r * 4 + c
            local_idx = r * 2 + c
            for s in range(7):
                U[global_idx_A, base_col + s] += U2[local_idx, s]
                
    # P2: A01
    for r in range(2):
        for c in range(2):
            global_idx_A = r * 4 + (2 + c)
            local_idx = r * 2 + c
            for s in range(7):
                U[global_idx_A, base_col + 7 + s] += U2[local_idx, s]
                
    # P3: A10
    for r in range(2):
        for c in range(2):
            global_idx_A = (2 + r) * 4 + c
            local_idx = r * 2 + c
            for s in range(7):
                U[global_idx_A, base_col + 14 + s] += U2[local_idx, s]
                
    # P4: A11
    for r in range(2):
        for c in range(2):
            global_idx_A = (2 + r) * 4 + (2 + c)
            local_idx = r * 2 + c
            for s in range(7):
                U[global_idx_A, base_col + 21 + s] += U2[local_idx, s]

    # V part
    # P1 (B0 = B[0:2, 4:6])
    for r in range(2):
        for c in range(2):
            global_idx_B = r * 6 + (4 + c)
            local_idx = r * 2 + c
            for s in range(7):
                V[global_idx_B, base_col + s] += V2[local_idx, s]
                
    # P2 (B1 = B[2:4, 4:6])
    for r in range(2):
        for c in range(2):
            global_idx_B = (2 + r) * 6 + (4 + c)
            local_idx = r * 2 + c
            for s in range(7):
                V[global_idx_B, base_col + 7 + s] += V2[local_idx, s]
                
    # P3 (B0)
    for r in range(2):
        for c in range(2):
            global_idx_B = r * 6 + (4 + c)
            local_idx = r * 2 + c
            for s in range(7):
                V[global_idx_B, base_col + 14 + s] += V2[local_idx, s]
                
    # P4 (B1)
    for r in range(2):
        for c in range(2):
            global_idx_B = (2 + r) * 6 + (4 + c)
            local_idx = r * 2 + c
            for s in range(7):
                V[global_idx_B, base_col + 21 + s] += V2[local_idx, s]

    # W part
    # C[:, 4:6]. C is 4x2.
    # C0 = rows 0-1, cols 4-5.
    # C1 = rows 2-3, cols 4-5.
    # W index = col_global * 4 + row_global.
    # For C0: col_global = 4 or 5. row_global = 0 or 1.
    # For C1: col_global = 4 or 5. row_global = 2 or 3.
    
    # C0 = P1 + P2
    for r in range(2):
        for c in range(2):
            global_idx_W = (4 + c) * 4 + r
            local_idx = r * 2 + c
            # P1
            for s in range(7):
                W[global_idx_W, base_col + s] += W2[local_idx, s]
            # P2
            for s in range(7):
                W[global_idx_W, base_col + 7 + s] += W2[local_idx, s]
                
    # C1 = P3 + P4
    for r in range(2):
        for c in range(2):
            global_idx_W = (4 + c) * 4 + (2 + r)
            local_idx = r * 2 + c
            # P3
            for s in range(7):
                W[global_idx_W, base_col + 14 + s] += W2[local_idx, s]
            # P4
            for s in range(7):
                W[global_idx_W, base_col + 21 + s] += W2[local_idx, s]

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
    # Simple validation
    U = np.array(res['U'])
    V = np.array(res['V'])
    W = np.array(res['W'])
    
    # Check random A, B
    A = np.random.rand(4, 4)
    B = np.random.rand(4, 6)
    C_true = A @ B
    
    C_calc = np.zeros((4, 6))
    for i in range(4):
        for k in range(4):
            for j in range(6):
                val = 0
                for r in range(U.shape[1]):
                    val += U[i*4+k, r] * V[k*6+j, r] * W[j*4+i, r]
                C_calc[i, j] = val
                
    error = np.linalg.norm(C_true - C_calc)
    print("Error:", error)
    if error < 1e-9:
        print("VALID")
    else:
        print("INVALID")
