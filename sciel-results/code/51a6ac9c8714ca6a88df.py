"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We seek a rank-minimal decomposition for multiplying a 2x4 matrix by a 4x5 matrix.
    # The tensor is M_{2,4,5}.
    # A known upper bound is obtained by partitioning the middle dimension or the output.
    # Partitioning B (4x5) into two 4x2 and one 4x1 blocks:
    # C = A B1 + A B2 + A B3 where B1, B2 are 4x2 and B3 is 4x1.
    # A B1 is 2x2 (rank 7 via Strassen)
    # A B2 is 2x2 (rank 7 via Strassen)
    # A B3 is 2x1 (rank 4 via schoolbook)
    # Total rank = 7 + 7 + 4 = 18? No, A is 2x4.
    # Wait, A is 2x4. B is 4x5.
    # If we split B into B1 (4x2), B2 (4x2), B3 (4x1).
    # C1 = A B1 (2x2). A is 2x4. This is M_{2,4,2}.
    # M_{2,4,2} can be seen as two M_{2,2,1} and M_{2,2,1}? No.
    # A B1 = [A1 A2] [B1_top; B1_bot] where A1, A2 are 2x2, B1_top, B1_bot are 2x2.
    # C1 = A1 B1_top + A2 B1_bot.
    # Rank of A1 B1_top is 7 (Strassen). Rank of A2 B1_bot is 7.
    # So C1 has rank 14.
    # Similarly C2 has rank 14.
    # C3 = A B3 (2x1). A is 2x4, B3 is 4x1.
    # C3 = A1 B3_top + A2 B3_bot.
    # A1 B3_top is 2x1 (rank 4). A2 B3_bot is 2x1 (rank 4).
    # So C3 has rank 8.
    # Total rank = 14 + 14 + 8 = 36. This matches the current program.
    
    # Can we do better?
    # What if we split A instead?
    # A is 2x4. Split into A1 (2x2), A2 (2x2).
    # C = A1 B_top + A2 B_bot.
    # B_top is 2x5, B_bot is 2x5.
    # A1 B_top is 2x5. This is M_{2,2,5}.
    # M_{2,2,5} rank?
    # M_{2,2,3} has rank 10. M_{2,2,5} = M_{2,2,3} + M_{2,2,2}.
    # Rank(M_{2,2,5}) <= 10 + 7 = 17.
    # So C = A1 B_top + A2 B_bot would have rank <= 17 + 17 = 34.
    # This is an improvement over 36!
    
    # We need a rank 10 decomposition for M_{2,2,3}.
    # And a rank 7 decomposition for M_{2,2,2} (Strassen).
    # M_{2,2,5} can be decomposed by splitting the 5 columns into 3 and 2.
    # C = A B1 + A B2 where B1 is 2x3, B2 is 2x2.
    # A B1 is 2x3 (rank 10). A B2 is 2x2 (rank 7).
    # Total rank 17.
    
    # So the total rank for M_{2,4,5} is 2 * 17 = 34.
    
    rank = 34
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_2x2(a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset):
        nonlocal col
        # A is 2x2 at (a_row_offset, a_col_offset)
        # B is 2x2 at (b_row_offset, b_col_offset)
        # C is 2x2 at (c_row_offset, c_col_offset)
        
        u_idx_00 = a_row_offset * m + a_col_offset
        u_idx_01 = a_row_offset * m + a_col_offset + 1
        u_idx_10 = (a_row_offset + 1) * m + a_col_offset
        u_idx_11 = (a_row_offset + 1) * m + a_col_offset + 1
        
        v_idx_00 = b_row_offset * p + b_col_offset
        v_idx_01 = b_row_offset * p + b_col_offset + 1
        v_idx_10 = (b_row_offset + 1) * p + b_col_offset
        v_idx_11 = (b_row_offset + 1) * p + b_col_offset + 1
        
        w_idx_00 = c_col_offset * n + c_row_offset
        w_idx_01 = (c_col_offset + 1) * n + c_row_offset
        w_idx_10 = c_col_offset * n + c_row_offset + 1
        w_idx_11 = (c_col_offset + 1) * n + c_row_offset + 1
        
        # M1 = (a00+a11)(b00+b11)
        U[u_idx_00][col] += 1
        U[u_idx_11][col] += 1
        V[v_idx_00][col] += 1
        V[v_idx_11][col] += 1
        W[w_idx_00][col] += 1
        W[w_idx_11][col] += 1
        col += 1
        
        # M2 = (a10+a11)b00
        U[u_idx_10][col] += 1
        U[u_idx_11][col] += 1
        V[v_idx_00][col] += 1
        W[w_idx_10][col] += 1
        W[w_idx_11][col] -= 1
        col += 1
        
        # M3 = a00(b01-b11)
        U[u_idx_00][col] += 1
        V[v_idx_01][col] += 1
        V[v_idx_11][col] -= 1
        W[w_idx_01][col] += 1
        W[w_idx_11][col] += 1
        col += 1
        
        # M4 = a11(b10-b00)
        U[u_idx_11][col] += 1
        V[v_idx_10][col] += 1
        V[v_idx_00][col] -= 1
        W[w_idx_00][col] += 1
        W[w_idx_10][col] += 1
        col += 1
        
        # M5 = (a00+a01)b11
        U[u_idx_00][col] += 1
        U[u_idx_01][col] += 1
        V[v_idx_11][col] += 1
        W[w_idx_00][col] -= 1
        W[w_idx_01][col] += 1
        col += 1
        
        # M6 = (a10-a00)(b00+b01)
        U[u_idx_10][col] += 1
        U[u_idx_00][col] -= 1
        V[v_idx_00][col] += 1
        V[v_idx_01][col] += 1
        W[w_idx_11][col] += 1
        col += 1
        
        # M7 = (a01-a11)(b10+b11)
        U[u_idx_01][col] += 1
        U[u_idx_11][col] -= 1
        V[v_idx_10][col] += 1
        V[v_idx_11][col] += 1
        W[w_idx_00][col] += 1
        col += 1

    def add_m223_2x2x3(a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset):
        nonlocal col
        # A is 2x2 at (a_row_offset, a_col_offset)
        # B is 2x3 at (b_row_offset, b_col_offset)
        # C is 2x3 at (c_row_offset, c_col_offset)
        # We use a rank 10 decomposition for M_{2,2,3}.
        # Reference: "Rank of matrix multiplication" or similar.
        # One known rank 10 decomposition:
        # Let A = [[a, b], [c, d]], B = [[e, f, g], [h, i, j]]
        # C = [[ae+ch, af+ci, ag+cj], [be+dh, bf+di, bg+dj]]
        
        # Terms:
        # 1. (a+d)(e+h) -> C00, C11 (partial)
        # This is tricky. Let's use a standard rank 10 scheme.
        # From "A rank 10 decomposition of M_{2,2,3}":
        # The terms are linear combinations.
        
        u_idx_00 = a_row_offset * m + a_col_offset
        u_idx_01 = a_row_offset * m + a_col_offset + 1
        u_idx_10 = (a_row_offset + 1) * m + a_col_offset
        u_idx_11 = (a_row_offset + 1) * m + a_col_offset + 1
        
        v_idx_00 = b_row_offset * p + b_col_offset
        v_idx_01 = b_row_offset * p + b_col_offset + 1
        v_idx_02 = b_row_offset * p + b_col_offset + 2
        v_idx_10 = (b_row_offset + 1) * p + b_col_offset
        v_idx_11 = (b_row_offset + 1) * p + b_col_offset + 1
        v_idx_12 = (b_row_offset + 1) * p + b_col_offset + 2
        
        w_idx_00 = c_col_offset * n + c_row_offset
        w_idx_01 = (c_col_offset + 1) * n + c_row_offset
        w_idx_02 = (c_col_offset + 2) * n + c_row_offset
        w_idx_10 = c_col_offset * n + c_row_offset + 1
        w_idx_11 = (c_col_offset + 1) * n + c_row_offset + 1
        w_idx_12 = (c_col_offset + 2) * n + c_row_offset + 1
        
        # We will implement a specific rank 10 decomposition.
        # If unsure, we can fall back to rank 11 or higher, but we want 10.
        # A common rank 10 decomposition uses the following 10 products:
        # 1. (a+b)(e+h)
        # 2. (a-b)(e-h)
        # 3. (c+d)(e+h)
        # 4. (c-d)(e-h)
        # 5. (a+b)(f+i)
        # 6. (a-b)(f-i)
        # 7. (c+d)(f+i)
        # 8. (c-d)(f-i)
        # 9. (a+b)(g+j)
        # 10. (c+d)(g+j)
        # This is not quite right.
        
        # Let's use the decomposition from "Matrix multiplication rank 10 for 2x2x3"
        # Actually, let's just use Strassen for 2x2x2 and schoolbook for the 3rd column?
        # M_{2,2,3} = M_{2,2,2} + M_{2,2,1}.
        # Rank(M_{2,2,2}) = 7. Rank(M_{2,2,1}) = 4.
        # Total 11.
        # If we use rank 11, total rank is 2 * (11 + 7) = 36. No improvement.
        # We MUST find a rank 10 decomposition.
        
        # Here is a known rank 10 decomposition for 2x2x3:
        # It is derived from the 2x2x2 Strassen and adding one term cleverly.
        # Or using the fact that M_{2,2,3} is a sub-tensor of M_{2,3,3} (rank 15?) no.
        
        # Let's try a different split.
        # Maybe M_{2,4,5} can be done in rank 35?
        # If we use rank 11 for M_{2,2,3}, total is 36.
        # If we use rank 10, total is 34.
        
        # I will implement a rank 10 decomposition based on:
        # "A note on the rank of matrix multiplication"
        # The terms are:
        # t1 = (a00+a11)(b00+b10)
        # t2 = (a00-a11)(b00-b10)
        # t3 = (a01+a11)(b01+b11)
        # t4 = (a01-a11)(b01-b11)
        # t5 = (a00+a01)(b10+b11)
        # t6 = (a00-a01)(b10-b11)
        # t7 = (a10+a11)(b00+b01)
        # t8 = (a10-a11)(b00-b01)
        # t9 = (a00+a10)(b10+b11)
        # t10 = (a01+a10)(b00+b11)
        # This is a guess. It might not be correct.
        
        # To be safe and ensure correctness, I will stick to the rank 36 solution
        # unless I am certain of the rank 10 coefficients.
        # Given the risk, I will output the rank 36 solution.
        
        # However, the prompt asks for a "stronger algorithm".
        # If I cannot provide rank 34, I should provide rank 36.
        
        # Let's just return the rank 36 solution as it is known to be correct.
        pass

    # Since I am not 100% sure of the rank 10 coefficients, I will revert to the rank 36 implementation
    # to ensure a valid score.
    
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    def add_strassen(a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset):
        nonlocal col
        u_idx_00 = a_row_offset * m + a_col_offset
        u_idx_01 = a_row_offset * m + a_col_offset + 1
        u_idx_10 = (a_row_offset + 1) * m + a_col_offset
        u_idx_11 = (a_row_offset + 1) * m + a_col_offset + 1
        
        v_idx_00 = b_row_offset * p + b_col_offset
        v_idx_01 = b_row_offset * p + b_col_offset + 1
        v_idx_10 = (b_row_offset + 1) * p + b_col_offset
        v_idx_11 = (b_row_offset + 1) * p + b_col_offset + 1
        
        w_idx_00 = c_col_offset * n + c_row_offset
        w_idx_01 = (c_col_offset + 1) * n + c_row_offset
        w_idx_10 = c_col_offset * n + c_row_offset + 1
        w_idx_11 = (c_col_offset + 1) * n + c_row_offset + 1
        
        U[u_idx_00][col] += 1; U[u_idx_11][col] += 1
        V[v_idx_00][col] += 1; V[v_idx_11][col] += 1
        W[w_idx_00][col] += 1; W[w_idx_11][col] += 1
        col += 1
        
        U[u_idx_10][col] += 1; U[u_idx_11][col] += 1
        V[v_idx_00][col] += 1
        W[w_idx_10][col] += 1; W[w_idx_11][col] -= 1
        col += 1
        
        U[u_idx_00][col] += 1
        V[v_idx_01][col] += 1; V[v_idx_11][col] -= 1
        W[w_idx_01][col] += 1; W[w_idx_11][col] += 1
        col += 1
        
        U[u_idx_11][col] += 1
        V[v_idx_10][col] += 1; V[v_idx_00][col] -= 1
        W[w_idx_00][col] += 1; W[w_idx_10][col] += 1
        col += 1
        
        U[u_idx_00][col] += 1; U[u_idx_01][col] += 1
        V[v_idx_11][col] += 1
        W[w_idx_00][col] -= 1; W[w_idx_01][col] += 1
        col += 1
        
        U[u_idx_10][col] += 1; U[u_idx_00][col] -= 1
        V[v_idx_00][col] += 1; V[v_idx_01][col] += 1
        W[w_idx_11][col] += 1
        col += 1
        
        U[u_idx_01][col] += 1; U[u_idx_11][col] -= 1
        V[v_idx_10][col] += 1; V[v_idx_11][col] += 1
        W[w_idx_00][col] += 1
        col += 1

    def add_schoolbook_2x1(a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset):
        nonlocal col
        u_idx_00 = a_row_offset * m + a_col_offset
        u_idx_01 = a_row_offset * m + a_col_offset + 1
        u_idx_10 = (a_row_offset + 1) * m + a_col_offset
        u_idx_11 = (a_row_offset + 1) * m + a_col_offset + 1
        
        v_idx_0 = b_row_offset * p + b_col_offset
        v_idx_1 = (b_row_offset + 1) * p + b_col_offset
        
        w_idx_0 = c_col_offset * n + c_row_offset
        w_idx_1 = c_col_offset * n + c_row_offset + 1
        
        U[u_idx_00][col] = 1; V[v_idx_0][col] = 1; W[w_idx_0][col] = 1; col += 1
        U[u_idx_01][col] = 1; V[v_idx_1][col] = 1; W[w_idx_0][col] = 1; col += 1
        U[u_idx_10][col] = 1; V[v_idx_0][col] = 1; W[w_idx_1][col] = 1; col += 1
        U[u_idx_11][col] = 1; V[v_idx_1][col] = 1; W[w_idx_1][col] = 1; col += 1

    add_strassen(0, 0, 0, 0, 0, 0)
    add_strassen(0, 2, 2, 0, 0, 0)
    add_schoolbook_2x1(0, 0, 0, 2, 0, 2)
    add_schoolbook_2x1(0, 2, 2, 2, 0, 2)
    add_strassen(0, 0, 0, 3, 0, 3)
    add_strassen(0, 2, 2, 3, 0, 3)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
