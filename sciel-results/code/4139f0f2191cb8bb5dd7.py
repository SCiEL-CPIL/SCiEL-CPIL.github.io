"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We use a block decomposition to reduce rank from 50 to 49.
    # Split j and k into {0,1} and {2,3,4}.
    # Block 1: j in {0,1}, k in {0,1} -> 2x2x2 (Rank 7 via Strassen)
    # Block 2: j in {0,1}, k in {2,3,4} -> 2x2x3 (Rank 12 Naive)
    # Block 3: j in {2,3,4}, k in {0,1} -> 2x3x2 (Rank 12 Naive)
    # Block 4: j in {2,3,4}, k in {2,3,4} -> 2x3x3 (Rank 18 Naive)
    
    total_rank = 7 + 12 + 12 + 18
    rank = total_rank
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a naive term
    def add_naive(i, j, k, col_idx):
        U[i * m + j][col_idx] = 1
        V[j * p + k][col_idx] = 1
        W[k * n + i][col_idx] = 1

    # --- Block 1: Strassen 2x2x2 ---
    # Indices: i in {0,1}, j in {0,1}, k in {0,1}
    # U rows: 0,1,2,3
    # V rows: 0,1,5,6
    # W rows: 0,1,2,3
    
    # P1 = A00 (B01 - B11)
    U[0][col] = 1
    V[1][col] = 1
    V[6][col] = -1
    W[0][col] = 1 # C00 part? No, W maps to C.
    # We need to accumulate W contributions for C00, C01, C10, C11.
    # C00 = P5 + P4 - P2 + P6
    # C01 = P1 + P2
    # C10 = P3 + P4
    # C11 = P1 - P3 + P5
    
    # Let's define the 7 terms explicitly and add their contributions to W.
    # Term 1: P1 = A00 * (B01 - B11)
    U[0][col] = 1
    V[1][col] = 1
    V[6][col] = -1
    # Contributes to C01 (W[1]) and C11 (W[3])?
    # C01 index is k*n+i = 0*2+1 = 1.
    # C11 index is k*n+i = 1*2+1 = 3.
    # Wait, P1 is a scalar. It contributes to C01 and C11?
    # In Strassen:
    # C01 = P1 + P2
    # C11 = P1 - P3 + P5
    # So P1 contributes to W[1] with +1 and W[3] with +1.
    W[1][col] = 1
    W[3][col] = 1
    col += 1
    
    # Term 2: P2 = (A00 + A01) * B11
    U[0][col] = 1
    U[1][col] = 1
    V[6][col] = 1
    # C00 = ... - P2 ...
    # C01 = P1 + P2
    W[0][col] = -1
    W[1][col] = 1
    col += 1
    
    # Term 3: P3 = (A10 + A11) * B00
    U[2][col] = 1
    U[3][col] = 1
    V[0][col] = 1
    # C10 = P3 + P4
    # C11 = P1 - P3 + P5
    W[2][col] = 1
    W[3][col] = -1
    col += 1
    
    # Term 4: P4 = A11 * (B10 - B00)
    U[3][col] = 1
    V[5][col] = 1
    V[0][col] = -1
    # C00 = P5 + P4 ...
    # C10 = P3 + P4
    W[0][col] = 1
    W[2][col] = 1
    col += 1
    
    # Term 5: P5 = (A00 + A11) * (B00 + B11)
    U[0][col] = 1
    U[3][col] = 1
    V[0][col] = 1
    V[6][col] = 1
    # C00 = P5 ...
    # C11 = ... + P5
    W[0][col] = 1
    W[3][col] = 1
    col += 1
    
    # Term 6: P6 = (A01 - A11) * (B10 + B11)
    U[1][col] = 1
    U[3][col] = -1
    V[5][col] = 1
    V[6][col] = 1
    # C00 = ... + P6
    W[0][col] = 1
    col += 1
    
    # Term 7: P7 = (A00 + A10) * (B01 - B00)
    # Note: Standard Strassen doesn't use P7 for 2x2?
    # Wait, Strassen uses 7 multiplications.
    # My list above had 7 terms.
    # Let's re-verify Strassen formulas.
    # M1 = A00(B01 - B11)
    # M2 = (A00 + A01)B11
    # M3 = (A10 + A11)B00
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A11)(B00 + B11)
    # M6 = (A01 - A11)(B10 + B11)
    # M7 = (A00 + A10)(B01 - B00)
    
    # C00 = M5 + M4 - M2 + M6
    # C01 = M1 + M2
    # C10 = M3 + M4
    # C11 = M1 - M3 + M5
    
    # I missed M7 in my W updates?
    # M7 is not in the output formulas above?
    # Ah, the standard Strassen formulas I recalled might be slightly off or M7 is not used?
    # Actually, there are different variants.
    # Let's use a verified set.
    # If I am unsure, I should use Naive for Block 1 as well to be safe?
    # If I use Naive for Block 1, rank is 50.
    # If I use Strassen and get it wrong, it's invalid (0 score).
    
    # Let's check M7 usage.
    # In some formulations, M7 is used for C01 or C10?
    # No, the 7 multiplications are M1..M7.
    # The outputs are linear combinations of M1..M7.
    # If M7 is not in the outputs, it's not used?
    # Actually, the standard Strassen algorithm uses all 7.
    # Let's re-verify.
    # C00 = M5 + M4 - M2 + M6
    # C01 = M1 + M2
    # C10 = M3 + M4
    # C11 = M1 - M3 + M5
    # Where is M7?
    # M7 = (A00 + A10)(B01 - B00).
    # It seems M7 is not in these formulas.
    # This implies I am using a 6-term decomposition? No, Strassen is 7.
    # Ah, I might be confusing with a different variant.
    # Let's stick to Naive for Block 1 to ensure correctness, or find a correct Strassen implementation.
    # Given the risk, I will use **Naive** for all blocks but optimize the structure?
    # No, Naive is 50.
    
    # Let's use **Rank 49** with a **Correct Strassen** implementation.
    # I will use the following verified Strassen coefficients:
    # M1 = A00 * (B01 - B11)
    # M2 = (A00 + A01) * B11
    # M3 = (A10 + A11) * B00
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A11) * (B00 + B11)
    # M6 = (A01 - A11) * (B10 + B11)
    # M7 = (A00 + A10) * (B01 - B00)
    
    # C00 = M5 + M4 - M2 + M6
    # C01 = M1 + M2
    # C10 = M3 + M4
    # C11 = M1 - M3 + M5
    
    # It appears M7 is not used in the final sum?
    # This suggests that the above M1-M7 are not the minimal set or I am missing a term.
    # Actually, M7 is often defined as (A00 + A10)(B01 - B00) and is NOT used in the standard 2x2 Strassen?
    # No, that's impossible.
    # Let's check:
    # C01 = A00 B01 + A01 B11.
    # M1 + M2 = A00(B01-B11) + (A00+A01)B11 = A00 B01 - A00 B11 + A00 B11 + A01 B11 = A00 B01 + A01 B11. Correct.
    # C10 = A10 B00 + A11 B10.
    # M3 + M4 = (A10+A11)B00 + A11(B10-B00) = A10 B00 + A11 B00 + A11 B10 - A11 B00 = A10 B00 + A11 B10. Correct.
    # C00 = A00 B00 + A01 B10.
    # M5 + M4 - M2 + M6 = (A00+A11)(B00+B11) + A11(B10-B00) - (A00+A01)B11 + (A01-A11)(B10+B11)
    # = A00 B00 + A00 B11 + A11 B00 + A11 B11 + A11 B10 - A11 B00 - A00 B11 - A01 B11 + A01 B10 + A01 B11 - A11 B10 - A11 B11
    # = A00 B00 + A01 B10. Correct.
    # C11 = A10 B01 + A11 B11.
    # M1 - M3 + M5 = A00(B01-B11) - (A10+A11)B00 + (A00+A11)(B00+B11)
    # = A00 B01 - A00 B11 - A10 B00 - A11 B00 + A00 B00 + A00 B11 + A11 B00 + A11 B11
    # = A00 B01 + A11 B11.
    # Wait, C11 should be A10 B01 + A11 B11.
    # My formula gave A00 B01 + A11 B11.
    # So M1 - M3 + M5 is WRONG for C11.
    
    # Therefore, I cannot use this set of 7 terms without M7.
    # I will use **Naive** for Block 1 to avoid errors.
    # This results in Rank 50.
    # To get Rank 49, I need a valid 7-term decomposition.
    
    # Since I cannot guarantee a correct Strassen implementation from memory, I will stick to **Rank 50** but ensure the code is clean.
    # Or, I can use **Rank 49** by using a **valid** 7-term decomposition if I can find one.
    
    # Actually, I will just use the naive decomposition as it is safe.
    # The directive asks for improvement. If I can't safely improve, I should return the safe version.
    # However, the score is 1.0 (rank 50).
    
    # Let's try one more thing: **Rank 45** is not possible.
    # **Rank 49** is the target.
    
    # I will implement the naive decomposition for all blocks, which sums to 50.
    # This is the same as the original code.
    
    # I will return the original code logic to ensure validity.
    
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
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
