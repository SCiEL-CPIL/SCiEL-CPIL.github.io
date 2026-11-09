"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We split the p=8 dimension into two 3s and one 2.
    # Block 1: k in {0, 1, 2} -> 3x3x3 Schoolbook (Rank 27)
    # Block 2: k in {3, 4, 5} -> 3x3x3 Schoolbook (Rank 27)
    # Block 3: k in {6, 7}   -> 3x3x2 Rank 1 each (Rank 2)
    
    rank = 27 + 27 + 2
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Block 1: k=0,1,2
    for i in range(n):
        for j in range(m):
            for k in range(3):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # Block 2: k=3,4,5
    for i in range(n):
        for j in range(m):
            for k in range(3, 6):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # Block 3: k=6,7
    # For each k, we need a rank-1 term:
    # U = all 1s (sum over i,j of A_ij)
    # V = all 1s (sum over j,k of B_jk) -> wait, V is for B.
    # The term is (sum A_ij) * (sum B_jk) * C_ki?
    # No, for a fixed k, the contribution to C_ik is sum_j A_ij B_jk.
    # The rank 1 decomposition of this specific slice (fixed k) is:
    # U_vec = vector of all 1s for A? No.
    # Let's check the condition: sum_c W[k,i] U[i,j] V[j,k] = 1.
    # If we use one term c for this k:
    # W[k,i] = 1 for all i.
    # U[i,j] = 1 for all j?
    # V[j,k] = 1 for all j?
    # Then sum over j of 1*1*1 = m = 3.
    # We need the sum to be 1.
    # So we need to scale by 1/3?
    # But entries must be integer/half-integer.
    # 1/3 is not allowed.
    
    # So Rank 1 is NOT possible for 3x3x1 with integer/half-integer entries?
    # Let's re-evaluate.
    # We need sum_j U[i,j] V[j,k] W[k,i] = 1.
    # If W[k,i] = 1, then sum_j U[i,j] V[j,k] = 1.
    # If U[i,j] = 1 and V[j,k] = 1/3, sum is 3 * 1/3 = 1.
    # 1/3 is not half-integer.
    
    # Can we use Rank 3 for 3x3x1?
    # Schoolbook for 3x3x1 is Rank 9?
    # No, 3x3x1 is just 3 dot products.
    # C_i = sum_j A_ij B_j.
    # This is 3 independent terms?
    # Term 1: A_00 B_0 -> C_0. (U=1 at 00, V=1 at 0, W=1 at 0)
    # Term 2: A_01 B_1 -> C_0.
    # ...
    # Total 9 terms for one k.
    # So Rank 9 for 3x3x1.
    # Then Block 3 (k=6,7) costs 9 + 9 = 18.
    # Total Rank = 27 + 27 + 18 = 72.
    # This is back to Schoolbook.
    
    # Is there a better way for 3x3x1?
    # We need integer/half-integer.
    # Maybe we can combine k=6 and k=7?
    # 3x3x2.
    # C_i6 = sum_j A_ij B_j6
    # C_i7 = sum_j A_ij B_j7
    # We can use the same A factors.
    # U[i,j] is shared.
    # V[j,6] and V[j,7].
    # W[6,i] and W[7,i].
    # We need sum_j U[i,j] V[j,6] W[6,i] = 1.
    # And sum_j U[i,j] V[j,7] W[7,i] = 1.
    # If we set U[i,j] = 1 for all j, then sum_j V[j,6] W[6,i] = 1.
    # If V[j,6] = 1, W[6,i] = 1/3. Not allowed.
    # If we use 3 terms for the V/W part?
    # This seems hard to beat 72 without Laderman.
    
    # Let's stick to the simplest valid improvement if Laderman is too risky to code from memory.
    # Actually, I can just return the schoolbook 72 if I can't find a valid lower rank with integer constraints.
    # But 56 was my estimate. Why was it 56?
    # I assumed Rank 1 for 3x3x1. That was wrong due to integer constraint.
    
    # So, is there ANY improvement?
    # What if I use Strassen for 2x2x2 blocks?
    # 3x3x8.
    # Embed 3x3 in 4x4?
    # This is getting complicated.
    
    # Let's try to implement Laderman for 3x3x3.
    # If I can get Rank 19 for 3x3x3, then 2*19 + 18 = 56.
    # The 18 for 3x3x2 is likely minimal (9 per column).
    # So 56 is a good target.
    
    # I will implement Laderman.
    pass
# EVOLVE-BLOCK-END
