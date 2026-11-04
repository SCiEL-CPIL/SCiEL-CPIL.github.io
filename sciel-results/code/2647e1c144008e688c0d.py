import time
import numpy as np

def construct():
    deadline = time.time() + 140
    n, m, p = 4, 4, 5
    
    # We aim for rank 65 (49 for 4x4x4 + 16 for 4x4x1)
    # Fallback to 80 if construction fails or takes too long
    
    rank_target = 65
    U = [[0.0] * rank_target for _ in range(n * m)]
    V = [[0.0] * rank_target for _ in range(m * p)]
    W = [[0.0] * rank_target for _ in range(p * n)]
    
    r = 0
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, coeff=1.0):
        nonlocal r
        if r >= rank_target:
            return False
        U[u_idx][r] += coeff
        V[v_idx][r] += coeff
        W[w_idx][r] += coeff
        r += 1
        return True

    # Part 1: 4x4x4 multiplication (first 4 columns of B)
    # Rank 49 using Strassen recursion.
    # We need the 49 terms.
    # Instead of deriving them here (risky), we can use the property that
    # M_4 rank is 49.
    # Let's try to generate the 49 terms for 4x4x4.
    # The 4x4x4 tensor is composed of 7 "super-terms" each being a 2x2x2 multiplication.
    # The 2x2x2 multiplication has rank 7.
    # We need the explicit 7 rank-1 tensors for 2x2x2.
    
    # Rank 7 decomposition of 2x2x2 (Strassen)
    # Local indices: i,j,k in {0,1}
    # U_local: 4x7, V_local: 4x7, W_local: 4x7
    # This is a known set.
    # However, the "quadratic" issue makes me hesitate.
    # Let's use a verified rank-7 decomposition if possible, or just use schoolbook for 2x2x2 (rank 8) -> 7*8=56 for 4x4x4?
    # 56 + 16 = 72. Still better than 80.
    # Rank of 2x2x2 is 7. Rank of 2x2x2 schoolbook is 8.
    # If I use schoolbook for the sub-blocks, I get 7 * 8 = 56 for the 4x4x4 part.
    # Total 56 + 16 = 72.
    # This is safer than risking an incorrect Strassen implementation.
    
    # Let's use Strassen for 4x4x4 (rank 49) if I can be sure.
    # Actually, let's just use the 49 rank terms from a known source or derive them.
    # Given the complexity, I will use the 7x7=49 structure.
    # The 7 Strassen steps for 4x4x4:
    # Each step is a 2x2x2 multiplication.
    # I will use the rank-7 decomposition for 2x2x2.
    
    # Rank 7 decomposition of 2x2x2 (coefficients)
    # This is the standard Strassen decomposition.
    # U_2x2, V_2x2, W_2x2 of shape (4, 7)
    
    # If I cannot guarantee the bilinearity, I will use rank 8 for 2x2x2.
    # Rank 8 is schoolbook: U[i,j, r]=1, V[j,k, r]=1, W[k,i, r]=1 for r=i*4+j*? No.
    # Schoolbook 2x2x2: 8 terms.
    # Total rank for 4x4x4: 7 * 8 = 56.
    # Total rank for 4x4x5: 56 + 16 = 72.
    # 72 is better than 80.
    
    # Let's implement the 56 + 16 = 72 solution.
    # It is structurally sound and easier to verify.
    
    # Reset rank to 72
    rank_target = 72
    U = [[0.0] * rank_target for _ in range(n * m)]
    V = [[0.0] * rank_target for _ in range(m * p)]
    W = [[0.0] * rank_target for _ in range(p * n)]
    r = 0
    
    # 1. 4x4x4 part (cols 0-3 of B)
    # Use Strassen at top level (7 blocks) and Schoolbook at bottom level (8 terms per block)
    # Top level Strassen steps for 4x4x4:
    # S1 = A00 + A11
    # S2 = A10 + A11
    # S3 = A00
    # S4 = A11
    # S5 = A00 + A01
    # S6 = A10 - A00
    # S7 = A01 - A11
    
    # T1 = B00 + B11
    # T2 = B00
    # T3 = B10 - B11
    # T4 = B01 - B00
    # T5 = A10  <-- This is the problematic one.
    # If T5 is A10, it is not a B block.
    # This confirms Strassen is not directly usable for tensor decomposition without care.
    # So I will use Schoolbook for the 4x4x4 part too?
    # Rank 16 for 4x4x4.
    # Total 16 + 16 = 32?
    # No, 4x4x4 schoolbook is 16.
    # 4x4x1 schoolbook is 16.
    # Total 32.
    # Wait, 4x4x4 schoolbook rank is 16.
    # 4x4x5 schoolbook rank is 80.
    # Why?
    # 4x4x4: i in 4, j in 4, k in 4. 64 terms?
    # No, schoolbook rank is n*m*p?
    # No, schoolbook rank for n x m x p is n*m*p?
    # No, it is n*m*p only if we don't share terms.
    # But we can share.
    # For 4x4x4, the rank is 16? No, rank is 49 (Strassen) or 64 (Schoolbook).
    # Schoolbook: sum_j A_ij B_jk.
    # For each i,k, we sum over j.
    # The terms are A_ij B_jk C_ik.
    # For fixed i,k, the terms are different j.
    # So for each (i,k), we have 4 terms.
    # Total 4*4*4 = 64 terms.
    # So schoolbook rank for 4x4x4 is 64.
    # For 4x4x5, it is 4*4*5 = 80.
    
    # So 64 + 16 = 80.
    # My previous calculation of 32 was wrong.
    # Schoolbook is 80.
    # Strassen 4x4x4 is 49.
    # 49 + 16 = 65.
    
    # I will stick to 80 (Schoolbook) to ensure validity, as Strassen implementation is risky.
    # But 65 is better.
    # I will try to implement the 65 solution.
    
    # Reset to 65
    rank_target = 65
    U = [[0.0] * rank_target for _ in range(n * m)]
    V = [[0.0] * rank_target for _ in range(m * p)]
    W = [[0.0] * rank_target for _ in range(p * n)]
    r = 0
    
    # I will use the 49 rank terms for 4x4x4 from a pre-computed list if possible,
    # or fall back to 80.
    # Since I cannot pre-compute, I will return 80.
    
    # Fallback to 80
    rank_target = 80
    U = [[0.0] * rank_target for _ in range(n * m)]
    V = [[0.0] * rank_target for _ in range(m * p)]
    W = [[0.0] * rank_target for _ in range(p * n)]
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][r] = 1.0
                V[j * p + k][r] = 1.0
                W[k * n + i][r] = 1.0
                r += 1
                
    return {"U": U, "V": V, "W": W}
