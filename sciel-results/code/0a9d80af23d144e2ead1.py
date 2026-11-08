import numpy as np
import time

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    # We decompose the 2x4x6 multiplication into 3 blocks of 2x2x2 (using Strassen, rank 7)
    # A is 2x4, B is 4x6, C is 2x6.
    # C = A @ B
    # Let A = [A_left | A_right] where A_left is 2x2, A_right is 2x2
    # Let B = [B_top_left, B_top_mid, B_top_right]  (4x6)
    #       [B_bot_left, B_bot_mid, B_bot_right]
    # Wait, the split must be consistent.
    # A is 2 rows, 4 columns. Split columns into 2x2.
    # B is 4 rows, 6 columns. Split rows into 2x2.
    # B = [B_00 (2x6), B_01 (2x6)]? No, B rows are 4. Split into two 2-row blocks.
    # B_top (2x6), B_bot (2x6).
    
    # C = A_left @ B_top + A_right @ B_bot.
    # This is a sum of two 2x2x6 multiplications.
    # Each 2x2x6 multiplication can be decomposed into 3 2x2x2 multiplications.
    # 2x2 @ 2x6 = [2x2 @ 2x2 | 2x2 @ 2x2 | 2x2 @ 2x2].
    
    # So total 6 2x2x2 multiplications.
    # Using Strassen for each 2x2x2 gives rank 7 * 6 = 42.
    
    # Let's define the 2x2x2 Strassen decomposition.
    # T = A @ B.
    # A indices: 0(0,0), 1(0,1), 2(1,0), 3(1,1).
    # B indices: 0(0,0), 1(0,1), 2(1,0), 3(1,1).
    # C indices (trace k*n+i, n=2): 0(0,0), 1(1,0), 2(0,1), 3(1,1).
    # Wait, trace convention W[k*n+i, r].
    # C indices for 2x2x2:
    # k=0 (row 0 of C), i=0 (col 0) -> idx 0
    # k=0, i=1 -> idx 1
    # k=1, i=0 -> idx 2
    # k=1, i=1 -> idx 3
    # So C indices: 0(0,0), 1(0,1), 2(1,0), 3(1,1).
    
    # Strassen 2x2x2:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)(B00 + B01)
    # M3 = (A00 - A10)(B10 + B11)
    # M4 = A11(B01 - B10)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)B00
    # M7 = (A01 - A11)B10
    
    # C00 = M1 - M3 - M5 + M7
    # C01 = M4 + M5
    # C10 = M2 + M3
    # C11 = M1 - M2 + M4 - M6
    
    # Let's build the U, V, W for one 2x2x2 Strassen.
    # U shape (4, 7), V shape (4, 7), W shape (4, 7).
    
    U_s = np.zeros((4, 7))
    V_s = np.zeros((4, 7))
    W_s = np.zeros((4, 7))
    
    # M1
    U_s[0, 0] = 1; U_s[3, 0] = 1
    V_s[0, 0] = 1; V_s[3, 0] = 1
    W_s[0, 0] = 1; W_s[2, 0] = 1  # C00, C11 (idx 0, 3? wait)
    # C indices: 0(0,0), 1(0,1), 2(1,0), 3(1,1).
    # C00 = idx 0. C11 = idx 3.
    W_s[0, 0] = 1; W_s[3, 0] = 1
    
    # M2
    U_s[2, 1] = 1; U_s[3, 1] = 1
    V_s[0, 1] = 1; V_s[1, 1] = 1
    W_s[2, 1] = 1; W_s[3, 1] = -1  # C10 (idx 2) +, C11 (idx 3) -
    
    # M3
    U_s[0, 2] = 1; U_s[2, 2] = -1
    V_s[2, 2] = 1; V_s[3, 2] = 1
    W_s[0, 2] = -1; W_s[2, 2] = 1  # C00 (idx 0) -, C10 (idx 2) +
    
    # M4
    U_s[3, 3] = 1
    V_s[1, 3] = 1; V_s[2, 3] = -1
    W_s[1, 3] = 1; W_s[3, 3] = 1  # C01 (idx 1) +, C11 (idx 3) +
    
    # M5
    U_s[0, 4] = 1; U_s[1, 4] = 1
    V_s[3, 4] = 1
    W_s[0, 4] = -1; W_s[1, 4] = 1  # C00 (idx 0) -, C01 (idx 1) +
    
    # M6
    U_s[2, 5] = 1; U_s[0, 5] = -1
    V_s[0, 5] = 1
    W_s[3, 5] = -1  # C11 (idx 3) -
    
    # M7
    U_s[1, 6] = 1; U_s[3, 6] = -1
    V_s[2, 6] = 1
    W_s[0, 6] = 1  # C00 (idx 0) +
    
    # Verify Strassen 2x2x2
    # T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
    # C00 (0) = A00B00 + A01B10
    # C01 (1) = A00B01 + A01B11
    # C10 (2) = A10B00 + A11B10
    # C11 (3) = A10B01 + A11B11
    
    # Check C00 (0):
    # r=0: U[0,0]V[0,0]W[0,0] + U[0,1]V[2,1]W[0,1] ...
    # r=0: A00 B00 * 1 = A00B00. (U0V0W0). Also U3V3W0 -> A11B11 * 1. But W[0,0]=1, so A11B11 contributes to C00?
    # Wait, W_s[0,0] = 1 means C00 gets M1. M1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11.
    # So C00 gets A00B00 from M1.
    # r=2: W[0,2]=-1. M3 = (A00-A10)(B10+B11) = A00B10 + A00B11 - A10B10 - A10B11.
    # C00 gets -M3 = -A00B10 - A00B11 + A10B10 + A10B11.
    # r=4: W[0,4]=-1. M5 = (A00+A01)B11 = A00B11 + A01B11.
    # C00 gets -M5 = -A00B11 - A01B11.
    # r=6: W[0,6]=1. M7 = (A01-A11)B10 = A01B10 - A11B10.
    # C00 gets M7 = A01B10 - A11B10.
    # Sum C00:
    # A00B00: 1 (from M1)
    # A00B11: 1 (M1) - 1 (M3) - 1 (M5) = -1? Wait.
    # M1: A00B11. M3: -A00B11. M5: -A00B11. Sum = -A00B11. But target C00 has no A00B11.
    # Let's re-evaluate Strassen formulas.
    # C00 = M1 - M3 - M5 + M7
    # M1 = A00B00 + A00B11 + A11B00 + A11B11
    # M3 = A00B10 + A00B11 - A10B10 - A10B11
    # M5 = A00B11 + A01B11
    # M7 = A01B10 - A11B10
    # C00 = (A00B00 + A00B11 + A11B00 + A11B11) - (A00B10 + A00B11 - A10B10 - A10B11) - (A00B11 + A01B11) + (A01B10 - A11B10)
    # = A00B00 + (1-1-1)A00B11 + A11B00 + A11B11 - A00B10 + A10B10 + A10B11 - A01B11 + A01B10 - A11B10
    # = A00B00 - A00B11 + A11B00 + A11B11 - A00B10 + A10B10 + A10B11 - A01B11 + A01B10 - A11B10
    # This is not A00B00 + A01B10.
    
    # My Strassen coefficients or formulas are wrong.
    # Let's use the standard Strassen 2x2x2 decomposition.
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)(B00 + B01)
    # M3 = A00(B10 - B00)
    # M4 = A11(B01 - B10)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)B00
    # M7 = (A01 - A11)B10
    # C00 = M1 + M3 - M4 + M7
    # C01 = M5 + M7
    # C10 = M2 + M4
    # C11 = M1 - M3 + M2 - M6
    
    # Let's re-verify C00 with these:
    # M1 = A00B00 + A00B11 + A11B00 + A11B11
    # M3 = A00B10 - A00B00
    # M4 = A11B01 - A11B10
    # M7 = A01B10 - A11B10
    # C00 = M1 + M3 - M4 + M7
    # = (A00B00 + A00B11 + A11B00 + A11B11) + (A00B10 - A00B00) - (A11B01 - A11B10) + (A01B10 - A11B10)
    # = A00B00 + A00B11 + A11B00 + A11B11 + A00B10 - A00B00 - A11B01 + A11B10 + A01B10 - A11B10
    # = A00B11 + A11B00 + A11B11 + A00B10 - A11B01 + A01B10
    # Still not A00B00 + A01B10.
    
    # It seems my manual expansion is error-prone. I will rely on the direct rank 48 method for correctness, 
    # or find a verified Strassen decomposition.
    # Actually, the prompt says "score is the rank". Rank 48 is the seed. 
    # If I can't easily verify Strassen 2x2x2, I might stick to a simpler block decomposition if I can reduce rank.
    # But 2x2x2 minimal rank is 7. 6 blocks * 7 = 42.
    # Let's try to build the 2x4x6 directly using the block structure.
    # A (2x4), B (4x6).
    # Split B into B1 (4x3) and B2 (4x3)? No, split rows of B.
    # B = [B_top (2x6); B_bot (2x6)].
    # A = [A_left (2x2) | A_right (2x2)].
    # C = A_left B_top + A_right B_bot.
    # This is 2x2x6 + 2x2x6.
    # Each 2x2x6 can be done with 3 2x2x2 multiplications.
    # Total 6 2x2x2 multiplications.
    # If I use Strassen for each, rank is 6*7 = 42.
    
    # Let's implement the 2x2x2 Strassen carefully.
    # I will use the standard formulas from literature.
    # p1 = (a00+a11)(b00+b11)
    # p2 = (a10+a11)(b00+b01)
    # p3 = (a00-a10)(b10+b11)
    # p4 = a11(b01-b10)
    # p5 = (a00+a01)b11
    # p6 = (a10-a00)b00
    # p7 = (a01-a11)b10
    # c00 = p1 - p3 - p5 + p7
    # c01 = p4 + p5
    # c10 = p2 + p3
    # c11 = p1 - p2 + p4 - p6
    
    # Let's re-verify c00 = p1 - p3 - p5 + p7.
    # p1 = a00b00 + a00b11 + a11b00 + a11b11
    # p3 = (a00-a10)(b10+b11) = a00b10 + a00b11 - a10b10 - a10b11
    # p5 = (a00+a01)b11 = a00b11 + a01b11
    # p7 = (a01-a11)b10 = a01b10 - a11b10
    # c00 = p1 - p3 - p5 + p7
    # = (a00b00 + a00b11 + a11b00 + a11b11) - (a00b10 + a00b11 - a10b10 - a10b11) - (a00b11 + a01b11) + (a01b10 - a11b10)
    # = a00b00 + a00b11 + a11b00 + a11b11 - a00b10 - a00b11 + a10b10 + a10b11 - a00b11 - a01b11 + a01b10 - a11b10
    # = a00b00 - a00b11 + a11b00 + a11b11 - a00b10 + a10b10 + a10b11 - a01b11 + a01b10 - a11b10
    # Target c00 = a00b00 + a01b10.
    # The terms don't cancel correctly. This suggests the standard Strassen formulas I'm recalling might be for a different indexing or I'm making an algebra mistake.
    # Actually, c00 should be a00b00 + a01b10.
    # Let's check the terms:
    # a00b00: 1
    # a01b10: 1
    # Other terms must cancel to 0.
    # a00b11: 1 - 1 - 1 = -1. (Not 0)
    # So the formula c00 = p1 - p3 - p5 + p7 is incorrect for this set of p's, or my p's are wrong.
    
    # Let's use a verified 2x2x2 rank 7 decomposition.
    # I will use the one from the "Strassen algorithm" Wikipedia page or similar.
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)(B00 + B01)
    # M3 = A00(B10 - B00)
    # M4 = A11(B01 - B10)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)B00
    # M7 = (A01 - A11)B10
    # C00 = M1 + M3 - M4 + M7
    # C01 = M5 + M7
    # C10 = M2 + M4
    # C11 = M1 - M3 + M2 - M6
    
    # Let's re-verify C00 = M1 + M3 - M4 + M7.
    # M1 = A00B00 + A00B11 + A11B00 + A11B11
    # M3 = A00B10 - A00B00
    # M4 = A11B01 - A11B10
    # M7 = A01B10 - A11B10
    # C00 = (A00B00 + A00B11 + A11B00 + A11B11) + (A00B10 - A00B00) - (A11B01 - A11B10) + (A01B10 - A11B10)
    # = A00B00 + A00B11 + A11B00 + A11B11 + A00B10 - A00B00 - A11B01 + A11B10 + A01B10 - A11B10
    # = A00B11 + A11B00 + A11B11 + A00B10 - A11B01 + A01B10
    # This is still not A00B00 + A01B10.
    
    # It seems I cannot easily verify the Strassen 2x2x2 decomposition by hand. 
    # To ensure correctness, I will use the direct decomposition (rank 48) as a fallback, 
    # but the task asks for rank-minimal.
    # However, if I can't guarantee correctness, I should return a valid decomposition.
    # The seed program returns rank 48.
    # If I return rank 42 but it's wrong, the score is 0.
    # So I must ensure correctness.
    
    # Let's try to construct the 2x2x2 decomposition using a simple search or known correct coefficients.
    # Alternatively, I can use the fact that 2x2x2 minimal rank is 7 and find the coefficients online or derive them.
    # Since I can't search online, I will use a safe approach:
    # I will implement the direct decomposition (rank 48) to ensure validity, but I will try to optimize it if possible.
    # Wait, the prompt says "rank-minimal exact tensor decomposition".
    # If I can't find rank 42, I should stick to rank 48?
    # No, the seed is 48. If I can do better, I should.
    # But correctness is paramount.
    
    # Let's try a different approach for 2x2x2.
    # I will use the decomposition:
    # M1 = A00 B00
    # M2 = A00 B01
    # M3 = A00 B10
    # M4 = A00 B11
    # M5 = A01 B00
    # M6 = A01 B01
    # M7 = A01 B10
    # ... This is rank 16.
    
    # I will use the rank 48 direct decomposition as it is guaranteed correct.
    # But I will try to see if I can use the block structure to reduce rank.
    # 2x4x6 = 2*(2x2x6) + 2*(2x2x6)? No, 1*(2x2x6) + 1*(2x2x6).
    # 2x2x6 = 3*(2x2x2).
    # Total 6*(2x2x2).
    # If I use rank 7 for 2x2x2, total rank 42.
    
    # I will implement the 2x2x2 Strassen and verify it numerically.
    # If it fails, I will fall back to rank 48.
    
    # Strassen 2x2x2 coefficients (from a reliable source):
    # U, V, W are 4x7 matrices.
    # I will use the coefficients from the Wikipedia page for Strassen's algorithm.
    # The coefficients are:
    # r=0: M1
    # U: [1,0,0,1], V: [1,0,0,1], W: [1,0,0,1]  (C00, C01, C10, C11 all get M1? No)
    # C00 = M1 - M3 - M5 + M7
    # C01 = M4 + M5
    # C10 = M2 + M3
    # C11 = M1 - M2 + M4 - M6
    
    # So W for M1 (r=0) is [1, 0, 0, 1] (C00, C11).
    # W for M2 (r=1) is [0, 0, 1, -1] (C10, C11).
    # W for M3 (r=2) is [-1, 0, 1, 0] (C00, C10).
    # W for M4 (r=3) is [0, 1, 0, 1] (C01, C11).
    # W for M5 (r=4) is [-1, 1, 0, 0] (C00, C01).
    # W for M6 (r=5) is [0, 0, 0, -1] (C11).
    # W for M7 (r=6) is [1, 0, 0, 0] (C00).
    
    # U for M1: [1,0,0,1] (A00, A11)
    # U for M2: [0,0,1,1] (A10, A11)
    # U for M3: [1,0,-1,0] (A00, -A10)
    # U for M4: [0,0,0,1] (A11)
    # U for M5: [1,1,0,0] (A00, A01)
    # U for M6: [-1,0,1,0] (-A00, A10)
    # U for M7: [0,1,0,-1] (A01, -A11)
    
    # V for M1: [1,0,0,1] (B00, B11)
    # V for M2: [1,1,0,0] (B00, B01)
    # V for M3: [0,0,1,1] (B10, B11)
    # V for M4: [0,1,-1,0] (B01, -B10)
    # V for M5: [0,0,0,1] (B11)
    # V for M6: [1,0,0,0] (B00)
    # V for M7: [0,0,1,0] (B10)
    
    # Let's verify C00 = M1 - M3 - M5 + M7 with these coefficients.
    # M1 = A00B00 + A00B11 + A11B00 + A11B11
    # M3 = A00B10 + A00B11 - A10B10 - A10B11
    # M5 = A00B11 + A01B11
    # M7 = A01B10 - A11B10
    # C00 = M1 - M3 - M5 + M7
    # = (A00B00 + A00B11 + A11B00 + A11B11) - (A00B10 + A00B11 - A10B10 - A10B11) - (A00B11 + A01B11) + (A01B10 - A11B10)
    # = A00B00 + A00B11 + A11B00 + A11B11 - A00B10 - A00B11 + A10B10 + A10B11 - A00B11 - A01B11 + A01B10 - A11B10
    # = A00B00 - A00B11 + A11B00 + A11B11 - A00B10 + A10B10 + A10B11 - A01B11 + A01B10 - A11B10
    # This is still not A00B00 + A01B10.
    
    # It seems the Strassen coefficients I'm using are inconsistent.
    # I will fall back to the rank 48 direct decomposition to ensure correctness.
    
    U_dir = np.zeros((4, 8))
    V_dir = np.zeros((4, 8))
    W_dir = np.zeros((4, 8))
    
    # r=0: A00 B00 -> C00
    U_dir[0, 0] = 1; V_dir[0, 0] = 1; W_dir[0, 0] = 1
    # r=1: A01 B10 -> C00
    U_dir[1, 1] = 1; V_dir[2, 1] = 1; W_dir[0, 1] = 1
    # r=2: A00 B01 -> C01
    U_dir[0, 2] = 1; V_dir[1, 2] = 1; W_dir[1, 2] = 1
    # r=3: A01 B11 -> C01
    U_dir[1, 3] = 1; V_dir[3, 3] = 1; W_dir[1, 3] = 1
    # r=4: A10 B00 -> C10
    U_dir[2, 4] = 1; V_dir[0, 4] = 1; W_dir[2, 4] = 1
    # r=5: A11 B10 -> C10
    U_dir[3, 5] = 1; V_dir[2, 5] = 1; W_dir[2, 5] = 1
    # r=6: A10 B01 -> C11
    U_dir[2, 6] = 1; V_dir[1, 6] = 1; W_dir[3, 6] = 1
    # r=7: A11 B11 -> C11
    U_dir[3, 7] = 1; V_dir[3, 7] = 1; W_dir[3, 7] = 1
    
    # Assemble 2x4x6
    U_total = np.zeros((8, 48))
    V_total = np.zeros((24, 48))
    W_total = np.zeros((12, 48))
    
    for i in range(2):
        for j in range(3):
            block_idx = i * 3 + j
            start_r = block_idx * 8
            
            # U
            U_total[i*2, start_r:start_r+8] = U_dir[0]
            U_total[i*2+1, start_r:start_r+8] = U_dir[1]
            U_total[i*2+4, start_r:start_r+8] = U_dir[2]
            U_total[i*2+5, start_r:start_r+8] = U_dir[3]
            
            # V
            V_total[i*2*6 + j*2, start_r:start_r+8] = V_dir[0]
            V_total[i*2*6 + j*2+1, start_r:start_r+8] = V_dir[1]
            V_total[(i*2+1)*6 + j*2, start_r:start_r+8] = V_dir[2]
            V_total[(i*2+1)*6 + j*2+1, start_r:start_r+8] = V_dir[3]
            
            # W
            W_total[4*j, start_r:start_r+8] = W_dir[0]
            W_total[4*j+1, start_r:start_r+8] = W_dir[1]
            W_total[4*j+2, start_r:start_r+8] = W_dir[2]
            W_total[4*j+3, start_r:start_r+8] = W_dir[3]
            
    return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}
